"""Everything the new runs share: paths, the two missions, the protocol and the model.

One combined model is used for BOTH missions. Its shared part (TCN, attention, graph layers,
command encoder, prediction head) has identical shapes on M1 and M2, so it can be federated
later; each mission keeps its own channel / command embeddings, thresholds and detection rule.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Channel set: "light" (ESA's lightweight subsets) or "full" (every channel). The full bundle
# ships a SET file containing "full"; NR_SET overrides it.
_set_file = f"{ROOT}/SET"
SET = os.environ.get("NR_SET") or (open(_set_file).read().strip() if os.path.exists(_set_file) else "light")
assert SET in ("light", "full", "sub"), SET
# "sub": one model per subsystem (core/subsets.json; NR_SUB names the model). Reads the full-set
# data (only its own channels), writes out_sub/<model>/; command caches are shared per mission.
SUB = os.environ.get("NR_SUB", "") if SET == "sub" else ""
assert SET != "sub" or SUB, "NR_SET=sub needs NR_SUB=<model in core/subsets.json>"
DATA = os.environ.get("NR_DATA", f"{ROOT}/data" if SET == "light" else f"{ROOT}/data_full")
OUT = os.environ.get("NR_OUT", {"light": f"{ROOT}/out", "full": f"{ROOT}/out_full",
                                "sub": f"{ROOT}/out_sub/{SUB}"}[SET])
CACHE = os.environ.get("NR_CACHE", f"{ROOT}/out_sub/cache" if SET == "sub" else f"{OUT}/cache")
SMOKE_ROWS = int(os.environ.get("NR_SMOKE_ROWS", 0))   # plumbing test: last N train / first N test rows
# transfer learning / onboarding: train on only the first N days of the mission (the test split
# is unchanged), optionally starting from another model's shared weights (embeddings stay new)
TRAIN_DAYS = float(os.environ.get("NR_TRAIN_DAYS", 0))
INIT = os.environ.get("NR_INIT", "")
TRAIN_FROM_END = os.environ.get("NR_TRAIN_FROM_END", "") == "1"   # the LAST TRAIN_DAYS (just before deployment)
FREEZE = os.environ.get("NR_FREEZE", "")                          # "tcn": transferred TCN kept fixed
VENDOR = f"{ROOT}/vendor"                      # ESA's metric package + pure-Python deps
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

TC_TAU = 1800.0     # command recency = exp(-seconds since firing / 1800)
LATENCY = 70        # max rows any decision may look ahead (ESA Telemanom's batch size)

# Selection folds: a model trained on everything before `start` predicts [start, end), which
# it has never seen. Settings are chosen on fold A and fold B together (out-of-fold).
_LIGHT = {"M1": range(41, 47), "M2": range(18, 29)}
_FULL_N = {"M1": 76, "M2": 100}


def _targets(raw, chs):
    """ESA scores only Target=YES channels (full set: M1 58 of 76, M2 47 of 100)."""
    import csv
    f = f"{raw}/channels.csv"
    if not os.path.exists(f):
        return list(chs)
    yes = {r["Channel"] for r in csv.DictReader(open(f)) if r.get("Target", "").strip().upper() == "YES"}
    return [c for c in chs if c in yes]


SUBSETS = {}
if SET == "sub":
    import json as _json
    SUBSETS = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "subsets.json")))
    assert SUB in SUBSETS, f"unknown model {SUB}; choose from {list(SUBSETS)}"


def _num(c):
    return int(c.split("_")[1])


MISSIONS = {}
for _mk, _grid, _folds in (("M1", 30, {"A": ("2003-01-01", "2005-01-01"), "B": ("2005-01-01", "2007-01-01")}),
                           ("M2", 18, {"A": ("2000-10-01", "2001-04-01"), "B": ("2001-04-01", "2001-10-01")})):
    if SET == "sub" and SUBSETS[SUB]["mission"] != _mk:
        continue
    _raw = f"{DATA}/raw/ESA-Mission{_mk[1]}"
    if SET == "sub":       # targets + context, in channel order; only the targets are scored
        _chs = sorted(SUBSETS[SUB]["targets"] + SUBSETS[SUB]["context"], key=_num)
    else:
        _chs = [f"channel_{i}" for i in (_LIGHT[_mk] if SET == "light" else range(1, _FULL_N[_mk] + 1))]
    # subsystems: scored = this model's own targets (a context channel may be another
    # subsystem's target; it is an input here and scored by its own model)
    _tg = _targets(_raw, _chs) if SET != "sub" else [c for c in _chs if c in SUBSETS[SUB]["targets"]]
    MISSIONS[_mk] = dict(
        raw=_raw, chs=_chs, targets=_tg, grid=_grid, folds=_folds,
        # full set / subsystems only (the lightweight runs stay exactly as tested):
        sd_floor=float(os.environ.get("NR_SD_FLOOR", 0.0 if SET == "light" else 1e-3)),   # near-constant channels
        silence_mult=float(os.environ.get("NR_SILENCE_MULT", 0 if SET == "light" else 6)),  # per-channel cadence
        # subsystems: forecast loss on target channels only (context channels are inputs)
        loss_targets_only=SET == "sub",
        # subsystems: STD-k off on channels with training std < sd_floor (the near-constant-
        # channel bug of the full run; see core/evaluate_fix.py)
        std_floor_fix=SET == "sub",
    )
if SET == "sub":
    _m = MISSIONS[SUBSETS[SUB]["mission"]]
    assert _m["targets"] == sorted(SUBSETS[SUB]["targets"], key=_num), "model targets not in its channels"
    assert set(_m["targets"]) <= set(_targets(_m["raw"], _m["chs"])), "a model target is not Target=YES"
if os.environ.get("NR_TEST_DROP_TARGETS"):          # smoke-test hook: exercise target subsetting
    for _m in MISSIONS.values():
        _m["targets"] = _m["targets"][:-int(os.environ["NR_TEST_DROP_TARGETS"])]
SEARCH = os.environ.get("NR_SEARCH", "coordinate" if SET in ("light", "sub") else "steepest")

# The combined model (all agreed changes). Shared shapes are identical for M1 and M2.
MODEL = dict(width=64, layers=8, act="gelu", dropout=0.1,      # 64 filters, 8.5 h look-back
             att="gatv2", graph="dir_anti", dynamic=True,      # GATv2 + STGLR one-way + dynamic
             tc_mode="path")                                   # telecommands' own path
TRAIN = dict(tc_min_priority=3 if SET == "sub" else 2,   # >= 2: M1 30, M2 123; subsystems >= 3 (ESA's filter): M1 11, M2 4
             mask_labels=True,       # labelled training rows left out of the loss (ESA R4)
             epochs=30, batches=300, patience=8, lr=1e-3)
SMOKE_TRAIN = dict(epochs=1, batches=6, patience=1)            # --smoke: plumbing test only
if os.environ.get("NR_SPARSE"):                                # test hook: force the attention path
    MODEL["sparse"] = os.environ["NR_SPARSE"] == "1"


def job_dir(mk, job, seed):
    """out/<mission>/<job>_s<seed>: model.pt, err_*.npy, meta.json, train.log, progress.json"""
    return f"{OUT}/{mk}/{job}_s{seed}"
