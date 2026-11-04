"""Same test evaluation as core.evaluate, with the 7 scorings run in parallel (one process
each) instead of one after another on a single core. Our 3 scorings are printed as soon as
they finish; the fixed-limit baselines get --ool-wait minutes, then are recorded as nan.

  python -m core.evaluate_par --mission M1 [--ool-wait 20]
"""
import argparse
import contextlib
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context

import numpy as np

from .config import OUT, MISSIONS, job_dir
from .data import load_split, label_mask, fire_seconds, target_idx
from . import detect as D
from . import evaluate as E
from .period import Period, assemble, spans_of, tele_key
from .esa import ESAScores, ChannelAwareFScore, ADTQC, BETA, series
from .score import Scorer, SEL, SEL_ANOM, SHOW, fmt, add_aspects

_S = {}


PARTS = ("global", "chan", "adtqc")
ZERO = {"global_EW_F_0.50": 0.0, "global_EW_precision": 0.0, "global_EW_recall": 0.0,
        "global_alarming_precision": 0.0, "global_AFF_F_0.50": 0.0, "chan_channel_F0.50": 0.0,
        "chan_subsystem_F0.50": 0.0, "adtqc_Total": 0.0}


def score_part(S, P, sel, part):
    """One of the three metric groups of Scorer.score (same calls, same order of keys)."""
    y = S.y
    with contextlib.redirect_stdout(io.StringIO()):
        if part == "global":
            return {f"global_{k}": float(v) for k, v in ESAScores(betas=BETA, select_labels=sel).score(
                y.drop(columns=["Channel"]), series(S.ts, P.max(axis=1))).items()}
        yp = {c: series(S.ts, P[:, i]) for i, c in enumerate(S.chs)}
        if part == "chan":
            try:
                return {f"chan_{k}": float(v) for k, v in ChannelAwareFScore(beta=BETA, select_labels=sel).score(
                    y.copy(), yp, S.sub).items()}
            except Exception:
                return {"chan_channel_F0.50": 0.0, "chan_subsystem_F0.50": 0.0}
        try:
            return {f"adtqc_{k}": float(v) for k, v in ADTQC(select_labels=sel).score(y.copy(), yp).items()}
        except Exception:
            return {"adtqc_Total": 0.0}


def _part(a):
    name, part = a
    mask, sel = _S["tasks"][name]
    return name, part, score_part(_S["S"], mask, sel, part)


def merge(S, P, sel, parts):
    """Combine the three groups as Scorer.score does (incl. its two early exits)."""
    if not S.y.Category.isin(sel["Category"]).any():
        return {"S5": float("nan")}
    if not P.any():
        return add_aspects(dict(ZERO))
    out = {}
    for p in PARTS:
        out.update(parts[p])
    return add_aspects(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", required=True, choices=list(MISSIONS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() or 1, 1))
    ap.add_argument("--ool-wait", type=float, default=20, dest="ool_wait")
    a = ap.parse_args()
    mk = a.mission
    t0 = time.time()
    frz = json.load(open(f"{OUT}/{mk}/frozen.json"))
    cfg = frz["detector"]
    seeds = [int(s) for s in a.seeds.split(",")]
    runs = [job_dir(mk, "final", s) for s in seeds]
    metas = [json.load(open(f"{r}/meta.json")) for r in runs]
    m0 = metas[0]
    etr = np.mean([np.load(f"{r}/err_train.npy").astype(np.float32) for r in runs], axis=0)
    ete = np.mean([np.load(f"{r}/err_test.npy").astype(np.float32) for r in runs], axis=0)
    tr, te = load_split(mk, "train"), load_split(mk, "test")
    tix = target_idx(mk)
    assert etr.shape[1] == len(tix), "error file / target channels mismatch"
    sub = lambda x: np.asarray(x)[:, tix]                  # noqa: E731
    trv, trc, tev, tec = sub(tr["vals"]), sub(tr["count"]), sub(te["vals"]), sub(te["count"])
    cat = lambda x, y: np.concatenate([np.asarray(x[:-1]), np.asarray(y)])   # noqa: E731
    ts = cat(tr["ts"], te["ts"])
    fit_end = len(tr["ts"]) - 1
    P = Period(mk, "test", cat(etr, ete), cat(trv, tev), cat(trc, tec), ts,
               np.asarray(m0["mu"])[tix], np.asarray(m0["sd"])[tix], m0["rf"], fit_end,
               fit_end, len(ts))
    del etr, ete
    if cfg.get("drift"):
        P.build_drift((cfg["drift"][0],))
    if cfg["thr"] != "off":                 # baselines built one span at a time (memory)
        P.zs[(cfg["W"], cfg["spans"])] = P.z_and_thresholds(cfg["W"], cfg["spans"], lazy=True)
    if cfg.get("tele"):
        k = tele_key(cfg["tele"])
        E._G["P"] = P
        with ProcessPoolExecutor(min(a.workers, P.C), mp_context=get_context("fork")) as pool:
            for c, res in pool.map(E._tele_task, [(c, k[1], {k}) for c in range(P.C)]):
                P.tele.setdefault(k, np.zeros((P.n, P.C), np.uint8))[:, c] = res[k]
    P.release_history()                     # scoring needs only the stored arrays
    print(f"[{mk}] test period ready: {P.n:,} rows ({time.time() - t0:.0f}s)", flush=True)

    S = Scorer(mk, ts[P.e0:P.e1])
    X, parts = assemble(P, cfg, parts=True)
    Xt, n_cmd = X, 0
    if frz["typing"] != "none":
        tw, tp = map(int, frz["typing"].split("_"))
        Xt, n_cmd = D.typed(X, ts[P.e0:P.e1].astype(np.int64) / 1e9, fire_seconds(mk, tp), tw)
    nominal = ~label_mask(mk, tr["ts"], chs=MISSIONS[mk]["targets"])
    masks = {}
    for name, q in (("minmax", None), ("p0.1", 0.001)):
        lo, hi = D.ool_limits(trv, trc, nominal, q)
        masks[name] = D.ool_mask(tev, tec, lo, hi)
    _S.update(S=S, tasks={"head": (X, SEL), "anom": (X, SEL_ANOM), "typed": (Xt, SEL_ANOM),
                          "minmax_h": (masks["minmax"], SEL), "minmax_a": (masks["minmax"], SEL_ANOM),
                          "p0.1_h": (masks["p0.1"], SEL), "p0.1_a": (masks["p0.1"], SEL_ANOM)})
    names = list(_S["tasks"]) if frz["typing"] != "none" else [n for n in _S["tasks"] if n != "typed"]
    ours = {"head", "anom", "typed"} & set(names)
    # trivial cases (no selected labels / no alarms) need no scoring
    jobs = [(n, p) for n in names for p in PARTS
            if S.y.Category.isin(_S["tasks"][n][1]["Category"]).any() and _S["tasks"][n][0].any()]
    jobs.sort(key=lambda j: j[0] not in ours)               # our scores first
    got_parts = {n: {} for n in names}
    got = {}
    t1, deadline, beat = time.time(), None, time.time()

    def finish(n):
        got[n] = merge(S, _S["tasks"][n][0], _S["tasks"][n][1], got_parts[n])
        print(f"[{mk}] scored {n:9s} ({time.time() - t1:.0f}s): {fmt(got[n])}", flush=True)

    for n in names:                                          # the trivial ones
        if not any(j[0] == n for j in jobs):
            finish(n)
    pool = ProcessPoolExecutor(max(min(len(jobs), a.workers), 1), mp_context=get_context("fork"))
    futs = {pool.submit(_part, j): j for j in jobs}
    pending = set(futs)
    while pending:
        done, pending = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
        for f in done:
            n, p, r = f.result()
            got_parts[n][p] = r
            if len(got_parts[n]) == len(PARTS):
                finish(n)
        if deadline is None and ours <= set(got):
            json.dump({k: got[k] for k in ours}, open(f"{OUT}/{mk}/result_ours.json", "w"),
                      indent=1, default=float)
            deadline = time.time() + a.ool_wait * 60
            print(f"[{mk}] our scores saved -> {OUT}/{mk}/result_ours.json; waiting <= "
                  f"{a.ool_wait:.0f} min for the fixed-limit baselines", flush=True)
        if time.time() - beat > 60 and pending:
            beat = time.time()
            print(f"[{mk}] still scoring ({time.time() - t1:.0f}s): "
                  + ", ".join(sorted(f"{futs[f][0]}/{futs[f][1]}" for f in pending)), flush=True)
        if deadline is not None and time.time() > deadline and pending:
            print(f"[{mk}] baselines not finished in time, recorded as nan: "
                  f"{sorted({futs[f][0] for f in pending})}", flush=True)
            for pr in list(pool._processes.values()):
                pr.kill()
            break
    pool.shutdown(wait=False, cancel_futures=True)
    nan = {k: float("nan") for k in SHOW}
    head, anom = got["head"], got["anom"]
    typed = got.get("typed", anom)
    ool = {name: dict(headline=got.get(f"{name}_h", nan), anom=got.get(f"{name}_a", nan),
                      mask=masks[name], alarm_time=float(masks[name].max(1).mean()))
           for name in masks}

    cats = ("Anomaly", "Rare Event")
    tt = ts[P.e0:P.e1]
    caught_ours = E.caught(X, S, tt, cats)
    by = {k: E.caught(v, S, tt, cats) for k, v in parts.items()}
    oo = E.caught(ool["minmax"]["mask"], S, tt, cats)
    catof = S.y.drop_duplicates("ID").set_index("ID").Category.to_dict()
    ev = {}
    for c in cats:
        ids = [i for i in caught_ours if catof[i] == c]
        ev[c] = dict(n=len(ids), ours=sum(caught_ours[i] for i in ids), ool=sum(oo[i] for i in ids),
                     ours_not_ool=sum(caught_ours[i] and not oo[i] for i in ids),
                     ool_not_ours=sum(oo[i] and not caught_ours[i] for i in ids),
                     by_detector={k: sum(v[i] for i in ids) for k, v in by.items()})
    res = dict(mission=mk, seeds=seeds, detector=cfg, typing=frz["typing"],
               headline=head, anom=anom, anom_typed=typed, n_commanded_alarms=n_cmd,
               ool={k: dict(headline=v["headline"], anom=v["anom"], alarm_time=v["alarm_time"])
                    for k, v in ool.items()},
               events=ev, n_alarm_rows=int(X.max(1).sum()), alarm_time=float(X.max(1).mean()),
               test_rows=P.n,
               model=dict(n_params=m0["n_params"], rf=m0["rf"], epochs=[m["epochs"] for m in metas],
                          train_s=[m["train_s"] for m in metas], device=m0["device"]),
               secs=time.time() - t0, evaluator="evaluate_par")
    json.dump(res, open(f"{OUT}/{mk}/result.json", "w"), indent=1, default=float)
    # per-channel alarms (headline and typed), for the mission-level score over all subsystems
    np.savez_compressed(f"{OUT}/{mk}/alarms_test_mask.npz", X=X.astype(np.uint8),
                        Xt=np.asarray(Xt, np.uint8), chs=np.array(S.chs))
    E.write_tables(mk, S, tt, X, parts, ool["minmax"]["mask"],
                   Xt if frz["typing"] != "none" else None)
    print(f"[{mk}] TEST  headline   {fmt(head)}")
    print(f"[{mk}] TEST  anomaly    {fmt(anom)}")
    print(f"[{mk}] TEST  uncommanded {fmt(typed)}")
    for c, e in ev.items():
        print(f"[{mk}] {c}: {e['ours']}/{e['n']} caught (out-of-limit: {e['ool']}; ours-only "
              f"{e['ours_not_ool']}; by detector {e['by_detector']})")
    print(f"[{mk}] done ({time.time() - t0:.0f}s) -> {OUT}/{mk}/result.json", flush=True)


if __name__ == "__main__":
    main()
