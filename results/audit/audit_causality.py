"""Full causality audit of the final run. Read-only on data and models; prints evidence only.
  python audit_causality.py    (timeeval env, FED_OUT=fed/out_final)
Rule tested: the prediction for row t may use rows < t only; alarms may look ahead at most 70 rows (ESA rule).
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.environ.setdefault("FED_OUT", os.path.join(ROOT, "fed", "out_final"))
sys.path.insert(0, os.path.join(ROOT, "fed"))
sys.argv = sys.argv[:1]
import numpy as np, pandas as pd, torch
import fedsim as F
from core.config import LATENCY, MODEL
from core.model import GDN2
from core.train import seg_bs, make_batch, series_errors
from core.features import CmdFeats
from core.detect import runs_of
from core.period import assemble as ORIG_ASSEMBLE
import config as C
torch.set_num_threads(4)
SATS, dev = F.SATS, torch.device("cpu")
RPD = 2880
day = lambda r: r / RPD
T0 = time.time()
def say(*a): print(f"[{(time.time()-T0)/60:5.1f}m]", *a, flush=True)
RES = json.load(open(f"{F.OUT}/results.json"))
D = {s: F.inputs(s) for s in SATS}
rf = GDN2(20, 0, 12, **MODEL).rf
seg, bs = seg_bs(rf)
say("rf", rf, "seg", seg, "bs", bs, "es", D["SAT_1"]["es"], "v0", D["SAT_1"]["v0"], "ntr", D["SAT_1"]["ntr"], "rows", len(D["SAT_1"]["vals"]))

# ---- 1. every training batch replayed
say("== 1 BATCH REPLAY: every batch of the final run")
EP = {"SAT_1": 12, "SAT_2": 10, "SAT_3": 12, "SAT_4": 12}
tot, mx = 0, 0
for i, s in enumerate(SATS):
    d = D[s]; starts = np.arange(d["a"], d["es"] - seg - 1)
    jobs = [("solo", ep, 150) for ep in range(EP[s])] + [("fed", 1000 * r + i, 150) for r in range(12)] + [("pers", 777 + k, 25) for k in range(4)]
    m = 0
    for name, seed, n in jobs:
        rng = np.random.default_rng(seed)
        for b in range(n):
            st = rng.choice(starts, bs, replace=False)
            m = max(m, int(st.max()) + seg - 1); tot += 1
            if s == "SAT_1" and name == "fed" and seed == 0 and b == 0:
                say("  SAT_1 round 1 batch 1 windows start day:", np.round(np.sort(st) / RPD, 1).tolist(), "each", seg, "rows =", round(seg * 30 / 3600, 1), "h")
    say(f"  {s}: last row touched by any training batch {m} (day {day(m):.2f}); validation starts {d['v0']} (day {day(d['v0']):.1f}); test starts {d['ntr']} (day {day(d['ntr']):.1f})")
    assert m < d["es"] <= d["v0"] < d["ntr"]
    va = np.linspace(d["es"], d["v0"] - seg - 1, 80).astype(int)
    assert va.min() >= d["es"] and va.max() + seg - 1 < d["v0"]
    r2 = d["vals"][:d["v0"]]
    assert np.array_equal(d["mu"], r2.mean(0)) and np.allclose(d["sd"], r2.std(0) + 1e-3)
say("  batches replayed:", tot, "| best-round check rows inside [es, v0) | mu/sd equal rows before v0: PASS")

# ---- 2. dataset rows come only from samples at or before the row time
say("== 2 RESAMPLING: each grid value vs raw telemetry; count and time-since from truncated raw")
q = {c[0]: c[8] for c in C.CHANNELS}
for s in SATS:
    d = D[s]; g = d["ts"].astype("datetime64[ns]").astype(np.int64) / 1e9
    cut = 250 * RPD; bad = badc = 0; nchk = 0
    for j, c in enumerate(d["chs"]):
        raw = pd.read_csv(f"{d['raw']}/channels/{c}.csv.gz")
        t = pd.to_datetime(raw.timestamp).dt.tz_localize(None).values.astype("datetime64[ns]").astype(np.int64) / 1e9
        v = raw.value.values
        i = np.searchsorted(t, g, "right") - 1
        ok = i >= 0
        tol = 0.51 * q[d["names"][c]] + 1e-5 * np.abs(v[i[ok]])
        bad += int((np.abs(d["vals"][ok, j] - v[i[ok]]) > tol).sum()); nchk += int(ok.sum())
        tc = t[t <= g[cut - 1]]                                   # raw file cut at day 250
        hi = np.searchsorted(tc, g[:cut], "right"); lo = np.searchsorted(tc, g[:cut] - 30.0, "right")
        badc += int(((hi - lo) != d["cnt"][:cut, j]).sum())
        ts_ = np.where(hi > 0, g[:cut] - tc[np.clip(hi - 1, 0, len(tc) - 1)], 1e6)
        badc += int((np.abs(ts_ - d["tsince"][:cut, j]) > 1e-2).sum())
    say(f"  {s}: {nchk:,} values checked, mismatches {bad}; count/time-since differences with raw cut at day 250: {badc}")

# ---- 3. model: gradient test + future replaced/removed
say("== 3 MODEL")
def load(arm, s):
    m = F.new_model(D[s], dev)
    st = torch.load(f"{F.OUT}/models/fed_state.pt", map_location="cpu")[s] if arm == "fed" else torch.load(f"{F.OUT}/models/{arm}_{s}.pt", map_location="cpu")
    m.load_state_dict(st); return m
for arm in ("solo", "fed", "pers"):
    for s in SATS:
        d = D[s]; m = load(arm, s)
        # gradient test, dropout on, as in training
        m.train()
        xs, ys, ws, cf = make_batch(d["X"], d["cmd"], d["W"], np.array([100000]), seg, rf, d["X"].shape[1])
        leak = 0.0; prev = []
        for p in (0, 500, seg - rf - 1):
            x = torch.from_numpy(xs).clone().requires_grad_(True); c = torch.from_numpy(cf).clone().requires_grad_(True)
            out = m.forward_segments(x, c)
            out[:, p, :].sum().backward()
            leak = max(leak, float(x.grad[:, :, rf + p:].abs().max()))           # the target row and every later row
            oth = c.grad.clone(); oth[:, p] = 0; leak = max(leak, float(oth.abs().max()))
            prev.append(float(x.grad[:, :, rf + p - 1].abs().max()))
        m.eval()
        ev = pd.read_csv(f"{d['raw']}/truth/events_truth.csv"); ev = ev[(ev.Split == "test") & ev.Visible & (ev.Category == "Anomaly")]
        fv = np.datetime64(pd.Timestamp(ev.FirstVisible.iloc[0]).tz_localize(None))
        n = len(d["vals"]); rows = [int(np.searchsorted(d["ts"], fv)), d["ntr"] + (n - d["ntr"]) // 2, n - 9 * RPD]
        worst = 0.0; stored = 0.0; chg = 0.0
        E = np.load(f"{F.OUT}/errors/{arm}_{s}.npy", mmap_mode="r")
        for T in rows:
            lo = T - 1500 - rf - 2; hi = T + 5
            Xs = np.ascontiguousarray(d["X"][lo:hi]); tt = d["ts"][lo:hi]
            full = series_errors(m, Xs, CmdFeats(tt, d["cmds"]), dev, rf + 1, len(Xs))
            rng = np.random.default_rng(T)
            X2 = Xs.copy(); X2[T - lo:] = rng.normal(0, 50, X2[T - lo:].shape)
            fake = [(c_, p_, np.sort(np.r_[x[x < d["ts"][T]], d["ts"][T] + (rng.integers(0, 900, 5)).astype("timedelta64[s]")])) for c_, p_, x in d["cmds"]]
            pert = series_errors(m, X2, CmdFeats(tt, fake), dev, rf + 1, len(Xs))
            X3 = np.ascontiguousarray(Xs[:T - lo]); cutc = [(c_, p_, x[x < d["ts"][T]]) for c_, p_, x in d["cmds"]]
            trunc = series_errors(m, X3, CmdFeats(tt[:T - lo], cutc), dev, rf + 1, len(X3))
            k = T - lo - (rf + 1)
            worst = max(worst, float(np.abs(full[:k] - pert[:k]).max()), float(np.abs(full[:k] - trunc).max()))
            chg = max(chg, float(np.abs(full[k:] - pert[k:]).max()))
            ref = np.asarray(E[lo + rf + 1:T], np.float32); a = full[:k][:, d["tix"]]
            stored = max(stored, float(np.median(np.abs(ref - a) / (np.abs(a) + 1e-2))))
        say(f"  {arm:4s} {s}: gradient into target row or later {leak:.1e} (into last past row {min(prev):.1e}); error change before cut when future altered/deleted {worst:.1e}; change at/after cut {chg:.1f}; median gap to stored errors {stored:.1e}")

# ---- 4,5,6 detector
def on_alarm(P, cfg):
    """Zero-look-ahead version: an alarm starts only once data up to that row confirms it."""
    cap = {}
    F.assemble = lambda P_, c_: (lambda r: (cap.update(comp=r[1], c=c_), r[0])[1])(ORIG_ASSEMBLE(P_, c_, parts=True))
    A = F.alarms(P, cfg); F.assemble = ORIG_ASSEMBLE
    core = cap["comp"]["B"].astype(bool); close, minlen = cfg["close"], cfg["minlen"]
    return A, online(core, close, minlen, P.cmask(cfg["mask"], cfg["smask"]) if (cfg["mask"] or cfg["smask"]) else None)
def online(core, close, minlen, cm):
    n, Cc = core.shape; out = np.zeros((n, Cc), np.uint8)
    for c in range(Cc):
        idx = np.flatnonzero(core[:, c])
        if not len(idx): continue
        for g in np.split(idx, np.flatnonzero(np.diff(idx) - 1 > close) + 1):
            conf = g[g - g[0] + 1 >= minlen]
            if len(conf): out[conf[0]:min(g[-1] + 1 + close, n), c] = 1
    if cm is not None: out[cm] = 0
    return out
def summ(d, per, A, tf):
    P, ts, y = per
    a = F.B.score(d["raw"], ts, A, d["tg"], F.B.SEL)["global_EW_F_0.50"]; b = F.B.score(d["raw"], ts, A, d["tg"], F.B.SEL_ANOM).get("global_EW_F_0.50", np.nan)
    ev = F.event_table(d, A, ts); fa = F.false_alarms(ts, A, y)
    return a, b, int(ev.detected.sum()), fa["n_false_alarm_events"], float(ev.delay_min.median())
say("== 4 DETECTOR: cuts placed inside alarms, all arms and both selections")
say("== 6 STRICT RESCORING is printed per arm below (diagnostic only)")
strict = {}
for arm in ("solo", "fed", "pers"):
    PER = F.load_periods(D, arm, rf); peers = [PER[s]["val"][0] for s in SATS]; rng = np.random.default_rng(7)
    for s in SATS:
        d = D[s]; E = np.load(f"{F.OUT}/errors/{arm}_{s}.npy"); tf = RES[arm][s]["train_families"]
        cf = {k: RES[arm][s][k]["cfg"] for k in F.SELS}
        fullA = {k: F.alarms(PER[s]["test"][0], cf[k]) for k in F.SELS}
        for k in F.SELS:
            A, Aon = on_alarm(PER[s]["test"][0], cf[k])
            assert np.array_equal(A, fullA[k])
            Adel = np.zeros_like(A); Adel[LATENCY:] = A[:-LATENCY]
            strict[(arm, s, k)] = (summ(d, PER[s]["test"], A, tf), summ(d, PER[s]["test"], Aon, tf), summ(d, PER[s]["test"], Adel, tf))
            say(f"  {arm:4s} {s} {k:9s} reported {np.round(strict[(arm, s, k)][0], 3).tolist()} zero-look-ahead {np.round(strict[(arm, s, k)][1], 3).tolist()} delayed-70 {np.round(strict[(arm, s, k)][2], 3).tolist()}")
        r = runs_of(fullA["federated"].any(1)); r = r[(r[:, 0] > 2000) & (r[:, 0] < len(fullA["federated"]) - 200)]
        picks = r[rng.choice(len(r), min(2, len(r)), replace=False)] if len(r) else []
        cuts = [int(a + 3) for a, b in picks] + [int(a + (b - a) // 2) for a, b in picks[:1]]
        for cut in cuts:
            P = F.periods(d, E[:d["ntr"] + cut], rf)["test"][0]; P.peers = peers; P.ref = PER[s]["val"][0]
            for k in F.SELS:
                A = F.alarms(P, cf[k]); safe = cut - LATENCY
                dif = np.flatnonzero((A[:cut] != fullA[k][:cut]).any(1))
                before = int((dif < safe).sum()); reach = int(cut - dif.min()) if len(dif) else 0
                say(f"  {arm:4s} {s} {k:9s} cut {cut}: rows differing before limit {before}; farthest look-ahead {reach} rows")
                assert before == 0
        del E
    del PER
agg = {}
for (arm, s, k), v in strict.items(): agg.setdefault((arm, k), []).append(v)
for (arm, k), v in agg.items():
    for i, nm in enumerate(("reported", "zero-look-ahead", "delayed-70")):
        a = [x[i] for x in v]
        say(f"  MEAN {arm:4s} {k:9s} {nm:15s} all {np.mean([x[0] for x in a]):.3f} anom {np.nanmean([x[1] for x in a]):.3f} caught {sum(x[2] for x in a)}/59 FA {sum(x[3] for x in a)} median delay min {np.nanmedian([x[4] for x in a]):.1f}")

say("== 5 TEST LABELS SCRAMBLED: settings and alarms must not change")
orig = F.build_labels_df; t0 = D["SAT_1"]["ts"][D["SAT_1"]["ntr"]]
def scr(raw, ts):
    lab = orig(raw, ts)
    if ts.iloc[0] >= t0:
        sh = np.timedelta64(40, "D"); lab = lab.copy(); lab["StartTime"] = lab.StartTime + sh; lab["EndTime"] = lab.EndTime + sh
    return lab
F.build_labels_df = scr
for arm in ("pers", "solo"):
    PER = F.load_periods(D, arm, rf)
    fc, _ = F.search([PER[s]["val"] for s in SATS], axes=dict(F.AXES, ceil=["own", "fleet"]))
    same = fc == RES[arm]["fed_selection"]["cfg"]; ls = []
    for s in SATS:
        lc, _ = F.search([PER[s]["val"]]); ls.append(lc == RES[arm][s]["local"]["cfg"])
    say(f"  {arm}: fleet settings unchanged {same}; own settings unchanged {ls}")
    del PER
F.build_labels_df = orig

say("== 7 RUL: rerun with data cut at each day")
import rul as R
raw = f"{F.ROOT if hasattr(F,'ROOT') else ROOT}/data/sim_fleet/SAT_2"
ratio = R.daily_ratio(raw); H, Z, _ = R.hi(ratio, R.REF_DAYS["SAT"]); res = json.load(open(f"{F.OUT}/rul.json"))
pred = pd.read_csv(f"{F.OUT}/rul_predictions.csv"); pe = pred[pred.kind == "exp"].set_index("day")
a_day, th, w = res["first_alert_day"], res["failure_threshold"], res["alert_wheel"]; dmax = 0.0; n = 0
for t in range(a_day, 356):
    Ht, Zt, _ = R.hi(ratio.iloc[:t + 1], R.REF_DAYS["SAT"])
    lo = max(a_day - 10, t - 30); days = np.arange(t + 1)
    p = R.fit_rul(Ht[w].values[lo:t + 1], days[lo:t + 1], th, "exp")
    if np.isfinite(p) and np.isfinite(pe.loc[t, "pred_rul"]): dmax = max(dmax, abs(p - pe.loc[t, "pred_rul"])); n += 1
    assert np.allclose(Ht.values, H.iloc[:t + 1].values, equal_nan=True)
Zs = Z.rolling(7, min_periods=3).median(); run3 = ((Zs > 6).astype(int).rolling(3).sum() >= 3)[w]
say(f"  {n} daily predictions rerun with later days deleted: max change {dmax:.2e}; first alert day recomputed {int(np.argmax(run3.values))} vs reported {a_day}; last day used for a prediction 355, failure day {res['true_fail_day']:.2f}")
say("ALL ASSERTIONS PASSED")
