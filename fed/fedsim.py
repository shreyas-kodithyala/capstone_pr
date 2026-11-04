"""Basilisk fleet v2: solo vs federated vs federated + personalised GDN2, scored with ESA-ADB's metrics.

Stages (each saved to OUT and skipped when present, so the script resumes):
  inputs  per satellite: channel tensor over train + test, command features, loss mask (labelled
          rows left out of the loss, ESA R4), normalisation from the training rows only
  solo    one GDN2 per satellite, trained on its own rows [0, v0) (v0 = start of the validation window)
  fed     FedAvg of the shared layers (GDN2.shared_state) over R rounds; channel / command
          embeddings stay on each satellite; rotating leader aggregates; same number of local
          gradient steps per satellite as the solo arm
  pers    federated model + short local fine-tune of all weights on the satellite's own rows
  errors  forecast errors on every row (train + test) for each arm
  select  detector settings (ESA detection stack, core/period.py) chosen on the validation window
            local      each satellite maximises its own validation F0.5
            federated  the fleet maximises the mean validation F0.5; satellites exchange only one
                       score per candidate setting (no telemetry, no labels); each satellite still
                       applies the setting to its own error scale (past-only robust z)
  score   the test half once: ESA event-wise F0.5 (Anomaly + Rare Event, and Anomaly only),
          channel-aware F0.5, alarming precision, ADTQC, recall per fault family, families never
          seen locally, detection delay, fault-family attribution from the alarmed channels

  conda activate timeeval
  caffeinate -i nice -n 10 python fedsim.py            # all stages
"""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, (os.environ.get("NEW_RUNS") or (os.path.join(ROOT, "ext", "new_runs") if os.path.isdir(os.path.join(ROOT, "ext", "new_runs")) else os.path.join(os.path.dirname(ROOT), "ESA", "Main", "new_runs"))))
sys.path.insert(0, os.path.join(ROOT, "sim"))
import torch  # noqa: E402
from core.config import MODEL, MISSIONS  # noqa: E402
from core.model import GDN2  # noqa: E402
from core.features import CmdFeats, channel_tensor  # noqa: E402
from core.train import fit, series_errors, pick_device, Loader, loss_fn, to, seg_bs, make_batch  # noqa: E402
from core.esa import ESAScores, BETA, series, build_labels_df  # noqa: E402
from core.period import Period, assemble  # noqa: E402
from core.detect import runs_of  # noqa: E402
import config as C  # noqa: E402
import model_check as MC  # noqa: E402
import baselines as B  # noqa: E402

SATS = ["SAT_1", "SAT_2", "SAT_3", "SAT_4"]
VAL_DAYS = 45
RPD = 2880                                   # 30 s rows per day
OUT = os.environ.get("FED_OUT", os.path.join(HERE, "out_v2"))
FIT_DAYS = int(os.environ.get("FED_FIT_DAYS", 0))   # onboard memory: local rows kept before validation, 0 = all
MISSIONS["SIM"] = dict(grid=30)
TRN = dict(epochs=12, batches=150, patience=4, lr=1e-3, smoke=False)
FED = dict(rounds=12, local_batches=150, patience=4, lr=1e-3)
PERS = dict(batches=100, lr=3e-4, check=25)
START = dict(L=0, W="all", close=60, cross=0.3, drift=0, hyst=1.0, minlen=10, onset=0, spans="s15", std=0,
             thr="k2", tele=None, mask=20, smask=5, ceil="own", latch=0)
# Round 4 detector: thr "k<x>" = x times each channel's own noise ceiling (99.99th percentile of z on
# nominal validation rows, command-masked), so quiet channels are not judged by the noisiest channel's
# scale. "a<z>" (one absolute z for every channel) stays a candidate. `latch` holds an alarm on for that
# many rows after its last exceedance, so one disturbance is one alarm. It is causal; `close` fills gaps
# and so looks ahead, which is why `close` stays within the 70-row limit (CLAUDE.md section 5).
AXES = dict(thr=["k0.7", "k1", "k1.5", "k2", "k3", "k5", "a20", "a50", "a120"], W=["1m", "all"], spans=["s15", "multi"],
            std=[0, 6, 10], hyst=[1.0, 0.7], cross=[0, 0.3], close=[10, 60], minlen=[3, 10, 30],
            drift=[0, "d6", "d10"], mask=[0, 10, 20, 30], smask=[0, 5, 10], latch=[0, 120, 240])
# Command mask (round 3): for `mask` minutes after a commanded manoeuvre every channel's residual is
# ignored, and for `smask` minutes after a load switch the power channels'. The model sees that a slew was commanded, not
# its angle, so the transient cannot be forecast; round 2 traced 83-100 % of false alarms to this.
# Past-only, telecommands only, no labels. Calibrations and safe mode stay unmasked (they are events).
MANOEUVRE = ("SLEW_TARGET", "POINT_NADIR")
SWITCHING = ("PAYLOAD_ON", "PAYLOAD_OFF", "TX_ON", "TX_OFF")
SELS = ("local", "federated")
FAM_OF_CH = {}                                           # fault-family attribution from channel names
T0 = time.time()


def log(s):
    line = f"[{(time.time() - T0) / 60:7.1f} min] {s}"
    print(line, flush=True)
    open(f"{OUT}/log.txt", "a").write(line + "\n")


def group_of(name):
    if name.startswith("rw"):
        return 0
    if name.startswith("gyro") or name.startswith("att"):
        return 1
    return 2 if name.startswith("mag") else 3


def own_ceiling(R, W, sp, m, sm):
    """Per-channel noise ceiling on one satellite's validation window (labels only exclude events)."""
    k = (W, sp, m, sm)
    if k not in R.ceil:
        Z = R.zs[(W, sp)][0]
        ok = R.nom[:, None] & ~R.cmask(m, sm)
        R.ceil[k] = np.array([max(np.quantile(Z[ok[:, j], j], 0.9999), 3.0) for j in range(Z.shape[1])], np.float32)
    return R.ceil[k]


def alarms(P, cfg):
    """assemble() + calibrated thresholds + command masks + causal alarm latch."""
    c = dict(cfg)
    m, sm, la, ceil = c.pop("mask", 0), c.pop("smask", 0), c.pop("latch", 0), c.pop("ceil", "own")
    W, sp0, t = c["W"], c["spans"], c["thr"]
    if m or sm:
        c["spans"] = f"{sp0}|m{m}s{sm}"
        if (W, c["spans"]) not in P.zs:
            Z, th = P.zs[(W, sp0)]
            Zm = Z.copy()
            Zm[P.cmask(m, sm)] = 0
            P.zs[(W, c["spans"])] = (Zm, th)
    if t.startswith("k"):
        c["thr"] = f"{t}|{ceil}|m{m}s{sm}"
        thrs = P.zs[(W, c["spans"])][1]
        if c["thr"] not in thrs:
            base = own_ceiling(P.ref, W, sp0, m, sm) if ceil == "own" else \
                np.max([own_ceiling(q, W, sp0, m, sm) for q in P.peers], 0)     # 19 numbers per satellite
            thrs[c["thr"]] = np.float32(float(t[1:])) * base
    A = assemble(P, c)
    if m or sm:
        A[P.cmask(m, sm)] = 0
    if la:
        g = A.any(1)
        idx = np.arange(len(g))
        last = np.maximum.accumulate(np.where(g, idx, -10 ** 9))
        for a, b in runs_of((idx - last <= la) & ~g):
            A[a:b] = A[a - 1]
    return A


def command_mask(d, ts, gix):
    """-> f(mask, smask minutes) = (rows, channels) bool, True where a recent command explains the residual."""
    t = ts.values
    cm = {c: x for c, _, x in d["cmds"]}
    ids = {n: f"telecommand_{i + 1}" for i, n in enumerate(C.TC_NAMES)}
    memo = {}

    def rows(names, m):
        hit = np.zeros(len(t) + 1, np.int32)
        for n in names:
            x = cm[ids[n]]
            np.add.at(hit, np.searchsorted(t, x), 1)
            np.add.at(hit, np.searchsorted(t, x + np.timedelta64(int(m * 60), "s")), -1)
        return np.cumsum(hit[:-1]) > 0

    def f(m, sm):
        if (m, sm) not in memo:
            M = np.repeat(rows(MANOEUVRE, m)[:, None], len(gix), 1)
            M[:, gix == 3] |= rows(SWITCHING, sm)[:, None]
            memo[(m, sm)] = M
        return memo[(m, sm)]
    return f


def fam_of(name):
    if name.startswith("rw") or name.startswith("mag"):
        return "F1"
    if name.startswith("gyro") or name.startswith("att"):
        return "F2"
    return "F3"


# ------------------------------------------------------------------ inputs
def inputs(sat):
    p = f"{OUT}/cache/{sat}.pkl"
    if os.path.exists(p):
        d = pickle.load(open(p, "rb"))
    else:
        tr, te = MC.arrays(sat, "train"), MC.arrays(sat, "test")
        ntr = len(tr["vals"])
        v0 = ntr - VAL_DAYS * RPD
        cat = lambda k: np.concatenate([tr[k], te[k]])
        vals, cnt, tsince, ts = cat("vals"), cat("count"), cat("tsince"), cat("ts")
        a0 = max(v0 - FIT_DAYS * RPD, 0) if FIT_DAYS else 0
        mu, sd = vals[a0:v0].mean(0), vals[a0:v0].std(0) + 1e-3
        W = np.ones((len(vals), vals.shape[1]), np.uint8)
        for j, c in enumerate(tr["chs"]):
            col = f"is_anomaly_{c}"
            if col in tr["lab"]:
                W[:ntr, j] = tr["lab"][col].values == 0
                W[ntr:, j] = te["lab"][col].values == 0
        names = pd.read_csv(f"{tr['raw']}/channels.csv").set_index("Channel").Name.to_dict()
        d = dict(vals=vals, cnt=cnt, tsince=tsince, ts=ts, mu=mu, sd=sd, W=W, ntr=ntr, v0=v0, chs=tr["chs"],
                 tg=tr["tg"], cmds=tr["cmds"], raw=tr["raw"], names=names)
        pickle.dump(d, open(p, "wb"), protocol=4)
    d["raw"] = os.path.join(MC.FLEET, sat)              # the cache must never point at another fleet
    d["a"] = max(d["v0"] - FIT_DAYS * RPD, 0) if FIT_DAYS else 0
    d["es"] = d["a"] + int(0.85 * (d["v0"] - d["a"]))
    d["X"] = channel_tensor(d["vals"], d["cnt"], d["tsince"], d["mu"], d["sd"])
    d["cmd"] = CmdFeats(d["ts"], d["cmds"])
    d["tix"] = [d["chs"].index(c) for c in d["tg"]]
    d["tser"] = pd.Series(d["ts"])
    return d


# ------------------------------------------------------------------ training
def new_model(d, dev):
    return GDN2(d["X"].shape[1], 0, d["cmd"].n, **MODEL).to(dev)


def val_batches(d, rf):
    seg, bs = seg_bs(rf)
    va = np.linspace(d["es"], d["v0"] - seg - 1, 80).astype(int)
    return [make_batch(d["X"], d["cmd"], d["W"], va[i:i + bs], seg, rf, d["X"].shape[1]) for i in range(0, len(va), bs)]


def val_loss(m, vb, dev):
    m.eval()
    vl = vn = 0.0
    with torch.no_grad():
        for xs, ys, ws, cf in vb:
            t = [None if a is None else torch.from_numpy(a) for a in (xs, ys, ws, cf)]
            vl += loss_fn(m.forward_segments(to(dev, t[0]), to(dev, t[3])), to(dev, t[1]), to(dev, t[2])).item() * len(xs)
            vn += len(xs)
    return vl / vn


def local_steps(m, opt, d, n, seed, dev):
    rf = m.rf
    seg, bs = seg_bs(rf)
    starts = np.arange(d["a"], d["es"] - seg - 1)
    m.train()
    tl = 0.0
    for xs, ys, ws, cf in Loader(d["X"], d["cmd"], d["W"], starts, bs, seg, rf, d["X"].shape[1], n, seed, False):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(m.forward_segments(to(dev, xs), to(dev, cf)), to(dev, ys), to(dev, ws))
        loss.backward()
        opt.step()
        tl += loss.item()
    return tl / n


def fit_window(d, dev, log):
    """core.train.fit's recipe on the satellite's own window (Adam, patience, best validation state)."""
    torch.manual_seed(0)
    np.random.seed(0)
    m = new_model(d, dev)
    opt = torch.optim.Adam(m.parameters(), lr=TRN["lr"])
    vb = val_batches(d, m.rf)
    best, bad, state, n = float("inf"), 0, None, 0
    for ep in range(TRN["epochs"]):
        tl = local_steps(m, opt, d, TRN["batches"], ep, dev)
        vl = val_loss(m, vb, dev)
        n = ep + 1
        log(f"ep{n}: train={tl:.5f} val={vl:.5f}")
        if vl < best - 1e-6:
            best, bad, state = vl, 0, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= TRN["patience"]:
                break
    m.load_state_dict(state)
    return m, dict(best_val=best, epochs=n)


def run_solo(D, dev):
    for s in D:
        p = f"{OUT}/models/solo_{s}.pt"
        if os.path.exists(p):
            continue
        d = D[s]
        t0 = time.time()
        if FIT_DAYS:
            m, info = fit_window(d, dev, lambda x: log(f"solo {s} {x}"))
        else:
            m, info = fit(d["X"], d["cmd"], d["W"], d["X"].shape[1], d["cmd"].n, d["v0"], 0, TRN, dev,
                          lambda x: log(f"solo {s} {x.strip()}"), lambda **k: None)
        torch.save(m.state_dict(), p)
        log(f"solo {s}: {info['epochs']} epochs, best val {info['best_val']:.4f}, {(time.time() - t0) / 60:.0f} min")


def fed_worker(s, conn):
    """One satellite: its own data, model and optimiser. Only shared layers cross the pipe."""
    torch.set_num_threads(2)
    dev = pick_device()
    d = inputs(s)
    torch.manual_seed(1 + SATS.index(s))                 # embeddings differ per satellite
    m = new_model(d, dev)
    opt = torch.optim.Adam(m.parameters(), lr=FED["lr"])
    vb = val_batches(d, m.rf)
    cpu = lambda st: {k: v.detach().cpu().clone() for k, v in st.items()}
    while True:
        msg = conn.recv()
        if msg[0] == "train":
            m.load_state_dict(msg[1], strict=False)
            tl = local_steps(m, opt, d, msg[2], msg[3], dev)
            conn.send((tl, cpu(m.shared_state())))
        elif msg[0] == "eval":
            m.load_state_dict(msg[1], strict=False)
            conn.send(val_loss(m, vb, dev))
        elif msg[0] == "state":
            conn.send(cpu(m.state_dict()))
        else:
            return


def run_fed():
    p = f"{OUT}/models/fed_state.pt"
    if os.path.exists(p):
        return
    ctx = mp.get_context("spawn")
    links = {}
    for s in SATS:
        a, b = ctx.Pipe()
        w = ctx.Process(target=fed_worker, args=(s, b), daemon=True)
        w.start()
        links[s] = a
    torch.manual_seed(0)
    m0 = GDN2(20, 0, 12, **MODEL)
    keys = list(m0.shared_state().keys())
    g = {k: v.detach().clone() for k, v in m0.shared_state().items()}            # one common start
    best, bad, hist = float("inf"), 0, []
    for r in range(FED["rounds"]):
        t0 = time.time()
        leader = SATS[r % len(SATS)]
        for i, s in enumerate(SATS):
            links[s].send(("train", g, FED["local_batches"], 1000 * r + i))
        got = {s: links[s].recv() for s in SATS}
        tls = {s: got[s][0] for s in SATS}
        states = [got[s][1] for s in SATS]
        # leader aggregates the shared layers only (FedAvg, equal weights: same steps per satellite)
        g = {k: (torch.stack([st[k].float() for st in states]).mean(0).to(states[0][k].dtype)
                 if states[0][k].is_floating_point() else states[0][k]) for k in keys}
        for s in SATS:
            links[s].send(("eval", g))
        vls = {s: links[s].recv() for s in SATS}
        mv = float(np.mean(list(vls.values())))
        hist.append(dict(round=r + 1, leader=leader, train=tls, val=vls, mean_val=mv, sec=time.time() - t0))
        log(f"fed round {r + 1}: leader {leader}, mean val {mv:.4f} " +
            " ".join(f"{s}={v:.4f}" for s, v in vls.items()) + f" ({time.time() - t0:.0f}s)")
        if mv < best - 1e-6:
            best, bad = mv, 0
            for s in SATS:
                links[s].send(("state",))
            torch.save({s: links[s].recv() for s in SATS}, p + ".tmp")
        else:
            bad += 1
            if bad >= FED["patience"]:
                break
    for s in SATS:
        links[s].send(("stop",))
    os.replace(p + ".tmp", p)
    json.dump(hist, open(f"{OUT}/fed_history.json", "w"), indent=1)


def run_pers(D, dev):
    fs = torch.load(f"{OUT}/models/fed_state.pt", map_location="cpu")
    for s in D:
        p = f"{OUT}/models/pers_{s}.pt"
        if os.path.exists(p):
            continue
        m = new_model(D[s], dev)
        m.load_state_dict(fs[s])
        vb = val_batches(D[s], m.rf)
        opt = torch.optim.Adam(m.parameters(), lr=PERS["lr"])
        best = val_loss(m, vb, dev)
        best_state, n0 = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, best
        for k in range(PERS["batches"] // PERS["check"]):
            local_steps(m, opt, D[s], PERS["check"], 777 + k, dev)
            v = val_loss(m, vb, dev)
            if v < best:
                best, best_state = v, {k_: v_.detach().cpu().clone() for k_, v_ in m.state_dict().items()}
        torch.save(best_state, p)
        log(f"pers {s}: val {n0:.4f} -> {best:.4f}")


def errors(D, dev, arm):
    fs = torch.load(f"{OUT}/models/fed_state.pt", map_location="cpu") if arm == "fed" else None
    for s in D:
        p = f"{OUT}/errors/{arm}_{s}.npy"
        if os.path.exists(p):
            continue
        m = new_model(D[s], dev)
        m.load_state_dict(fs[s] if arm == "fed" else torch.load(f"{OUT}/models/{arm}_{s}.pt", map_location="cpu"))
        E = series_errors(m, D[s]["X"], D[s]["cmd"], dev)[:, D[s]["tix"]]
        np.save(p, np.clip(E, 0, 6e4).astype(np.float16))
        log(f"errors {arm} {s} done")


# ------------------------------------------------------------------ detection
def periods(d, E, rf):
    tix = d["tix"]
    out = {}
    for name, e0, e1 in (("val", d["v0"], d["ntr"]), ("test", d["ntr"], len(E))):
        P = Period("SIM", name, E.astype(np.float32), d["vals"][:, tix], d["cnt"][:, tix], d["ts"], d["mu"][tix],
                   d["sd"][tix], rf + d["a"], d["v0"], e0, e1)
        P.build_tz()
        for W in AXES["W"]:
            for sp in AXES["spans"]:
                P.zs[(W, sp)] = P.z_and_thresholds(W, sp)
        P.build_drift(("d",))
        P.release_history()
        P.gix = np.array([group_of(d["names"][c]) for c in d["tg"]])
        ts = d["tser"].iloc[e0:e1].reset_index(drop=True)
        P.cmask = command_mask(d, ts, P.gix)
        lab = build_labels_df(d["raw"], ts)
        y = lab[lab["Channel"].isin(d["tg"])].reset_index(drop=True)
        P.ceil = {}
        out[name] = (P, ts, y)
    v, t, y = out["val"]
    near = np.zeros(len(t), bool)                       # labelled events + 6 h of settling
    for a, b in zip(y.StartTime.values, y.EndTime.values):
        i0, i1 = np.searchsorted(t.values, [a, b + np.timedelta64(6, "h")])
        near[i0:i1 + 1] = True
    v.nom = ~near
    v.ref = out["test"][0].ref = v
    return out


def f05(per, cfg):
    """Selection objective on validation: mean of F0.5 over all events and over anomalies only, so a
    few loud calibrations cannot decide the setting on their own."""
    P, ts, y = per
    A = alarms(P, cfg)
    sc = []
    for sel in (B.SEL, B.SEL_ANOM):
        if y.Category.isin(sel["Category"]).any():
            if not A.any():                 # no alarms: F0.5 is 0 (ESA's code cannot take an empty list)
                sc.append(0.0)
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                sc.append(float(ESAScores(betas=BETA, select_labels=sel).score(
                    y.drop(columns=["Channel"]), series(ts, A.max(1)))["EW_F_0.50"]))
    return float(np.mean(sc)) if sc else np.nan


def search(pers, passes=2, axes=AXES):
    """Coordinate search; objective = mean validation F0.5 over `pers`."""
    cache = {}

    def obj(cfg):
        k = json.dumps(cfg, sort_keys=True)
        if k not in cache:
            cache[k] = float(np.nanmean([f05(p, cfg) for p in pers]))
        return cache[k]
    cur = dict(START)
    best = obj(cur)
    for _ in range(passes):
        moved = False
        for ax, vals in axes.items():
            for v in vals:
                c = dict(cur, **{ax: v})
                f = obj(c)
                if f > best + 1e-9:
                    best, cur, moved = f, c, True
        if not moved:
            break
    return cur, best


def event_table(d, A, ts, cat="Anomaly"):
    """Per test event: detected?, delay, family attributed from the alarmed channels."""
    tru = pd.read_csv(f"{d['raw']}/truth/events_truth.csv")
    tru = tru[(tru.Split == "test") & tru.Visible & (tru.Category == cat)]
    t = ts.values
    rows = []
    for _, e in tru.iterrows():
        a = np.datetime64(pd.Timestamp(e.FirstVisible).tz_localize(None))
        b = np.datetime64(pd.Timestamp(e.LastVisible).tz_localize(None))
        i0, i1 = np.searchsorted(t, [a, b + np.timedelta64(30, "s")])
        hit = A[i0:i1].any(1)
        det = bool(hit.any())
        delay = float((np.argmax(hit)) * 0.5) if det else np.nan             # minutes after first visible
        fam = ""
        if det:
            chs = A[i0:i1].any(0)
            names = [d["names"][d["tg"][j]] for j in np.flatnonzero(chs)]
            votes = pd.Series([fam_of(n) for n in names]).value_counts()
            fam = votes.index[0]
        rows.append(dict(id=e.ID, code=e.Code, family=e.FaultFamily, detected=det, delay_min=delay, attributed=fam))
    return pd.DataFrame(rows)


def false_alarms(ts, A, y):
    """Alarm events (OR over channels, as ESA scores them) that touch no labelled event, and the
    share of nominal time under alarm. These two numbers are what ESA's precision penalises."""
    t = ts.values
    gt = np.zeros(len(t), bool)
    for a, b in zip(y.StartTime.values, y.EndTime.values):
        i0, i1 = np.searchsorted(t, [a, b])
        gt[i0:min(i1 + 1, len(t))] = True
    g = A.any(1)
    r = runs_of(g)
    fp = [(s, e) for s, e in r if not gt[s:e].any()]
    return dict(n_alarm_events=int(len(r)), n_false_alarm_events=int(len(fp)),
                false_alarm_event_share=float(len(fp) / max(len(r), 1)),
                false_alarm_time_share=float((g & ~gt).sum() / max((~gt).sum(), 1)),
                false_alarms_per_30d=float(len(fp) / (len(t) / RPD / 30.0)),
                median_false_alarm_min=float(np.median([(e - s) * 0.5 for s, e in fp])) if fp else 0.0)


def score_test(d, per, cfg, train_fams):
    P, ts, y = per
    A = alarms(P, cfg)
    r = dict(cfg=cfg, all=B.score(d["raw"], ts, A, d["tg"], B.SEL), anomalies=B.score(d["raw"], ts, A, d["tg"], B.SEL_ANOM))
    ev = event_table(d, A, ts)
    r["events"] = ev.to_dict("records")
    if len(ev):
        r["recall_by_family"] = ev.groupby("family").detected.mean().round(3).to_dict()
        new = ev[~ev.family.isin(train_fams)]
        r["recall_unseen_families"] = float(new.detected.mean()) if len(new) else None
        r["recall_seen_families"] = float(ev[ev.family.isin(train_fams)].detected.mean()) if len(new) < len(ev) else None
        r["n_anomalies"], r["n_detected"] = int(len(ev)), int(ev.detected.sum())
        det = ev[ev.detected]
        r["family_attribution_acc"] = float((det.attributed == det.family).mean()) if len(det) else None
        r["median_delay_min"] = float(det.delay_min.median()) if len(det) else None
    r["alarm_rows_frac"] = float(A.any(1).mean())
    r.update(false_alarms(ts, A, y))
    rv = event_table(d, A, ts, "Rare Event")
    r["n_rare"], r["n_rare_detected"] = int(len(rv)), int(rv.detected.sum()) if len(rv) else 0
    return r


def load_periods(D, arm, rf):
    PER = {s: periods(D[s], np.load(f"{OUT}/errors/{arm}_{s}.npy"), rf) for s in SATS}
    vals = [PER[s]["val"][0] for s in SATS]
    for s in SATS:
        PER[s]["val"][0].peers = PER[s]["test"][0].peers = vals
    return PER


def detect_all(D, arms):
    rf = GDN2(20, 0, 12, **MODEL).rf
    for arm in arms:
        if os.path.exists(f"{OUT}/results_{arm}.json"):
            continue
        log(f"detection: {arm}")
        PER = load_periods(D, arm, rf)
        loc = {s: search([PER[s]["val"]]) for s in SATS}
        fed_cfg, fed_val = search([PER[s]["val"] for s in SATS], axes=dict(AXES, ceil=["own", "fleet"]))
        out = dict(fed_selection=dict(cfg=fed_cfg, mean_val=fed_val))
        for s in SATS:
            tru = pd.read_csv(f"{D[s]['raw']}/truth/events_truth.csv")
            tf = sorted(set(tru[(tru.Split == "train") & (tru.Category == "Anomaly")].FaultFamily))
            out[s] = dict(train_families=tf,
                          local=dict(val=loc[s][1], **score_test(D[s], PER[s]["test"], loc[s][0], tf)),
                          federated=dict(val=f05(PER[s]["val"], fed_cfg), **score_test(D[s], PER[s]["test"], fed_cfg, tf)))
            for k in SELS:
                r = out[s][k]
                log(f"  {arm:5s} {s} {k:11s} val {r['val']:.3f} | TEST F0.5 all {r['all'].get('global_EW_F_0.50', np.nan):.3f} "
                    f"anom {r['anomalies'].get('global_EW_F_0.50', np.nan):.3f} | recall seen {r.get('recall_seen_families')} "
                    f"unseen {r.get('recall_unseen_families')} | {r.get('n_detected')}/{r.get('n_anomalies')}")
        json.dump(out, open(f"{OUT}/results_{arm}.json", "w"), indent=1, default=str)
        del PER


def matched(D, arm):
    """Score this arm's model with the settings the fleet chose for each arm: the setting is then the
    same for every arm, so differences come from the model alone. Settings are from validation."""
    rf = GDN2(20, 0, 12, **MODEL).rf
    cfgs = {k: json.load(open(f"{OUT}/results_{k}.json"))["fed_selection"]["cfg"] for k in ("solo", "fed", "pers")}
    out = {}
    PER = load_periods(D, arm, rf)
    for s in SATS:
        per = PER[s]["test"]
        tru = pd.read_csv(f"{D[s]['raw']}/truth/events_truth.csv")
        tf = sorted(set(tru[(tru.Split == "train") & (tru.Category == "Anomaly")].FaultFamily))
        out[s] = {k: {q: v for q, v in score_test(D[s], per, c, tf).items() if q != "events"} for k, c in cfgs.items()}
    json.dump(out, open(f"{OUT}/matched_{arm}.json", "w"), indent=1, default=str)


def fan(stage, flag, items):
    """Run one process per satellite (or arm); they share the GPU."""
    ps = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--stages", stage, flag, i]) for i in items]
    rc = [p.wait() for p in ps]
    if any(rc):
        raise SystemExit(f"stage {stage} failed: {dict(zip(items, rc))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="inputs,solo,fed,pers,errors,detect")
    ap.add_argument("--sat", default="", help="internal: this process owns one satellite")
    ap.add_argument("--arm", default="", help="internal: this process scores one arm")
    a = ap.parse_args()
    for sub in ("cache", "models", "errors"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    torch.set_num_threads(2)
    st = a.stages.split(",")
    if a.sat:
        dev = pick_device()
        D = {a.sat: inputs(a.sat)}
        d = D[a.sat]
        log(f"inputs {a.sat}: rows {len(d['X']):,} (train {d['ntr']:,}, fit < {d['v0']:,}), "
            f"{d['X'].shape[1]} channels, {d['cmd'].n} commands")
        if "solo" in st:
            run_solo(D, dev)
        if "pers" in st:
            run_pers(D, dev)
        if "errors" in st:
            for arm in ("solo", "fed", "pers"):
                errors(D, dev, arm)
        return
    if a.arm:
        D = {s: inputs(s) for s in SATS}
        if "detect" in st:
            detect_all(D, (a.arm,))
        if "matched" in st:
            matched(D, a.arm)
        return
    if "inputs" in st:
        fan("inputs", "--sat", SATS)
    if "solo" in st:
        fan("solo", "--sat", SATS)
    if "fed" in st:
        run_fed()
    if "pers" in st:
        fan("pers", "--sat", SATS)
    if "errors" in st:
        fan("errors", "--sat", SATS)
    if "detect" in st:
        arms = ("solo", "fed", "pers")
        fan("detect", "--arm", arms)
        json.dump({k: json.load(open(f"{OUT}/results_{k}.json")) for k in arms},
                  open(f"{OUT}/results.json", "w"), indent=1)
        fan("matched", "--arm", arms)
    log("done")


if __name__ == "__main__":
    main()
