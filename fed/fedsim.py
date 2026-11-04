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
import os
import pickle
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
import model_check as MC  # noqa: E402
import baselines as B  # noqa: E402

SATS = ["SAT_1", "SAT_2", "SAT_3", "SAT_4"]
VAL_DAYS = 45
RPD = 2880                                   # 30 s rows per day
OUT = os.environ.get("FED_OUT", os.path.join(HERE, "out_v2"))
MISSIONS["SIM"] = dict(grid=30)
TRN = dict(epochs=12, batches=150, patience=4, lr=1e-3, smoke=False)
FED = dict(rounds=12, local_batches=150, patience=4, lr=1e-3)
PERS = dict(batches=100, lr=3e-4, check=25)
START = dict(L=0, W="all", close=60, cross=0.3, drift=0, hyst=1.0, minlen=10, onset=0, spans="s15", std=0,
             thr="a20", tele=None)                       # ESA Mission-2 frozen setting
AXES = dict(thr=["a5", "a8", "a12", "a20", "a30", "a50"], W=["1m", "all"], spans=["s15", "multi"],
            std=[0, 6, 10], hyst=[1.0, 0.7], cross=[0, 0.3], close=[10, 60], minlen=[1, 3, 10],
            drift=[0, "d6", "d10"])
FAM_OF_CH = {}                                           # fault-family attribution from channel names
T0 = time.time()


def log(s):
    line = f"[{(time.time() - T0) / 60:7.1f} min] {s}"
    print(line, flush=True)
    open(f"{OUT}/log.txt", "a").write(line + "\n")


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
        mu, sd = vals[:v0].mean(0), vals[:v0].std(0) + 1e-3
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
    es = int(0.85 * d["v0"])
    va = np.linspace(es, d["v0"] - seg - 1, 80).astype(int)
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
    starts = np.arange(0, int(0.85 * d["v0"]) - seg - 1)
    m.train()
    tl = 0.0
    for xs, ys, ws, cf in Loader(d["X"], d["cmd"], d["W"], starts, bs, seg, rf, d["X"].shape[1], n, seed, False):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(m.forward_segments(to(dev, xs), to(dev, cf)), to(dev, ys), to(dev, ws))
        loss.backward()
        opt.step()
        tl += loss.item()
    return tl / n


def run_solo(D, dev):
    for s in SATS:
        p = f"{OUT}/models/solo_{s}.pt"
        if os.path.exists(p):
            continue
        d = D[s]
        t0 = time.time()
        m, info = fit(d["X"], d["cmd"], d["W"], d["X"].shape[1], d["cmd"].n, d["v0"], 0, TRN, dev,
                      lambda x: log(f"solo {s} {x.strip()}"), lambda **k: None)
        torch.save(m.state_dict(), p)
        log(f"solo {s}: {info['epochs']} epochs, best val {info['best_val']:.4f}, {(time.time() - t0) / 60:.0f} min")


def run_fed(D, dev):
    p = f"{OUT}/models/fed_state.pt"
    if os.path.exists(p):
        return
    torch.manual_seed(0)
    M = {s: new_model(D[s], dev) for s in SATS}
    keys = list(M[SATS[0]].shared_state().keys())
    g = {k: v.detach().clone() for k, v in M[SATS[0]].shared_state().items()}   # one common start
    opt = {s: torch.optim.Adam(M[s].parameters(), lr=FED["lr"]) for s in SATS}
    vb = {s: val_batches(D[s], M[s].rf) for s in SATS}
    best, bad, hist = float("inf"), 0, []
    for r in range(FED["rounds"]):
        t0 = time.time()
        leader = SATS[r % len(SATS)]
        states, tls = [], {}
        for i, s in enumerate(SATS):
            M[s].load_state_dict(g, strict=False)
            tls[s] = local_steps(M[s], opt[s], D[s], FED["local_batches"], 1000 * r + i, dev)
            states.append({k: v.detach().clone() for k, v in M[s].shared_state().items()})
        # leader aggregates the shared layers only (FedAvg, equal weights: same steps per satellite)
        g = {k: (torch.stack([st[k].float() for st in states]).mean(0).to(states[0][k].dtype)
                 if states[0][k].is_floating_point() else states[0][k]) for k in keys}
        vls = {}
        for s in SATS:
            M[s].load_state_dict(g, strict=False)
            vls[s] = val_loss(M[s], vb[s], dev)
        mv = float(np.mean(list(vls.values())))
        hist.append(dict(round=r + 1, leader=leader, train=tls, val=vls, mean_val=mv, sec=time.time() - t0))
        log(f"fed round {r + 1}: leader {leader}, mean val {mv:.4f} " +
            " ".join(f"{s}={v:.4f}" for s, v in vls.items()) + f" ({time.time() - t0:.0f}s)")
        if mv < best - 1e-6:
            best, bad = mv, 0
            torch.save({s: {k: v.detach().cpu().clone() for k, v in M[s].state_dict().items()} for s in SATS}, p + ".tmp")
        else:
            bad += 1
            if bad >= FED["patience"]:
                break
    os.replace(p + ".tmp", p)
    json.dump(hist, open(f"{OUT}/fed_history.json", "w"), indent=1)


def run_pers(D, dev):
    fs = torch.load(f"{OUT}/models/fed_state.pt", map_location="cpu")
    for s in SATS:
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
    for s in SATS:
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
                   d["sd"][tix], rf, d["v0"], e0, e1)
        P.build_tz()
        for W in AXES["W"]:
            for sp in AXES["spans"]:
                P.zs[(W, sp)] = P.z_and_thresholds(W, sp)
        P.build_drift(("d",))
        P.release_history()
        ts = d["tser"].iloc[e0:e1].reset_index(drop=True)
        lab = build_labels_df(d["raw"], ts)
        y = lab[lab["Channel"].isin(d["tg"])].reset_index(drop=True)
        out[name] = (P, ts, y)
    return out


def f05(per, cfg, sel=B.SEL):
    P, ts, y = per
    if not y.Category.isin(sel["Category"]).any():
        return np.nan
    A = assemble(P, cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        r = ESAScores(betas=BETA, select_labels=sel).score(y.drop(columns=["Channel"]), series(ts, A.max(1)))
    return float(r["EW_F_0.50"])


def search(pers, passes=2):
    """Coordinate search from the ESA Mission-2 setting; objective = mean validation F0.5 over `pers`."""
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
        for ax, vals in AXES.items():
            for v in vals:
                c = dict(cur, **{ax: v})
                f = obj(c)
                if f > best + 1e-9:
                    best, cur, moved = f, c, True
        if not moved:
            break
    return cur, best


def event_table(d, A, ts):
    """Per test anomaly: detected?, delay, family attributed from the alarmed channels."""
    tru = pd.read_csv(f"{d['raw']}/truth/events_truth.csv")
    tru = tru[(tru.Split == "test") & tru.Visible & (tru.Category == "Anomaly")]
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


def score_test(d, per, cfg, train_fams):
    P, ts, y = per
    A = assemble(P, cfg)
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
    return r


def detect_all(D, arms):
    res = json.load(open(f"{OUT}/results.json")) if os.path.exists(f"{OUT}/results.json") else {}
    rf = GDN2(20, 0, 12, **MODEL).rf
    for arm in arms:
        if arm in res:
            continue
        log(f"detection: {arm}")
        PER = {s: periods(D[s], np.load(f"{OUT}/errors/{arm}_{s}.npy"), rf) for s in SATS}
        loc = {s: search([PER[s]["val"]]) for s in SATS}
        fed_cfg, fed_val = search([PER[s]["val"] for s in SATS])
        out = dict(fed_selection=dict(cfg=fed_cfg, mean_val=fed_val))
        for s in SATS:
            tru = pd.read_csv(f"{D[s]['raw']}/truth/events_truth.csv")
            tf = sorted(set(tru[(tru.Split == "train") & (tru.Category == "Anomaly")].FaultFamily))
            out[s] = dict(train_families=tf,
                          local=dict(val=loc[s][1], **score_test(D[s], PER[s]["test"], loc[s][0], tf)),
                          federated=dict(val=f05(PER[s]["val"], fed_cfg), **score_test(D[s], PER[s]["test"], fed_cfg, tf)))
            for k in ("local", "federated"):
                r = out[s][k]
                log(f"  {arm:5s} {s} {k:9s} val {r['val']:.3f} | TEST F0.5 all {r['all'].get('global_EW_F_0.50', np.nan):.3f} "
                    f"anom {r['anomalies'].get('global_EW_F_0.50', np.nan):.3f} | recall seen {r.get('recall_seen_families')} "
                    f"unseen {r.get('recall_unseen_families')} | {r.get('n_detected')}/{r.get('n_anomalies')}")
        res[arm] = out
        json.dump(res, open(f"{OUT}/results.json", "w"), indent=1, default=str)
        del PER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="inputs,solo,fed,pers,errors,detect")
    a = ap.parse_args()
    for sub in ("cache", "models", "errors"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    torch.set_num_threads(3)
    dev = pick_device()
    st = a.stages.split(",")
    D = {}
    for s in SATS:
        D[s] = inputs(s)
        log(f"inputs {s}: rows {len(D[s]['X']):,} (train {D[s]['ntr']:,}, fit < {D[s]['v0']:,}), "
            f"{D[s]['X'].shape[1]} channels, {D[s]['cmd'].n} commands")
    if "solo" in st:
        run_solo(D, dev)
    if "fed" in st:
        run_fed(D, dev)
    if "pers" in st:
        run_pers(D, dev)
    if "errors" in st:
        for arm in ("solo", "fed", "pers"):
            errors(D, dev, arm)
    if "detect" in st:
        detect_all(D, ("solo", "fed", "pers"))
    log("done")


if __name__ == "__main__":
    main()
