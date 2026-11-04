"""Model check: our ESA-ADB detector (GDN2 forecaster) on the simulated satellites.

Same network and training code as the ESA-ADB runs (ESA/Main/new_runs/core, imported unchanged):
availability-aware inputs (value, gap flag, time since last sample), telecommand path, labelled
training rows left out of the loss. Detection is a light, past-only rule (the Mission-2 style
robust-z rule): forecast error -> EWMA -> robust z against the training errors -> per-channel
threshold -> close gaps < 10 rows, drop runs < 3 rows. Its two settings (EWMA span, z threshold)
are chosen on the validation month (last 30 days of the training half); the test half is scored
once with ESA-ADB's metrics. The full ESA detection stack (Telemanom pruning etc.) is not used here.

  conda activate timeeval
  python model_check.py --sats SAT_4,SAT_2 --epochs 12 --batches 200
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FLEET = os.environ.get("SIM_FLEET") or os.path.join(ROOT, "data", "sim_fleet")
OUT = os.path.join(FLEET, "validation", "model_check")
sys.path.insert(0, (os.environ.get("NEW_RUNS") or (os.path.join(ROOT, "ext", "new_runs") if os.path.isdir(os.path.join(ROOT, "ext", "new_runs")) else os.path.join(os.path.dirname(ROOT), "ESA", "Main", "new_runs"))))
sys.path.insert(0, HERE)
import torch  # noqa: E402
from core.config import MODEL  # noqa: E402
from core.features import CmdFeats, channel_tensor  # noqa: E402
from core.train import fit, series_errors, pick_device  # noqa: E402
import baselines as B  # noqa: E402

VAL_DAYS = 30


def arrays(sat, part):
    raw = f"{FLEET}/{sat}"
    df = pd.read_csv(f"{raw}/resampled_30s_{part}.csv.gz")
    ch = pd.read_csv(f"{raw}/channels.csv")
    chs = ch.Channel.tolist()
    ts = pd.to_datetime(df.timestamp).dt.tz_localize(None).values.astype("datetime64[ns]")
    g = ts.astype(np.int64) / 1e9
    vals = df[chs].values.astype(np.float32)
    cnt = np.zeros_like(vals, dtype=np.float32)
    tsince = np.zeros_like(vals, dtype=np.float32)
    for j, c in enumerate(chs):
        t = pd.to_datetime(pd.read_csv(f"{raw}/channels/{c}.csv.gz", usecols=[0]).timestamp).dt.tz_localize(None)
        t = t.values.astype("datetime64[ns]").astype(np.int64) / 1e9
        hi = np.searchsorted(t, g, "right")
        lo = np.searchsorted(t, g - 30.0, "right")
        cnt[:, j] = hi - lo
        last = t[np.clip(hi - 1, 0, len(t) - 1)]
        tsince[:, j] = np.where(hi > 0, g - last, 1e6)
    tcs = pd.read_csv(f"{raw}/telecommands.csv")
    cmds = []
    for c in tcs.Telecommand:
        f = pd.read_csv(f"{raw}/telecommands/{c}.csv")
        cmds.append((c, 3, pd.to_datetime(f.timestamp).dt.tz_localize(None).values.astype("datetime64[ns]")))
    lab = df[[f"is_anomaly_{c}" for c in chs if f"is_anomaly_{c}" in df.columns]]
    tg = ch.Channel[ch.Target == "YES"].tolist()
    return dict(ts=ts, ts_str=df.timestamp, vals=vals, count=cnt, tsince=tsince, cmds=cmds, chs=chs, tg=tg,
                lab=lab, raw=raw)


def ewma(e, span):
    if span <= 1:
        return e
    a = 2.0 / (span + 1)
    out = np.empty_like(e)
    s = e[0].astype(np.float64)
    for i in range(len(e)):
        s = a * e[i] + (1 - a) * s
        out[i] = s
    return out


def ewma_fast(e, span):
    import pandas as pd_
    return pd_.DataFrame(e).ewm(span=span, adjust=False).mean().values if span > 1 else e


def post(P, close=10, minlen=3):
    P = P.copy()
    for j in range(P.shape[1]):
        x = P[:, j]
        d = np.diff(np.r_[0, x.astype(np.int8), 0])
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        for k in range(len(st) - 1):                       # close short gaps
            if st[k + 1] - en[k] < close:
                x[en[k]:st[k + 1]] = 1
        d = np.diff(np.r_[0, x.astype(np.int8), 0])
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        for a, b in zip(st, en):                           # drop short runs
            if b - a < minlen:
                x[a:b] = 0
        P[:, j] = x
    return P


def detect(Z, tau):
    return post((Z > tau).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sats", default="SAT_2,SAT_3,SAT_4,SAT_5,SAT_1")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batches", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = pick_device()
    torch.set_num_threads(2)
    res = json.load(open(f"{OUT}/results.json")) if os.path.exists(f"{OUT}/results.json") else {}
    for sat in a.sats.split(","):
        t0 = time.time()
        log = lambda s: print(f"[{sat} {time.time() - t0:6.0f}s] {s}", flush=True)
        tr, te = arrays(sat, "train"), arrays(sat, "test")
        log(f"inputs: train {tr['vals'].shape}, test {te['vals'].shape}, {len(tr['cmds'])} commands")
        mu = tr["vals"].mean(0)
        sd = tr["vals"].std(0) + 1e-3
        X = channel_tensor(tr["vals"], tr["count"], tr["tsince"], mu, sd)
        Xt = channel_tensor(te["vals"], te["count"], te["tsince"], mu, sd)
        W = np.ones((len(X), X.shape[1]), np.uint8)
        for j, c in enumerate(tr["chs"]):
            col = f"is_anomaly_{c}"
            if col in tr["lab"]:
                W[:, j] = (tr["lab"][col].values == 0)
        cmd, cmdt = CmdFeats(tr["ts"], tr["cmds"]), CmdFeats(te["ts"], te["cmds"])
        trn = dict(epochs=a.epochs, batches=a.batches, patience=4, lr=1e-3, smoke=False)
        model, info = fit(X, cmd, W, X.shape[1], cmd.n, len(X), 0, trn, dev, log, lambda **k: None)
        tix = [tr["chs"].index(c) for c in tr["tg"]]
        Etr = series_errors(model, X, cmd, dev)[:, tix]
        Ete = series_errors(model, Xt, cmdt, dev)[:, tix]
        torch.save(model.state_dict(), f"{OUT}/{sat}_model.pt")
        log(f"trained ({info['epochs']} epochs, best val {info['best_val']:.4f}); errors done")
        # robust z against nominal training errors (first part of training, excluding validation month)
        v0 = len(Etr) - VAL_DAYS * 2880
        rf = model.rf
        nomr = W[rf:v0][:, tix].astype(bool)
        best = None
        for span in (1, 5, 15):
            Str = ewma_fast(Etr, span)
            med = np.array([np.median(Str[rf:v0][nomr[:, j], j]) for j in range(len(tix))])
            iqr = np.array([np.subtract(*np.percentile(Str[rf:v0][nomr[:, j], j], [75, 25])) for j in range(len(tix))]) + 1e-6
            Zv = (Str[v0:] - med) / iqr
            for tau in (5, 8, 12, 20, 30, 50):
                Pv = detect(Zv, tau)
                s = B.score(tr["raw"], tr["ts_str"].iloc[v0:], Pv, tr["tg"], B.SEL)
                f = s.get("global_EW_F_0.50", 0.0)
                if best is None or f > best[0]:
                    best = (f, span, tau, med, iqr)
        f_val, span, tau, med, iqr = best
        Zt = (ewma_fast(Ete, span) - med) / iqr
        Pt = detect(Zt, tau)
        r = dict(val_F05=f_val, span=span, tau=tau, epochs=info["epochs"], minutes=(time.time() - t0) / 60,
                 test_all=B.score(te["raw"], te["ts_str"], Pt, te["tg"], B.SEL),
                 test_anomalies=B.score(te["raw"], te["ts_str"], Pt, te["tg"], B.SEL_ANOM))
        res[sat] = r
        log(f"chosen on validation: span={span} tau={tau} (val F0.5 {f_val:.3f}); TEST event-wise F0.5 "
            f"all={r['test_all'].get('global_EW_F_0.50', float('nan')):.3f} "
            f"anomalies={r['test_anomalies'].get('global_EW_F_0.50', float('nan')):.3f}")
        json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
