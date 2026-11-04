"""Test evaluation -- run ONCE, with the model and the detector setting frozen.

  python -m core.evaluate --mission M1 --seeds 0 [--workers 14]

* With several final seeds, their forecast errors are averaged (an ensemble) first.
* Scores: headline (anomalies + rare events), anomaly-only, and anomaly-only on uncommanded
  alarms (alarm typing chosen in selection).
* Out-of-limit baseline: fixed per-channel limits from nominal training data (what operators
  use today) -- how many events it catches vs our detector.
* Which detector caught each event (first-hit attribution).
Writes out/<mk>/result.json.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd

from .config import OUT, MISSIONS, SET, job_dir
from .data import load_split, label_mask, fire_seconds, target_idx
from . import detect as D
from .period import Period, assemble, spans_of, tele_key
from .score import Scorer, SEL, SEL_ANOM, SHOW, fmt

_G = {}


def _tele_task(a):
    c, span, combos = a
    return c, _G["P"].telemanom(c, span, combos)


def caught(P_mask, scorer, ts, cats):
    """{event ID: True/False} -- any alarm on an affected channel during the event."""
    y = scorer.y[scorer.y.Category.isin(cats)]
    t = ts.astype("datetime64[ns]")
    out = {}
    for eid, g in y.groupby("ID"):
        hit = False
        for _, r in g.iterrows():
            c = scorer.chs.index(r.Channel)
            a = max(int(np.searchsorted(t, np.datetime64(r.StartTime), "right")) - 1, 0)
            b = int(np.searchsorted(t, np.datetime64(r.EndTime), "right"))
            if P_mask[a:max(b, a + 1), c].any():
                hit = True
                break
        out[eid] = hit
    return out


def write_tables(mk, S, tt, X, parts, ool_mask, Xtyped):
    """events_test.csv: one row per labelled test event (caught? first-alarm delay? by which
    detector? caught by fixed limits?). alarms_test.csv: one row per alarm (merged over
    channels): start, end, duration, channels, commanded (typed) or not, overlapping events."""
    import csv
    t = tt.astype("datetime64[ns]")
    with open(f"{OUT}/{mk}/events_test.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["event_id", "category", "start", "end", "minutes", "n_channels", "caught",
                    "first_alarm", "delay_minutes", "caught_by", "fixed_limits_caught"])
        for eid, g in S.y.groupby("ID"):
            st, en = g.StartTime.min(), g.EndTime.max()
            hit_rows, by, oo = [], set(), False
            for _, r in g.iterrows():
                c = S.chs.index(r.Channel)
                a = max(int(np.searchsorted(t, np.datetime64(r.StartTime), "right")) - 1, 0)
                b = max(int(np.searchsorted(t, np.datetime64(r.EndTime), "right")), a + 1)
                h = np.flatnonzero(X[a:b, c])
                if len(h):
                    hit_rows.append(a + h[0])
                by |= {k for k, v in parts.items() if v[a:b, c].any()}
                oo |= bool(ool_mask[a:b, c].any())
            fa = t[min(hit_rows)] if hit_rows else None
            w.writerow([eid, g.Category.iloc[0], st, en, round((en - st).total_seconds() / 60, 2),
                        g.Channel.nunique(), bool(hit_rows), fa if fa is not None else "",
                        round((pd.Timestamp(fa) - st).total_seconds() / 60, 2) if fa is not None else "",
                        "+".join(sorted(by)), oo])
    g = X.max(1)
    r = D.runs_of(g)
    ev = S.y.drop_duplicates("ID")[["ID", "StartTime", "EndTime"]].values
    with open(f"{OUT}/{mk}/alarms_test.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start", "end", "minutes", "channels", "commanded", "overlapping_events"])
        for s0, e0 in r:
            a, b = pd.Timestamp(t[s0]), pd.Timestamp(t[min(e0, len(t) - 1)])
            chs = [S.chs[c] for c in range(X.shape[1]) if X[s0:e0, c].any()]
            cmd = "" if Xtyped is None else (not Xtyped[s0:e0].any())
            hits = [i for i, es, ee in ev if es <= b and ee >= a]
            w.writerow([a, b, round((b - a).total_seconds() / 60, 2), "+".join(chs), cmd, "+".join(hits)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", required=True, choices=list(MISSIONS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1))
    a = ap.parse_args()
    mk = a.mission
    t0 = time.time()
    frz = json.load(open(f"{OUT}/{mk}/frozen.json"))
    cfg = frz["detector"]
    seeds = [int(s) for s in a.seeds.split(",")]
    runs = [job_dir(mk, "final", s) for s in seeds]
    metas = [json.load(open(f"{r}/meta.json")) for r in runs]
    m0 = metas[0]
    # ---- ensemble (mean of the seeds' absolute forecast errors) ----
    etr = np.mean([np.load(f"{r}/err_train.npy").astype(np.float32) for r in runs], axis=0)
    ete = np.mean([np.load(f"{r}/err_test.npy").astype(np.float32) for r in runs], axis=0)
    tr, te = load_split(mk, "train"), load_split(mk, "test")
    tix = target_idx(mk)                                   # the channels ESA scores
    assert etr.shape[1] == len(tix), "error file / target channels mismatch"
    sub = lambda x: np.asarray(x)[:, tix]                  # noqa: E731
    trv, trc, tev, tec = sub(tr["vals"]), sub(tr["count"]), sub(te["vals"]), sub(te["count"])
    # the last training row and the first test row share a timestamp: keep the test one
    cat = lambda x, y: np.concatenate([np.asarray(x[:-1]), np.asarray(y)])   # noqa: E731
    ts = cat(tr["ts"], te["ts"])
    fit_end = len(tr["ts"]) - 1
    P = Period(mk, "test", cat(etr, ete), cat(trv, tev), cat(trc, tec), ts,
               np.asarray(m0["mu"])[tix], np.asarray(m0["sd"])[tix], m0["rf"], fit_end,
               fit_end, len(ts))
    P.build_tz(spans_of(cfg["spans"]))
    if cfg.get("drift"):
        P.build_drift((cfg["drift"][0],))
    if cfg["thr"] != "off":
        P.zs[(cfg["W"], cfg["spans"])] = P.z_and_thresholds(cfg["W"], cfg["spans"])
    if cfg.get("tele"):
        k = tele_key(cfg["tele"])
        _G["P"] = P
        with ProcessPoolExecutor(min(a.workers, P.C), mp_context=get_context("fork")) as pool:
            for c, res in pool.map(_tele_task, [(c, k[1], {k}) for c in range(P.C)]):
                P.tele.setdefault(k, np.zeros((P.n, P.C), np.uint8))[:, c] = res[k]
    print(f"[{mk}] test period ready: {P.n:,} rows, seeds {seeds} ({time.time() - t0:.0f}s)", flush=True)

    S = Scorer(mk, ts[P.e0:P.e1])
    X, parts = assemble(P, cfg, parts=True)
    head, anom = S.score(X, SEL), S.score(X, SEL_ANOM)
    typed = anom
    n_cmd = 0
    if frz["typing"] != "none":
        tw, tp = map(int, frz["typing"].split("_"))
        Xt, n_cmd = D.typed(X, ts[P.e0:P.e1].astype(np.int64) / 1e9, fire_seconds(mk, tp), tw)
        typed = S.score(Xt, SEL_ANOM)

    # ---- out-of-limit baseline: limits from nominal training rows ----
    nominal = ~label_mask(mk, tr["ts"], chs=MISSIONS[mk]["targets"])
    ool = {}
    for name, q in (("minmax", None), ("p0.1", 0.001)):
        lo, hi = D.ool_limits(trv, trc, nominal, q)
        M = D.ool_mask(tev, tec, lo, hi)
        ool[name] = dict(headline=S.score(M, SEL), anom=S.score(M, SEL_ANOM), mask=M,
                         alarm_time=float(M.max(1).mean()))

    # ---- event-level comparison and per-detector attribution ----
    cats = ("Anomaly", "Rare Event")
    tt = ts[P.e0:P.e1]
    ours = caught(X, S, tt, cats)
    by = {k: caught(v, S, tt, cats) for k, v in parts.items()}
    oo = caught(ool["minmax"]["mask"], S, tt, cats)      # the realistic limits: normal envelope
    catof = S.y.drop_duplicates("ID").set_index("ID").Category.to_dict()
    ev = {}
    for c in cats:
        ids = [i for i in ours if catof[i] == c]
        ev[c] = dict(n=len(ids), ours=sum(ours[i] for i in ids), ool=sum(oo[i] for i in ids),
                     ours_not_ool=sum(ours[i] and not oo[i] for i in ids),
                     ool_not_ours=sum(oo[i] and not ours[i] for i in ids),
                     by_detector={k: sum(v[i] for i in ids) for k, v in by.items()})
    res = dict(mission=mk, seeds=seeds, detector=cfg, typing=frz["typing"],
               headline=head, anom=anom, anom_typed=typed, n_commanded_alarms=n_cmd,
               ool={k: dict(headline=v["headline"], anom=v["anom"], alarm_time=v["alarm_time"])
                    for k, v in ool.items()},
               events=ev, n_alarm_rows=int(X.max(1).sum()), alarm_time=float(X.max(1).mean()),
               test_rows=P.n,
               model=dict(n_params=m0["n_params"], rf=m0["rf"], epochs=[m["epochs"] for m in metas],
                          train_s=[m["train_s"] for m in metas], device=m0["device"]),
               secs=time.time() - t0)
    json.dump(res, open(f"{OUT}/{mk}/result.json", "w"), indent=1, default=float)
    write_tables(mk, S, tt, X, parts, ool["minmax"]["mask"],
                 Xt if frz["typing"] != "none" else None)
    if SET == "sub":       # per-channel alarms, for the mission-level combination (core.combine)
        np.savez_compressed(f"{OUT}/{mk}/alarms_test_mask.npz", X=X, chs=np.array(S.chs),
                            Xt=Xt if frz["typing"] != "none" else X)
    print(f"[{mk}] TEST  headline   {fmt(head)}")
    print(f"[{mk}] TEST  anomaly    {fmt(anom)}")
    print(f"[{mk}] TEST  uncommanded {fmt(typed)}")
    for c, e in ev.items():
        print(f"[{mk}] {c}: {e['ours']}/{e['n']} caught (out-of-limit: {e['ool']}; ours-only "
              f"{e['ours_not_ool']}; by detector {e['by_detector']})")


if __name__ == "__main__":
    main()
