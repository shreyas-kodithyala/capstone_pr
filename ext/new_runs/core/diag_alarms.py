"""Where does the test alarm time come from? (read-only diagnosis, ~5 min)

  python -m core.diag_alarms --mission M1

Rebuilds the frozen detector on the test period exactly as core.evaluate does, then prints
per detector: share of test time in alarm, the channels responsible, alarm time per year,
the longest alarms, and whether the responsible channels stopped reporting.
Writes nothing except out/<mk>/diag_alarms.txt.
"""
import argparse
import contextlib
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd

from .config import OUT, MISSIONS, job_dir
from .data import load_split, label_mask, target_idx
from . import detect as D
from . import evaluate as E
from .period import Period, assemble, spans_of, tele_key


class Tee:
    def __init__(self, *f):
        self.f = f

    def write(self, s):
        for x in self.f:
            x.write(s)
            x.flush()

    def flush(self):
        pass


def report(name, M, lab, labg, tt, chs, trc, tec, mk, vsd=None):
    g = M.any(1)
    print(f"=== {name}: alarm time {g.mean():.2%} of test; of that inside labelled events "
          f"{(g & labg).sum() / max(g.sum(), 1):.2%}")
    if not g.any():
        return
    frac = M.mean(0)
    fp = (M & ~lab).mean(0)
    top = np.argsort(-fp)[:10]
    print("  channels with most FALSE alarm time (alarm outside labels):")
    for c in top:
        if frac[c] == 0:
            break
        ok_tr = (trc[:, c] > 0).mean()
        ok_te = (tec[:, c] > 0).mean()
        sdtxt = "" if vsd is None else f"  train std {vsd[c]:.2e}"
        print(f"    {chs[c]:12s} alarm {frac[c]:7.2%}  false {fp[c]:7.2%}  "
              f"data present train {ok_tr:6.1%} / test {ok_te:6.1%}{sdtxt}")
    yr = pd.Series(g, index=tt).groupby(tt.year).mean()
    print("  alarm time per year: " + "  ".join(f"{y}: {v:.0%}" for y, v in yr.items()))
    runs = D.runs_of(g.astype(np.uint8))
    if len(runs):
        L = runs[:, 1] - runs[:, 0]
        o = np.argsort(-L)[:8]
        print(f"  {len(runs):,} alarms; longest:")
        for s, e in runs[o]:
            cs = M[s:e].mean(0)
            who = ", ".join(f"{chs[c]} {cs[c]:.0%}" for c in np.argsort(-cs)[:4] if cs[c] > 0)
            print(f"    {tt[s]} -> {tt[min(e, len(tt) - 1)]}  ({(e - s) * 30 / 86400 if mk == 'M1' else (e - s) * 18 / 86400:.1f} d)"
                  f"  labelled {labg[s:e].mean():.0%}  channels: {who}")
    print()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="M1", choices=list(MISSIONS))
    ap.add_argument("--workers", type=int, default=44)
    a = ap.parse_args()
    mk = a.mission
    sys.stdout = Tee(sys.__stdout__, open(f"{OUT}/{mk}/diag_alarms.txt", "w"))
    t0 = time.time()
    frz = json.load(open(f"{OUT}/{mk}/frozen.json"))
    cfg = frz["detector"]
    print(f"[{mk}] frozen detector: {cfg}  typing {frz['typing']}")
    r = job_dir(mk, "final", 0)
    m0 = json.load(open(f"{r}/meta.json"))
    etr = np.load(f"{r}/err_train.npy").astype(np.float32)
    ete = np.load(f"{r}/err_test.npy").astype(np.float32)
    tr, te = load_split(mk, "train"), load_split(mk, "test")
    tix = target_idx(mk)
    chs = [tr["chs"][i] for i in tix]
    sub = lambda x: np.asarray(x)[:, tix]                  # noqa: E731
    trv, trc, tev, tec = sub(tr["vals"]), sub(tr["count"]), sub(te["vals"]), sub(te["count"])
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
        E._G["P"] = P
        with ProcessPoolExecutor(min(a.workers, P.C), mp_context=get_context("fork")) as pool:
            for c, res in pool.map(E._tele_task, [(c, k[1], {k}) for c in range(P.C)]):
                P.tele.setdefault(k, np.zeros((P.n, P.C), np.uint8))[:, c] = res[k]
    X, parts = assemble(P, cfg, parts=True)
    tt = pd.to_datetime(ts[P.e0:P.e1])
    lab = label_mask(mk, ts[P.e0:P.e1], chs=chs)            # any labelled event, per channel
    labg = lab.any(1)
    print(f"[{mk}] test {tt[0]} -> {tt[-1]}, {len(tt):,} rows, labelled rows {labg.mean():.2%} "
          f"({time.time() - t0:.0f}s)\n")

    ctx = (lab, labg, tt, chs, trc, tec, mk, P.vsd)
    report("FINAL (all detectors)", X, *ctx)
    for k, v in parts.items():
        report(f"detector {k}", v, *ctx)
    print(f"[{mk}] diag done ({time.time() - t0:.0f}s) -> {OUT}/{mk}/diag_alarms.txt")


if __name__ == "__main__":
    main()
