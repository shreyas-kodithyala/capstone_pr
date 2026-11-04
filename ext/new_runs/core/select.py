"""Choose ONE detector setting per mission, on data the models have never seen.

  python -m core.select --mission M1 --workers 14 [--smoke]

Two folds per mission (config.MISSIONS[mk]["folds"]): the fold-A model predicts fold A's
period, the fold-B model predicts fold B's. Every candidate setting is scored on
  * fold A, fold B (each alone) and
  * the pooled out-of-fold period A+B (more events -> a steadier F0.5).
Rule ("must work in every period"):
  phase I   maximise pooled event-wise F0.5
  phase II  maximise min(S5 on A, S5 on B) among settings whose pooled F0.5 is within 0.02
            of the best;   final pick: same rule over everything evaluated (tie: pooled S5)
Then alarm typing (commanded vs uncommanded) is chosen by anomaly-only S5 on A+B.
Coordinate search: each axis's values are evaluated in parallel across CPU cores. Phase I
only needs the pooled event-wise F0.5, so it is scored cheaply (global metrics only); every
setting within 0.02 of the best is then fully scored before phase II. This picks exactly the
setting the all-full-scoring search picks, with far fewer full scorings.
Memory: the candidate set ("axes profile") is sized to this machine. "full" when it fits;
"big" (fewer windows, span 15 only, 12 Telemanom variants) / "tiny" when it does not (e.g.
M1 S6, 42 channels, on a 29 GB Kaggle machine). NR_AXES forces one; NR_RAM_GB sets the budget.
Writes out/<mk>/frozen.json.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from .config import OUT, MISSIONS, SEARCH, job_dir
from .data import load_split, row_of, row_end, fire_seconds, target_idx
from . import detect as D
from .period import (Period, assemble, WINDOWS_DAYS, QS, TELE_PS, MINERRS, BACKS, SPANS,
                     spans_of, tele_key)
from .score import Scorer, SEL, SEL_ANOM, SHOW, fmt

W_AXIS = list(WINDOWS_DAYS) + ["1dx", "1wx", "1mx", "3mx"]
TELE = [f"{p}/{sp}/{me}/{bk}" for p in TELE_PS for sp in (5, 105) for me in MINERRS for bk in BACKS]
COMMON = dict(W=W_AXIS, L=[0, 10, 20, 40], std=[0, 5, 8, 12],
              drift=[0, "d4", "d6", "d8", "w4", "w6", "w8"], close=[0, 10, 30, 60],
              minlen=[1, 3, 5, 10], onset=[0, 3, 10], hyst=[1.0, 0.7, 0.5, 0.3],
              cross=[0.0, 0.7, 0.5, 0.3])
THR_Q = [f"{q}/{m}" for q in QS for m in ("global", "perchan")]
AXES = {
    # M1 always keeps Telemanom pruning (project rule: it is M1's precision engine)
    "M1": dict(thr=["off"] + THR_Q + ["a10", "a20", "a40"], spans=["multi", "s15"],
               tele=TELE, **COMMON),
    # M2 never uses Telemanom pruning (project rule: 0.9644 -> 0.014 with it)
    "M2": dict(thr=THR_Q + ["a5", "a10", "a20", "a40"], spans=["s15", "multi"], tele=[None],
               **COMMON),
}
START = {
    "M1": dict(thr="a20", W="1d", spans="multi", tele="0.3/105/0.007/25", L=10, std=8, drift=0,
               close=60, minlen=3, onset=0, hyst=1.0, cross=0.7),
    "M2": dict(thr="a20", W="all", spans="s15", tele=None, L=0, std=0, drift=0, close=0,
               minlen=1, onset=0, hyst=1.0, cross=0.0),
}
SMOKE_AXES = {
    "M1": dict(thr=["a20", "q0.9999/global"], W=["1d", "1wx"], spans=["multi"],
               tele=["0.3/105/0.007/25", "0.13/5/0.0/100"], L=[10], std=[8], drift=[0, "d6"],
               close=[60], minlen=[3], onset=[0, 3], hyst=[1.0, 0.5], cross=[0.7]),
    "M2": dict(thr=["a20", "q0.9999/global"], W=["all", "1wx"], spans=["s15"], tele=[None],
               L=[0], std=[0, 5], drift=[0, "d6"], close=[0], minlen=[1], onset=[0],
               hyst=[1.0], cross=[0.0, 0.5]),
}
_W6 = ["1d", "1w", "1m", "6m", "all", "1wx"]
PROFILES = {
    "full": AXES,
    "big": {"M1": dict(AXES["M1"], spans=["s15"], W=_W6,
                       tele=[f"{p}/105/{me}/{bk}" for p in TELE_PS for me in MINERRS for bk in (100, 25)]),
            "M2": dict(AXES["M2"], spans=["s15"], W=_W6)},
    "tiny": {"M1": dict(AXES["M1"], spans=["s15"], W=["6m", "all", "1wx"],
                        tele=[f"{p}/105/0.007/25" for p in TELE_PS]),
             "M2": dict(AXES["M2"], spans=["s15"], W=["6m", "all", "1wx"])},
}
TYPING = [(t, p) for p in (3, 2) for t in (600, 1800, 3600, 7200)]

_G = {}          # state shared with forked workers: periods, scorers, mission


def key(cfg):
    return json.dumps(cfg, sort_keys=True)


# ------------------------------------------------------------------ worker tasks
def _z_task(a):
    name, W, spans = a
    return name, (W, spans), _G["P"][name].z_and_thresholds(W, spans)


def _tele_task(a):
    name, c, span, combos = a
    return name, c, _G["P"][name].telemanom(c, span, combos)


def _eval_task(cfg):
    PA, PB = _G["P"]["A"], _G["P"]["B"]
    xa, xb = assemble(PA, cfg), assemble(PB, cfg)
    S = _G["S"]
    ra, rb = S["A"].score(xa), S["B"].score(xb)
    rp = S["pool"].score(np.vstack([xa, xb]))
    return key(cfg), dict(pool=rp, A=ra, B=rb, rows=int(xa.max(1).sum() + xb.max(1).sum()))


def _eval_cheap(cfg):
    """Phase I of the steepest search: pooled event-wise metrics only (no channel / timing
    scoring, which dominates the cost at 47-58 scored channels)."""
    PA, PB = _G["P"]["A"], _G["P"]["B"]
    xa, xb = assemble(PA, cfg), assemble(PB, cfg)
    rp = _G["S"]["pool"].score(np.vstack([xa, xb]), full=False)
    return key(cfg), dict(pool=rp, rows=int(xa.max(1).sum() + xb.max(1).sum()))


# ------------------------------------------------------------------ building the folds
def load_fold(mk, fold, seed=0):
    rd = job_dir(mk, f"fold{fold}", seed)
    meta = json.load(open(f"{rd}/meta.json"))
    d = load_split(mk, "train")
    s, e = MISSIONS[mk]["folds"][fold]
    e0, e1 = row_of(d["ts"], s), row_end(d["ts"], e)
    assert meta["fit_end"] == e0 and meta["err_end"] == e1, "fold boundaries changed"
    err = np.load(f"{rd}/err_train.npy").astype(np.float32)    # scored channels only
    tix = target_idx(mk)
    assert err.shape[1] == len(tix), "error file / target channels mismatch"
    return Period(mk, fold, err, np.asarray(d["vals"][:e1])[:, tix], np.asarray(d["count"][:e1])[:, tix],
                  d["ts"], np.asarray(meta["mu"])[tix], np.asarray(meta["sd"])[tix],
                  meta["rf"], meta["fit_end"], e0, e1)


def build_base(periods, axes, log):
    """Histograms for every smoothing span + drift statistics. Runs in the parent BEFORE any
    worker is forked, so the workers inherit them (copy-on-write) instead of rebuilding."""
    t0 = time.time()
    need_sp = sorted({s for sp in axes["spans"] for s in spans_of(sp)})
    for P in periods.values():
        P.build_tz(need_sp)
        P.build_drift(sorted({d[0] for d in axes["drift"] if d}))
    log(f"  baselines built ({time.time() - t0:.0f}s)")


def precompute(pool, periods, axes, log, tpool=None):
    """All Z / thresholds / Telemanom masks the axes can ask for (parallel). Z tasks run on
    `pool` (memory-limited), Telemanom channel jobs on `tpool` (all cores)."""
    t0 = time.time()
    tasks = [(n, W, sp) for n in periods for W in axes["W"] for sp in axes["spans"]]
    for name, k, v in pool.map(_z_task, tasks):
        periods[name].zs[k] = v
    pool = tpool or pool
    log(f"  {len(tasks)} Detector-B baselines ({time.time() - t0:.0f}s)")
    combos = {tele_key(t) for t in axes["tele"] if t}
    if combos:
        tasks = [(n, c, sp, combos) for n in periods for c in range(periods[n].C)
                 for sp in sorted({k[1] for k in combos})]
        for name, c, res in pool.map(_tele_task, tasks):
            P = periods[name]
            for k, v in res.items():
                P.tele.setdefault(k, np.zeros((P.n, P.C), np.uint8))[:, c] = v
        log(f"  Telemanom: {len(combos)} settings x {len(tasks)} channel jobs ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ memory
def ram_budget():
    """Bytes the search may use: NR_RAM_GB if set, else 80 % of the container / machine."""
    if os.environ.get("NR_RAM_GB"):
        return float(os.environ["NR_RAM_GB"]) * 2 ** 30
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(f).read().strip()
            if v != "max" and int(v) < 2 ** 50:
                return 0.8 * int(v)
        except Exception:
            pass
    return 0.8 * os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def need_bytes(P, axes):
    """-> (bytes held at the end of the precompute, of which freed before the search,
    temporary bytes per Z worker, per search worker)."""
    spans = {s for sp in axes["spans"] for s in spans_of(sp)}
    nz = len(axes["W"]) * len(axes["spans"])
    ntele = len({tele_key(t) for t in axes["tele"] if t})
    ndrift = len({d[0] for d in axes["drift"] if d})
    held = free = ztmp = etmp = 0
    for p in P.values():
        e1, n, C, nb = p.e1, p.n, p.C, -(-p.e1 // p.R)
        ci = p._count_all.itemsize
        hist = e1 * C * (8 + ci)                                              # err, vals, count
        base = len(spans) * C * (e1 * 4 + (nb + 1) * 512 * 4)                 # baselines per span
        held += hist + n * C * 4 + base + nz * n * C * 4 + ntele * n * C + ndrift * n * C * 4
        free += hist + base - n * C * (4 + ci)                                # scored rows stay
        ztmp = max(ztmp, 4 * e1 * C * 4)          # one full-length Z + quantile copies
        etmp += 12 * n * C                         # one setting's alarm arrays, both folds
    return held, free, ztmp, etmp


def rss_gb():
    import resource
    k = 1 if os.uname().sysname == "Darwin" else 1024
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * k / 2 ** 30,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * k / 2 ** 30)


# ------------------------------------------------------------------ search
def objective(r, best_ew=None):
    ew = r["pool"].get("global_EW_F_0.50", 0.0)
    if best_ew is None:
        return ew
    if ew < best_ew - 0.02:
        return -1.0 + ew
    return min(np.nan_to_num(r["A"].get("S5", 0.0)), np.nan_to_num(r["B"].get("S5", 0.0)))


def coord(pool, axes, start, cache, obj, passes, log, tag, fn=_eval_task, cheap=None, floor=None):
    """fn scores a setting. With `cheap` and `floor`: a setting whose cheap pooled event-wise
    F0.5 is already below `floor` is not scored again (the objective cannot choose it, and its
    value -1 + F0.5 is known from the cheap score)."""
    cur = dict(start)

    def run(cfgs):
        out, todo = {}, []
        for c in cfgs:
            k = key(c)
            if k in cache:
                out[k] = cache[k]
            elif cheap is not None and k in cheap and cheap[k]["pool"].get("global_EW_F_0.50", 0) < floor:
                out[k] = cheap[k]
            else:
                todo.append(c)
        for k, r in pool.map(fn, todo, chunksize=1):
            cache[k] = r
            out[k] = r
        return [out[key(c)] for c in cfgs]

    best = obj(run([cur])[0])
    for ps in range(passes):
        moved = False
        for ax, vals in axes.items():
            cands = [dict(cur, **{ax: v}) for v in vals if v != cur[ax]]
            if not cands:
                continue
            for c, r in zip(cands, run(cands)):
                f = obj(r)
                if f > best + 1e-9:
                    best, cur, moved = f, c, True
            log(f"    {tag} pass{ps + 1} {ax:>6} -> {str(cur[ax]):>18}  obj={best:.4f}  ({len(cache)} evals)")
        if not moved:
            break
    return cur, best


def steepest(pool, axes, start, cache, obj, fn, log, tag, max_iter=15):
    """Steepest ascent: every single-setting change of the current point is scored at once
    (all cores busy), then the best one is taken. Stops when no change improves."""
    cur = dict(start)

    def run(cfgs):
        todo = [c for c in cfgs if key(c) not in cache]
        for k, r in pool.map(fn, todo, chunksize=1):
            cache[k] = r
        return [cache[key(c)] for c in cfgs]

    best = obj(run([cur])[0])
    for it in range(max_iter):
        cands = [dict(cur, **{ax: v}) for ax, vals in axes.items() for v in vals if v != cur[ax]]
        res = run(cands)
        j = int(np.argmax([obj(r) for r in res]))
        f = obj(res[j])
        if f <= best + 1e-9:
            break
        changed = [ax for ax in axes if cands[j][ax] != cur[ax]][0]
        best, cur = f, cands[j]
        log(f"    {tag} step {it + 1}: {changed} -> {cur[changed]}  obj={best:.4f}  ({len(cache)} evals)")
    return cur, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", required=True, choices=list(MISSIONS))
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    mk = a.mission
    out = f"{OUT}/{mk}/frozen.json"
    os.makedirs(f"{OUT}/{mk}", exist_ok=True)
    logf = open(f"{OUT}/{mk}/select.log", "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    t0 = time.time()
    P = {f: load_fold(mk, f, a.seed) for f in ("A", "B")}
    budget = ram_budget()
    if a.smoke:
        prof, axes = "smoke", SMOKE_AXES[mk]
    else:
        order = ["full", "big", "tiny"] if os.environ.get("NR_AXES", "auto") == "auto" else [os.environ["NR_AXES"]]
        for prof in order:
            axes = PROFILES[prof][mk]
            held, free, ztmp, etmp = need_bytes(P, axes)
            if (held + ztmp <= budget and held - free + etmp <= budget) or prof == order[-1]:
                break
    held, free, ztmp, etmp = need_bytes(P, axes)
    zw = int(max(1, min(a.workers, (budget - held) // max(ztmp, 1))))
    ew_ = int(max(1, min(a.workers, (budget - held + free) // max(etmp, 1))))
    log(f"[{mk}] axes profile '{prof}': est. {held / 2**30:.1f} GB after precompute ({(held - free) / 2**30:.1f} GB"
        f" in the search) + {ztmp / 2**30:.2f} GB per Z worker / {etmp / 2**30:.2f} GB per search worker;"
        f" budget {budget / 2**30:.1f} GB -> workers: Z {zw}, Telemanom {a.workers}, search {ew_}")
    start = dict(START[mk])
    for ax, vals in axes.items():                     # keep the start point inside the axes
        if start[ax] not in vals:
            start[ax] = vals[0]
    assert P["A"].e1 == P["B"].e0, "folds must be contiguous"
    ts = P["B"].ts
    S = {"A": Scorer(mk, ts[P["A"].e0:P["A"].e1]), "B": Scorer(mk, ts[P["B"].e0:P["B"].e1]),
         "pool": Scorer(mk, ts[P["A"].e0:P["B"].e1])}
    log(f"[{mk}] select: events A={S['A'].n_events} B={S['B'].n_events} pooled={S['pool'].n_events}"
        f" | workers={a.workers} ({time.time() - t0:.0f}s)")
    _G.update(P=P, S=S)
    build_base(P, axes, log)
    ctx = get_context("fork")
    with ProcessPoolExecutor(zw, mp_context=ctx) as zpool, ProcessPoolExecutor(a.workers, mp_context=ctx) as tpool:
        precompute(zpool, P, axes, log, tpool)
    for p in P.values():                  # the search reads only the stored arrays
        p.release_history()
    import gc
    gc.collect()
    log(f"  peak memory so far: {rss_gb()[0]:.1f} GB (this process), {rss_gb()[1]:.1f} GB (largest worker)")
    cache, cheap = {}, {}
    with ProcessPoolExecutor(ew_, mp_context=ctx) as pool:             # re-fork: sees precomputed blocks
        if SEARCH == "steepest":
            c1, ew = steepest(pool, axes, start, cheap, objective, _eval_cheap, log, "I ",
                              max_iter=2 if a.smoke else 15)
            log(f"  phase I: best pooled F0.5 {ew:.4f} ({len(cheap)} cheap evals, {time.time() - t0:.0f}s)")
            o2 = lambda r: objective(r, ew)                              # noqa: E731
            top = sorted(cheap.items(), key=lambda kr: -kr[1]["pool"].get("global_EW_F_0.50", 0))[:16]
            for k, r in pool.map(_eval_task, [json.loads(k) for k, _ in top], chunksize=1):
                cache[k] = r
            c2s = max(((json.loads(k), r) for k, r in cache.items()), key=lambda kr: o2(kr[1]))[0]
            steepest(pool, axes, c2s, cache, o2, _eval_task, log, "II", max_iter=2 if a.smoke else 15)
        else:
            passes = 1 if a.smoke else 3
            c1, ew = coord(pool, axes, start, cheap, objective, passes, log, "I ", fn=_eval_cheap)
            near = [json.loads(k) for k, r in cheap.items() if r["pool"].get("global_EW_F_0.50", 0) >= ew - 0.02]
            log(f"  phase I: best pooled F0.5 {ew:.4f} ({len(cheap)} cheap evals, {time.time() - t0:.0f}s);"
                f" full scoring of the {len(near)} settings within 0.02")
            for k, r in pool.map(_eval_task, near, chunksize=1):
                cache[k] = r
            o2 = lambda r: objective(r, ew)                              # noqa: E731
            c2s = max(((json.loads(k), r) for k, r in cache.items()), key=lambda kr: o2(kr[1]))[0]
            coord(pool, axes, c2s, cache, o2, passes, log, "II", fn=_eval_task, cheap=cheap, floor=ew - 0.02)
    best_ew = max([r["pool"].get("global_EW_F_0.50", 0) for r in list(cache.values()) + list(cheap.values())])
    ranked = sorted(((json.loads(k), r) for k, r in cache.items()
                     if r["pool"].get("global_EW_F_0.50", 0) >= best_ew - 0.02),
                    key=lambda kr: (objective(kr[1], best_ew), kr[1]["pool"]["S5"]),
                    reverse=True)
    cfg, r = ranked[0]
    log(f"  SELECTED {cfg}")
    for part in ("pool", "A", "B"):
        log(f"    {part:>4}: {fmt(r[part])}")

    # ---- alarm typing: commanded (expected) vs uncommanded (anomaly alarm) ----
    G = np.vstack([assemble(P["A"], cfg), assemble(P["B"], cfg)])
    tsec = ts[P["A"].e0:P["B"].e1].astype(np.int64) / 1e9
    typing = {"none": S["pool"].score(G, SEL_ANOM)}
    for t, p in TYPING:
        Gt, n = D.typed(G, tsec, fire_seconds(mk, p), t)
        typing[f"{t}_{p}"] = dict(S["pool"].score(Gt, SEL_ANOM), n_commanded=n)
    tbest = max(typing, key=lambda k: (typing[k].get("S5", 0) if np.isfinite(typing[k].get("S5", np.nan)) else -1))
    log("  typing (anomaly-only S5): " + " ".join(f"{k}={v.get('S5', float('nan')):.3f}" for k, v in typing.items())
        + f" -> {tbest}")
    json.dump(dict(mission=mk, detector=cfg, typing=tbest,
                   metrics={p: {k: r[p].get(k) for k in SHOW} for p in ("pool", "A", "B")},
                   typing_scores={k: {kk: v.get(kk) for kk in SHOW + ["n_commanded"]} for k, v in typing.items()},
                   events={k: s.n_events for k, s in S.items()}, n_evals=len(cache) + len(cheap),
                   search=SEARCH, axes_profile=prof, axes={k: v for k, v in axes.items()},
                   secs=time.time() - t0, smoke=a.smoke,
                   top10=[dict(cfg=c, pool_EW=x["pool"]["global_EW_F_0.50"], pool_S5=x["pool"]["S5"],
                               S5_A=x["A"]["S5"], S5_B=x["B"]["S5"]) for c, x in ranked[:10]]),
              open(out, "w"), indent=1)
    import csv
    keys = ["global_EW_F_0.50", "global_EW_precision", "global_EW_recall", "chan_channel_F0.50",
            "chan_subsystem_F0.50", "global_alarming_precision", "adtqc_Total", "global_AFF_F_0.50", "S5"]
    with open(f"{OUT}/{mk}/search_trace.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        axes_names = list(axes)
        w.writerow(["order"] + axes_names + [f"{p}_{k}" for p in ("pool", "A", "B") for k in keys] + ["alarm_rows"])
        for i, (k, r) in enumerate(list(cheap.items()) + list(cache.items())):
            c = json.loads(k)
            w.writerow([i] + [c.get(x) for x in axes_names] +
                       [r.get(p, {}).get(kk) for p in ("pool", "A", "B") for kk in keys] + [r["rows"]])
    log(f"  frozen -> {out} ({len(cheap)} cheap + {len(cache)} full scorings, {time.time() - t0:.0f}s, "
        f"peak memory {rss_gb()[0]:.1f} GB + largest worker {rss_gb()[1]:.1f} GB); "
        f"every setting tried: {OUT}/{mk}/search_trace.csv")


if __name__ == "__main__":
    main()
