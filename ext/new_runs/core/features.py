"""Model inputs.

Channel nodes: [value_z, gap_flag, log10(1 + tsince) / 7] per row (as published).
Command path:  5 numbers per command per row, from that command's firing times up to the row:
               recency exp(-dt/1800), log time since last firing, log #firings in last
               1 h / 6 h / 24 h.

M2 has 123 priority>=2 commands firing ~1,500 times a day; computing their features for every
batch on the fly is what made the command-path model CPU-bound. `precompute` writes the whole
(T, n_cmd, 5) table once per split as float16 .npy (M2 ~3.8 GB, M1 ~2.2 GB); training jobs
memory-map it, so concurrent jobs share one copy through the page cache.

  python -m core.features --mission M1 [--split train|test|both]   # build the caches
  python -m core.features --selftest                                # cache == on-the-fly
"""
import argparse
import os
import time

import numpy as np

from .config import CACHE, SMOKE_ROWS, TC_TAU, TRAIN, TRAIN_DAYS
from .data import command_times, load_split

LOG_DT_MAX = np.log1p(1e7)
WINDOWS = (3600.0, 21600.0, 86400.0)


def channel_tensor(vals, count, tsince, mu, sd):
    T, C = vals.shape
    X = np.empty((T, C, 3), np.float32)
    X[:, :, 0] = (np.asarray(vals) - mu) / sd
    X[:, :, 1] = np.asarray(count) == 0
    X[:, :, 2] = np.log10(1.0 + np.asarray(tsince)) / 7.0
    return X


def seconds(ts):
    return ts.astype("datetime64[ns]").astype(np.int64) / 1e9


def _features(tt, fires, out):
    """Fill out[(rows), j, 0:5] for query times tt (seconds) and per-command firing times."""
    for j, f in enumerate(fires):
        i_now = np.searchsorted(f, tt, "right")
        last = f[np.maximum(i_now - 1, 0)] if len(f) else np.zeros_like(tt)
        dt = np.where(i_now > 0, np.minimum(tt - last, 1e7), 1e7)
        out[:, j, 0] = np.exp(-dt / TC_TAU)
        out[:, j, 1] = np.log1p(dt) / LOG_DT_MAX
        for k, w in enumerate(WINDOWS):
            out[:, j, 2 + k] = np.log1p(i_now - np.searchsorted(f, tt - w, "right")) / 4.0


def cache_path(mk, split, pmin):
    if SMOKE_ROWS or TRAIN_DAYS:  # row-sliced splits: a full-length cache would be misaligned
        return None
    return f"{CACHE}/{mk}_{split}_cmd_p{pmin}.npy"


class CmdFeats:
    """Command features for any rows: read from the cache if it exists, else computed."""

    def __init__(self, ts, cmds, cache=None):
        self.t = seconds(ts)
        self.f = [seconds(c[2]) for c in cmds]
        self.n = len(cmds)
        self.cache = np.load(cache, mmap_mode="r") if cache and os.path.exists(cache) else None

    def at(self, rows, dtype=np.float32):
        rows = np.asarray(rows)
        if self.cache is not None:
            return np.asarray(self.cache[rows.ravel()], dtype).reshape(*rows.shape, self.n, 5)
        out = np.empty((rows.size, self.n, 5), np.float32)
        _features(self.t[rows.ravel()], self.f, out)
        return out.astype(dtype, copy=False).reshape(*rows.shape, self.n, 5)


_W = {}


def _chunk(a_b):
    """Worker: fill rows [a, b) of the memory-mapped cache file."""
    a, b = a_b
    t, fires, tmp = _W["t"], _W["fires"], _W["tmp"]
    mm = np.load(tmp, mmap_mode="r+")
    buf = np.empty((b - a, len(fires), 5), np.float32)
    _features(t[a:b], fires, buf)
    mm[a:b] = buf
    mm.flush()
    return b - a


def precompute(mk, split, pmin, chunk=200_000, workers=None):
    """Write the (T, n_cmd, 5) float16 cache, rows split across all cores."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    path = cache_path(mk, split, pmin)
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ts = load_split(mk, split)["ts"]
    cmds = command_times(mk, pmin)
    tmp = path + ".tmp.npy"
    mm = np.lib.format.open_memmap(tmp, "w+", np.float16, (len(ts), len(cmds), 5))
    del mm
    _W.update(t=seconds(ts), fires=[seconds(c[2]) for c in cmds], tmp=tmp)
    jobs = [(a, min(a + chunk, len(ts))) for a in range(0, len(ts), chunk)]
    workers = workers or max(1, min(len(jobs), os.cpu_count() or 1))
    with ProcessPoolExecutor(workers, mp_context=get_context("fork")) as ex:
        assert sum(ex.map(_chunk, jobs)) == len(ts)
    os.replace(tmp, path)
    return path


def build_inputs(mk, d, mu, sd, pmin=None):
    """-> (X (T, C, 3) float32, CmdFeats) for one split dict from data.load_split."""
    pmin = TRAIN["tc_min_priority"] if pmin is None else pmin
    X = channel_tensor(d["vals"], d["count"], d["tsince"], mu, sd)
    split = d.get("split", "train")
    return X, CmdFeats(d["ts"], command_times(mk, pmin), cache_path(mk, split, pmin))


def _selftest():
    """The cache and the on-the-fly path must give the same numbers (to float16 precision)."""
    import tempfile
    from . import features as F
    for mk in ("M1", "M2"):
        d = load_split(mk, "train")
        n = 50_000
        ts = d["ts"][-n:]
        cmds = command_times(mk, 2)
        fly = CmdFeats(ts, cmds)
        with tempfile.TemporaryDirectory() as tmp:
            p = f"{tmp}/c.npy"
            mm = np.lib.format.open_memmap(p, "w+", np.float16, (n, len(cmds), 5))
            buf = np.empty((n, len(cmds), 5), np.float32)
            F._features(seconds(ts), [seconds(c[2]) for c in cmds], buf)
            mm[:] = buf
            del mm
            cached = CmdFeats(ts, cmds, p)
            rows = np.random.default_rng(0).integers(0, n, (4, 64))
            a, b = fly.at(rows), cached.at(rows)
            err = np.abs(a - b).max()
            print(f"{mk}: {len(cmds)} commands, max |cache - on-the-fly| = {err:.2e}")
            assert err < 4e-3, err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission")
    ap.add_argument("--split", default="both", choices=["train", "test", "both"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        for sp in (("train", "test") if a.split == "both" else (a.split,)):
            t0 = time.time()
            p = precompute(a.mission, sp, TRAIN["tc_min_priority"], workers=a.workers or None)
            print(f"{a.mission} {sp}: {p} ({os.path.getsize(p)/1e9:.2f} GB, {time.time()-t0:.0f}s)",
                  flush=True)
