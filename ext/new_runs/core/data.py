"""Data access. All arrays are memory-mapped .npy files written by make_bundle.py.

  data/<mk>_<split>_{vals,count,tsince,ts}.npy   lightweight channels, official 30 s / 18 s grid
  data/<mk>_commands.npz                          every telecommand's firing times (raw pickles)
  data/raw/ESA-Mission{1,2}/*.csv                 labels, anomaly types, channels, commands
Labels are only ever used as evaluation targets or to drop rows from the training loss.
"""
import json
import os

import numpy as np
import pandas as pd

from .config import DATA, MISSIONS, SET, SMOKE_ROWS, TRAIN_DAYS, TRAIN_FROM_END

SEL_CATS = ("Rare Event", "Anomaly")


def _rows(split, ts):
    """Row slice of a split: everything; for onboarding runs the first TRAIN_DAYS of the
    training split; for a plumbing test the last / first SMOKE_ROWS."""
    T = len(ts)
    if TRAIN_DAYS and split == "train" and TRAIN_FROM_END:
        return slice(int(np.searchsorted(ts, ts[-1] - np.timedelta64(int(TRAIN_DAYS * 86400), "s"))), T)
    if TRAIN_DAYS and split == "train":
        return slice(0, int(np.searchsorted(ts, ts[0] + np.timedelta64(int(TRAIN_DAYS * 86400), "s"))))
    if not SMOKE_ROWS or SMOKE_ROWS >= T:
        return slice(0, T)
    return slice(T - SMOKE_ROWS, T) if split == "train" else slice(0, SMOKE_ROWS)


def _cols(mk):
    """Subsystem runs: row indices of this model's channels in the channel-major files. The
    files hold every channel, or only those listed in data_full/<mk>_channels.json (laptop
    extracts made by make_sub_data.py)."""
    if SET != "sub":
        return None
    f = f"{DATA}/{mk}_channels.json"
    have = json.load(open(f)) if os.path.exists(f) else None
    return [have.index(c) if have else int(c.split("_")[1]) - 1 for c in MISSIONS[mk]["chs"]]


def _arr(p, name, rows=slice(None), cols=None):
    """(T, C) array. The full bundle stores each channel contiguously ('_cm', shape (C, T)):
    it compresses 2-40x better; the transposed memory map is read as (T, C) at no cost.
    With cols, only those channels are read into memory (contiguous reads)."""
    if os.path.exists(f"{p}_{name}_cm.npy"):
        a = np.load(f"{p}_{name}_cm.npy", mmap_mode="r")[:, rows]
        return (a if cols is None else a[cols]).T
    a = np.load(f"{p}_{name}.npy", mmap_mode="r")[rows]
    return a if cols is None else a[:, cols]


def load_split(mk, split):
    p = f"{DATA}/{mk}_{split}"
    ts = np.load(f"{p}_ts.npy").astype("datetime64[ns]")
    r, c = _rows(split, ts), _cols(mk)
    return dict(vals=_arr(p, "vals", r, c), count=_arr(p, "count", r, c), tsince=_arr(p, "tsince", r, c),
                ts=ts[r], chs=MISSIONS[mk]["chs"], split=split)


def target_idx(mk):
    """Column indices (into MISSIONS[mk]['chs']) of the channels ESA scores."""
    chs = MISSIONS[mk]["chs"]
    return [chs.index(c) for c in MISSIONS[mk]["targets"]]


def row_of(ts, when):
    """First row with timestamp >= when."""
    return int(np.searchsorted(ts, np.datetime64(pd.Timestamp(when))))


def row_end(ts, when):
    """Exclusive end row for a period ending at `when` (the whole split if when >= last)."""
    w = np.datetime64(pd.Timestamp(when))
    return len(ts) if w >= ts[-1] else int(np.searchsorted(ts, w))


def events(mk, cats=SEL_CATS):
    raw = MISSIONS[mk]["raw"]
    L = pd.read_csv(f"{raw}/labels.csv", parse_dates=["StartTime", "EndTime"])
    A = pd.read_csv(f"{raw}/anomaly_types.csv")
    L = L.merge(A[["ID", "Category"]], on="ID", how="left")
    for c in ("StartTime", "EndTime"):
        L[c] = L[c].dt.tz_localize(None)
    return L[L.Channel.isin(MISSIONS[mk]["chs"]) & L.Category.isin(cats)].reset_index(drop=True)



def label_mask(mk, ts, cats=SEL_CATS, chs=None):
    """(T, C) bool: rows labels.csv marks for each channel. Row i covers [ts[i], ts[i+1])."""
    chs = MISSIONS[mk]["chs"] if chs is None else chs
    col = {c: i for i, c in enumerate(chs)}
    out = np.zeros((len(ts), len(chs)), bool)
    for _, r in events(mk, cats).iterrows():
        if r.Channel not in col:            # e.g. a context channel's own labels, when only the
            continue                        # targets' mask is asked for
        a = max(int(np.searchsorted(ts, np.datetime64(r.StartTime), "right")) - 1, 0)
        b = int(np.searchsorted(ts, np.datetime64(r.EndTime), "right"))
        out[a:max(b, a + 1), col[r.Channel]] = True
    return out


def command_times(mk, min_priority):
    """[(name, priority, sorted datetime64[ns] firing times)] for Priority >= min_priority."""
    z = np.load(f"{DATA}/{mk}_commands.npz")
    out = []
    for name, pr, a, b in zip(z["names"], z["priority"], z["offsets"][:-1], z["offsets"][1:]):
        if pr >= min_priority:
            out.append((str(name), int(pr), z["times"][a:b].astype("datetime64[ns]")))
    return out


def fire_seconds(mk, min_priority):
    """All firing times (seconds, sorted) of commands with Priority >= min_priority."""
    t = [c[2] for c in command_times(mk, min_priority)]
    if not t:
        return np.zeros(0)
    return np.sort(np.concatenate(t).astype(np.int64) / 1e9)
