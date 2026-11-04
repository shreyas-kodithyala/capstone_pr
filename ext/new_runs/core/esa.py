"""ESA-ADB's own metric classes (vendored, unchanged) + the input preparation of its harness.

Validated in the main project: this path reproduces the official Docker scores to 1e-12.
"""
import numpy as np
import pandas as pd

from . import config  # noqa: F401  (puts vendor/ on sys.path)
from timeeval.metrics import ESAScores, ChannelAwareFScore, ADTQC  # noqa: E402,F401

BETA = 0.5


def change_points(pred):
    """Rows where a 0/1 series changes value, plus first and last row. ESA's metrics only use
    alarm starts / ends, so scoring these rows equals scoring all rows (much less memory)."""
    pred = np.asarray(pred)
    keep = np.empty(len(pred), bool)
    keep[0] = True
    np.not_equal(pred[1:], pred[:-1], out=keep[1:])
    keep[-1] = True
    return np.flatnonzero(keep)


def series(ts, pred):
    """One 0/1 series in ESA's format: DataFrame[Timestamp, Score] at change points."""
    pred = np.asarray(pred).astype(np.uint8)
    i = change_points(pred)
    return pd.DataFrame({"Timestamp": ts.iloc[i].reset_index(drop=True), "Score": pred[i]})


def build_labels_df(raw, ts):
    """Ground truth exactly as ESA's harness builds it: events fully inside the scored range,
    joined with the last four anomaly_types columns by ID (not de-duplicated)."""
    lab = pd.read_csv(f"{raw}/labels.csv", parse_dates=["StartTime", "EndTime"])
    for c in ("StartTime", "EndTime"):
        lab[c] = lab[c].apply(lambda t: t.tz_localize(None) if t.tzinfo else t)
    lab = lab[(lab["StartTime"] >= ts.min()) & (lab["EndTime"] <= ts.max())]
    at = pd.read_csv(f"{raw}/anomaly_types.csv")
    cols = list(at.columns[-4:])
    for c in cols:
        lab[c] = ""
    for _, row in at.iterrows():
        lab.loc[lab["ID"] == row["ID"], cols] = row[cols].values
    return lab


def subsystems_mapping(raw, channels):
    ch = pd.read_csv(f"{raw}/channels.csv")
    ch = ch[ch["Channel"].isin(channels)]
    return {s: [*v] for s, v in ch.groupby("Subsystem")["Channel"]}
