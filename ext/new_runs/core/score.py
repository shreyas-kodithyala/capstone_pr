"""Scoring with ESA's metric classes + the five-aspect summary and the selection rule.

Aspects (each in [0, 1]):
  a1 event-wise F0.5   a2 mean(channel-aware, subsystem-aware F0.5)   a3 alarming precision
  a4 ADTQC (timing)    a5 affiliation F0.5                             S5 = mean(a1..a5)
"""
import contextlib
import io

import numpy as np
import pandas as pd

from .config import MISSIONS
from .esa import ESAScores, ChannelAwareFScore, ADTQC, BETA, series, build_labels_df, \
    subsystems_mapping

SEL = {"Category": ["Rare Event", "Anomaly"]}     # ESA's headline selection
SEL_ANOM = {"Category": ["Anomaly"]}              # ESA's anomaly-only selection
KEYS = ["global_EW_F_0.50", "chan_F", "global_alarming_precision", "adtqc_Total",
        "global_AFF_F_0.50"]
SHOW = ["global_EW_F_0.50", "global_EW_precision", "global_EW_recall", "chan_channel_F0.50",
        "chan_subsystem_F0.50", "global_alarming_precision", "adtqc_Total", "global_AFF_F_0.50",
        "S5"]


class Scorer:
    """Label table for one scored period, built once; scores (T, C) 0/1 predictions."""

    def __init__(self, mk, ts, chs=None):
        m = MISSIONS[mk]
        self.chs, raw = (m["targets"] if chs is None else chs), m["raw"]
        self.ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)
        lab = build_labels_df(raw, self.ts)
        self.y = lab[lab["Channel"].isin(self.chs)].reset_index(drop=True)
        self.sub = subsystems_mapping(raw, self.chs)
        self.n_events = {k: int(self.y[self.y.Category.isin(v["Category"])].ID.nunique())
                         for k, v in (("sel", SEL), ("anom", SEL_ANOM))}

    def score(self, P, sel=SEL, full=True):
        """All labels are passed and `sel` picks the scored ones, as ESA's harness does: an
        alarm on an unselected event is then neither a hit nor a false alarm."""
        if not self.y.Category.isin(sel["Category"]).any():
            return {"S5": float("nan")}
        if not P.any():
            return add_aspects({"global_EW_F_0.50": 0.0, "global_EW_precision": 0.0,
                                "global_EW_recall": 0.0, "global_alarming_precision": 0.0,
                                "global_AFF_F_0.50": 0.0, "chan_channel_F0.50": 0.0,
                                "chan_subsystem_F0.50": 0.0, "adtqc_Total": 0.0})
        out = {}
        y = self.y
        with contextlib.redirect_stdout(io.StringIO()):
            for k, v in ESAScores(betas=BETA, select_labels=sel).score(
                    y.drop(columns=["Channel"]), series(self.ts, P.max(axis=1))).items():
                out[f"global_{k}"] = float(v)
            if not full:
                return out
            yp = {c: series(self.ts, P[:, i]) for i, c in enumerate(self.chs)}
            try:
                for k, v in ChannelAwareFScore(beta=BETA, select_labels=sel).score(
                        y.copy(), {c: d.copy() for c, d in yp.items()}, self.sub).items():
                    out[f"chan_{k}"] = float(v)
            except Exception:
                out["chan_channel_F0.50"] = out["chan_subsystem_F0.50"] = 0.0
            try:
                for k, v in ADTQC(select_labels=sel).score(
                        y.copy(), {c: d.copy() for c, d in yp.items()}).items():
                    out[f"adtqc_{k}"] = float(v)
            except Exception:
                out["adtqc_Total"] = 0.0
        return add_aspects(out)


def _nz(x):
    return 0.0 if x is None or not np.isfinite(x) else float(x)


def add_aspects(r):
    r["chan_F"] = 0.5 * (_nz(r.get("chan_channel_F0.50")) + _nz(r.get("chan_subsystem_F0.50")))
    r["S5"] = float(np.mean([_nz(r.get(k)) for k in KEYS]))
    return r


def fmt(r, keys=SHOW):
    return "  ".join(f"{k.replace('global_', '').replace('chan_', '')}="
                     f"{r.get(k, float('nan')):.4f}" for k in keys)
