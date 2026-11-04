"""Per-subsystem results + one mission-level score from all subsystem models.

  NR_SET=full python -m core.combine          # after run_sub.py; writes out_sub/SUMMARY.md

Each target channel belongs to exactly one model, so the mission's alarm matrix is the models'
per-channel alarms side by side; it is scored over ALL target channels with ESA's metrics,
exactly like a full-set run. Channels whose model has not run yet stay silent (listed).
"""
import json
import os

import numpy as np

from .config import ROOT, MISSIONS, SET
from .data import load_split
from .score import Scorer, SEL, SEL_ANOM, SHOW, fmt

SUBDIR = f"{ROOT}/out_sub"
COLS = [("global_EW_F_0.50", "EW F0.5"), ("global_EW_precision", "P"), ("global_EW_recall", "R"),
        ("chan_channel_F0.50", "chan F0.5"), ("chan_subsystem_F0.50", "subsys F0.5"),
        ("global_alarming_precision", "alarm P"), ("adtqc_Total", "ADTQC"), ("global_AFF_F_0.50", "aff F0.5")]


def row(r):
    return " | ".join("-" if k not in r or r[k] != r[k] else f"{r[k]:.3f}" for k, _ in COLS)


def main():
    assert SET == "full", "run with NR_SET=full (scores over every target channel)"
    subsets = json.load(open(f"{ROOT}/core/subsets.json"))
    md = ["# Per-subsystem models", "",
          "Detector setting chosen per model on its two validation folds (never on the test split).", ""]
    for mk in ("M1", "M2"):
        models = {m: v for m, v in subsets.items() if v["mission"] == mk}
        if not any(os.path.exists(f"{SUBDIR}/{m}/{mk}/result.json") for m in models):
            continue                                   # nothing of this mission has run here
        try:
            load_split(mk, "test")
        except FileNotFoundError:
            print(f"[{mk}] test timestamps not available here; combined score skipped")
            continue
        md += [f"## {mk}", "", "| model | targets | context | test events (anomalies) | "
               + " | ".join(n for _, n in COLS) + " | train min |",
               "|" + "---|" * (len(COLS) + 5)]
        tg = MISSIONS[mk]["targets"]
        ts = load_split(mk, "test")["ts"]
        X = np.zeros((len(ts), len(tg)), np.uint8)
        Xt = np.zeros_like(X)
        missing = []
        for m, v in models.items():
            f = f"{SUBDIR}/{m}/{mk}/result.json"
            if not os.path.exists(f):
                missing += v["targets"]
                md.append(f"| {m} | {len(v['targets'])} | {len(v['context'])} | not run yet |")
                continue
            r = json.load(open(f))
            z = np.load(f"{SUBDIR}/{m}/{mk}/alarms_test_mask.npz")
            assert z["X"].shape[0] == len(ts), (m, z["X"].shape, len(ts))
            for i, c in enumerate(z["chs"]):
                X[:, tg.index(str(c))] = z["X"][:, i]
                Xt[:, tg.index(str(c))] = z["Xt"][:, i]
            ev = r["events"]
            n = sum(e["n"] for e in ev.values())
            md.append(f"| {m} | {len(v['targets'])} | {len(v['context'])} | {n} ({ev['Anomaly']['n']}) | "
                      f"{row(r['headline'])} | {sum(r['model']['train_s']) / 60:.0f} |")
        S = Scorer(mk, ts, chs=tg)
        head, anom, typed = S.score(X, SEL), S.score(X, SEL_ANOM), S.score(Xt, SEL_ANOM)
        md += ["", f"**{mk} combined, all {len(tg)} target channels** "
               f"({S.n_events['sel']} events, {S.n_events['anom']} anomalies)", "",
               "| selection | " + " | ".join(n for _, n in COLS) + " |", "|" + "---|" * (len(COLS) + 1),
               f"| anomalies + rare events | {row(head)} |", f"| anomalies | {row(anom)} |",
               f"| anomalies, uncommanded alarms | {row(typed)} |", ""]
        if missing:
            md += [f"Not run yet (silent in the combined score): {len(missing)} channels.", ""]
        print(f"[{mk}] combined headline {fmt(head)}")
        json.dump(dict(headline=head, anom=anom, anom_typed=typed, missing=missing),
                  open(f"{SUBDIR}/{mk}_combined.json", "w"), indent=1, default=float)
    open(f"{SUBDIR}/SUMMARY.md", "w").write("\n".join(md) + "\n")
    print(f"-> {SUBDIR}/SUMMARY.md")


if __name__ == "__main__":
    main()
