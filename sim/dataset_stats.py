"""Dataset characteristics next to ESA-ADB's: python dataset_stats.py [fleet_dir]"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "data", "sim_fleet")
ref = json.load(open(os.path.join(HERE, "esa_ref_labels.json")))
for m in ("M1", "M2"):
    print(f"ESA {m}: {ref[m]['pct_time_anomaly']:.2f} % anomalous, {ref[m]['pct_time_rare']:.2f} % rare, "
          f"{ref[m]['events_per_year']['Anomaly']:.1f} anomalies/yr, {ref[m]['events_per_year']['Rare Event']:.1f} rare/yr")
print("sat split | anomalies rare | families | % rows anomaly (without run-to-failure) | % rows rare | type mix")
for s in ("SAT_1", "SAT_2", "SAT_3", "SAT_4"):
    r = f"{FLEET}/{s}"
    t = pd.read_csv(f"{r}/truth/events_truth.csv")
    ty = pd.read_csv(f"{r}/anomaly_types.csv")
    lab = pd.read_csv(f"{r}/labels.csv", parse_dates=["StartTime", "EndTime"])
    wear = set(t[t.Code.isin(["W2", "W3"])].ID)
    for part in ("train", "test"):
        df = pd.read_csv(f"{r}/resampled_30s_{part}.csv.gz", usecols=lambda c: c == "timestamp" or c.startswith("is_anomaly"))
        v = df.drop(columns="timestamp").values
        ts = pd.to_datetime(df.timestamp).dt.tz_localize(None)
        keep = (v == 1).any(1)
        w = pd.Series(False, index=ts.index)
        for a, b in lab[lab.ID.isin(wear)][["StartTime", "EndTime"]].itertuples(index=False):
            a, b = a.tz_localize(None), b.tz_localize(None)
            w |= (ts >= a) & (ts <= b)
        e = t[(t.Split == part) & t.Visible]
        an, ra = e[e.Category == "Anomaly"], e[e.Category == "Rare Event"]
        mix = (ty[ty.ID.isin(an.ID)].Dimensionality.str[0] + ty[ty.ID.isin(an.ID)].Locality.str[0]).value_counts().to_dict()
        print(f"{s} {part:5s} | {len(an)} {len(ra)} | {an.FaultFamily.value_counts().to_dict()} | {100 * keep.mean():.2f} "
              f"({100 * (keep & ~w.values).mean():.2f}) | {100 * (v == 2).any(1).mean():.2f} | {mix}")
