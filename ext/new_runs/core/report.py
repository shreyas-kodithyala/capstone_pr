"""out/REPORT.md -- every number comes from a result file, nothing is typed in by hand
except ESA's published bests.

  python -m core.report
"""
import json
import os
import time

from .config import OUT, ROOT, MODEL, TRAIN, SET, MISSIONS

REF = f"{ROOT}/reference"
ROWS = [("event-wise F0.5 (headline)", "global_EW_F_0.50"), ("  precision", "global_EW_precision"),
        ("  recall", "global_EW_recall"), ("channel-aware F0.5", "chan_channel_F0.50"),
        ("subsystem-aware F0.5", "chan_subsystem_F0.50"),
        ("alarming precision", "global_alarming_precision"), ("ADTQC (timing)", "adtqc_Total"),
        ("affiliation F0.5", "global_AFF_F_0.50"), ("S5 (mean of the 5 aspects)", "S5")]
ESA_BEST = {
    "light": {"M1": {"global_EW_F_0.50": "0.786 (Telemanom-ESA-Pruned)", "chan_channel_F0.50": "0.837",
                     "global_alarming_precision": "0.875", "adtqc_Total": "0.901", "global_AFF_F_0.50": "0.734"},
              "M2": {"global_EW_F_0.50": "0.949 (Windowed iForest; SMED 0.971)"}},
    # full M1: ESA-ADB paper tables, as recorded in results/M1_full_summary.md
    "full": {"M1": {"global_EW_F_0.50": "0.061 (Telemanom-ESA-Pruned)", "global_EW_precision": "0.050",
                    "global_EW_recall": "0.967 (iForest)", "chan_channel_F0.50": "0.477", "chan_subsystem_F0.50": "0.689",
                    "global_alarming_precision": "0.206", "adtqc_Total": "0.752 (DC-VAE-ESA)",
                    "global_AFF_F_0.50": "0.607"},
             "M2": {}},
}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def v(d, k):
    x = None if d is None else d.get(k)
    return "—" if x is None else f"{x:.4f}"


def s5(d):
    """S5 for reference files that predate it."""
    if d is None:
        return None
    if "S5" in d:
        return d
    ks = ["global_EW_F_0.50", "global_alarming_precision", "adtqc_Total", "global_AFF_F_0.50"]
    if not all(k in d for k in ks + ["chan_channel_F0.50", "chan_subsystem_F0.50"]):
        return d
    a2 = 0.5 * (d["chan_channel_F0.50"] + d["chan_subsystem_F0.50"])
    return dict(d, S5=(sum(d[k] for k in ks) + a2) / 5)


def anom_view(d):
    return None if d is None else {k[5:]: x for k, x in d.items() if k.startswith("anom_")}


def main():
    L = [f"# New runs — combined model, both missions ({'lightweight subsets' if SET == 'light' else 'FULL channel sets'})", "",
         f"Input / scored channels: " + ", ".join(f"{k} {len(m['chs'])} / {len(m['targets'])}" for k, m in MISSIONS.items()), "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Model: `{MODEL}`; training: `{TRAIN}`.", ""]
    for mk in ("M1", "M2"):
        r = load(f"{OUT}/{mk}/result.json")
        f = load(f"{OUT}/{mk}/frozen.json")
        if SET == "light":
            v1 = s5(load(f"{REF}/v1_{mk}_full_scoreboard.json"))
            prev = load(f"{REF}/final_{mk}_baseline.json")
        else:                                   # full set: v1's Kaggle run (M1 only, non-causal)
            v1 = s5(load(f"{REF}/v1_{mk}full_result.json"))
            prev = None
        L += [f"## {mk}", ""]
        if r is None:
            L += ["(no test result yet)", ""]
            continue
        cols = [("v1 published" if SET == "light" else "v1 full-set run (Kaggle, non-causal)", v1),
                ("v1 model + causal detectors (claimable)", prev and prev["headline"]),
                ("fixed limits (normal min/max)", r["ool"]["minmax"]["headline"]),
                (f"**NEW combined model** ({len(r['seeds'])} seed)", r["headline"])]
        L += ["### Test, anomalies + rare events (ESA headline)", "",
              "| metric | " + " | ".join(c for c, _ in cols) + " | ESA best published |",
              "|" + "---|" * (len(cols) + 2)]
        for name, k in ROWS:
            L.append(f"| {name} | " + " | ".join(v(d, k) for _, d in cols) + f" | {ESA_BEST[SET][mk].get(k, '—')} |")
        acols = [("v1 published", anom_view(v1)), ("v1 model + causal detectors", prev and prev["anom_all"]),
                 ("fixed limits", r["ool"]["minmax"]["anom"]), ("NEW: all alarms", r["anom"]),
                 ("NEW: uncommanded alarms only", r["anom_typed"])]
        L += ["", "### Test, anomalies only (ESA's second label view)", "",
              "| metric | " + " | ".join(c for c, _ in acols) + " |", "|" + "---|" * (len(acols) + 1)]
        for name, k in ROWS:
            L.append(f"| {name} | " + " | ".join(v(d, k) for _, d in acols) + " |")
        L += ["", f"Share of test time in alarm: ours {r['alarm_time']:.2%}; fixed limits (normal min/max) "
                  f"{r['ool']['minmax']['alarm_time']:.2%}; fixed limits (0.1-99.9 percentiles) "
                  f"{r['ool']['p0.1']['alarm_time']:.2%} (F0.5 {r['ool']['p0.1']['headline']['global_EW_F_0.50']:.4f}).",
              "", "### Events caught on test", "",
              "| category | events | ours | fixed limits | ours but not limits | limits but not ours | by detector |",
              "|---|---|---|---|---|---|---|"]
        for c, e in r["events"].items():
            L.append(f"| {c} | {e['n']} | {e['ours']} | {e['ool']} | {e['ours_not_ool']} | "
                     f"{e['ool_not_ours']} | {e['by_detector']} |")
        if f:
            L += ["", "### Detector setting (chosen on unseen folds before the test)", "",
                  f"`{f['detector']}` · alarm typing `{f['typing']}` · {f['n_evals']} settings evaluated",
                  "", "| selection period | F0.5 | channel F0.5 | alarming P | ADTQC | affiliation | S5 |",
                  "|---|---|---|---|---|---|---|"]
            for p in ("A", "B", "pool"):
                m = f["metrics"][p]
                L.append(f"| {p} ({f['events'][p]}) | {v(m, 'global_EW_F_0.50')} | {v(m, 'chan_channel_F0.50')} | "
                         f"{v(m, 'global_alarming_precision')} | {v(m, 'adtqc_Total')} | "
                         f"{v(m, 'global_AFF_F_0.50')} | {v(m, 'S5')} |")
        mm = r["model"]
        L += ["", f"Model: {mm['n_params']:,} parameters, look-back {mm['rf']} rows, epochs {mm['epochs']}, "
                  f"trained on {mm['device']} in {[round(x / 60) for x in mm['train_s']]} min.", ""]
    t = load(f"{OUT}/timing.json")
    if t:
        L += ["## Run time", "", f"Total {t['total_min']:.0f} min on {t['hardware']}.", "",
              "| job | minutes |", "|---|---|"] + [f"| {k} | {x:.1f} |" for k, x in t["jobs"].items()]
    open(f"{OUT}/REPORT.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
