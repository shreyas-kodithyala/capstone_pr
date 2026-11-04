"""Build all_metrics.md and per_event_detection.csv from the stored final results. No model runs."""
import json, os
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{H}/results.json"))
S = ["SAT_1", "SAT_2", "SAT_3", "SAT_4"]
ARM = dict(solo="Solo", fed="Plain FL", pers="FL + personalization")
SEL = dict(local="own settings", federated="fleet settings")
f = lambda x, n=3: "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"
K = [("F0.5", "global_EW_F_0.50"), ("Precision", "global_EW_precision"), ("Recall", "global_EW_recall"),
     ("Alarm precision", "global_alarming_precision"), ("AFF F0.5", "global_AFF_F_0.50"),
     ("Channel F0.5", "chan_channel_F0.50"), ("Subsystem F0.5", "chan_subsystem_F0.50"), ("ADTQC", "adtqc_Total")]
out = ["# All final-fleet metrics", "",
       "Built by `make_all_metrics.py` from `results.json` and `matched_*.json`. Test days 182-364, scored once, one seed.", ""]
for blk, title in (("all", "All events (anomalies + rare events)"), ("anomalies", "Anomalies only")):
    out += [f"## Fleet means: {title}", "", "| Arm | Settings | " + " | ".join(k for k, _ in K) + " |", "|---|---|" + "---|" * len(K)]
    for a in ARM:
        for s in SEL:
            v = [float(np.nanmean([R[a][q][s][blk].get(key, np.nan) for q in S])) for _, key in K]
            out.append(f"| {ARM[a]} | {SEL[s]} | " + " | ".join(f(x) for x in v) + " |")
    out.append("")
out += ["## Fleet totals", "", "| Arm | Settings | Caught of 59 | False alarms | Rare events caught of 30 | Unseen caught of 27 | Seen caught of 32 |", "|---|---|---|---|---|---|---|"]
rows = []
for a in ARM:
    for s in SEL:
        c = fa = rr = un = se = 0
        for q in S:
            r = R[a][q][s]; tf = R[a][q]["train_families"]
            ev = pd.DataFrame(r["events"]); c += int(ev.detected.sum()); fa += r["n_false_alarm_events"]; rr += r["n_rare_detected"]
            un += int(ev[~ev.family.isin(tf)].detected.sum()); se += int(ev[ev.family.isin(tf)].detected.sum())
            for e in r["events"]:
                rows.append(dict(arm=a, settings=s, satellite=q, **e))
        out.append(f"| {ARM[a]} | {SEL[s]} | {c} | {fa} | {rr} | {un} | {se} |")
out.append("")
pd.DataFrame(rows).to_csv(f"{H}/per_event_detection.csv", index=False)
out += ["## Per satellite", "", "| Satellite | Arm | Settings | F0.5 all | F0.5 anomalies | Precision anom | Recall anom | Channel F0.5 all | Subsystem F0.5 all | Alarm precision all | ADTQC all | Caught | False alarms | False-alarm time % | Rare caught | Median delay min | Family attribution |", "|" + "---|" * 17]
for q in S:
    for a in ARM:
        for s in SEL:
            r = R[a][q][s]; A, N = r["all"], r["anomalies"]
            out.append(f"| {q} | {ARM[a]} | {SEL[s]} | {f(A['global_EW_F_0.50'])} | {f(N['global_EW_F_0.50'])} | {f(N['global_EW_precision'])} | {f(N['global_EW_recall'])} | {f(A['chan_channel_F0.50'])} | {f(A['chan_subsystem_F0.50'])} | {f(A['global_alarming_precision'])} | {f(A.get('adtqc_Total'))} | {r['n_detected']}/{r['n_anomalies']} | {r['n_false_alarm_events']} | {f(100*r['false_alarm_time_share'],2)} | {r['n_rare_detected']}/{r['n_rare']} | {f(r.get('median_delay_min'),1)} | {f(r.get('family_attribution_acc'),2)} |")
out.append("")
out += ["## Recall by fault family", "", "| Satellite | Trained on | Arm | Settings | F1 wheel | F2 sensor | F3 power/thermal |", "|---|---|---|---|---|---|---|"]
for q in S:
    for a in ARM:
        for s in SEL:
            b = R[a][q][s]["recall_by_family"]
            out.append(f"| {q} | {' '.join(R[a][q]['train_families'])} | {ARM[a]} | {SEL[s]} | {f(b.get('F1'),2)} | {f(b.get('F2'),2)} | {f(b.get('F3'),2)} |")
out.append("")
out += ["## Same settings for all models (matched)", "", "Each row uses the fleet setting chosen by the named arm. Cells: F0.5 all / F0.5 anomalies / caught / false alarms.", "", "| Setting from | Satellite | Solo model | Plain FL model | FL + personalization model |", "|---|---|---|---|---|"]
M = {a: json.load(open(f"{H}/matched_{a}.json")) for a in ARM}
cell = lambda r: f"{f(r['all']['global_EW_F_0.50'])} / {f(r['anomalies']['global_EW_F_0.50'])} / {r['n_detected']} / {r['n_false_alarm_events']}"
for k in ARM:
    for q in S:
        out.append(f"| {ARM[k]} | {q} | " + " | ".join(cell(M[a][q][k]) for a in ARM) + " |")
    m = lambda a, b: float(np.mean([M[a][q][k][b]["global_EW_F_0.50"] for q in S]))
    out.append(f"| {ARM[k]} | **mean** | " + " | ".join(f"{f(m(a,'all'))} / {f(m(a,'anomalies'))}" for a in ARM) + " |")
out.append("")
out += ["## Settings chosen", "", "| Arm | Scope | Setting |", "|---|---|---|"]
keep = ("thr", "ceil", "W", "spans", "close", "minlen", "cross", "hyst", "drift", "std", "mask", "smask", "latch")
fm = lambda c: " ".join(f"{k}={c[k]}" for k in keep)
for a in ARM:
    out.append(f"| {ARM[a]} | fleet | {fm(R[a]['fed_selection']['cfg'])} |")
    for q in S:
        out.append(f"| {ARM[a]} | {q} own | {fm(R[a][q]['local']['cfg'])} |")
open(f"{H}/all_metrics.md", "w").write("\n".join(out) + "\n")
