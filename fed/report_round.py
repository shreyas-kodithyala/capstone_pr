"""Print every metric of one round, per satellite: python report_round.py [out_dir]"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FED_OUT", os.path.join(HERE, "out_v2"))
FLEET = os.environ.get("SIM_FLEET") or os.path.join(os.path.dirname(HERE), "data", "sim_fleet")
R = json.load(open(f"{OUT}/results.json"))
SELS = ("local", "federated")
f = lambda v, n=3: "-" if v is None else (f"{v:.{n}f}" if isinstance(v, (int, float)) else str(v))
g = lambda d, k: d.get(f"global_{k}")
tag = lambda c: (f"{c.get('thr')} {c.get('ceil', '')} m{c.get('mask', 0)} s{c.get('smask', 0)} c{c.get('close')} h{c.get('latch', 0)} "
                 f"l{c.get('minlen')} W{c.get('W')} {c.get('spans')}")
HDR = ("arm sel | val | F.5all P R | F.5anom P R | alarmP | caught | seen unseen | rare | FAev FA/30d FAshare FAtime% "
       "| delay attr | chanF.5 subF.5 affF.5 ADTQCafter | setting")
for s in ("SAT_1", "SAT_2", "SAT_3", "SAT_4"):
    print(f"\n## {s} (trained on {R['solo'][s]['train_families']})\n{HDR}")
    for arm in ("solo", "fed", "pers"):
        for sel in SELS:
            r = R[arm][s][sel]
            a, n = r["all"], r["anomalies"]
            print(f"{arm:4s} {sel:11s} | {f(r['val'])} | {f(g(a, 'EW_F_0.50'))} {f(g(a, 'EW_precision'))} {f(g(a, 'EW_recall'))} | "
                  f"{f(g(n, 'EW_F_0.50'))} {f(g(n, 'EW_precision'))} {f(g(n, 'EW_recall'))} | {f(g(n, 'alarming_precision'))} | "
                  f"{r.get('n_detected')}/{r.get('n_anomalies')} | {f(r.get('recall_seen_families'), 2)} {f(r.get('recall_unseen_families'), 2)} | "
                  f"{r.get('n_rare_detected')}/{r.get('n_rare')} | {r.get('n_false_alarm_events')} {f(r.get('false_alarms_per_30d'), 1)} "
                  f"{f(r.get('false_alarm_event_share'), 2)} {f(100 * r.get('false_alarm_time_share', 0), 2)} | "
                  f"{f(r.get('median_delay_min'), 0)} {f(r.get('family_attribution_acc'), 2)} | "
                  f"{f(n.get('chan_channel_F0.50'))} {f(n.get('chan_subsystem_F0.50'))} {f(g(n, 'AFF_F_0.50'))} {f(n.get('adtqc_AfterRate'), 2)} | "
                  f"{tag(r['cfg'])}")
    print("by family:", {f"{arm}/{sel}": R[arm][s][sel].get("recall_by_family") for arm in R for sel in SELS})
if os.path.exists(f"{OUT}/matched_solo.json"):
    M = {a: json.load(open(f"{OUT}/matched_{a}.json")) for a in ("solo", "fed", "pers")}
    print("\n## Same setting for every arm (model effect only): F0.5 anomalies / caught / false alarms")
    for k in ("solo", "fed", "pers"):
        print(f"setting chosen by the fleet on {k} errors: {tag(R[k]['fed_selection']['cfg'])}")
        for s in ("SAT_1", "SAT_2", "SAT_3", "SAT_4"):
            print(f"  {s}: " + " | ".join(
                f"{a} {f(g(M[a][s][k]['anomalies'], 'EW_F_0.50'))} {M[a][s][k].get('n_detected')}/{M[a][s][k].get('n_anomalies')} FA {M[a][s][k].get('n_false_alarm_events')}"
                for a in ("solo", "fed", "pers")))
print("\nfederated settings:", {a: R[a]["fed_selection"] for a in R})
if os.path.exists(f"{OUT}/rul.json"):
    u = json.load(open(f"{OUT}/rul.json"))
    print("\nRUL:", {k: u.get(k) for k in ("alert_wheel", "true_wheel", "first_alert_day", "true_fail_day", "lead_time_days",
                                          "true_phase2_visible_day", "failure_threshold")})
    print("exp:", u.get("exp"), "\nexp_min:", u.get("exp_min"), "\nlin:", u.get("lin"))
    print("reliable:", {k: u.get(k) for k in ("reliable_from_day", "reliable_lead_days", "milestones")})
b = f"{FLEET}/validation/baselines.json"
if os.path.exists(b):
    B = json.load(open(b))
    print("\nbaselines (anomalies only, F0.5):", {s: {k: round(v["anomalies"].get("global_EW_F_0.50", 0), 3)
                                                     for k, v in B[s].items() if "anomalies" in v} for s in B})
