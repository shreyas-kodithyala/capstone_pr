"""Build viz/mission_replay.html: one offline page that plays the whole mission from the saved data.

Telemetry comes from the 30 s dataset, kept as 3-hour average, low and high, so spikes stay visible.
Telecommands are every firing. Events come from the truth tables, and caught or missed comes from the
stored final results. Nothing is simulated or scored again.

  python viz_dashboard.py
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FLEET = os.path.join(ROOT, "data", "sim_fleet")
SATS = ["SAT_1", "SAT_2", "SAT_3", "SAT_4"]
ROLES = ["Survey, healthy", "Imaging, wheel wear", "Survey, power", "Imaging, heavy ops"]
EPOCH = pd.Timestamp("2025-01-01", tz="UTC")
BIN = 360                                       # 30 s rows per 3-hour bin
NAMES = dict(A1="Wheel friction spike", A2="Wheel drive dropout", W1="Wear friction spike", W2="Wheel degradation",
             W3="Wheel failure", B1="Gyro bias drift", B2="Gyro spikes", B4="Star-tracker outage",
             C1="Solar strings open", C3="Unit latch-up", D1="Heater stuck on", GYRO_CAL="Gyro calibration",
             INERTIAL_CAL="Inertial calibration", STR_RESET="Star-tracker reset", SAFE="Safe-mode drill",
             AUTO_DUMP="Autonomous momentum dump")


def days(x):
    return (pd.to_datetime(x, utc=True) - EPOCH).dt.total_seconds() / 86400


def sig(a):
    return [float(f"{v:.5g}") for v in a]


def main():
    ch = pd.read_csv(f"{FLEET}/SAT_1/channels.csv")
    ch = ch[ch.Target == "YES"]
    units = dict(zip(ch.Name, ch["Physical Unit"]))
    cols, names = ch.Channel.tolist(), ch.Name.tolist()
    tel = []
    for s in SATS:
        df = pd.concat([pd.read_csv(f"{FLEET}/{s}/resampled_30s_{p}.csv.gz", usecols=cols) for p in ("train", "test")], ignore_index=True)
        g = np.arange(len(df)) // BIN
        gb = df.groupby(g)
        m, lo, hi = gb.mean(), gb.min(), gb.max()
        tel.append([[sig(m[c].values), sig(lo[c].values), sig(hi[c].values)] for c in cols])
        print(s, "telemetry", len(m), flush=True)
    nb = len(tel[0][0][0])

    d = pd.read_csv(os.path.join(ROOT, "results", "final", "per_event_detection.csv"))
    got = {(r.satellite, r.id, r.arm, r.settings): bool(r.detected) for r in d.itertuples()}
    ev, cmdT, cmdC = [], [], []
    tc = pd.read_csv(f"{FLEET}/SAT_1/telecommands.csv")
    cmdNames = tc.Name.tolist()
    for i, s in enumerate(SATS):
        t = pd.read_csv(f"{FLEET}/{s}/truth/events_truth.csv")
        t = t[t.Visible].copy()
        t["a"], t["b"] = days(t.FirstVisible), days(t.LastVisible)
        for x in t.itertuples():
            A = x.Category == "Anomaly"
            b = 365.0 if x.Code == "W3" else float(x.b)
            ev.append(dict(s=i, cat="A" if A else "R", code=x.Code, name=NAMES.get(x.Code, x.Code), comp="" if not A or pd.isna(x.Component) else x.Component,
                           a=round(float(x.a), 4), b=round(b, 4), split=x.Split, solo=got.get((s, x.ID, "solo", "local"), False),
                           fed=got.get((s, x.ID, "fed", "federated"), False), pers=got.get((s, x.ID, "pers", "federated"), False)))
        ts, cs = [], []
        for k, c in enumerate(tc.Telecommand):
            f = pd.read_csv(f"{FLEET}/{s}/telecommands/{c}.csv")
            v = (pd.to_datetime(f.timestamp, utc=True) - EPOCH).dt.total_seconds().round().astype(int).values
            ts += v.tolist()
            cs += [k] * len(v)
        o = np.argsort(ts, kind="stable")
        cmdT.append(np.array(ts)[o].tolist())
        cmdC.append(np.array(cs)[o].tolist())
    ev.sort(key=lambda e: e["a"])

    r = pd.read_csv(os.path.join(ROOT, "results", "final", "rul_predictions.csv"))
    r = r[r.kind == "exp"].sort_values("day")
    rul = [[int(a), round(float(b), 3), None if not np.isfinite(c) else round(float(c), 3)] for a, b, c in zip(r.day, r.true_rul, r.pred_rul)]

    R = json.load(open(os.path.join(ROOT, "results", "final", "results.json")))
    score = []
    for lab, a, k in (("Solo, own settings", "solo", "local"), ("Plain FL, fleet settings", "fed", "federated"),
                      ("FL + personalization, own settings", "pers", "local"), ("FL + personalization, fleet settings", "pers", "federated")):
        c = sum(R[a][s][k]["n_detected"] for s in SATS)
        fa = sum(R[a][s][k]["n_false_alarm_events"] for s in SATS)
        f = np.mean([R[a][s][k]["all"]["global_EW_F_0.50"] for s in SATS])
        score.append([lab, int(c), int(fa), f"{f:.3f}"])

    jumps = [[116.4, "Day 116.4: training data ends"], [137, "Day 137: settings window"], [182, "Day 182: frozen test starts"],
             [190, "Day 190: hidden wear starts on SAT_2"], [256, "Day 256: first RUL alert"], [257.4, "Day 257: first labelled wear spike"],
             [296, "Day 296: degradation becomes visible"], [356.05, "Day 356: just before wheel failure"], [356.2, "Day 356: after wheel failure"]]
    card = [dict(n="rw2_current", l="Wheel 2 current", u="A"), dict(n="att_err", l="Attitude error", u="deg"),
            dict(n="bat_voltage", l="Battery", u="V"), dict(n="sa_power", l="Solar power", u="W")]
    data = dict(sats=SATS, roles=ROLES, ch=names, chName=[f"{n} ({units[n]})" for n in names], nb=nb, tel=tel, events=ev, cmdT=cmdT,
                cmdC=cmdC, cmdNames=cmdNames, rul=rul, score=score, jumps=jumps, card=card)
    html = open(os.path.join(HERE, "viz_dashboard_template.html"), encoding="utf-8").read()
    html = html.replace("/*DATA*/", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    os.makedirs(os.path.join(ROOT, "viz"), exist_ok=True)
    p = os.path.join(ROOT, "viz", "mission_replay.html")
    open(p, "w", encoding="utf-8").write(html)
    print("wrote", p, round(os.path.getsize(p) / 1e6, 1), "MB", "events", len(ev), "commands", sum(len(x) for x in cmdT))


if __name__ == "__main__":
    main()
