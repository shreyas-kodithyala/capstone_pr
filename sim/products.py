"""Labels + export for one satellite (ESA-ADB layout, CSV).

Labelling rules (ESA-ADB Supp. Table 1 and Appendix A.7, applied exactly):
  * Anomaly: a fault from the fault library. A channel is labelled where the fault changed it by
    more than LABEL_K x its noise sigma, computed against a fault-free version that shares the
    same noise: physics faults vs. the Basilisk twin run, sensor / power faults vs. the same run
    without that fault. Start = first visible sample (physical onset kept in truth/events_truth.csv).
    Visible samples less than 10 min apart form one segment; one event ID per fault.
  * Rare Event: commanded calibrations, star-tracker resets, safe-mode drills and uncommanded
    autonomous momentum dumps; labelled on the channels each operation drives.
  * Only target channels are labelled. Event types (uni/multivariate, global/local,
    point/subsequence) are computed with ESA's rules.
"""
import json
import os

import numpy as np
import pandas as pd

import config as C
from operations import Timeline
from physics import load_truth
from telemetry import CH, Telemetry

TARGETS = [n for n in C.CH_NAMES if CH[n]["target"]]
AOCS_MOVE = [f"rw{k}_speed" for k in range(1, 5)] + [f"rw{k}_current" for k in range(1, 5)] + \
    ["gyro_x", "gyro_y", "gyro_z", "att_err", "mag_x", "mag_y", "mag_z", "sa_power"]
RARE_MAP = {
    "gyro_calibration": AOCS_MOVE,
    "inertial_calibration": AOCS_MOVE,
    "safe_mode_drill": AOCS_MOVE,
    "autonomous_momentum_dump": [f"rw{k}_speed" for k in range(1, 5)] + ["mag_x", "mag_y", "mag_z"],
}
POWER_CH = [n for n in ("sa_power", "bat_voltage", "bus_load", "bat_temp") if n in C.CH_NAMES]
ATT = ["att_err"]
GAP = 600.0


def thr(name):
    return max(C.LABEL_K * CH[name]["sigma"], 1.5 * CH[name]["q"])


def segments(t, mask, t0, t1):
    i0, i1 = np.searchsorted(t, [t0, t1])
    idx = np.nonzero(mask[i0:i1])[0] + i0
    if len(idx) == 0:
        return []
    br = np.nonzero(np.diff(t[idx]) > GAP)[0]
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br], idx[-1]]
    return [(float(t[a]), float(t[b])) for a, b in zip(starts, ends)]


def diff_segments(tel, ch_a, ch_b, names, t0, t1):
    out = {}
    for n in names:
        if n not in TARGETS or n not in ch_a or n not in ch_b:
            continue
        t = tel.times[n]
        i0, i1 = np.searchsorted(t, [t0, t1])
        m = np.zeros(len(t), bool)
        m[i0:i1] = np.abs(ch_a[n][i0:i1].astype(np.float64) - ch_b[n][i0:i1]) > thr(n)
        s = segments(t, m, t0, t1)
        if s:
            out[n] = s
    return out


def ts(epoch, t):
    return (np.datetime64(epoch) + (np.asarray(t) * 1000).astype("timedelta64[ms]"))


def iso(epoch, t):
    return np.datetime_as_string(ts(epoch, t), unit="ms") + "Z"


def make(name, spec, seed, phys_dir, twin_dir, out_root, log=print, tl=None):
    tl = tl or Timeline(name, spec, seed).build()
    plog = json.load(open(f"{phys_dir}/physlog.json"))
    tel = Telemetry(tl, seed)
    log(f"{name}: loading truth")
    P = tel.physical(load_truth(phys_dir), np.load(f"{phys_dir}/ticks.npy"))
    post = [f for f in tl.faults if f["kind"] != "physics"]
    physf = [f for f in tl.faults if f["kind"] == "physics"]
    log(f"{name}: power / thermal + channels")
    pw_clean = tel.power_thermal(P, [])
    clean = tel.channels(P, pw_clean, [])
    pw_final = tel.power_thermal(P, post) if post else pw_clean
    final = tel.channels(P, pw_final, post) if post else clean

    events = []           # dict(id, category, cls, ..., segs={channel: [(s, e)]})
    # ---------------------------------------------------------- physics faults (twin)
    if physf:
        log(f"{name}: twin comparison")
        Pw = tel.physical(load_truth(twin_dir), np.load(f"{twin_dir}/ticks.npy"))
        twin = tel.channels(Pw, tel.power_thermal(Pw, []), [])
        del Pw
        if any(f["code"].startswith("W") for f in physf):
            _phase2(tl, tel, clean, twin, physf)
            physf = [f for f in tl.faults if f["kind"] == "physics"]
        for f in physf:
            end = f["end"] + C.SETTLE_H["physics"] * 3600
            f["segs"] = diff_segments(tel, clean, twin, TARGETS, f["start"], min(end, tl.T))
            events.append(f)
        del twin
    # ---------------------------------------------------------- sensor / power faults
    for f in post:
        if f["kind"] == "power":
            pw_f = tel.power_thermal(P, [f])
            ch_f = tel.channels(P, pw_f, [f], names=POWER_CH)
            names = POWER_CH
        else:
            names = [f["component"]] if f["code"] != "B4" else ATT
            ch_f = tel.channels(P, pw_clean, [f], names=names)
        end = f["end"] + C.SETTLE_H["sensor" if f["kind"] == "sensor" else "power"] * 3600
        f["segs"] = diff_segments(tel, ch_f, clean, names, f["start"], min(end, tl.T))
        events.append(f)
    # ---------------------------------------------------------- rare events
    rare = [dict(r) for r in tl.rare]
    rare += [dict(cls="autonomous_momentum_dump", key="AUTO_DUMP", start=a, end=b, tc=None, commanded=False)
             for a, b in plog["auto_dumps"]]
    for r in rare:
        r.update(category="Rare Event", severity=0, family="R", kind="ops", subsystem="AOCS",
                 component="aocs" if r["key"] != "STR_RESET" else "star_tracker")
        if r["key"] == "STR_RESET":
            names = ATT
            ch_w = tel.channels(P, pw_clean, [], names=names, skip_reset=r["start"])
            r["segs"] = diff_segments(tel, clean, ch_w, names, r["start"], r["end"] + 600)
        else:
            segs = {}
            for n in RARE_MAP[r["cls"]]:
                t = tel.times[n]
                i0, i1 = np.searchsorted(t, [r["start"], r["end"]])
                if i1 > i0:
                    segs[n] = [(float(t[i0]), float(t[i1 - 1]))]
            r["segs"] = segs
        events.append(r)
    for e in events:
        e.setdefault("category", "Anomaly")
    events = sorted(events, key=lambda e: e["start"])
    for i, e in enumerate(events):
        e["id"] = f"id_{i + 1}"
    log(f"{name}: writing")
    write(tl, tel, final, events, plog, out_root, seed)
    return events


def _phase2(tl, tel, clean, twin, physf):
    """SAT_5 / archive: phase 2 starts when the wear signature becomes persistently visible
    (daily median |faulty - twin| > threshold on the wheel's temperature or current, outside spikes)."""
    w = tl.wear
    k = w["wheel"] + 1
    spikes = [(s["start"], s["end"] + 7200) for s in w["spikes"]]
    t_lo = w["t_spikes"] or 0.0
    t_hi = w["t_fail"] or tl.T
    t_p2 = None
    for d0 in np.arange(np.floor(t_lo / 86400) * 86400, t_hi, 86400.0):
        vis = False
        for n in (f"rw{k}_current",):
            t = tel.times[n]
            i0, i1 = np.searchsorted(t, [d0, d0 + 86400])
            m = np.ones(i1 - i0, bool)
            for a, b in spikes:
                m &= ~((t[i0:i1] >= a) & (t[i0:i1] < b))
            if m.sum() > 10 and np.median(np.abs(clean[n][i0:i1][m] - twin[n][i0:i1][m])) > thr(n):
                vis = True
        if vis:
            t_p2 = float(d0)
            break
    w["t_phase2_visible"] = t_p2
    keep = []
    for f in tl.faults:
        if f["code"] == "W2":
            if t_p2 is None:
                continue
            f["start"] = t_p2
            f["split"] = "train" if t_p2 < C.SPLIT_DAY * 86400 else "test"
        if f["code"] == "W1" and t_p2 is not None and f["start"] >= t_p2:
            continue                                   # spikes after phase 2 belong to the W2 event
        keep.append(f)
    if t_p2 is not None and not any(f["code"] == "W2" for f in keep):
        e2 = w["t_fail"] or tl.T
        keep.append(dict(code="W2", cls="rw_degradation", family="W", kind="physics", subsystem="AOCS", severity=2,
                         mode=C.FAULTS["W2"]["mode"], start=t_p2, end=e2, component=f"rw{k}", phase=2,
                         split="train" if t_p2 < C.SPLIT_DAY * 86400 else "test"))
    tl.faults = sorted(keep, key=lambda x: x["start"])


def write(tl, tel, final, events, plog, out_root, seed):
    E = C.EPOCH
    d = f"{out_root}/{tl.name}"
    for sub in ("channels", "telecommands", "truth"):
        os.makedirs(f"{d}/{sub}", exist_ok=True)
    # ---- metadata
    rows = [dict(Channel=C.CH_ID[n], Name=n, Description=CH[n]["desc"], Subsystem=C.SUBSYS_ID[CH[n]["subsystem"]],
                 SubsystemName=CH[n]["subsystem"], **{"Physical Unit": CH[n]["unit"]}, Group=CH[n]["group"],
                 Target="YES" if CH[n]["target"] else "NO", NativeRate_s=CH[n]["rate"]) for n in C.CH_NAMES]
    pd.DataFrame(rows).to_csv(f"{d}/channels.csv", index=False)
    pd.DataFrame([dict(Telecommand=C.TC_ID[n], Name=n, Description=desc, Priority=3)
                  for n, desc in C.TELECOMMANDS]).to_csv(f"{d}/telecommands.csv", index=False)
    # ---- raw channels
    fmt = {}
    for n in C.CH_NAMES:
        q = CH[n]["q"]
        dec = max(0, int(np.ceil(-np.log10(q)))) if q < 1 else 0
        fmt[n] = f"%.{dec}f"
        pd.DataFrame({"timestamp": iso(E, tel.times[n]), "value": final[n]}).to_csv(
            f"{d}/channels/{C.CH_ID[n]}.csv.gz", index=False, float_format=fmt[n])
    # ---- telecommands
    for n in C.TC_NAMES:
        tt = np.array(sorted(t for t, c in tl.tcs if c == n))
        pd.DataFrame({"timestamp": iso(E, tt), "value": np.ones(len(tt), int)}).to_csv(
            f"{d}/telecommands/{C.TC_ID[n]}.csv", index=False)
    # ---- labels + anomaly types (ESA rules)
    lab, typ, tru = [], [], []
    labelled = {n: np.zeros(len(tel.times[n]), bool) for n in C.CH_NAMES}
    for e in events:
        for n, ss in e.get("segs", {}).items():
            for a, b in ss:
                i0, i1 = np.searchsorted(tel.times[n], [a, b + 1e-6])
                labelled[n][i0:i1] = True
    nom = {n: (final[n][~labelled[n]].min(), final[n][~labelled[n]].max()) for n in TARGETS}
    for e in events:
        segs = e.get("segs", {})
        vis = bool(segs)
        if vis:
            for n, ss in segs.items():
                for a, b in ss:
                    lab.append(dict(ID=e["id"], Channel=C.CH_ID[n], StartTime=iso(E, a), EndTime=iso(E, b)))
            glob = False
            for n, ss in segs.items():
                for a, b in ss:
                    i0, i1 = np.searchsorted(tel.times[n], [a, b + 1e-6])
                    v = final[n][i0:i1]
                    if len(v) and (v.min() < nom[n][0] or v.max() > nom[n][1]):
                        glob = True
            point = all(b - a <= 90.0 for ss in segs.values() for a, b in ss)
            typ.append(dict(ID=e["id"], Class=e["cls"], Subclass=e.get("component", ""), Category=e["category"],
                            Dimensionality="Univariate" if len(segs) == 1 else "Multivariate",
                            Locality="Global" if glob else "Local", Length="Point" if point else "Subsequence"))
        first = min((a for ss in segs.values() for a, b in ss), default=np.nan)
        last = max((b for ss in segs.values() for a, b in ss), default=np.nan)
        tru.append(dict(ID=e["id"], Category=e["category"], Class=e["cls"], Code=e.get("code", e.get("key")),
                        Family=e.get("family"), FaultFamily=C.FAMILY.get(e.get("family"), "R"), Component=e.get("component", ""), Subsystem=e.get("subsystem", ""),
                        Severity=e.get("severity", 0), Commanded=e.get("commanded", False),
                        Telecommand=e.get("tc") or "", Mode=e.get("mode", ""),
                        PhysicalStart=iso(E, e["start"]), PhysicalEnd=iso(E, e["end"]),
                        FirstVisible=iso(E, first) if vis else "", LastVisible=iso(E, last) if vis else "",
                        Visible=vis, NChannels=len(segs), Channels=" ".join(C.CH_ID[n] for n in segs),
                        Split="train" if e["start"] < C.SPLIT_DAY * 86400 else "test",
                        Phase=e.get("phase", ""), Params=json.dumps({k: e[k] for k in
                                                                    ("mult", "frac", "amp", "n", "strings", "factor", "watts")
                                                                    if k in e})))
    pd.DataFrame(lab, columns=["ID", "Channel", "StartTime", "EndTime"]).to_csv(f"{d}/labels.csv", index=False)
    pd.DataFrame(typ, columns=["ID", "Class", "Subclass", "Category", "Dimensionality", "Locality", "Length"]).to_csv(
        f"{d}/anomaly_types.csv", index=False)
    pd.DataFrame(tru).to_csv(f"{d}/truth/events_truth.csv", index=False)
    # ---- truth: health / RUL (degrading satellites and archive)
    if tl.wear:
        w = tl.wear
        th = np.arange(0.0, tl.T, 3600.0)
        wd = np.interp(th / 86400.0, np.arange(len(w["w_daily"])), w["w_daily"])
        tf = w["t_fail"]
        if tf:
            wd[th >= tf] = C.WEAR_W["fail"]
        tp2 = w.get("t_phase2_visible") or w["t_phase2"]
        phase = np.zeros(len(th), int)
        if w["t_spikes"] is not None:
            phase[th >= w["t_spikes"]] = 1
        if tp2 is not None:
            phase[th >= tp2] = 2
        if tf:
            phase[th >= tf] = 3
        rul = (tf - th) / 86400.0 if tf else np.full(len(th), np.nan)
        rul = np.where(phase == 3, 0.0, rul)
        tier = np.where(phase == 3, 3, np.where((phase == 2) & (rul < 14), 3, np.where(phase == 2, 2, np.where(phase == 1, 1, 0))))
        k = w["wheel"]
        fc0 = plog["var"]["fc"][k]
        pd.DataFrame(dict(timestamp=iso(E, th), wheel=f"rw{k + 1}", wear=wd.round(4),
                          coulomb_friction_Nm=(fc0 * (1 + wd)).round(6), phase=phase, rul_days=np.round(rul, 3),
                          alert_tier=tier)).to_csv(f"{d}/truth/health.csv", index=False)
        json.dump({k_: v for k_, v in w.items() if k_ != "w_daily"}, open(f"{d}/truth/wear.json", "w"), indent=1)
    # ---- truth: timeline
    pd.DataFrame([dict(timestamp=iso(E, t), mode=m) for t, m in tl.modes]).to_csv(f"{d}/truth/aocs_modes.csv", index=False)
    pd.DataFrame([dict(start=iso(E, a), end=iso(E, b), kind=k) for a, b, k in tl.loads]).to_csv(
        f"{d}/truth/load_windows.csv", index=False)
    json.dump(dict(name=tl.name, seed=seed, spec=tl.spec, orbit=dict(inc_deg=float(np.degrees(tl.orbit.inc)),
              raan0_deg=float(np.degrees(tl.orbit.raan0)), period_min=float(tl.orbit.period / 60)),
              unit_variation=plog["var"], auto_dumps=len(plog["auto_dumps"]), fail_lock_t=plog["fail_lock_t"],
              days=tl.days, split_day=C.SPLIT_DAY, val_days=C.VAL_DAYS),
              open(f"{d}/truth/satellite.json", "w"), indent=1, default=str)
    # ---- 30 s resampled train / test (ESA preprocessing: zero-order hold, labels per channel)
    g = np.arange(0.0, tl.T, C.GRID)
    cols = {"timestamp": iso(E, g)}
    for n in C.CH_NAMES:
        t = tel.times[n]
        i = np.searchsorted(t, g, "right") - 1
        cols[C.CH_ID[n]] = final[n][np.clip(i, 0, len(t) - 1)]
    for n in C.TC_NAMES:
        tt = np.array(sorted(t for t, c in tl.tcs if c == n))
        hit = np.zeros(len(g), np.int8)
        if len(tt):
            hit[np.clip(np.ceil(tt / C.GRID).astype(int), 0, len(g) - 1)] = 1
        cols[C.TC_ID[n]] = hit
    for n in TARGETS:
        lab_g = np.zeros(len(g), np.int8)
        for e in events:
            code = 1 if e["category"] == "Anomaly" else 2
            for a, b in e.get("segs", {}).get(n, []):
                i0, i1 = np.searchsorted(g, [a, b + C.GRID])
                lab_g[i0:i1] = 1 if code == 1 else np.where(lab_g[i0:i1] == 1, 1, 2)
        cols[f"is_anomaly_{C.CH_ID[n]}"] = lab_g
    df = pd.DataFrame(cols)
    s = int(C.SPLIT_DAY * 86400 / C.GRID)
    for part, sl in (("train", slice(0, s)), ("test", slice(s, None))):
        sub = df.iloc[sl]
        sub.to_csv(f"{d}/resampled_30s_{part}.csv.gz", index=False, float_format="%.6g")
