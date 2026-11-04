"""Validation of the simulated fleet dataset (part 1, Basilisk environment).

  python validate.py        -> data/sim_fleet/validation/{validation_part1.json, figures}

Checks (published methods):
  P1 momentum conservation   torque-free Basilisk run (wheels + controller only): |dH|/|H|
  P2 orbit rate              gyro_z in nadir pointing vs mean motion n
  P3 eclipse duration        measured eclipse per orbit vs analytic cylindrical-shadow formula
                             (Wertz & Larson; beta angle from the Sun and orbit normal)
  P4 actuator limits         |wheel speed| < 6000 rpm, |current| <= 2 A (0.2 Nm / k_t), HR16 datasheet
  P5 battery behaviour       discharge in eclipse, charge in sunlight, depth of discharge per orbit,
                             voltage within 8S Li-ion limits
  R1 dataset statistics      ESA-ADB Table 1 quantities for our satellites
  R2 realism vs real data    catch22 features (Lubba et al. 2019) of 1-day windows: ours vs ESA-ADB
                             Mission 1 / Mission 2 lightweight channels (real telemetry)
  W  degradation / RUL       SAT_5 health truth, phases, early-warning lead time of a simple indicator
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402
from envmodels import RE  # noqa: E402

ROOT = os.path.dirname(HERE)
FLEET = os.path.join(ROOT, "data", "sim_fleet")
OUTD = os.path.join(FLEET, "validation")
ESA_DATA = os.path.join((os.environ.get("NEW_RUNS") or (os.path.join(ROOT, "ext", "new_runs") if os.path.isdir(os.path.join(ROOT, "ext", "new_runs")) else os.path.join(os.path.dirname(ROOT), "ESA", "Main", "new_runs"))), "data")
ESA_RAW = os.path.join(os.path.dirname(ROOT), "ESA")
SATS = list(C.FLEET)
CHID = C.CH_ID
INV = {v: k for k, v in CHID.items()}


def load(sat, part):
    return pd.read_csv(f"{FLEET}/{sat}/resampled_30s_{part}.csv.gz", parse_dates=["timestamp"])


# ------------------------------------------------------------------ P1
def momentum_test():
    from Basilisk.architecture import messaging
    from Basilisk.fswAlgorithms import attTrackingError, inertial3D, mrpFeedback, rwMotorTorque
    from Basilisk.simulation import reactionWheelStateEffector, simpleNav, spacecraft
    from Basilisk.utilities import SimulationBaseClass, macros, simIncludeRW
    sim = SimulationBaseClass.SimBaseClass()
    p = sim.CreateNewProcess("p")
    p.addTask(sim.CreateNewTask("t", macros.sec2nano(C.DT_DYN)))
    sc = spacecraft.Spacecraft(); sc.ModelTag = "m"
    sc.hub.mHub = C.SC["mass"]; sc.hub.IHubPntBc_B = np.array(C.SC["inertia"]).reshape(3, 3).tolist()
    sc.hub.omega_BN_BInit = [[0.01], [-0.02], [0.015]]
    sim.AddModelToTask("t", sc)
    rwF = simIncludeRW.rwFactory()
    b = np.radians(C.SC["rw_beta_deg"])
    for k, a in enumerate(C.SC["rw_az_deg"]):
        a = np.radians(a)
        rwF.create("Honeywell_HR16", [np.cos(b) * np.cos(a), np.cos(b) * np.sin(a), np.sin(b)], maxMomentum=50.0,
                   Omega=1200.0 * (-1) ** k)
    rwE = reactionWheelStateEffector.ReactionWheelStateEffector(); rwE.ModelTag = "rw"
    rwF.addToSpacecraft("m", rwE, sc); sim.AddModelToTask("t", rwE, 2)
    for k in range(4):
        rwE.ReactionWheelData[k].fCoulomb = C.SC["rw_fc"]; rwE.ReactionWheelData[k].cViscous = C.SC["rw_cv"]
    nav = simpleNav.SimpleNav(); nav.scStateInMsg.subscribeTo(sc.scStateOutMsg); sim.AddModelToTask("t", nav)
    i3 = inertial3D.inertial3D(); i3.sigma_R0N = [0.2, -0.1, 0.3]; sim.AddModelToTask("t", i3)
    err = attTrackingError.attTrackingError(); err.attNavInMsg.subscribeTo(nav.attOutMsg)
    err.attRefInMsg.subscribeTo(i3.attRefOutMsg); sim.AddModelToTask("t", err)
    vc = messaging.VehicleConfigMsgPayload(); vc.ISCPntB_B = C.SC["inertia"]; vcm = messaging.VehicleConfigMsg().write(vc)
    rp = rwF.getConfigMessage()
    ctl = mrpFeedback.mrpFeedback(); ctl.guidInMsg.subscribeTo(err.attGuidOutMsg); ctl.vehConfigInMsg.subscribeTo(vcm)
    ctl.rwParamsInMsg.subscribeTo(rp); ctl.rwSpeedsInMsg.subscribeTo(rwE.rwSpeedOutMsg)
    ctl.K, ctl.Ki, ctl.P = C.SC["K"], -1.0, C.SC["P"]; sim.AddModelToTask("t", ctl)
    mt = rwMotorTorque.rwMotorTorque(); mt.controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    mt.vehControlInMsg.subscribeTo(ctl.cmdTorqueOutMsg); mt.rwParamsInMsg.subscribeTo(rp)
    rwE.rwMotorCmdInMsg.subscribeTo(mt.rwMotorTorqueOutMsg); sim.AddModelToTask("t", mt)
    lg = sc.logger(["totRotAngMomPntC_N", "totRotEnergy"], macros.sec2nano(10.0))
    sim.AddModelToTask("t", lg)
    sim.InitializeSimulation(); sim.ConfigureStopTime(macros.sec2nano(5740.0)); sim.ExecuteSimulation()
    H = np.array(lg.totRotAngMomPntC_N)
    return dict(H0_Nms=float(np.linalg.norm(H[0])),
                max_rel_drift=float(np.max(np.linalg.norm(H - H[0], axis=1)) / np.linalg.norm(H[0])),
                note="1 orbit, 30 deg slew, wheels with friction, controller active, no external torques")


# ------------------------------------------------------------------ P2-P5, R1
def physics_checks(sat):
    from envmodels import Orbit
    from operations import epoch_unix
    spec = C.FLEET[sat]
    o = Orbit(spec["orbit"], epoch_unix())
    df = pd.concat([load(sat, "train"), load(sat, "test")], ignore_index=True)
    t = (df.timestamp - df.timestamp.iloc[0]).dt.total_seconds().values
    res = {}
    nadir = df[CHID["aocs_mode"]].values == 0
    lab_any = (df[[c for c in df.columns if c.startswith("is_anomaly_")]] > 0).any(axis=1).values
    clean = nadir & ~lab_any
    res["orbit_rate_deg_s"] = dict(measured_median_gyro_z=float(np.median(df[CHID["gyro_z"]].values[clean])),
                                   analytic_n=float(np.degrees(o.n)))
    # eclipse durations per orbit vs analytic (cylindrical): t_e = (P/pi) acos(sqrt(h^2+2Rh)/(a cos beta))
    ecl = df[CHID["sa_power"]].values < 5.0            # v2: no eclipse flag; solar power ~0 in shadow
    d = np.diff(np.r_[0, ecl.astype(int), 0])
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    dur = (en - st) * 30.0
    mid = t[np.clip((st + en) // 2, 0, len(t) - 1)]
    s, _ = o.sun_P(o.t0 + mid)
    hn = np.array([np.sin(o.raan0) * np.sin(o.inc), -np.cos(o.raan0) * np.sin(o.inc), np.cos(o.inc)])
    beta = np.arcsin(np.clip(s @ hn, -1, 1))
    h = o.a - RE
    x = np.sqrt(h ** 2 + 2 * RE * h) / (o.a * np.cos(beta))
    te = np.where(x < 1, o.period / np.pi * np.arccos(np.clip(x, -1, 1)), 0.0)
    ok = dur > 300
    res["eclipse"] = dict(n_eclipses=int(ok.sum()), mean_measured_min=float(dur[ok].mean() / 60) if ok.any() else 0.0,
                          mean_analytic_min=float(te[ok].mean() / 60) if ok.any() else 0.0,
                          mean_abs_diff_min=float(np.mean(np.abs(dur[ok] - te[ok])) / 60) if ok.any() else 0.0,
                          eclipse_fraction=float(ecl.mean()))
    rpm = df[[CHID[f"rw{k}_speed"] for k in range(1, 5)]].values
    cur = df[[CHID[f"rw{k}_current"] for k in range(1, 5)]].values
    res["actuator_limits"] = dict(max_abs_rpm=float(np.abs(rpm).max()), limit_rpm=6000.0,
                                  max_abs_current_A=float(np.abs(cur).max()), limit_current_A=2.0)
    v = df[CHID["bat_voltage"]].values
    per, drop = [], []
    for a, b in zip(st, en):
        if b - a > 10 and not lab_any[a:b].any():
            drop.append(v[a + 1] - v[b - 1])
            w = v[max(a - 4, 0):min(b + 4, len(v))]
            per.append(w.max() - w.min())
    res["battery"] = dict(frac_eclipses_voltage_falls=float(np.mean(np.array(drop) > 0)) if drop else None,
                          voltage_swing_per_eclipse_median_V=float(np.median(per)) if per else None,
                          voltage_min=float(v.min()), voltage_max=float(v.max()), limit_v=[24.0, 33.6 + 0.1])
    return res, df


def dataset_stats(sat, df):
    lab = pd.read_csv(f"{FLEET}/{sat}/labels.csv")
    typ = pd.read_csv(f"{FLEET}/{sat}/anomaly_types.csv")
    tru = pd.read_csv(f"{FLEET}/{sat}/truth/events_truth.csv")
    lab_cols = [c for c in df.columns if c.startswith("is_anomaly_")]
    L = df[lab_cols].values
    any_lab = (L > 0).any(1)
    raw_n = sum(len(pd.read_csv(f, usecols=[1])) for f in glob.glob(f"{FLEET}/{sat}/channels/*.csv.gz"))
    ts = pd.to_datetime(pd.read_csv(f"{FLEET}/{sat}/channels/{CHID['rw1_speed']}.csv.gz", usecols=[0]).timestamp)
    dt = np.diff(ts.values.astype("datetime64[ns]").astype(np.int64)) / 1e9
    typ_v = typ.merge(tru[["ID", "Split"]], on="ID")
    out = dict(raw_samples=int(raw_n), grid_rows=int(len(df)), channels=len(C.CH_NAMES),
               target_channels=sum(1 for c in C.CHANNELS if c[5]), telecommands=len(C.TC_NAMES),
               tc_executions=int(sum(len(pd.read_csv(f)) for f in glob.glob(f"{FLEET}/{sat}/telecommands/*.csv"))),
               annotated_rows_pct=float(100 * any_lab.mean()),
               anomaly_rows_pct=float(100 * (L == 1).any(1).mean()),
               events=int(typ.ID.nunique()), anomalies=int((typ.Category == "Anomaly").sum()),
               rare_events=int((typ.Category == "Rare Event").sum()),
               invisible_faults=int((~tru.Visible & (tru.Category == "Anomaly")).sum()),
               uni_multi=[int((typ.Dimensionality == "Univariate").sum()), int((typ.Dimensionality == "Multivariate").sum())],
               global_local=[int((typ.Locality == "Global").sum()), int((typ.Locality == "Local").sum())],
               point_subseq=[int((typ.Length == "Point").sum()), int((typ.Length == "Subsequence").sum())],
               classes=int(typ.Class.nunique()),
               train_test_events=[int((typ_v.Split == "train").sum()), int((typ_v.Split == "test").sum())],
               sampling_interval_rw1_speed=dict(median_s=float(np.median(dt)), cv=float(dt.std() / dt.mean()),
                                                max_gap_s=float(dt.max())))
    return out


# ------------------------------------------------------------------ R2 catch22
def catch22_compare(fleet_dfs, n_win=150, win=2880, seed=0):
    import pycatch22
    rng = np.random.default_rng(seed)

    def feats(x):
        x = np.asarray(x, float)
        if np.nanstd(x) < 1e-12:
            return None
        return pycatch22.catch22_all(x.tolist())["values"]

    ref = os.path.join(HERE, "esa_ref_catch22.npy")   # catch22 features of the real windows (no ESA data needed)
    if os.path.exists(ref) and not os.path.exists(f"{ESA_DATA}/M1_train_vals.npy"):
        real = list(np.load(ref))
    else:
        real = []
        for mk in ("M1", "M2"):
            V = np.load(f"{ESA_DATA}/{mk}_train_vals.npy", mmap_mode="r")
            for _ in range(n_win):
                c = rng.integers(V.shape[1]); i = rng.integers(0, V.shape[0] - win)
                f = feats(V[i:i + win, c])
                if f is not None:
                    real.append(f)
        np.save(ref, np.array(real))
    rng = np.random.default_rng(seed + 1)       # fleet windows drawn independently of the real ones
    ours = []
    tg = [CHID[n] for n in C.CH_NAMES if dict((c[0], c[5]) for c in C.CHANNELS)[n]]
    per_sat = max(1, n_win * 2 // len(fleet_dfs))
    for df in fleet_dfs:
        for _ in range(per_sat):
            c = tg[rng.integers(len(tg))]; i = rng.integers(0, len(df) - win)
            f = feats(df[c].values[i:i + win])
            if f is not None:
                ours.append(f)
    R, O = np.array(real), np.array(ours)
    R[~np.isfinite(R)] = 0; O[~np.isfinite(O)] = 0
    mu, sd = R.mean(0), R.std(0) + 1e-9
    Rz, Oz = (R - mu) / sd, (O - mu) / sd
    # nearest-neighbour distance of our windows to real windows vs real-to-real
    def nn(A, B, same=False):
        d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
        if same:
            np.fill_diagonal(d, np.inf)
        return d.min(1)
    rr, orr = nn(Rz, Rz, True), nn(Oz, Rz)
    thr95 = np.percentile(rr, 95)
    lo, hi = np.percentile(R, 1, 0), np.percentile(R, 99, 0)
    cover = ((O >= lo) & (O <= hi)).mean(0)
    # PCA figure
    U, S, Vt = np.linalg.svd(Rz - Rz.mean(0), full_matrices=False)
    pr, po = Rz @ Vt[:2].T, Oz @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pr[:, 0], pr[:, 1], s=8, alpha=0.5, label="ESA-ADB real telemetry (M1+M2)")
    ax.scatter(po[:, 0], po[:, 1], s=8, alpha=0.5, label="simulated fleet (ours)")
    ax.set_xlabel("PC1 (catch22, fitted on real windows)"); ax.set_ylabel("PC2"); ax.legend()
    ax.set_title("1-day windows in catch22 feature space")
    fig.tight_layout(); fig.savefig(f"{OUTD}/catch22_pca.png", dpi=120); plt.close(fig)
    return dict(n_real=len(R), n_sim=len(O), median_feature_coverage=float(np.median(cover)),
                per_feature_coverage=[float(x) for x in cover],
                frac_sim_within_real_nn95=float((orr <= thr95).mean()),
                frac_real_within_real_nn95=0.95,
                note="coverage = share of simulated windows whose feature lies in the real 1-99 % range")


# ------------------------------------------------------------------ W: degradation / RUL
def rul_check():
    return dict(note="v2: run-to-failure / RUL is evaluated in fed/rul.py (SAT_2 wheel, archive thresholds)")


def main():
    os.makedirs(OUTD, exist_ok=True)
    rep = dict(P1_momentum=momentum_test())
    print("P1", rep["P1_momentum"])
    dfs = []
    rep["physics"], rep["stats"] = {}, {}
    for sat in SATS:
        r, df = physics_checks(sat)
        rep["physics"][sat] = r
        rep["stats"][sat] = dataset_stats(sat, df)
        print(sat, json.dumps(r)[:300])
        dfs.append(df)
    if os.path.exists(f"{ESA_DATA}/M1_train_vals.npy") or os.path.exists(os.path.join(HERE, "esa_ref_catch22.npy")):
        rep["R2_catch22"] = catch22_compare(dfs)
    else:                                   # ESA arrays not on this machine (not in git)
        rep["R2_catch22"] = dict(skipped=f"{ESA_DATA}/M1_train_vals.npy not found")
    print("catch22", {k: v for k, v in rep["R2_catch22"].items() if k != "per_feature_coverage"})
    rep["W_rul"] = rul_check()
    print("RUL", rep["W_rul"])
    json.dump(rep, open(f"{OUTD}/validation_part1.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
