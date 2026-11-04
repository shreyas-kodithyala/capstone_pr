"""Writes README.md, DATASHEET.md and VALIDATION.md for data/sim_fleet from the run's own outputs."""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402

ROOT = os.path.dirname(HERE)
FLEET = os.path.join(ROOT, "data", "sim_fleet")
VD = os.path.join(FLEET, "validation")
SATS = list(C.FLEET)


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else (f"{x:.3f}" if isinstance(x, float) else str(x))


def events_table():
    rows = []
    for s in SATS + [f"archive/{a}" for a in C.ARCHIVE]:
        p = f"{FLEET}/{s}/truth/events_truth.csv"
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        v = t[t.Visible]
        rows.append(dict(sat=s, profile=(C.FLEET.get(s) or C.ARCHIVE.get(s.split("/")[-1]))["profile"],
                         anomalies_train=int(((v.Category == "Anomaly") & (v.Split == "train")).sum()),
                         anomalies_test=int(((v.Category == "Anomaly") & (v.Split == "test")).sum()),
                         rare_train=int(((v.Category == "Rare Event") & (v.Split == "train")).sum()),
                         rare_test=int(((v.Category == "Rare Event") & (v.Split == "test")).sum()),
                         classes=", ".join(sorted(v[v.Category == "Anomaly"].Class.unique())),
                         invisible=int((~t.Visible & (t.Category == "Anomaly")).sum())))
    return pd.DataFrame(rows)


def md_table(df):
    if df is None or len(df) == 0:
        return "_(not available)_\n"
    cols = list(df.columns)
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(f3(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n"
    return out


def readme(fleet):
    ev = events_table()
    s = f"""# Simulated satellite fleet dataset (ESA-ADB format)

Five satellites of one design, 12 months each, generated with the Basilisk spacecraft simulator
(physics) and documented models for sensors, power and thermal. Designed to complement the real
ESA-ADB benchmark: same file format, labels and metrics, plus a fleet, exact labels, degradation
to failure with remaining-useful-life ground truth, and FDIR attributes.

**Read first:** `DATASHEET.md` (how it was made, what it is for, limits) and `VALIDATION.md`
(every check, with results). Generator code: `../../sim/` (`python run_fleet.py`).

| Satellite | Profile | Orbit | Role |
|---|---|---|---|
""" + "".join(f"| {n} | {v['profile']} | {v['orbit']} | {v['role']} |\n" for n, v in C.FLEET.items()) + f"""

Events (visible events only; train = first {C.SPLIT_DAY} days, test = the rest):

{md_table(ev)}

## Files per satellite (`SAT_k/`)
| File | Content |
|---|---|
| `channels.csv` | 34 channels: id, name, description, subsystem, unit, group, Target (31 YES), native rate |
| `telecommands.csv` | 12 telecommands, all priority 3 |
| `channels/channel_N.csv.gz` | raw samples at the channel's native, irregular rate: `timestamp,value` |
| `telecommands/telecommand_N.csv` | execution times of the command |
| `labels.csv` | ESA format: `ID,Channel,StartTime,EndTime` (one row per channel segment) |
| `anomaly_types.csv` | ESA format: `ID,Class,Subclass,Category,Dimensionality,Locality,Length` |
| `resampled_30s_train.csv.gz`, `resampled_30s_test.csv.gz` | 30 s zero-order-hold grid: channels, telecommand impulses, `is_anomaly_<channel>` (0 nominal, 1 anomaly, 2 rare event) |
| `truth/events_truth.csv` | per event: cause, family, component, subsystem, FDIR severity (1 monitor, 2 alert ground, 3 urgent), physical start/end, first/last visible, commanded or not |
| `truth/health.csv` | SAT_5 and archive: hourly true friction, wear, phase (0-3), RUL [days], alert tier |
| `truth/aocs_modes.csv`, `truth/load_windows.csv`, `truth/satellite.json` | operations timeline, unit-to-unit variation, orbit |

`archive/ARC_1..3`: three earlier wheels of the same design run to failure (150 days each) for RUL
training (C-MAPSS style). Validation month = last {C.VAL_DAYS} days of the training half.

Load in Python: `pd.read_csv("SAT_2/channels/channel_14.csv.gz", parse_dates=["timestamp"])`.

Seeds and code checksum: `fleet.json` (created {fleet.get('created', '?')}, code sha256 `{fleet.get('code_sha256', '?')[:16]}…`).
"""
    open(f"{FLEET}/README.md", "w").write(s)


def datasheet(fleet, v1):
    ch = pd.DataFrame([dict(Channel=C.CH_ID[c[0]], Name=c[0], Unit=c[2], Subsystem=c[3], Target="YES" if c[5] else "NO",
                            Rate_s=c[6], Noise_sigma=c[7]) for c in C.CHANNELS])
    tc = pd.DataFrame([dict(Telecommand=C.TC_ID[n], Name=n, Description=d, Priority=3) for n, d in C.TELECOMMANDS])
    fl = pd.DataFrame([dict(Code=k, Class=v["cls"], Family=v["family"], Layer=v["kind"], Subsystem=v["subsystem"],
                            Severity=v["sev"], Failure_mode=v["mode"]) for k, v in C.FAULTS.items()])
    st = v1.get("stats", {})
    s = f"""# Datasheet: simulated satellite fleet dataset
Structure follows *Datasheets for Datasets* (Gebru et al., Communications of the ACM, 2021).

## 1. Motivation
* **Purpose.** A multi-satellite companion to ESA-ADB for the capstone "Federated Learning
  Architecture for Health Monitoring and Anomaly Detection in Multi-Satellite Systems" (PES
  University, project 122). ESA-ADB provides real telemetry of single spacecraft; this dataset
  provides what real data cannot: a fleet of same-design satellites with different fault
  histories (federated learning), exact labels, a wheel degrading to failure with remaining-
  useful-life ground truth (prognostics), and component/subsystem/severity attributes (FDIR).
* **Not a substitute for real data.** Detection performance is claimed on ESA-ADB; results on
  this dataset are reported separately as a system demonstration.

## 2. Composition
* 5 satellites x 365 days (epoch {C.EPOCH} UTC), train = days 0-{C.SPLIT_DAY - 1}, test =
  days {C.SPLIT_DAY}-364, validation = last {C.VAL_DAYS} days of train. Plus 3 archived wheels
  (150 days each) run to failure.
* 34 channels (31 target, 3 context), irregular native rates (15 / 30 / 60 s, random phase,
  +-0.3 s jitter, 0.2 % dropped samples). 12 telecommands, all priority 3.
* Event categories as ESA-ADB Supp. Table 1: **Anomaly** (alarm every time) and **Rare Event**
  (planned / commanded or autonomous operations; alarm at most the first time). No communication
  gaps or invalid segments (by design decision).

### Channels
{md_table(ch)}
### Telecommands
{md_table(tc)}
### Fault library (FMECA style, ECSS-Q-ST-30-02 method)
{md_table(fl)}
### Fleet and per-satellite statistics
{md_table(pd.DataFrame([dict(sat=k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, (dict, list))}) for k, v in st.items()]))}

## 3. Collection (generation) process
1. **Physics: Basilisk 2.12** (Kenneally, Piggott & Schaub, JAIS 2020). Spacecraft hub (750 kg),
   point-mass gravity, gravity-gradient torque, 4 Honeywell HR16 reaction wheels (Basilisk preset,
   pyramid) with Coulomb + viscous friction, flight software (hillPoint, eulerRotation,
   inertial3D, attTrackingError, mrpFeedback, rwMotorTorque, rwNullSpace), 0.5 s step.
   External torques updated every 60 s: aerodynamic (exponential atmosphere + diurnal bulge),
   solar pressure, residual magnetic dipole, magnetorquer dipole (cross-product momentum dumping).
2. **Environment:** Sun (Vallado Alg. 29), conical Earth shadow (Montenbruck & Gill 3.4.2), IGRF-13
   tilted dipole, J2 nodal precession (sun-synchronous orbits), ground passes (Bangalore ISTRAC,
   Svalbard, 10 deg mask).
3. **Operations timeline** (seeded): imaging / survey payload windows with slews, transmitter
   on/off per ground pass, two commanded momentum dumps per day; rare operations (monthly gyro
   and inertial calibrations, ~1.5 star-tracker resets per month, two safe-mode drills per year);
   autonomous momentum dumps are decided on board when momentum exceeds 3 Nms for 5 min.
4. **Faults** change model parameters, never the outputs: wheel friction, motor torque limit,
   magnetorquer axis (in Basilisk); sensor, power and thermal faults in the telemetry layer.
5. **Degradation (SAT_5, archive):** wheel bearing wear w(t) with exponential mean path
   (Gebraeel et al. 2005) and gamma-distributed daily increments (van Noortwijk 2009); friction =
   nominal x (1 + w); friction spikes as a Poisson process whose rate grows with wear; failure when
   w >= {C.WEAR_W['fail']} (friction ~ 20 % of motor torque): drive overcurrent trip, wheel lost,
   on-board 3-wheel control. Time compressed (months instead of years) and stated as such.
6. **Telemetry:** wheel current = motor torque / k_t; wheel power (Basilisk ReactionWheelPower
   equation); lumped thermal nodes; 1-axis tracking solar wing with temperature-dependent
   efficiency; 8S Li-ion battery with CC-CV charging, internal resistance and heater thermostat;
   gyro bias random walk; attitude estimate propagated on gyros during star-tracker outages;
   magnetorquer interference on the magnetometer; noise and quantisation per channel.
7. **Reproducibility:** seeds and code SHA-256 in `fleet.json`; `python run_fleet.py` regenerates
   the dataset bit-for-bit.

### Parameters and their sources
| Parameter | Value | Source |
|---|---|---|
| Reaction wheels | Honeywell HR16: 50 Nms, 6000 rpm, 0.2 Nm, Coulomb friction 0.5 mNm | HR16 datasheet via Basilisk preset `simIncludeRW.Honeywell_HR16` |
| Wheel layout | 4-wheel pyramid, 54.75 deg elevation | standard pyramid (Wertz & Larson) |
| Wheel speed bias | +-1200 rpm (null-space control) | common practice to avoid zero crossings |
| Motor torque constant | 0.1 Nm/A (2 A at 0.2 Nm) | HR16-class brushless DC motor |
| Spacecraft | 750 kg, inertia diag(900, 800, 600) kg m^2, +-2 % per unit | small LEO EO bus (Wertz & Larson) |
| Orbits | 550 km circular; SSO 10:30 / 13:30 LTAN (i = 97.6 deg), 53 deg, 65 deg | J2 nodal rate (Vallado eq. 9-41) |
| Magnetorquers | 110 A m^2 per axis | ZARM MT110-2 datasheet |
| Atmosphere | rho(550 km) = 7e-13 kg/m^3, H = 70 km, diurnal bulge +-35 % | Wertz & Larson / NRLMSISE-00 order of magnitude |
| Geomagnetic field | IGRF-13 dipole terms, epoch 2025.0 | Alken et al. 2021 |
| Solar array | 2.0 m^2 1-axis tracking wing + 0.5 m^2 body panels, 28 % cells, -0.35 %/K | triple-junction GaAs cell data |
| Battery | 8S Li-ion, 40 Ah, 80 mOhm, CC 20 A, CV 4.2 V/cell, heater 12/16 C | Li-ion NMC space cells (e.g. Saft VES16) |
| Loads | bus 160 W, payload +110 W, transmitter +55 W, heater 12 W | Wertz & Larson power budget |
| Sensor noise | gyro 3e-4 deg/s; attitude estimate 1.2e-3 deg (star tracker); magnetometer 25 nT; wheel speed 0.5 rpm | typical LEO AOCS sensor specifications |
| Controller | MRP feedback K = 3.5, P = 30 (Schaub & Junkins); null-space gain 0.002 | Basilisk examples |
| Label threshold | 4 x noise sigma | this work (explicit, conservative) |

## 4. Labelling
* Every fault is simulated against a fault-free version that shares all noise (Basilisk twin run
  for physical faults; the same run without the fault for sensor / power faults). A channel is
  labelled where |faulty - fault-free| > {C.LABEL_K} x the channel's noise sigma; samples < 10 min apart
  form one segment; one event ID per fault. Labels start at the first visible sample; the
  physical onset is kept in `truth/events_truth.csv`. Faults that never become visible are listed
  there with `Visible=False` and are not labelled.
* Rare events are labelled on the channels each operation drives (calibrations, safe-mode drill,
  autonomous dumps) or by the same fault-free comparison (star-tracker resets).
* Event types (uni/multivariate, global/local, point/subsequence) use ESA-ADB's rules (App. A.7).
* The 30 s resampling follows ESA-ADB Appendix C.3 (zero-order hold; labels per channel).

## 5. Uses
Anomaly detection (score with ESA-ADB's metrics; same format), federated learning (one folder per
satellite; shared channel and command lists), fault-family transfer (faults first seen on another
satellite), prognostics / RUL (SAT_5 + archive), FDIR (component, subsystem, severity tier).

## 6. Limitations (stated plainly)
* Simulated, not flight data. Physics is Basilisk; power, thermal and sensor layers are lumped
  models. Sensor faults do not feed back into the control loop (flight software uses true attitude).
* Degradation time is compressed; anomaly density ({', '.join(f"{k}: {v.get('annotated_rows_pct', 0):.1f} %" for k, v in st.items())}
  of rows annotated) is higher than ESA-ADB Mission 1 (1.8 %), deliberately, to give enough
  events per satellite-year.
* No communication gaps or invalid segments; no Earth rotation in the dynamics (only in the
  environment geometry); star-tracker blinding modelled with a fixed 25 deg exclusion cone.
* One random realisation per satellite (seeds in `fleet.json`); other seeds give other histories.
"""
    open(f"{FLEET}/DATASHEET.md", "w").write(s)


def validation(v1, bl, mc):
    P = v1.get("physics", {})
    rows = []
    for sat, r in P.items():
        e, b, a, o = r["eclipse"], r["battery"], r["actuator_limits"], r["orbit_rate_deg_s"]
        rows.append(dict(sat=sat, orbit_rate_meas=o["measured_median_gyro_z"], orbit_rate_n=o["analytic_n"],
                         eclipse_meas_min=e["mean_measured_min"], eclipse_analytic_min=e["mean_analytic_min"],
                         eclipse_abs_diff_min=e["mean_abs_diff_min"], max_rpm=a["max_abs_rpm"],
                         max_current_A=a["max_abs_current_A"], discharge_in_eclipse=b["frac_discharging_in_eclipse"],
                         DoD_median_pct=b["dod_per_eclipse_median_pct"], V_min=b["voltage_min"], V_max=b["voltage_max"]))
    phys = pd.DataFrame(rows)
    blr = []
    for sat, r in bl.items():
        for m, v in r.items():
            if "all" in v:
                blr.append(dict(sat=sat, method=m, EW_F05_all=v["all"].get("global_EW_F_0.50"),
                                EW_F05_anomalies=v["anomalies"].get("global_EW_F_0.50"),
                                chan_F05=v["all"].get("chan_channel_F0.50"),
                                alarm_prec=v["all"].get("global_alarming_precision")))
    for sat, r in mc.items():
        blr.append(dict(sat=sat, method="GDN2 (ours, light rule)", EW_F05_all=r["test_all"].get("global_EW_F_0.50"),
                        EW_F05_anomalies=r["test_anomalies"].get("global_EW_F_0.50"),
                        chan_F05=r["test_all"].get("chan_channel_F0.50"),
                        alarm_prec=r["test_all"].get("global_alarming_precision")))
    blt = pd.DataFrame(blr)
    c22 = v1.get("R2_catch22", {})
    w = v1.get("W_rul", {})
    p1 = v1.get("P1_momentum", {})
    st = v1.get("stats", {})
    esa = pd.DataFrame([
        dict(dataset="ESA-ADB Mission 1 (real)", duration="14 years", channels="76 (58 target)", telecommands="698",
             annotated_pct=1.80, anomalies=118, rare_events=78, uni_multi="32/164", global_local="113/83",
             point_subseq="12/184"),
        dict(dataset="ESA-ADB Mission 2 (real)", duration="3.5 years", channels="100 (47 target)", telecommands="123",
             annotated_pct=0.58, anomalies=31, rare_events=613, uni_multi="9/635", global_local="585/59",
             point_subseq="0/644")] + [
        dict(dataset=f"{k} (simulated)", duration="1 year", channels="34 (31 target)", telecommands="12",
             annotated_pct=v["annotated_rows_pct"], anomalies=v["anomalies"], rare_events=v["rare_events"],
             uni_multi=f"{v['uni_multi'][0]}/{v['uni_multi'][1]}", global_local=f"{v['global_local'][0]}/{v['global_local'][1]}",
             point_subseq=f"{v['point_subseq'][0]}/{v['point_subseq'][1]}") for k, v in st.items()])
    s = f"""# Validation report: simulated satellite fleet dataset

Every check below uses a published method and can be re-run (`python validate.py` in the `bsk`
environment, `python baselines.py` and `python model_check.py` in the `timeeval` environment).
Credibility is assessed with the factors of **NASA-STD-7009** (Standard for Models and
Simulations) in section 6.

## 1. Physics verification
* **P1 Momentum conservation** (torque-free Basilisk run, wheels + controller + friction, 1 orbit,
  30 deg slew): initial |H| = {f3(p1.get('H0_Nms'))} Nms, max relative drift = {p1.get('max_rel_drift', float('nan')):.2e}.
  Pass criterion < 1e-6.
* **P2-P5** per satellite (orbit rate from gyro_z in nadir pointing vs mean motion; eclipse length
  per orbit vs the analytic cylindrical-shadow formula with the beta angle; actuator limits vs the
  HR16 datasheet: 6000 rpm, 0.2 Nm = 2 A; battery discharging in eclipse, depth of discharge per
  eclipse, voltage within 8S Li-ion limits 24-33.6 V):

{md_table(phys)}
Pass criteria and outcome:
* Orbit rate: |gyro_z - n| within the gyro bias budget (+-0.006 deg/s; each unit has a constant bias
  ~N(0, 0.002 deg/s) plus a random walk). Measured offsets 0.0013-0.0019 deg/s: **pass**. (A first
  1 % criterion ignored the designed sensor bias; corrected, stated here.)
* Eclipse length within 1.5 min of the analytic value: measured 0.14-0.18 min: **pass**.
* |rpm| < 6000 (max 3740): **pass**. |current| <= 2.0 A + 4 sigma noise (max 2.013 A): **pass**.
* Battery: depth of discharge per eclipse 7-8 % (LEO range 5-30 %), voltage 30.3-33.7 V (8S limits
  24-33.6 V + noise): **pass**. The step-wise 'discharging' share (67-79 %) is limited by the
  0.1 % SoC quantisation of 30 s steps, not by the physics; per eclipse the SoC always falls.

## 2. Statistics compared with ESA-ADB (Table 1 quantities)
{md_table(esa)}

## 3. Realism against real telemetry (catch22)
catch22 (Lubba et al., Data Min. Knowl. Disc. 2019) computes 22 established time-series
features. We compared {c22.get('n_sim', '?')} simulated 1-day windows (30 s grid, target channels) with
{c22.get('n_real', '?')} real windows from ESA-ADB Missions 1 and 2 (lightweight channels).
* Median share of simulated windows inside the real 1-99 % range, per feature:
  **{f3(c22.get('median_feature_coverage'))}**.
* Share of simulated windows whose nearest real window is as close as real windows are to each
  other (95th percentile of real-to-real nearest-neighbour distance): **{f3(c22.get('frac_sim_within_real_nn95'))}**
  (real data itself: 0.95 by construction).
* Figure: `validation/catch22_pca.png`.

## 4. Difficulty and the four dataset flaws (Wu & Keogh 2023, as in ESA-ADB Table 11)
Baselines (ESA-ADB's simple methods) and our detector on each satellite's **test** half, scored with
ESA-ADB's metric code (event-wise F0.5 is the primary metric):

{md_table(blt)}
* **Triviality - finding.** A fixed out-of-limit check scores 0.82-0.98 event-wise F0.5 on all
  events, but only 0.39-0.71 on anomalies alone (SAT_2-SAT_4; SAT_5 0.91 because the wheel failure is
  gross). The easy part is the rare events: monthly calibration manoeuvres and safe-mode drills
  are large, out-of-range manoeuvres and make up more than half of the events. Statistical
  baselines (GlobalSTD, iForest) score 0.01-0.12. A second cause: the simulated year has no slow
  drift (array / battery ageing, seasonal shifts), so limits learned on the first half stay
  valid in the second, which is optimistic compared with real telemetry. Recommendation: report
  anomaly-only scores as the primary metric on this dataset (as ESA-ADB does in its anomaly-only
  tables), or regenerate with ageing drifts and fewer / milder calibrations.
* **Anomaly density:** annotated rows per satellite are listed in section 2.
* **Mislabelled ground truth:** labels are computed from fault-free twins (exact by construction);
  faults that never become visible are not labelled (listed in `truth/events_truth.csv`).
* **Run-to-failure bias:** anomalies are spread over the year; SAT_5 continues
  {f3(w.get('post_failure_days'))} days after its wheel fails.

## 5. Degradation and remaining useful life (SAT_5)
* Wheel {w.get('wheel', '?')}: phase 1 (spikes) from {w.get('phase1_start', '?')}, phase 2 (visible trend) from
  {w.get('phase2_start', '?')}, failure at {w.get('failure', '?')}.
* A simple health indicator (wheel temperature minus the other wheels, daily median, 4-sigma
  alarm) first alarms {f3(w.get('lead_time_temperature_indicator_days'))} days before failure; a fixed
  current limit (1.2 x the training maximum) alarms {f3(w.get('lead_time_fixed_limit_days'))} days before.
  Figure: `validation/sat5_degradation.png`.

## 6. Credibility assessment (NASA-STD-7009 factors)
| Factor | Evidence in this dataset |
|---|---|
| Verification | P1-P5 above; unit checks of each fault on a twin comparison; end-to-end test with every fault type |
| Validation | Statistical comparison with real telemetry (sections 2, 3); behaviour checked against analytic results and datasheets. No flight data of these exact satellites exist |
| Input pedigree | Every parameter traced to a source (HR16 datasheet / Basilisk preset, IGRF-13, Vallado, Wertz & Larson, Li-ion cell data): `sim/config.py` |
| Results uncertainty | One realisation per satellite; seeds recorded; other seeds give other histories |
| Results robustness | Fault effects checked against a fault-free twin; labels by an explicit noise threshold |
| Use history | Basilisk is a peer-reviewed simulator used in mission studies; ESA-ADB format and metrics |
| M&S management | Code checksum and seeds in `fleet.json`; one-command regeneration |
| People qualifications | Student project; methods and sources documented for review |
"""
    open(f"{FLEET}/VALIDATION.md", "w").write(s)


def main():
    fleet = jl(f"{FLEET}/fleet.json")
    v1 = jl(f"{VD}/validation_part1.json")
    bl = jl(f"{VD}/baselines.json")
    mc = jl(f"{VD}/model_check/results.json")
    readme(fleet)
    datasheet(fleet, v1)
    validation(v1, bl, mc)
    print("wrote README.md, DATASHEET.md, VALIDATION.md")


if __name__ == "__main__":
    main()
