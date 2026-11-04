"""Fleet dataset configuration: one place for every number, with its source.

Design: 5 satellites of ONE design (same bus, channels, telecommands, flight software), each with
its own orbit, mission role, small unit-to-unit variation and its own fault history.

Sources for parameters (full list with values in specs/dataset_spec.md):
  [HR16]   Honeywell HR16 reaction wheel: Basilisk preset simIncludeRW.Honeywell_HR16
           (50 Nms, 6000 rpm, 0.2 Nm, Coulomb friction 5e-4 Nm)
  [BSK]    Basilisk (Kenneally, Piggott & Schaub, JAIS 2020): spacecraft, RW, gravity-gradient,
           MRP feedback (Schaub & Junkins), rwMotorTorque, rwNullSpace, hillPoint, eulerRotation
  [VAL]    Vallado, Fundamentals of Astrodynamics, 4th ed.: Sun position (Alg. 29), GMST, J2 node rate
  [MG]     Montenbruck & Gill, Satellite Orbits (2000), 3.4.2: conical Earth shadow model
  [IGRF]   IGRF-13 (Alken et al., Earth Planets Space 2021): dipole terms g10, g11, h11 + SV
  [WL]     Wertz & Larson, Space Mission Analysis and Design: disturbance torques, power budget,
           LEO battery depth of discharge, thermal ranges
  [MT110]  ZARM MT110-2 magnetorquer: 110 A m^2 linear dipole
  [LiIon]  Li-ion NMC cell (e.g. Saft VES16): 3.0-4.2 V, ~4.5 Ah; 8S pack typical for 28 V bus
"""
import numpy as np

# ---------------------------------------------------------------- time ----
EPOCH = "2025-01-01T00:00:00"          # UTC; all satellites share the mission calendar
DAYS = 365                             # 12 months per satellite
SPLIT_DAY = 182                        # train: days [0, 182), test: [182, 365)
VAL_DAYS = 45                          # last 45 days of train = validation (detector settings chosen here)
DT_DYN = 0.5                           # [s] dynamics + flight software step (probe: identical to 0.25 s)
DT_REC = 5.0                           # [s] truth recording step
DT_TICK = 60.0                         # [s] environment / disturbance / magnetorquer update step
GRID = 30                              # [s] resampling grid (ESA-ADB Mission 1: 0.033 Hz)

# ---------------------------------------------------------- spacecraft ----
SC = dict(
    mass=750.0,                                         # [kg] small LEO EO bus [WL]
    inertia=[900.0, 3.0, -5.0, 3.0, 800.0, 2.0, -5.0, 2.0, 600.0],   # [kg m^2], small products of inertia
    rw_beta_deg=54.75,                                  # pyramid elevation (standard 4-wheel pyramid)
    rw_az_deg=[45.0, 135.0, 225.0, 315.0],
    rw_maxH=50.0,                                       # [Nms] [HR16]
    rw_umax=0.2,                                        # [Nm]  [HR16]
    rw_fc=5.0e-4,                                       # [Nm] Coulomb friction [HR16 preset]
    rw_cv=2.0e-6,                                       # [Nm s/rad] viscous friction (bearing drag ~ 2.5e-4 Nm at 1200 rpm)
    rw_bias_rpm=1200.0,                                 # null-space speed bias (avoid zero crossings)
    rw_kt=0.1,                                          # [Nm/A] motor torque constant (HR16 class BLDC)
    rw_pbase=3.0,                                       # [W] wheel electronics base power
    rw_eta=0.80,                                        # electrical->mechanical efficiency (Basilisk ReactionWheelPower)
    mtq_max=110.0,                                      # [A m^2] [MT110]
    mtq_gain=5.0e-4,                                    # cross-product momentum-dump gain
    res_dipole=[0.2, -0.1, 0.3],                        # [A m^2] residual magnetic dipole [WL]
    cp_offset=[0.12, -0.05, 0.08],                      # [m] centre of pressure - centre of mass [WL]
    area_body=3.0, area_wing=4.0, cd=2.2,               # [m^2], drag coefficient [WL]
    K=30.0, P=150.0,                                    # MRP feedback gains: wn ~0.09-0.11 rad/s, zeta ~1 (agile EO bus; v1 used 3.5/30)
    null_gain=0.002,
)
EPS = dict(
    wing_area=2.0, body_panel_area=0.5, cell_eff=0.28,  # triple-junction GaAs [WL]
    temp_coeff=-0.0035, t_ref=28.0,                     # efficiency change per K above 28 C (GaAs)
    n_strings=12,
    bus_base=160.0, payload=110.0, tx=55.0, heater=12.0,  # [W] loads
    cells=8, cap_ah=40.0, r_int=0.08,                    # 8S Li-ion pack, 40 Ah, 80 mOhm [LiIon]
    i_charge_max=20.0, eta_charge=0.97,
    heater_on=12.0, heater_off=16.0,                     # battery thermostat [C]
)
# Li-ion NMC open-circuit voltage per cell vs SoC (typical curve)
OCV_SOC = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], float)
OCV_V = np.array([3.00, 3.35, 3.48, 3.58, 3.64, 3.68, 3.72, 3.78, 3.86, 3.95, 4.05, 4.10, 4.17])

# ------------------------------------------------------------- orbits ----
ALT = 550e3                     # [m] all satellites at 550 km circular
STATIONS = {"Bangalore": (13.03, 77.51), "Svalbard": (78.23, 15.40)}   # ISRO ISTRAC, KSAT
EL_MASK = 10.0                  # [deg]

# --------------------------------------------------------------- fleet ----
# v2 (4 Nov 2026): 4 satellites. profile drives the fault schedule, role the routine operations.
#   SAT_1 healthy    rare, isolated anomalies of every family (ESA-like sparsity)
#   SAT_2 wheel      F1 wheel faults in train; test adds F2 sensor faults (some overlapping F1);
#                    wheel 2 wears out over the year, visible ~2 months before it fails (RUL)
#   SAT_3 power      F3 power/thermal faults in train; test adds F2 and F1
#   SAT_4 ops        telecommand-heavy: many rare nominal events, few anomalies
FLEET = {
    "SAT_1": dict(profile="healthy", role="survey", orbit=dict(kind="sso", ltan=10.5, argl0=0.0)),
    "SAT_2": dict(profile="wheel", role="imaging", orbit=dict(kind="incl", inc=65.0, raan0=160.0, argl0=45.0)),
    "SAT_3": dict(profile="power", role="survey", orbit=dict(kind="incl", inc=53.0, raan0=40.0, argl0=0.0)),
    "SAT_4": dict(profile="ops", role="imaging", orbit=dict(kind="sso", ltan=13.5, argl0=180.0)),
}
# archived run-to-failure wheels of the same design (failure threshold for RUL, C-MAPSS style)
ARCHIVE = {
    "ARC_1": dict(profile="archive", role="survey", days=150, wear=dict(t_on=10, fail=120),
                  orbit=dict(kind="sso", ltan=10.5, argl0=200.0)),
    "ARC_2": dict(profile="archive", role="imaging", days=150, wear=dict(t_on=5, fail=95),
                  orbit=dict(kind="incl", inc=53.0, raan0=250.0, argl0=10.0)),
    "ARC_3": dict(profile="archive", role="survey", days=150, wear=dict(t_on=15, fail=135),
                  orbit=dict(kind="incl", inc=65.0, raan0=300.0, argl0=120.0)),
}
# SAT_2 wheel 2 degradation (days from epoch): hidden onset, exponential mean wear path (Gebraeel
# 2005) with gamma-distributed daily increments (van Noortwijk 2009); phases by friction level.
WEAR_SAT = "SAT_2"
SAT5_WEAR = dict(wheel=1, t_on=190.0, t_phase2=300.0, t_fail=354.0)
WEAR_W = dict(spikes=0.05, phase2=3.0, fail=79.0)    # w thresholds: friction = nominal * (1 + w)
# rare nominal operations per profile: (gyro cal / yr, inertial cal / yr, star-tracker resets / month,
# safe drills / yr). SAT_4 runs a campaign-heavy calibration plan.
RARE_RATE = dict(healthy=(4, 4, 0.5, 1), wheel=(4, 4, 0.5, 1), power=(4, 4, 0.5, 1), ops=(12, 12, 2.0, 4),
                 archive=(4, 4, 0.5, 0))
# slow ageing (all satellites): solar-cell power loss and battery capacity fade per year [WL; LiIon]
AGEING = dict(sa_per_year=0.0275, cap_per_year=0.02, rint_per_year=0.05)

# ------------------------------------------------------------ channels ----
# (name, description, unit, subsystem, group, target, rate_s, noise_sigma, quantum)
CHANNELS = []
# v2: 20 channels (19 targets + AOCS mode context); dependencies kept: wheel speed <-> current
# (torque), wheels <-> gyro / attitude error (control loop), magnetometer (momentum dumps),
# solar power -> battery voltage -> bus load, battery temperature (heater).
for k in range(4):
    CHANNELS.append((f"rw{k+1}_speed", f"Reaction wheel {k+1} speed", "rpm", "AOCS", 1, True, 15, 0.5, 0.1))
for k in range(4):
    CHANNELS.append((f"rw{k+1}_current", f"Reaction wheel {k+1} motor current", "A", "AOCS", 2, True, 15, 0.004, 0.001))
for a in "xyz":
    CHANNELS.append((f"gyro_{a}", f"Gyro body rate {a}", "deg/s", "AOCS", 5, True, 15, 3e-4, 1e-5))
CHANNELS.append(("att_err", "Total attitude error (estimate)", "deg", "AOCS", 6, True, 15, 1.2e-3, 1e-4))
for a in "xyz":
    CHANNELS.append((f"mag_{a}", f"Magnetometer {a}", "nT", "AOCS", 7, True, 30, 25.0, 1.0))
CHANNELS.append(("sa_power", "Solar array power", "W", "EPS", 10, True, 30, 1.5, 0.5))
CHANNELS.append(("bat_voltage", "Battery voltage", "V", "EPS", 11, True, 30, 0.02, 0.01))
CHANNELS.append(("bus_load", "Bus load power", "W", "EPS", 12, True, 30, 1.5, 0.5))
CHANNELS.append(("bat_temp", "Battery temperature", "degC", "THERMAL", 13, True, 60, 0.08, 0.05))
CHANNELS.append(("aocs_mode", "AOCS mode (0 nadir,1 target,2 cal,3 inertial,4 safe)", "enum", "AOCS", 15, False, 60, 0.0, 1.0))
CH_NAMES = [c[0] for c in CHANNELS]
CH_ID = {c[0]: f"channel_{i+1}" for i, c in enumerate(CHANNELS)}
SUBSYS_ID = {"AOCS": "subsystem_1", "EPS": "subsystem_2", "THERMAL": "subsystem_3"}

# --------------------------------------------------------- telecommands ----
# all priority 3 (agreed); every type occurs > 2 times in train and >= 1 time in test
TELECOMMANDS = [
    ("SLEW_TARGET", "Slew to imaging target (offset from nadir)"),
    ("POINT_NADIR", "Return to nadir pointing"),
    ("PAYLOAD_ON", "Payload power on"),
    ("PAYLOAD_OFF", "Payload power off"),
    ("TX_ON", "X-band transmitter on (ground pass)"),
    ("TX_OFF", "X-band transmitter off"),
    ("MTQ_DUMP", "Start 30 min magnetic momentum dump"),
    ("GYRO_CAL", "Gyro calibration manoeuvre sequence"),
    ("STR_RESET", "Star tracker reset"),
    ("SAFE_ENTER", "Enter sun-safe mode (drill)"),
    ("SAFE_EXIT", "Exit sun-safe mode"),
    ("INERTIAL_CAL", "Inertial-hold calibration (deep space)"),
]
TC_NAMES = [t[0] for t in TELECOMMANDS]
TC_ID = {t[0]: f"telecommand_{i+1}" for i, t in enumerate(TELECOMMANDS)}

# ------------------------------------------------------- fault library ----
# FMECA-style: component -> failure mode -> effect. kind: physics (Basilisk) or
# post-processing layer (sensor / power / thermal). dur in hours (min, max).
# severity: 1 monitor, 2 alert ground, 3 urgent.
FAULTS = {
    # Family A: actuators (physics)
    "A1": dict(cls="rw_friction_spike", family="A", kind="physics", subsystem="AOCS", dur=(0.5, 6.0), sev=2,
               mode="Bearing lubricant starvation transient: friction x(4-12)"),
    "A2": dict(cls="rw_drive_dropout", family="A", kind="physics", subsystem="AOCS", dur=(1 / 60, 5 / 60), sev=2,
               mode="Motor driver latch-up (SEU), auto-reset after 1-5 min"),
    "A3": dict(cls="rw_torque_limit", family="A", kind="physics", subsystem="AOCS", dur=(6.0, 24.0), sev=2,
               mode="Motor current limiter degraded: max torque 25-40 %"),
    "A4": dict(cls="mtq_axis_open", family="A", kind="physics", subsystem="AOCS", dur=(24.0, 96.0), sev=2,
               mode="Magnetorquer coil open circuit (one axis)"),
    # Family B: attitude sensors (post-processing)
    "B1": dict(cls="gyro_bias_instability", family="B", kind="sensor", subsystem="AOCS", dur=(2.0, 12.0), sev=2,
               mode="Gyro bias instability episode (thermal)"),
    "B2": dict(cls="gyro_spikes", family="B", kind="sensor", subsystem="AOCS", dur=(10 / 60, 40 / 60), sev=1,
               mode="Gyro single-event upsets (spikes)"),
    "B3": dict(cls="mag_stuck_axis", family="B", kind="sensor", subsystem="AOCS", dur=(1.0, 8.0), sev=1,
               mode="Magnetometer ADC frozen on one axis"),
    "B4": dict(cls="str_outage", family="B", kind="sensor", subsystem="AOCS", dur=(15 / 60, 1.0), sev=2,
               mode="Star tracker uncommanded reboot / loss of lock"),
    # Family C: power (post-processing)
    "C1": dict(cls="sa_string_open", family="C", kind="power", subsystem="EPS", dur=(6.0, 48.0), sev=2,
               mode="Solar string intermittent open circuit (cracked interconnect)"),
    "C2": dict(cls="bat_impedance_rise", family="C", kind="power", subsystem="EPS", dur=(6.0, 48.0), sev=2,
               mode="Battery cell impedance rise x3 (cell degradation episode)"),
    "C3": dict(cls="uncommanded_load", family="C", kind="power", subsystem="EPS", dur=(10 / 60, 1.0), sev=1,
               mode="Unit latch-up: +50-90 W until current limiter trips"),
    "C4": dict(cls="payload_no_power", family="C", kind="power", subsystem="EPS", dur=(0.0, 0.0), sev=2,
               mode="PAYLOAD_ON command, payload does not draw power (wrong response)"),
    # Family D: thermal (post-processing)
    "D1": dict(cls="heater_stuck_on", family="D", kind="power", subsystem="THERMAL", dur=(3.0, 18.0), sev=2,
               mode="Battery heater thermostat stuck closed"),
    "D2": dict(cls="temp_sensor_stuck", family="D", kind="sensor", subsystem="THERMAL", dur=(2.0, 12.0), sev=1,
               mode="Thermistor reading frozen"),
    # SAT_5 degradation (physics)
    "W1": dict(cls="rw_wear_spike", family="W", kind="physics", subsystem="AOCS", dur=(0.3, 3.0), sev=2,
               mode="Degrading bearing: friction spike (phase 1)"),
    "W2": dict(cls="rw_degradation", family="W", kind="physics", subsystem="AOCS", dur=None, sev=2,
               mode="Degrading bearing: friction trend (phase 2)"),
    "W3": dict(cls="rw_failure", family="W", kind="physics", subsystem="AOCS", dur=None, sev=3,
               mode="Bearing seizure, drive overcurrent trip: wheel lost (phase 3)"),
}
# fault counts per profile and split: {code: (n_train, n_test)}. v2 library in use:
#   F1 wheel   A1 friction spike, A2 drive dropout (+ W1-W3 wear on SAT_2)
#   F2 sensor  B1 gyro bias drift, B2 gyro spikes (point), B4 star-tracker outage
#   F3 power   C1 solar string open, C3 unit latch-up, D1 heater stuck on
FAMILY = dict(A="F1", W="F1", B="F2", C="F3", D="F3")
# Each satellite trains on a subset of families and meets the others only in test; every family
# is seen in training somewhere in the fleet (the case for federated learning).
PROFILE_FAULTS = {
    "healthy": {"B1": (1, 1), "B2": (2, 0), "B4": (1, 0), "A1": (0, 1), "A2": (0, 1), "C1": (0, 1),
                "C3": (0, 1), "D1": (0, 1)},
    "wheel": {"A1": (5, 3), "A2": (5, 2), "B1": (0, 3), "B2": (0, 2), "B4": (0, 3)},
    "power": {"C1": (3, 2), "C3": (4, 2), "D1": (3, 2), "B1": (0, 2), "B2": (0, 1), "B4": (0, 2),
              "A1": (0, 2), "A2": (0, 2)},
    "ops": {"A1": (2, 0), "A2": (1, 0), "B2": (1, 0), "B4": (1, 0), "C1": (0, 2), "C3": (0, 2), "D1": (0, 1)},
    "archive": {},
}
# rare (nominal) events, labelled 'Rare Event' (ESA Supp. Table 1)
RARE = {
    "GYRO_CAL": dict(cls="gyro_calibration", commanded=True),
    "INERTIAL_CAL": dict(cls="inertial_calibration", commanded=True),
    "STR_RESET": dict(cls="star_tracker_reset", commanded=True),
    "SAFE": dict(cls="safe_mode_drill", commanded=True),
    "AUTO_DUMP": dict(cls="autonomous_momentum_dump", commanded=False),
}
SETTLE_H = dict(physics=2.0, sensor=0.5, power=6.0)   # label window extension after fault end [h]
LABEL_K = 4.0                                          # visible if |faulty - clean| > 4 sigma_noise
