"""Replay the saved final-fleet Basilisk records in Vizard. Nothing is simulated again.

Reads data/sim_work/<sat>/phys/truth_*.npz (the records the dataset was made from) and the events and
telecommands in data/sim_fleet, then writes Vizard .bin files.

Frames. Each satellite's record is in its own frame P (envmodels.py), which turns slowly with that
orbit's J2 node rate. Position and attitude are rotated back to one shared inertial frame so that all
satellites, the Sun and the Earth share one scene. This is the same rotation the simulator used for its
Sun, eclipse, magnetic field and ground-pass geometry, so the scene matches what the dataset used.

Truth labels are used only to colour and caption the replay. They never reach any model.

  python viz_replay.py --clip year        (or a clip name below; --list prints them)
  python viz_replay.py --sats SAT_2 --start-day 355 --days 2 --step 5 --name my_clip
"""
import argparse
import glob
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import config as C  # noqa: E402
from envmodels import AU, RE, Orbit, dcm_to_mrp, gmst, mrp_to_dcm, rotz, station_ecef, sun_eci  # noqa: E402
from operations import epoch_unix  # noqa: E402

from Basilisk.architecture import messaging, sysModel  # noqa: E402
from Basilisk.simulation import vizInterface  # noqa: E402
from Basilisk.utilities import SimulationBaseClass, macros, vizSupport  # noqa: E402

SATS = ["SAT_1", "SAT_2", "SAT_3", "SAT_4"]
FLEET = os.path.join(ROOT, "data", "sim_fleet")
WORK = os.path.join(ROOT, "data", "sim_work")
OUT = os.path.join(ROOT, "viz", "clips")
DAY = 86400.0
EPOCH = datetime(2025, 1, 1, tzinfo=timezone.utc)

# trajectory colours (RGBA), highest priority last
COL = dict(nominal=(90, 170, 255, 255), slew=(0, 230, 200, 255), rare=(255, 200, 0, 255), anomaly=(255, 60, 60, 255),
           degrading=(255, 130, 0, 255), failed=(200, 0, 255, 255))
FAULT = dict(A1="Wheel friction spike", A2="Wheel drive dropout", W1="Wear friction spike", W2="Wheel degradation",
             W3="Wheel failure", B1="Gyro bias drift", B2="Gyro spikes", B4="Star-tracker outage",
             C1="Solar strings open", C3="Unit latch-up", D1="Heater stuck on")
RARE = dict(GYRO_CAL="Gyro calibration", INERTIAL_CAL="Inertial calibration", STR_RESET="Star-tracker reset",
            SAFE="Safe-mode drill", AUTO_DUMP="Autonomous momentum dump")
MILESTONES = [(116.45, "Weight-training data ends (day 116.4)"), (137.0, "Detector-settings window starts (day 137)"),
              (182.0, "Frozen test period starts (day 182)")]


def day_of(ts):
    return (pd.to_datetime(ts, utc=True) - pd.Timestamp(EPOCH)).total_seconds() / DAY


def load_record(sat, t0, t1, step):
    """Saved 5 s record for [t0, t1), keeping one row per `step` seconds."""
    keep = int(round(step / 5.0))
    parts = []
    for f in sorted(glob.glob(f"{WORK}/{sat}/phys/truth_*.npz")):
        z = np.load(f)
        t = np.round(z["t"])                               # the record stores time with rounding noise
        if t[-1] < t0 or t[0] >= t1:
            continue
        i = np.flatnonzero((t >= t0) & (t < t1) & (np.round(t / 5.0).astype(np.int64) % keep == 0))
        parts.append({k: z[k][i] for k in ("r_BN", "sigma_BN", "omega_BN", "Omega", "u")} | dict(t=t[i]))
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    _, first = np.unique(out["t"], return_index=True)      # each 80-day restart repeats one timestamp
    return {k: v[first] for k, v in out.items()}


def to_shared_frame(sat, rec):
    """Frame P -> shared inertial frame: r_N = Rz(dOmega) r_P, C_BN = C_BP Rz(-dOmega)."""
    o = Orbit(C.FLEET[sat]["orbit"], epoch_unix())
    t = rec["t"].astype(np.float64)
    dO = o.d_omega(epoch_unix() + t)
    r = rotz(dO, rec["r_BN"].astype(np.float64))
    v = np.gradient(r, t, axis=0)
    Cbp = mrp_to_dcm(rec["sigma_BN"].astype(np.float64))
    c, s = np.cos(dO), np.sin(dO)
    Rpn = np.zeros((len(t), 3, 3))
    Rpn[:, 0, 0], Rpn[:, 0, 1], Rpn[:, 1, 0], Rpn[:, 1, 1], Rpn[:, 2, 2] = c, s, -s, c, 1.0
    Cbn = Cbp @ Rpn
    sig = np.array([dcm_to_mrp(m) for m in Cbn])
    return dict(t=t, r=r, v=v, sigma=sig, omega=rec["omega_BN"].astype(np.float64), Omega=rec["Omega"].astype(np.float64),
                u=rec["u"].astype(np.float64))


def events(sat):
    e = pd.read_csv(f"{FLEET}/{sat}/truth/events_truth.csv")
    return e[e.Visible].copy()


def status_track(sat, t):
    """Colour index per frame from the truth timeline."""
    keys = list(COL)
    st = np.zeros(len(t), np.int8)

    def paint(a, b, k):
        st[np.searchsorted(t, a):np.searchsorted(t, b)] = np.maximum(st[np.searchsorted(t, a):np.searchsorted(t, b)], keys.index(k))
    m = pd.read_csv(f"{FLEET}/{sat}/truth/aocs_modes.csv")
    ms = (pd.to_datetime(m.timestamp, utc=True) - pd.Timestamp(EPOCH)).dt.total_seconds().values
    for a, b, code in zip(ms, np.r_[ms[1:], ms[-1] + 240], m["mode"].values):
        if code == 1:
            paint(a, a + 240, "slew")
    e = events(sat)
    for _, x in e.iterrows():
        a, b = day_of(x.FirstVisible) * DAY, day_of(x.LastVisible) * DAY
        k = "rare" if x.Category != "Anomaly" else "degrading" if x.Code == "W2" else "failed" if x.Code == "W3" else "anomaly"
        paint(a, max(b, a + 60), k)
    return st


def caught_table():
    p = os.path.join(ROOT, "results", "final", "per_event_detection.csv")
    d = pd.read_csv(p)
    return {(r.satellite, r.id, r.arm, r.settings): bool(r.detected) for r in d.itertuples()}


def captions(sats, t0, t1, commands):
    """-> sorted (time, text). Every labelled event; every telecommand when `commands` is set."""
    got = caught_table()
    out = []
    for s in sats:
        for _, x in events(s).iterrows():
            a = day_of(x.FirstVisible) * DAY
            if x.Category == "Anomaly":
                txt = f"{s} ANOMALY {x.Code}: {FAULT.get(x.Code, x.Code)} on {x.Component}, day {a / DAY:.2f}."
                if x.Split == "test":
                    so, fl = got.get((s, x.ID, "solo", "local")), got.get((s, x.ID, "pers", "federated"))
                    yn = lambda b: "caught" if b else "missed"
                    txt += f" Solo: {yn(so)}. FL + personalization: {yn(fl)}."
            else:
                txt = f"{s} RARE EVENT: {RARE.get(x.Code, x.Code)}, day {a / DAY:.2f}."
            out.append((a, txt))
        if commands:
            tc = pd.read_csv(f"{FLEET}/{s}/telecommands.csv")
            for c, n in zip(tc.Telecommand, tc.Name):
                f = pd.read_csv(f"{FLEET}/{s}/telecommands/{c}.csv")
                ts = (pd.to_datetime(f.timestamp, utc=True) - pd.Timestamp(EPOCH)).dt.total_seconds().values
                out += [(float(a), f"{s} telecommand {n}") for a in ts[(ts >= t0) & (ts < t1)]]
    if "SAT_2" in sats:
        out += [(190 * DAY, "SAT_2 wheel 2: hidden wear starts, day 190. Not yet visible in telemetry.")]
    out += [(d * DAY, f"Mission: {m}") for d, m in MILESTONES]
    return sorted(x for x in out if t0 <= x[0] < t1)


class Replay(sysModel.SysModel):
    """Writes one saved frame per task tick."""

    def __init__(self, data, tracks, t0, step, viz, caps, cap_s):
        super().__init__()
        self.d, self.tr, self.t0, self.step, self.viz, self.caps, self.cap_s = data, tracks, t0, step, viz, caps, cap_s
        self.names = list(data)
        self.sc = {s: messaging.SCStatesMsg() for s in self.names}
        self.rw = {s: [messaging.RWConfigLogMsg() for _ in range(4)] for s in self.names}
        self.col = {s: messaging.ColorMsg() for s in self.names}
        self.earth, self.sun = messaging.SpicePlanetStateMsg(), messaging.SpicePlanetStateMsg()
        b = np.radians(C.SC["rw_beta_deg"])
        self.G = [[np.cos(b) * np.cos(np.radians(a)), np.cos(b) * np.sin(np.radians(a)), np.sin(b)] for a in C.SC["rw_az_deg"]]
        self.i, self.ci, self.last, self.keep = 0, 0, {}, []

    def write_frame(self, now):
        i = self.i
        t = self.t0 + i * self.step
        tu = epoch_unix() + t
        g = float(gmst(tu))
        pe = messaging.SpicePlanetStateMsgPayload()
        pe.PlanetName = "earth"
        pe.PositionVector, pe.VelocityVector = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        c, s = np.cos(g), np.sin(g)
        pe.J20002Pfix = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
        pe.computeOrient = 1
        self.earth.write(pe, now, self.moduleID)
        su, rau = sun_eci(tu)
        ps = messaging.SpicePlanetStateMsgPayload()
        ps.PlanetName = "sun"
        ps.PositionVector = (np.asarray(su) * float(rau) * AU).tolist()
        ps.VelocityVector = [0.0, 0.0, 0.0]
        ps.J20002Pfix = np.eye(3).tolist()
        self.sun.write(ps, now, self.moduleID)
        for n in self.names:
            d = self.d[n]
            p = messaging.SCStatesMsgPayload()
            p.r_BN_N = p.r_CN_N = d["r"][i].tolist()
            p.v_BN_N = p.v_CN_N = d["v"][i].tolist()
            p.sigma_BN, p.omega_BN_B = d["sigma"][i].tolist(), d["omega"][i].tolist()
            self.sc[n].write(p, now, self.moduleID)
            for k in range(4):
                w = messaging.RWConfigLogMsgPayload()
                w.gsHat_B = self.G[k]
                w.rWB_B = [0.6 * x for x in self.G[k]]
                w.Omega, w.u_current = float(d["Omega"][i, k]), float(d["u"][i, k])
                w.Omega_max, w.u_max = 6000 * 2 * np.pi / 60, C.SC["rw_umax"]
                self.rw[n][k].write(w, now, self.moduleID)
            st = int(self.tr[n][i])
            if self.last.get(n) != st:
                q = messaging.ColorMsgPayload()
                q.colorRGBA = list(COL[list(COL)[st]])
                self.col[n].write(q, now, self.moduleID)
                self.last[n] = st

    def UpdateState(self, now):
        if self.i >= len(next(iter(self.d.values()))["t"]):
            return
        self.write_frame(now)
        t = self.t0 + self.i * self.step
        while self.ci < len(self.caps) and self.caps[self.ci][0] <= t + self.step:
            e = vizInterface.VizEventDialog()
            e.eventHandlerID = f"note{self.ci}"
            e.displayString = self.caps[self.ci][1]
            e.durationOfDisplay = int(self.cap_s * 1e9)
            e.useSimElapsedTimeForDuration = True
            e.hideOnSelection = True
            e.useConfirmationPanel = False
            self.keep.append(e)                        # the vector holds a pointer, so keep the object alive
            self.viz.vizEventDialogs.append(e)
            self.ci += 1
        self.i += 1


def build(name, sats, start_day, days, step, commands=False, cap_s=None, large=False):
    t0, t1 = start_day * DAY, (start_day + days) * DAY
    print(f"{name}: loading {sats}", flush=True)
    data = {s: to_shared_frame(s, load_record(s, t0, t1, step)) for s in sats}
    n = min(len(d["t"]) for d in data.values())
    data = {s: {k: v[:n] for k, v in d.items()} for s, d in data.items()}
    t = next(iter(data.values()))["t"]
    assert all(np.array_equal(d["t"], t) for d in data.values()) and np.all(np.diff(t) == step)
    tracks = {s: status_track(s, t) for s in sats}
    caps = captions(sats, t0, t1, commands)

    sim = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("p")
    proc.addTask(sim.CreateNewTask("t", macros.sec2nano(float(step))))
    rep = Replay(data, tracks, t[0], step, None, caps, cap_s or max(20.0, 60.0 * step))
    rep.ModelTag = "replay"
    sim.AddModelToTask("t", rep, 100)

    class W:
        def __init__(self, m):
            self.rwOutMsgs = m

    out = os.path.join(ROOT, "viz", "large") if large else OUT      # large files go in the Release, not git
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name + ".bin")
    class Entry(tuple):                                  # vizSupport reads ModelTag from the first entry
        ModelTag = property(lambda self: self[0])

    viz = vizSupport.enableUnityVisualization(sim, "t", [Entry((s, rep.sc[s])) for s in sats], saveFile=path,
                                              rwEffectorList=[W(rep.rw[s]) for s in sats],
                                              trueOrbitColorInMsgList=[rep.col[s].addSubscriber() for s in sats])
    rep.viz = viz
    for i, s in enumerate(sats):
        vizSupport.setActuatorGuiSetting(viz, spacecraftName=s, viewRWPanel=True, viewRWHUD=False, showRWLabels=True)
    earth, sun = vizInterface.GravBodyInfo(), vizInterface.GravBodyInfo()
    earth.bodyName, earth.mu, earth.radEquator, earth.radiusRatio = "earth", 3.986004418e14, RE, 1.0
    sun.bodyName, sun.mu, sun.radEquator, sun.radiusRatio = "sun", 1.32712440018e20, 6.957e8, 1.0
    viz.gravBodyInformation = vizInterface.GravBodyInfoVector([earth, sun])
    viz.spiceInMsgs = messaging.SpicePlanetStateMsgInMsgsVector([rep.earth.addSubscriber(), rep.sun.addSubscriber()])
    ep = messaging.EpochMsgPayload()
    d0 = EPOCH + timedelta(seconds=float(t[0]))
    ep.year, ep.month, ep.day, ep.hours, ep.minutes, ep.seconds = d0.year, d0.month, d0.day, d0.hour, d0.minute, float(d0.second)
    ep_msg = messaging.EpochMsg().write(ep)
    viz.epochInMsg.subscribeTo(ep_msg)
    viz.settings.showSpacecraftLabels = 1
    viz.settings.trueTrajectoryLinesOn = 1
    viz.settings.orbitLinesOn = -1
    for st, (la, lo) in C.STATIONS.items():
        e = station_ecef(la, lo)
        vizSupport.addLocation(viz, stationName=st, parentBodyName="earth", r_GP_P=e.tolist(), gHat_P=(e / np.linalg.norm(e)).tolist(),
                               fieldOfView=np.radians(160), color="pink", range=3.0e6)
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(float((n - 1) * step)))
    sim.ExecuteSimulation()
    del sim
    meta = dict(name=name, satellites=sats, start_day=start_day, days=days, step_s=step, frames=int(n), captions=len(caps),
                size_mb=round(os.path.getsize(path) / 1e6, 1))
    print(meta, flush=True)
    return meta


Q = [(0, 91), (91, 91), (182, 91), (273, 92)]
CLIPS = dict(
    # whole mission: every satellite, every day
    fleet_year_60s=dict(sats=SATS, start_day=0, days=365, step=60, large=True),
    **{f"fleet_Q{i + 1}_60s": dict(sats=SATS, start_day=a, days=n, step=60, large=True) for i, (a, n) in enumerate(Q)},
    **{f"{s}_year_30s": dict(sats=[s], start_day=0, days=365, step=30, commands=True, cap_s=300.0, large=True) for s in SATS},
    # key moments at the full 5 s record, with every telecommand captioned
    clip1_fleet_day200=dict(sats=SATS, start_day=200, days=0.25, step=5, commands=True),
    clip2_SAT_4_imaging=dict(sats=["SAT_4"], start_day=250, days=0.125, step=5, commands=True),
    clip3_SAT_4_gyro_calibration=dict(sats=["SAT_4"], start_day=225.38, days=0.08, step=5, commands=True),
    clip4_SAT_4_safe_drill=dict(sats=["SAT_4"], start_day=246.57, days=0.2, step=5, commands=True),
    clip5_SAT_1_friction_spike=dict(sats=["SAT_1"], start_day=204.74, days=0.17, step=5, commands=True),
    clip6_SAT_3_wheel_dropout=dict(sats=["SAT_3"], start_day=267.53, days=0.06, step=5, commands=True),
    clip7_SAT_2_wear_spike=dict(sats=["SAT_2"], start_day=286.84, days=0.14, step=5, commands=True),
    clip8_SAT_2_wheel_failure=dict(sats=["SAT_2"], start_day=356.02, days=0.25, step=5, commands=True),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="")
    ap.add_argument("--list", action="store_true", help="print clip names, one per line")
    ap.add_argument("--name", default="custom")
    ap.add_argument("--sats", default=",".join(SATS))
    ap.add_argument("--start-day", type=float, default=0.0)
    ap.add_argument("--days", type=float, default=1.0)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--commands", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(CLIPS))
    elif a.clip:
        build(a.clip, **CLIPS[a.clip])
    else:
        build(a.name, a.sats.split(","), a.start_day, a.days, a.step, a.commands)


if __name__ == "__main__":
    main()
