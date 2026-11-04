"""Vizard clips (saved .bin files, open any time in Vizard: Basilisk Message File -> Select).

  python viz_clips.py              # all clips -> sim/_VizFiles/
  python viz_clips.py --live normal   # stream one clip live to Vizard (Vizard: Live Streaming, tcp://localhost:5556)

Clips use the same simulator and settings as the dataset; each replays a short window of a
satellite's operations (shifted to start at t = 0) with the faults of interest:
  01_SAT2_normal_ops      imaging slews, payload windows, eclipse, magnetic momentum dump (4 h)
  02_SAT4_wheel_faults    friction spike on RW1, drive dropout on RW3, torque-limited slews (4 h)
  03_SAT5_wheel_failure   worn bearing: friction spikes, overcurrent trip, 3-wheel control (4 h)
  04_fleet                all 5 satellites in their orbits (2 orbits)
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402
from operations import Timeline  # noqa: E402
from physics import SatSim, PHYS_FAULT_KINDS  # noqa: E402

VIZ = os.path.join(HERE, "_VizFiles")
H = 3600.0


def window(name, seed, t_from, dur, extra=(), drop_faults=True):
    tl = Timeline(name, C.FLEET[name], seed).build()
    ev = [(t - t_from, k, d) for t, k, d in tl.phys if t_from <= t < t_from + dur
          and not (drop_faults and k in PHYS_FAULT_KINDS)]
    tl.phys = sorted(ev + list(extra), key=lambda e: e[0])
    tl.T = dur
    return tl


def busy_start(tl, n_slews=3, dur=4 * H):
    """First daytime window of `dur` that holds at least n slews and a momentum dump."""
    sl = np.array([t for t, k, d in tl.phys if k == "mode" and d["type"] == "offset"])
    dp = np.array([t for t, k, d in tl.phys if k == "dump"])
    for t0 in np.arange(86400.0, 20 * 86400.0, 1800.0):
        if ((sl >= t0 + 600) & (sl < t0 + dur - 600)).sum() >= n_slews and ((dp >= t0) & (dp < t0 + dur - 1800)).any():
            return t0
    return 86400.0


def clip(tag, tl, live=False):
    path = os.path.join(HERE, tag)
    sim = SatSim(tl, viz_path=None if live else path, viz_dt=5.0, live=live)
    try:
        from Basilisk.utilities import vizSupport
        vizSupport.setActuatorGuiSetting(sim.viz, viewRWPanel=True, viewRWHUD=True)
        sim.viz.settings.orbitLinesOn = 1
        sim.viz.settings.showSpacecraftLabels = 1
    except Exception as e:
        print("viz settings:", e)
    sim.run(os.path.join(VIZ, "_tmp_" + tag), log=lambda s: None)
    print("wrote", tag)


def fleet(live=False):
    from Basilisk.simulation import spacecraft
    from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, simIncludeGravBody, vizSupport
    from envmodels import MU, dcm_to_mrp
    sim = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("p")
    proc.addTask(sim.CreateNewTask("t", macros.sec2nano(10.0)))
    grav = simIncludeGravBody.gravBodyFactory()
    earth = grav.createEarth()
    earth.isCentralBody = True
    scs = []
    for i, (n, s) in enumerate(C.FLEET.items()):
        from envmodels import Orbit
        from operations import epoch_unix
        o = Orbit(s["orbit"], epoch_unix())
        sc = spacecraft.Spacecraft()
        sc.ModelTag = n
        sc.hub.mHub = C.SC["mass"]
        sc.hub.IHubPntBc_B = np.array(C.SC["inertia"]).reshape(3, 3).tolist()
        grav.addBodiesTo(sc)
        oe = orbitalMotion.ClassicElements()
        oe.a, oe.e, oe.i, oe.Omega, oe.omega, oe.f = o.a, 0.0, o.inc, o.raan0, 0.0, o.argl0
        r, v = orbitalMotion.elem2rv(MU, oe)
        sc.hub.r_CN_NInit, sc.hub.v_CN_NInit = r, v
        r, v = np.array(r).flatten(), np.array(v).flatten()
        rh = r / np.linalg.norm(r); hh = np.cross(r, v); hh /= np.linalg.norm(hh)
        sc.hub.sigma_BNInit = [[x] for x in dcm_to_mrp(np.array([rh, np.cross(hh, rh), hh]))]
        sc.hub.omega_BN_BInit = [[0.0], [0.0], [o.n]]
        sim.AddModelToTask("t", sc)
        scs.append(sc)
    viz = vizSupport.enableUnityVisualization(sim, "t", scs, saveFile=None if live else os.path.join(HERE, "04_fleet"),
                                              liveStream=live)
    viz.settings.orbitLinesOn = 1
    viz.settings.showSpacecraftLabels = 1
    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(2 * 5740.0))
    sim.ExecuteSimulation()
    print("wrote 04_fleet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="")
    a = ap.parse_args()
    os.makedirs(VIZ, exist_ok=True)
    os.chdir(HERE)
    clips = {}
    # 01: normal imaging operations
    tl2 = Timeline("SAT_2", C.FLEET["SAT_2"], 11).build()
    t0 = busy_start(tl2)
    clips["normal"] = ("01_SAT2_normal_ops", lambda: window("SAT_2", 11, t0, 4 * H))
    # 02: wheel faults on an imaging satellite
    tl4 = Timeline("SAT_4", C.FLEET["SAT_4"], 13).build()
    t4 = busy_start(tl4)
    ex4 = [(0.5 * H, "fric", dict(wheel=0, mult=10.0, key="a")), (2.0 * H, "fric_end", dict(wheel=0, key="a")),
           (2.4 * H, "umax", dict(wheel=2, frac=0.0)), (2.4 * H + 240, "umax_end", dict(wheel=2))]
    clips["faults"] = ("02_SAT4_wheel_faults", lambda: window("SAT_4", 13, t4, 4 * H, ex4))
    # 03: worn wheel fails (friction spikes, trip, 3-wheel control); imaging-like slews kept
    tl5 = Timeline("SAT_5", C.FLEET["SAT_5"], 17).build()
    t5 = busy_start(tl5, n_slews=0)
    ex5 = [(0.0, "wear", dict(wheel=1, w=40.0)),
           (0.3 * H, "fric", dict(wheel=1, mult=1.6, key="s")), (0.8 * H, "fric_end", dict(wheel=1, key="s")),
           (1.2 * H, "wear", dict(wheel=1, w=79.0)), (1.3 * H, "fail", dict(wheel=1)),
           (0.6 * H, "mode", dict(type="offset", angles=(0.0, 10.0, 25.0))), (0.75 * H, "mode", dict(type="nadir")),
           (2.6 * H, "mode", dict(type="offset", angles=(0.0, 10.0, 25.0))), (2.75 * H, "mode", dict(type="nadir"))]
    clips["failure"] = ("03_SAT5_wheel_failure", lambda: window("SAT_5", 17, t5, 4 * H, ex5))
    if a.live:
        if a.live == "fleet":
            fleet(live=True)
        else:
            tag, mk = clips[a.live]
            clip(tag, mk(), live=True)
        return
    for tag, mk in clips.values():
        clip(tag, mk())
    fleet()
    import shutil
    for d in os.listdir(VIZ):
        if d.startswith("_tmp_"):
            shutil.rmtree(os.path.join(VIZ, d), ignore_errors=True)


if __name__ == "__main__":
    main()
