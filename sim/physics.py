"""Basilisk physics run for one satellite: orbit, attitude, 4 HR16 wheels, flight software,
disturbance torques, magnetic momentum dumps, and physical faults / wear applied while it runs.

Flight software (Basilisk modules, standard chain from the Basilisk examples):
  hillPoint (nadir frame) -> eulerRotation (pointing offsets) -+
  inertial3D (inertial hold / sun-safe) ----------------------+-> attTrackingError -> mrpFeedback
  -> rwMotorTorque -> rwNullSpace (wheel speed bias +-1200 rpm) -> reactionWheelStateEffector
Dynamics: spacecraft hub, point-mass Earth gravity, GravityGradientEffector, extForceTorque
(aerodynamic + solar pressure + residual-dipole torques, magnetorquer torque m x B).

Physical faults change model parameters, never outputs: wheel Coulomb/viscous friction (A1, wear),
wheel motor torque limit (A2, A3, failure), magnetorquer axis health (A4).

Twin mode: same timeline without physics faults; autonomous dumps replayed from the faulty run.
"""
import bisect
import json
import os
import time

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import (attTrackingError, eulerRotation, hillPoint, inertial3D, mrpFeedback,
                                    rwMotorTorque, rwNullSpace)
from Basilisk.simulation import (GravityGradientEffector, extForceTorque, reactionWheelStateEffector, simpleNav,
                                 spacecraft)
from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, simIncludeGravBody, simIncludeRW

import config as C
from envmodels import MU, P_SUN, b_field_P, dcm_to_mrp, density, mrp_to_dcm, shadow

PHYS_FAULT_KINDS = ("fric", "fric_end", "umax", "umax_end", "mtq_fail", "mtq_fail_end", "wear", "fail")
H_AUTO = 3.0          # [Nms] autonomous momentum dump trigger (on-board FDIR threshold)
CHUNK = 80 * 86400.0  # [s] Basilisk piece length (< 2^53 ns)
H_PERSIST = 5         # ... sustained for 5 consecutive ticks (5 min): ignores slew transients
RPM = 2 * np.pi / 60


def unit_variation(seed):
    """Small unit-to-unit differences (same design, different units) - fixed per satellite."""
    rng = np.random.default_rng(seed + 7919)
    I = np.array(C.SC["inertia"]).reshape(3, 3)
    I = I * np.diag(rng.uniform(0.98, 1.02, 3)) + (I - np.diag(np.diag(I)))   # +-2 % principal
    return dict(
        inertia=I.flatten().tolist(),
        fc=(C.SC["rw_fc"] * rng.uniform(0.85, 1.15, 4)).tolist(),        # +-15 % bearing friction
        cv=(C.SC["rw_cv"] * rng.uniform(0.85, 1.15, 4)).tolist(),
        res_dipole=(np.array(C.SC["res_dipole"]) * rng.uniform(0.7, 1.3, 3)).tolist(),
        cp=(np.array(C.SC["cp_offset"]) * rng.uniform(0.8, 1.2, 3)).tolist(),
    )


class SatSim:
    def __init__(self, tl, viz_path=None, viz_dt=10.0, live=False):
        self.tl = tl
        self.var = unit_variation(tl.seed)
        self.orbit = tl.orbit
        self.t0u = tl.t0u
        self.viz_args = (viz_path, viz_dt, live)
        self.t_off = 0.0
        self.cur_angles, self.cur_i3 = (0.0, 0.0, 0.0), [0.0, 0.0, 0.0]
        self.cur_rates = (0.0, 0.0, 0.0)          # [deg/s] guidance rate feed-forward (profiled slews)
        self._build(None)

    def _build(self, init):
        """(Re)build the Basilisk simulation. Basilisk keeps time in uint64 ns and converts with a
        2^53 ns (~104 day) safety limit, so a year is run in 80-day pieces: each piece starts from
        the exact end state of the previous one (orbit, attitude, rates, wheel speeds) and all
        fault / wear / mode settings are re-applied (see run())."""
        viz_path, viz_dt, live = self.viz_args if init is None else (None, 10.0, False)
        tl = self.tl
        dyn = macros.sec2nano(C.DT_DYN)
        sim = self.sim = SimulationBaseClass.SimBaseClass()
        proc = sim.CreateNewProcess("p")
        proc.addTask(sim.CreateNewTask("dyn", dyn))

        sc = self.sc = spacecraft.Spacecraft()
        sc.ModelTag = tl.name
        sc.hub.mHub = C.SC["mass"]
        sc.hub.IHubPntBc_B = np.array(self.var["inertia"]).reshape(3, 3).tolist()
        sim.AddModelToTask("dyn", sc)
        grav = simIncludeGravBody.gravBodyFactory()
        earth = grav.createEarth()
        earth.isCentralBody = True
        grav.addBodiesTo(sc)
        # initial orbit (frame P) consistent with envmodels.Orbit.r_P(t0)
        oe = orbitalMotion.ClassicElements()
        oe.a, oe.e, oe.i = self.orbit.a, 0.0, self.orbit.inc
        oe.Omega, oe.omega, oe.f = self.orbit.raan0, 0.0, self.orbit.argl0
        r0, v0 = orbitalMotion.elem2rv(MU, oe)
        if init is None:
            sc.hub.r_CN_NInit, sc.hub.v_CN_NInit = r0, v0
            r0, v0 = np.array(r0).flatten(), np.array(v0).flatten()
            rh = r0 / np.linalg.norm(r0)
            hh = np.cross(r0, v0); hh /= np.linalg.norm(hh)
            th = np.cross(hh, rh)
            sc.hub.sigma_BNInit = [[x] for x in dcm_to_mrp(np.array([rh, th, hh]))]
            sc.hub.omega_BN_BInit = [[0.0], [0.0], [self.orbit.n]]
        else:
            sc.hub.r_CN_NInit = [[x] for x in init["r"]]
            sc.hub.v_CN_NInit = [[x] for x in init["v"]]
            sc.hub.sigma_BNInit = [[x] for x in init["sigma"]]
            sc.hub.omega_BN_BInit = [[x] for x in init["omega"]]

        gg = GravityGradientEffector.GravityGradientEffector()
        gg.ModelTag = "gg"
        gg.addPlanetName(earth.planetName)
        sc.addDynamicEffector(gg)
        sim.AddModelToTask("dyn", gg)
        ext = self.ext = extForceTorque.ExtForceTorque()
        ext.ModelTag = "ext"
        sc.addDynamicEffector(ext)
        sim.AddModelToTask("dyn", ext)

        # reaction wheels: Honeywell HR16 (Basilisk preset), pyramid
        b = np.radians(C.SC["rw_beta_deg"])
        self.G = np.array([[np.cos(b) * np.cos(np.radians(a)), np.cos(b) * np.sin(np.radians(a)), np.sin(b)]
                           for a in C.SC["rw_az_deg"]]).T          # 3 x 4
        bias = C.SC["rw_bias_rpm"] * np.array([1, -1, 1, -1])
        rwF = simIncludeRW.rwFactory()
        for k in range(4):
            om0 = float(bias[k]) if init is None else float(init["Omega"][k] / RPM)
            rwF.create("Honeywell_HR16", self.G[:, k].tolist(), maxMomentum=C.SC["rw_maxH"], Omega=om0,
                       label=f"RW{k + 1}")
        rwE = self.rwE = reactionWheelStateEffector.ReactionWheelStateEffector()
        rwE.ModelTag = "RW"
        rwF.addToSpacecraft(sc.ModelTag, rwE, sc)
        sim.AddModelToTask("dyn", rwE, 2)
        self.Js = C.SC["rw_maxH"] / (6000 * RPM)
        self.base_fc, self.base_cv = list(self.var["fc"]), list(self.var["cv"])
        for k in range(4):
            rwE.ReactionWheelData[k].fCoulomb = self.base_fc[k]
            rwE.ReactionWheelData[k].cViscous = self.base_cv[k]

        nav = simpleNav.SimpleNav()
        nav.scStateInMsg.subscribeTo(sc.scStateOutMsg)
        sim.AddModelToTask("dyn", nav)
        hp = hillPoint.hillPoint()
        hp.transNavInMsg.subscribeTo(nav.transOutMsg)
        sim.AddModelToTask("dyn", hp)
        eul = self.eul = eulerRotation.eulerRotation()
        eul.attRefInMsg.subscribeTo(hp.attRefOutMsg)
        eul.angleSet = [0.0, 0.0, 0.0]
        sim.AddModelToTask("dyn", eul)
        i3 = self.i3 = inertial3D.inertial3D()
        i3.sigma_R0N = [0.0, 0.0, 0.0]
        sim.AddModelToTask("dyn", i3)
        err = self.err = attTrackingError.attTrackingError()
        err.attNavInMsg.subscribeTo(nav.attOutMsg)
        err.attRefInMsg.subscribeTo(eul.attRefOutMsg)
        self.ref = "eul"
        sim.AddModelToTask("dyn", err)
        vc = messaging.VehicleConfigMsgPayload()
        vc.ISCPntB_B = C.SC["inertia"]                      # flight software uses the design inertia
        self.vcMsg = messaging.VehicleConfigMsg().write(vc)
        self.rwParams = rwF.getConfigMessage()
        ctl = mrpFeedback.mrpFeedback()
        ctl.guidInMsg.subscribeTo(err.attGuidOutMsg)
        ctl.vehConfigInMsg.subscribeTo(self.vcMsg)
        ctl.rwParamsInMsg.subscribeTo(self.rwParams)
        ctl.rwSpeedsInMsg.subscribeTo(rwE.rwSpeedOutMsg)
        ctl.K, ctl.Ki, ctl.P = C.SC["K"], -1.0, C.SC["P"]
        sim.AddModelToTask("dyn", ctl)
        mt = rwMotorTorque.rwMotorTorque()
        mt.controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        mt.vehControlInMsg.subscribeTo(ctl.cmdTorqueOutMsg)
        mt.rwParamsInMsg.subscribeTo(self.rwParams)
        av = messaging.RWAvailabilityMsgPayload()
        av.wheelAvailability = [messaging.AVAILABLE] * 4
        self.availMsg = messaging.RWAvailabilityMsg().write(av)
        mt.rwAvailInMsg.subscribeTo(self.availMsg)
        ctl.rwAvailInMsg.subscribeTo(self.availMsg)
        sim.AddModelToTask("dyn", mt)
        self.ns_ref = None
        ns = rwNullSpace.rwNullSpace()
        ns.OmegaGain = C.SC["null_gain"]
        ns.rwMotorTorqueInMsg.subscribeTo(mt.rwMotorTorqueOutMsg)
        ns.rwSpeedsInMsg.subscribeTo(rwE.rwSpeedOutMsg)
        cp = messaging.RWConstellationMsgPayload()
        cp.numRW = 4
        els = []
        for k in range(4):
            e = messaging.RWConfigElementMsgPayload()
            e.gsHat_B, e.Js, e.uMax = self.G[:, k].tolist(), self.Js, C.SC["rw_umax"]
            els.append(e)
        cp.reactionWheels = els
        self.rwConst = messaging.RWConstellationMsg().write(cp)
        ns.rwConfigInMsg.subscribeTo(self.rwConst)
        des = messaging.RWSpeedMsgPayload()
        des.wheelSpeeds = (bias * RPM).tolist()
        self.desMsg = messaging.RWSpeedMsg().write(des)
        ns.rwDesiredSpeedsInMsg.subscribeTo(self.desMsg)
        sim.AddModelToTask("dyn", ns)
        rwE.rwMotorCmdInMsg.subscribeTo(ns.rwMotorTorqueOutMsg)
        self.ns = ns
        self.mods = [nav, hp, ctl, mt, ns, grav, earth, gg]   # keep references alive

        rd = macros.sec2nano(C.DT_REC)
        self.rec_st = sc.scStateOutMsg.recorder(rd)
        self.rec_ws = rwE.rwSpeedOutMsg.recorder(rd)
        self.rec_ag = err.attGuidOutMsg.recorder(rd)
        self.rec_rw = [m.recorder(rd) for m in rwE.rwOutMsgs]
        for rc in [self.rec_st, self.rec_ws, self.rec_ag] + self.rec_rw:
            sim.AddModelToTask("dyn", rc)

        if viz_path or live:
            from Basilisk.utilities import vizSupport
            proc.addTask(sim.CreateNewTask("viz", macros.sec2nano(viz_dt)))
            self.viz = vizSupport.enableUnityVisualization(sim, "viz", sc, rwEffectorList=rwE,
                                                           saveFile=viz_path, liveStream=live)
        sim.InitializeSimulation()

    # --------------------------------------------------------------- helpers
    def set_ref(self, which):
        self.ref_want = which
        if which != self.ref:
            self.err.attRefInMsg.subscribeTo(self.eul.attRefOutMsg if which == "eul" else self.i3.attRefOutMsg)
            self.ref = which

    def apply_friction(self, k):
        if self.locked[k]:
            fc, cv = 0.0, 0.05
        else:
            m = (1 + self.wear[k]) * np.prod(list(self.mults[k].values()) or [1.0])
            fc, cv = self.base_fc[k] * m, self.base_cv[k] * m
        self.rwE.ReactionWheelData[k].fCoulomb = float(fc)
        self.rwE.ReactionWheelData[k].cViscous = float(cv)

    def apply_umax(self, k):
        f = 0.0 if self.failed[k] else self.umax_frac[k]
        self.rwE.ReactionWheelData[k].u_max = float(max(C.SC["rw_umax"] * f, 1e-6))

    def reapply(self):
        for k in range(4):
            self.apply_friction(k); self.apply_umax(k)
        if any(self.failed):
            av = messaging.RWAvailabilityMsgPayload()
            av.wheelAvailability = [messaging.UNAVAILABLE if f else messaging.AVAILABLE for f in self.failed]
            self.availMsg.write(av, 0)
            self.ns.OmegaGain = 0.0
        self.eul.angleSet = np.radians(self.cur_angles).tolist()
        self.eul.angleRates = np.radians(self.cur_rates).tolist()
        self.i3.sigma_R0N = list(self.cur_i3)
        want, self.ref = self.ref_want, "eul"
        self.set_ref(want)
        tau = getattr(self, "last_tau", np.zeros(3))
        self.ext.extTorquePntB_B = [[tau[0]], [tau[1]], [tau[2]]]

    def full_state(self):
        st = self.sc.scStateOutMsg.read()
        return dict(r=list(st.r_BN_N), v=list(st.v_BN_N), sigma=list(st.sigma_BN), omega=list(st.omega_BN_B),
                    Omega=np.array(self.rwE.rwSpeedOutMsg.read().wheelSpeeds[:4], float))

    def state(self):
        st = self.sc.scStateOutMsg.read()
        return (np.array(st.r_BN_N, float), np.array(st.v_BN_N, float), np.array(st.sigma_BN, float),
                np.array(self.rwE.rwSpeedOutMsg.read().wheelSpeeds[:4], float))

    def sunsafe_sigma(self, t):
        s, _ = self.orbit.sun_P(self.t0u + t)
        b1 = s / np.linalg.norm(s)
        z = np.array([0.0, 0.0, 1.0])
        b3 = np.cross(b1, z)
        b3 = b3 / np.linalg.norm(b3) if np.linalg.norm(b3) > 1e-6 else np.array([0.0, 1.0, 0.0])
        b2 = np.cross(b3, b1)
        return dcm_to_mrp(np.array([b1, b2, b3]))

    # ----------------------------------------------------------------- run
    def run(self, out_dir, twin=False, replay=None, t_end=None, log=print, follow=None):
        os.makedirs(out_dir, exist_ok=True)
        tl = self.tl
        T = float(t_end or tl.T)
        ev = [e for e in tl.phys if e[0] < T and not (twin and e[1] in PHYS_FAULT_KINDS)]
        if twin and replay:
            ev += [(a, "dump", dict(dur=b - a, auto=True)) for a, b in replay]
        ev.sort(key=lambda e: e[0])
        known = 0

        def sync(t):
            """Twin in follow mode: run behind the faulty run instead of after it. It may enter a
            day only once the faulty run is 2 days ahead, so every autonomous dump starting in that
            day (<= 1 h long) is already recorded. Same events, same order as a replay."""
            nonlocal known
            need, last, seen = min(t + 2 * 86400.0, T), -1.0, time.time()
            while True:
                if os.path.exists(f"{follow}/physlog.json"):
                    c = dict(t=T, auto=json.load(open(f"{follow}/physlog.json"))["auto_dumps"])
                else:
                    try:
                        c = json.load(open(f"{follow}/clock.json"))
                    except (OSError, ValueError):
                        c = dict(t=0.0, auto=[])
                if c["t"] >= need:
                    break
                if c["t"] != last:
                    last, seen = c["t"], time.time()
                elif time.time() - seen > 1800:
                    raise RuntimeError(f"faulty run stalled at t={last}")
                time.sleep(1.0)
            for a, b in c["auto"][known:]:
                assert a > t, (a, t)
                ev.insert(bisect.bisect_right(ev, a, key=lambda e: e[0]), (a, "dump", dict(dur=b - a, auto=True)))
            known = len(c["auto"])

        if follow:
            sync(0.0)
        ticks = np.arange(C.DT_TICK, T + 1e-6, C.DT_TICK)
        days = np.arange(86400.0, T + 1e-6, 86400.0)
        chunks = np.arange(CHUNK, T - 1e-6, CHUNK)
        stops = np.unique(np.concatenate([ticks, np.array([e[0] for e in ev] or [T]), days, chunks, [T]]))
        chunk_set = set(np.round(chunks, 3))
        self.ref_want = "eul"
        stops = stops[(stops > 0) & (stops <= T)]
        tick_set = set(np.round(ticks, 3))
        day_set = set(np.round(days, 3))

        self.wear = [0.0] * 4
        self.mults = [dict() for _ in range(4)]
        self.umax_frac = [1.0] * 4
        self.failed, self.locked = [False] * 4, [False] * 4
        self.mtq_health = np.ones(3)
        dump_until, dump_auto = -1.0, False
        hi_count = 0
        auto_dumps, cur_auto = [], None
        mode_type = "nadir"
        fail_lock_t = None
        tick_log = []
        buf = {k: [] for k in ("t", "sigma_BN", "omega_BN", "r_BN", "Omega", "sigma_BR", "u", "fric")}
        part = 0
        ei = 0
        res_dip = np.array(self.var["res_dipole"])
        cp = np.array(self.var["cp"])
        wall0 = time.time()

        def flush(final=False):
            nonlocal part
            st, ws, ag = self.rec_st, self.rec_ws, self.rec_ag
            n = st.size()
            if n == 0 and not (final and buf["t"]):
                return
            if n == 0:
                np.savez(f"{out_dir}/truth_{part:03d}.npz", **{k: np.concatenate(v) for k, v in buf.items()})
                for v in buf.values():
                    v.clear()
                part += 1
                return
            buf["t"].append(st.times() * 1e-9 + self.t_off)
            buf["sigma_BN"].append(np.asarray(st.sigma_BN, np.float32))
            buf["omega_BN"].append(np.asarray(st.omega_BN_B, np.float32))
            buf["r_BN"].append(np.asarray(st.r_BN_N, np.float32))
            buf["Omega"].append(np.asarray(ws.wheelSpeeds[:, :4], np.float32))
            buf["sigma_BR"].append(np.asarray(ag.sigma_BR, np.float32))
            buf["u"].append(np.stack([np.asarray(r.u_current, np.float32) for r in self.rec_rw], 1))
            buf["fric"].append(np.stack([np.asarray(r.frictionTorque, np.float32) for r in self.rec_rw], 1))
            for rc in [st, ws, ag] + self.rec_rw:
                rc.clear()
            if final or sum(len(x) for x in buf["t"]) >= 30 * 17280:
                np.savez(f"{out_dir}/truth_{part:03d}.npz", **{k: np.concatenate(v) for k, v in buf.items()})
                for v in buf.values():
                    v.clear()
                part += 1

        for t in stops:
            self.sim.ConfigureStopTime(macros.sec2nano(float(t - self.t_off)))
            self.sim.ExecuteSimulation()
            # ---- events due at t
            while ei < len(ev) and ev[ei][0] <= t + 1e-6:
                _, kind, d = ev[ei]
                ei += 1
                if kind == "mode":
                    mode_type = d["type"]
                    if d["type"] in ("nadir", "offset"):
                        self.cur_angles = tuple(d.get("angles", (0.0, 0.0, 0.0)))
                        self.cur_rates = tuple(d.get("rates", (0.0, 0.0, 0.0)))
                        self.eul.angleSet = np.radians(self.cur_angles).tolist()
                        self.eul.angleRates = np.radians(self.cur_rates).tolist()
                        self.set_ref("eul")
                    elif d["type"] == "slew_pt":      # profiled slew: guidance angle + rate update
                        self.cur_angles, self.cur_rates = tuple(d["angles"]), tuple(d["rates"])
                        self.eul.angleSet = np.radians(self.cur_angles).tolist()
                        self.eul.angleRates = np.radians(self.cur_rates).tolist()
                    elif d["type"] == "inertial_hold":
                        self.cur_i3 = self.state()[2].tolist()
                        self.i3.sigma_R0N = self.cur_i3
                        self.set_ref("i3")
                    elif d["type"] == "sunsafe":
                        self.cur_i3 = self.sunsafe_sigma(t).tolist()
                        self.i3.sigma_R0N = self.cur_i3
                        self.set_ref("i3")
                elif kind == "dump":
                    dump_until, dump_auto = t + d["dur"], bool(d.get("auto"))
                elif kind == "fric":
                    self.mults[d["wheel"]][d["key"]] = d["mult"]; self.apply_friction(d["wheel"])
                elif kind == "fric_end":
                    self.mults[d["wheel"]].pop(d["key"], None); self.apply_friction(d["wheel"])
                elif kind == "wear":
                    self.wear[d["wheel"]] = d["w"]; self.apply_friction(d["wheel"])
                elif kind == "umax":
                    self.umax_frac[d["wheel"]] = d["frac"]; self.apply_umax(d["wheel"])
                elif kind == "umax_end":
                    self.umax_frac[d["wheel"]] = 1.0; self.apply_umax(d["wheel"])
                elif kind == "mtq_fail":
                    self.mtq_health[d["axis"]] = 0.0
                elif kind == "mtq_fail_end":
                    self.mtq_health[d["axis"]] = 1.0
                elif kind == "fail":
                    # drive overcurrent trip; on-board FDIR (built-in, not operator recovery) flags the
                    # wheel unavailable: 3-wheel allocation, null-space speed management off
                    self.failed[d["wheel"]] = True; self.apply_umax(d["wheel"])
                    av = messaging.RWAvailabilityMsgPayload()
                    av.wheelAvailability = [messaging.UNAVAILABLE if self.failed[k] else messaging.AVAILABLE
                                            for k in range(4)]
                    self.availMsg.write(av, macros.sec2nano(float(t - self.t_off)))
                    self.ns.OmegaGain = 0.0
            # ---- environment / disturbance / magnetorquer update every tick
            if round(t, 3) in tick_set:
                r, v, sig, ws = self.state()
                tu = self.t0u + t
                Cbn = mrp_to_dcm(sig)
                s_P, rau = self.orbit.sun_P(tu)
                nu = float(shadow(r[None], s_P[None], np.array([rau]))[0])
                B_B = Cbn @ b_field_P(self.orbit, tu, r) * 1e-9
                s_B, v_B = Cbn @ s_P, Cbn @ v
                vn = np.linalg.norm(v_B)
                sp = np.hypot(s_B[0], s_B[1])
                nw = np.array([s_B[0] / sp, s_B[1] / sp, 0.0]) if sp > 1e-6 else np.array([1.0, 0, 0])
                A_d = C.SC["area_body"] + C.SC["area_wing"] * abs(nw @ v_B) / vn
                F = -0.5 * density(r, s_P) * C.SC["cd"] * A_d * vn * v_B
                A_s = 0.5 * C.SC["area_body"] + C.SC["area_wing"] * sp
                F = F - P_SUN / rau ** 2 * A_s * 1.3 * nu * s_B
                tau = np.cross(cp, F) + np.cross(res_dip, B_B)
                Hw = self.Js * (self.G @ ws)
                m_cmd = np.zeros(3)
                if t < dump_until:
                    b2 = B_B @ B_B
                    m_cmd = np.clip(C.SC["mtq_gain"] * np.cross(Hw, B_B) / b2, -C.SC["mtq_max"], C.SC["mtq_max"])
                    tau = tau + np.cross(m_cmd * self.mtq_health, B_B)
                else:
                    if cur_auto is not None:
                        auto_dumps.append((cur_auto, float(dump_until))); cur_auto = None
                    if not twin and mode_type in ("nadir", "offset") and np.linalg.norm(Hw) > H_AUTO \
                            and np.linalg.norm(self.err.attGuidOutMsg.read().sigma_BR) < 0.01:
                        hi_count += 1
                    else:
                        hi_count = 0
                    if hi_count >= H_PERSIST:
                        dump_until, dump_auto, cur_auto, hi_count = t + 1800.0, True, float(t), 0
                self.ext.extTorquePntB_B = [[tau[0]], [tau[1]], [tau[2]]]
                self.last_tau = tau
                for k in range(4):
                    if self.failed[k] and not self.locked[k] and abs(ws[k]) < 5 * RPM:
                        self.locked[k] = True; self.apply_friction(k); fail_lock_t = float(t)
                tick_log.append((t, *m_cmd, *(m_cmd * self.mtq_health), nu, np.linalg.norm(Hw),
                                 float(t < dump_until), float(dump_auto and t < dump_until)))
            if round(t, 3) in day_set or round(t, 3) in chunk_set:
                flush()
            if round(t, 3) in day_set:
                if follow:
                    sync(float(t))
                elif not twin:                  # publish progress + dumps for a following twin
                    with open(f"{out_dir}/clock.tmp", "w") as fh:
                        json.dump(dict(t=float(t), auto=auto_dumps), fh)
                    os.replace(f"{out_dir}/clock.tmp", f"{out_dir}/clock.json")
            if round(t, 3) in chunk_set:                # next 80-day piece from the exact end state
                init = self.full_state()
                self.t_off = float(t)
                self._build(init)
                self.reapply()
                d = int(round(t / 86400))
                if d % 10 == 0:
                    el = time.time() - wall0
                    log(f"{tl.name}{' twin' if twin else ''}: day {d}/{int(T / 86400)}  "
                        f"{el / 60:.1f} min elapsed, ~{el / d * (T / 86400 - d) / 60:.1f} min left")
                    json.dump(dict(day=d, of=int(T / 86400), wall_min=el / 60), open(f"{out_dir}/progress.json", "w"))
        flush(final=True)
        if cur_auto is not None:
            auto_dumps.append((cur_auto, float(dump_until)))
        tick = np.array(tick_log, np.float64)
        np.save(f"{out_dir}/ticks.npy", tick.astype(np.float32))
        meta = dict(name=tl.name, twin=twin, T=T, auto_dumps=auto_dumps, fail_lock_t=fail_lock_t,
                    var=self.var, wall_min=(time.time() - wall0) / 60, tick_cols=[
                        "t", "mcmd_x", "mcmd_y", "mcmd_z", "meff_x", "meff_y", "meff_z", "shadow", "Hw", "dumping", "auto"])
        json.dump(meta, open(f"{out_dir}/physlog.json", "w"), indent=1)
        return meta


def load_truth(out_dir):
    parts = sorted(f for f in os.listdir(out_dir) if f.startswith("truth_"))
    zs = [np.load(f"{out_dir}/{p}") for p in parts]
    d = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}
    _, keep = np.unique(np.round(d["t"], 3), return_index=True)   # piece boundaries repeat one sample
    return {k: v[keep] for k, v in d.items()}
