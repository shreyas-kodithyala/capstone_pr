"""Telemetry layer: Basilisk truth -> 34 sensor channels at their native, irregular rates.

  AOCS     wheel speed / current / temperature / power, gyro, attitude-error estimate,
           magnetometer, magnetorquer dipole, star tracker valid, AOCS mode
  EPS      solar array power, battery SoC / voltage, bus load, eclipse flag
  THERMAL  solar wing and battery temperature

Models (equations in specs/dataset_spec.md):
  wheel current   i = u / k_t                     (motor torque incl. friction compensation, from Basilisk)
  wheel power     P = |u Omega| / eta + P_base    (Basilisk ReactionWheelPower equation)
  wheel thermal   C dT/dt = Q - (T - T_struct)/R, Q = |u Omega|(1-eta)/eta + |f Omega| + 1 W
  solar array     P = S eff(T) (A_wing cos(theta_wing) n_ok/n + A_body |s_z|) nu   (1-axis tracking wing)
  battery         Li-ion 8S: SoC integration, V = 8 OCV(SoC) + I R, charge limit + CV taper, heater thermostat
  sensors         noise, quantisation, gyro bias random walk, star-tracker outages (attitude estimate
                  propagated on gyros), magnetorquer interference on the magnetometer
Sampling: native rate per channel, random phase, +-0.3 s jitter, 0.2 % dropped samples.
Noise is drawn per channel from the satellite seed, so faulty / clean / twin variants share it and
their differences isolate fault effects exactly (used for labelling).
"""
import numpy as np

import config as C
from envmodels import b_field_P, mrp_to_dcm, shadow

DEG = 180 / np.pi
RPM = 60 / (2 * np.pi)
CH = {c[0]: dict(desc=c[1], unit=c[2], subsystem=c[3], group=c[4], target=c[5], rate=c[6], sigma=c[7], q=c[8])
      for c in C.CHANNELS}


def interp(t, tp, fp):
    """Linear interpolation of (n,) or (n,k) arrays."""
    if fp.ndim == 1:
        return np.interp(t, tp, fp)
    return np.stack([np.interp(t, tp, fp[:, i]) for i in range(fp.shape[1])], 1)


def hold(t, tp, fp):
    """Zero-order hold (sample-and-hold) of fp at times tp."""
    i = np.clip(np.searchsorted(tp, t, "right") - 1, 0, len(tp) - 1)
    return fp[i]


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.where(d == 1)[0], np.where(d == -1)[0]


class Telemetry:
    def __init__(self, tl, seed):
        self.tl, self.seed = tl, seed
        self.T = tl.T
        rng = np.random.default_rng(seed + 31337)
        self.times = {}
        for i, name in enumerate(C.CH_NAMES):
            r = CH[name]["rate"]
            n = int(self.T // r)
            t = rng.uniform(0, r) + np.arange(n) * r + rng.uniform(-0.3, 0.3, n)
            keep = rng.random(n) > 0.002
            t = t[keep & (t >= 0) & (t < self.T)]
            self.times[name] = np.sort(t)
        self.noise = {n: np.random.default_rng(seed * 1000 + i).standard_normal(len(self.times[n])).astype(np.float32)
                      for i, n in enumerate(C.CH_NAMES)}
        u = np.random.default_rng(seed + 4242)
        self.unit = dict(gyro_bias=u.normal(0, 0.002, 3), mag_bias=u.normal(0, 80, 3),
                         gyro_rw=u.normal(0, 1.5e-6, (int(self.T // 600) + 2, 3)).cumsum(0),
                         eff=u.uniform(0.98, 1.02), cap=C.EPS["cap_ah"] * u.uniform(0.97, 1.03),
                         rint=C.EPS["r_int"] * u.uniform(0.9, 1.1), soc0=u.uniform(80, 95))
        self.grid = np.arange(0.0, self.T, 30.0)

    # ------------------------------------------------------------ physics
    def physical(self, tr, ticks):
        """Noise-free physical quantities from one Basilisk truth record."""
        tl, o = self.tl, self.tl.orbit
        t5 = tr["t"]
        g = self.grid
        P = {}
        r_g = interp(g, t5, tr["r_BN"].astype(np.float64))
        sig_g = interp(g, t5, tr["sigma_BN"].astype(np.float64))
        sun_g, rau = o.sun_P(tl.t0u + g)
        nu = shadow(r_g, sun_g, rau)
        Cg = mrp_to_dcm(sig_g)
        sB = np.einsum("nij,nj->ni", Cg, sun_g)
        P["nu"], P["sB"], P["S"] = nu, sB, 1361.0 / rau ** 2
        om = tr["Omega"].astype(np.float64)
        u = tr["u"].astype(np.float64)
        f = tr["fric"].astype(np.float64)
        prw = (np.abs(u * om) / C.SC["rw_eta"] + C.SC["rw_pbase"])
        qrw = np.abs(u * om) * (1 - C.SC["rw_eta"]) / C.SC["rw_eta"] + np.abs(f * om) + 1.0
        # 30 s averages for the thermal / power loop
        k = np.clip(np.searchsorted(g, t5, "right") - 1, 0, len(g) - 1)
        cnt = np.bincount(k, minlength=len(g)).clip(1)
        P["prw_g"] = np.bincount(k, prw.sum(1), minlength=len(g)) / cnt
        P["qrw_g"] = np.stack([np.bincount(k, qrw[:, j], minlength=len(g)) / cnt for j in range(4)], 1)
        P["t5"], P["om"], P["u"], P["prw"] = t5, om, u, prw.sum(1)
        P["sig"], P["sigBR"], P["w"] = tr["sigma_BN"], tr["sigma_BR"], tr["omega_BN"]
        P["r"] = tr["r_BN"]
        P["tick_t"], P["meff"] = ticks[:, 0].astype(np.float64), ticks[:, 4:7].astype(np.float64)
        return P

    def loads(self, mods):
        """Bus load on the 30 s grid: base + payload + transmitter (+ C3 latch-up, - C4)."""
        g = self.grid
        L = np.full(len(g), C.EPS["bus_base"])
        L += 4.0 * np.sin(2 * np.pi * g / 5400.0 + 0.7)            # avionics duty cycles
        for a, b, kind in self.tl.loads:
            if kind == "payload" and any(m["code"] == "C4" and m["start"] == a for m in mods):
                continue
            i0, i1 = np.searchsorted(g, [a, b])
            L[i0:i1] += C.EPS["payload"] if kind == "payload" else C.EPS["tx"]
        for m in mods:
            if m["code"] == "C3":
                i0, i1 = np.searchsorted(g, [m["start"], m["end"]])
                L[i0:i1] += m["watts"]
        return L

    def power_thermal(self, P, mods=()):
        """EPS + thermal integration on the 30 s grid (sequential: SoC limits, thermostat)."""
        g, E, U = self.grid, C.EPS, self.unit
        n = len(g)
        nu, sB, S = P["nu"], P["sB"], P["S"]
        cosw = np.hypot(sB[:, 0], sB[:, 1])
        body = np.abs(sB[:, 2])
        frac = np.ones(n)
        rmul = np.ones(n)
        heat_forced = np.zeros(n, bool)
        for m in mods:
            i0, i1 = np.searchsorted(g, [m["start"], m["end"]])
            if m["code"] == "C1":
                frac[i0:i1] = (E["n_strings"] - m["strings"]) / E["n_strings"]
            elif m["code"] == "C2":
                rmul[i0:i1] = m["factor"]
            elif m["code"] == "D1":
                heat_forced[i0:i1] = True
        load = self.loads(mods) + P["prw_g"]
        ocv_s, ocv_v = C.OCV_SOC, C.OCV_V
        out = np.empty((n, 10))
        soc, Tb, Tp, Ts = U["soc0"], 16.0, 20.0, 15.0
        Tw = [20.0] * 4
        heater = False
        dt = 30.0
        a_s, a_p = dt / 3000.0, dt / 600.0
        cap, rint, eff0 = U["cap"], U["rint"], E["cell_eff"] * U["eff"]
        Aw, Ab, ns = E["wing_area"], E["body_panel_area"], E["cells"]
        vmax = 4.2 * ns
        qrw = P["qrw_g"]
        for i in range(n):
            v = nu[i]
            Ts += a_s * ((8.0 + 10.0 * v + 0.02 * (load[i] - 250.0)) - Ts)
            Tp += a_p * ((-75.0 + 150.0 * v * (0.35 + 0.65 * cosw[i])) - Tp)
            age = g[i] / (365.0 * 86400.0)
            eff = eff0 * (1 - C.AGEING["sa_per_year"] * age) * (1 + E["temp_coeff"] * (Tp - E["t_ref"]))
            psa = S[i] * eff * (Aw * cosw[i] * frac[i] + Ab * body[i]) * v
            pl = load[i] + (E["heater"] if heater else 0.0)
            pn = psa - pl
            ocv = ns * np.interp(soc, ocv_s, ocv_v)
            R = rint * (1 + C.AGEING["rint_per_year"] * age) * rmul[i] * (1 + 0.01 * (20.0 - Tb))
            if pn >= 0:            # CC-CV charging: current limit, then 4.2 V/cell voltage limit
                I = min(pn / ocv, E["i_charge_max"], max(0.0, (vmax - ocv) / R))
            else:
                I = pn / ocv
            V = ocv + I * R
            soc = min(100.0, max(0.0, soc + 100.0 * I * dt / 3600.0 / (cap * (1 - C.AGEING["cap_per_year"] * age)) * (E["eta_charge"] if I > 0 else 1.0)))
            if heat_forced[i]:
                heater = True
            elif Tb < E["heater_on"]:
                heater = True
            elif Tb > E["heater_off"]:
                heater = False
            Tb += dt / 8000.0 * (I * I * R + (E["heater"] if heater else 0.0) + 2.0 * (Ts - Tb))
            for j in range(4):
                Tw[j] += dt / 900.0 * (qrw[i, j] - (Tw[j] - Ts) / 4.0)
            out[i] = (psa, soc, V, pl, Tp, Tb, *Tw)
        return dict(sa_power=out[:, 0], bat_soc=out[:, 1], bat_voltage=out[:, 2], bus_load=out[:, 3],
                    panel_temp=out[:, 4], bat_temp=out[:, 5], rw1_temp=out[:, 6], rw2_temp=out[:, 7],
                    rw3_temp=out[:, 8], rw4_temp=out[:, 9])

    # ------------------------------------------------------------ outages
    def str_outages(self, P, mods, skip_reset=None):
        """Star tracker invalid intervals: commanded resets, B4 faults, Sun in the 25 deg exclusion cone."""
        self._b4 = {m["start"] for m in mods if m["code"] == "B4"}
        outs = [(r["start"], r["start"] + r["outage"]) for r in self.tl.rare
                if r["key"] == "STR_RESET" and r["start"] != skip_reset]
        outs += [(m["start"], m["end"]) for m in mods if m["code"] == "B4"]
        bore = np.array([np.cos(np.radians(40)), 0.0, -np.sin(np.radians(40))])   # zenith, tilted 40 deg off-track
        blind = (P["sB"] @ bore > np.cos(np.radians(25))) & (P["nu"] > 0.5)
        a, b = runs(blind)
        outs += [(self.grid[i], self.grid[j - 1] + 30.0) for i, j in zip(a, b)]
        return sorted(outs)

    # ----------------------------------------------------------- channels
    def channels(self, P, pw, mods=(), names=None, skip_reset=None):
        """Noisy channel values (float32) at each channel's sample times."""
        names = names or C.CH_NAMES
        tl, o, U = self.tl, self.tl.orbit, self.unit
        out = {}
        t5 = P["t5"]
        outs = None
        for name in names:
            t = self.times[name]
            c = CH[name]
            z = self.noise[name]
            if name.startswith("rw") and name.endswith("_speed"):
                k = int(name[2]) - 1
                v = interp(t, t5, P["om"][:, k]) * RPM
            elif name.startswith("rw") and name.endswith("_current"):
                k = int(name[2]) - 1
                v = interp(t, t5, P["u"][:, k]) / C.SC["rw_kt"]
            elif name == "rw_power":
                v = interp(t, t5, P["prw"])
            elif name.startswith("gyro_"):
                a = "xyz".index(name[-1])
                v = interp(t, t5, P["w"][:, a].astype(np.float64)) * DEG + U["gyro_bias"][a] + \
                    interp(t, np.arange(U["gyro_rw"].shape[0]) * 600.0, U["gyro_rw"][:, a])
            elif name.startswith("att_err_"):
                a = "xyz".index(name[-1])
                s = interp(t, t5, P["sigBR"].astype(np.float64))
                sn = np.linalg.norm(s, axis=1)
                ang = np.where(sn > 1e-12, 4 * np.arctan(sn) / np.maximum(sn, 1e-12), 4.0)
                v = s[:, a] * ang * DEG
                if outs is None:
                    outs = self.str_outages(P, mods, skip_reset)
                v = v + self._est_drift(t, outs, a)
            elif name == "att_err":            # total attitude error (principal angle of the estimate)
                s = interp(t, t5, P["sigBR"].astype(np.float64))
                sn = np.linalg.norm(s, axis=1)
                ang = np.where(sn > 1e-12, 4 * np.arctan(sn) / np.maximum(sn, 1e-12), 4.0)
                if outs is None:
                    outs = self.str_outages(P, mods, skip_reset)
                vec = s * ang[:, None] * DEG + np.stack([self._est_drift(t, outs, a) for a in range(3)], 1)
                v = np.linalg.norm(vec, axis=1)
            elif name.startswith("mag_"):
                a = "xyz".index(name[-1])
                if "_magB" not in P:
                    tm = self.times["mag_x"]
                    r = interp(tm, t5, P["r"].astype(np.float64))
                    sg = interp(tm, t5, P["sig"].astype(np.float64))
                    P["_magB"] = np.einsum("nij,nj->ni", mrp_to_dcm(sg), b_field_P(o, tl.t0u + tm, r))
                    P["_magM"] = hold(tm, P["tick_t"] - 60.0, P["meff"])
                v = np.interp(t, self.times["mag_x"], P["_magB"][:, a]) + U["mag_bias"][a] + \
                    2.0 * np.interp(t, self.times["mag_x"], P["_magM"][:, a])
            elif name.startswith("mtq_"):
                a = "xyz".index(name[-1])
                v = hold(t, P["tick_t"] - 60.0, P["meff"][:, a])
            elif name == "str_valid":
                if outs is None:
                    outs = self.str_outages(P, mods, skip_reset)
                v = np.ones(len(t))
                for a_, b_ in outs:
                    i0, i1 = np.searchsorted(t, [a_, b_])
                    v[i0:i1] = 0.0
            elif name == "eclipse_flag":
                v = hold(t, self.grid, (P["nu"] < 0.5).astype(float))
            elif name == "aocs_mode":
                tm = np.array([0.0] + [m[0] for m in tl.modes])
                cm = np.array([0.0] + [m[1] for m in tl.modes])
                v = hold(t, tm, cm)
            else:                                              # EPS / thermal from the 30 s loop
                v = hold(t, self.grid, pw[name])
            v = np.asarray(v, np.float64) + c["sigma"] * z
            if name == "sa_power":
                v = np.maximum(v, 0.0)
            elif name == "bat_soc":
                v = np.clip(v, 0.0, 100.0)
            out[name] = v
        self._sensor_faults(out, mods, names)
        for name in out:
            q = CH[name]["q"]
            out[name] = (np.round(out[name] / q) * q).astype(np.float32)
        return out

    def _est_drift(self, t, outs, axis):
        """Attitude-estimate error while the star tracker is out: gyro-propagated drift (random
        residual bias 1e-5 deg/s + angle random walk), snapping back when fixes resume."""
        e = np.zeros(len(t))
        for a, b in outs:
            i0, i1 = np.searchsorted(t, [a, b])
            if i1 <= i0:
                continue
            rr = np.random.default_rng(int(a) * 3 + axis + self.seed)
            dtv = t[i0:i1] - a
            # uncommanded reboot (B4): the estimator also loses its bias estimate, so it free-runs on
            # the full gyro bias (~0.2 deg/h class) instead of a calibrated residual
            g = 6.0 if a in getattr(self, "_b4", ()) else 1.0
            e[i0:i1] = g * (rr.normal(0, 1e-5) * dtv + rr.normal(0, 2e-4) * np.sqrt(dtv / 60.0) * rr.standard_normal())
        return e

    def _sensor_faults(self, out, mods, names):
        for m in mods:
            c, comp = m["code"], m.get("component", "")
            if c in ("B1", "B2") and comp in out:
                t = self.times[comp]
                i0, i1 = np.searchsorted(t, [m["start"], m["end"]])
                if i1 <= i0:
                    continue
                if c == "B1":       # smooth bias excursion with a random-walk texture
                    x = (t[i0:i1] - m["start"]) / (m["end"] - m["start"])
                    env = np.sin(np.pi * x) ** 0.6
                    rr = np.random.default_rng(int(m["start"]))
                    tex = 1 + 0.3 * np.cumsum(rr.normal(0, 0.05, i1 - i0))
                    out[comp][i0:i1] += m["amp"] * env * tex
                else:               # single-event upsets: isolated spikes
                    rr = np.random.default_rng(int(m["start"]))
                    idx = rr.choice(np.arange(i0, i1), size=min(m["n"], i1 - i0), replace=False)
                    out[comp][idx] += rr.choice([-1, 1], len(idx)) * m["amp"] * rr.uniform(0.5, 1.0, len(idx))
            elif c in ("B3", "D2") and comp in out:
                t = self.times[comp]
                i0, i1 = np.searchsorted(t, [m["start"], m["end"]])
                if i1 > i0:
                    out[comp][i0:i1] = out[comp][max(i0 - 1, 0)]


def fault_kind(m):
    """Which layer a fault lives in: physics (Basilisk), power (EPS/thermal loop) or sensor."""
    return m["kind"]
