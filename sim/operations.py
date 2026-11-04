"""Operations timeline for one satellite: telecommands, physics events, faults, rare events, wear.

Everything is drawn from the satellite's seed, so the timeline is reproducible. Times are seconds
from the mission epoch (config.EPOCH).

Routine operations (unlabelled, frequent):  imaging / survey payload windows, slews, ground-pass
transmitter on/off, commanded magnetic momentum dumps (2 per day).
Rare nominal events (labelled 'Rare Event'): gyro calibration (monthly), inertial-hold calibration
(monthly), star tracker resets (~1.5 / month), safe-mode drills (2 / year), autonomous momentum
dumps (uncommanded, decided on board by the physics run).
Faults (labelled 'Anomaly'): drawn per profile from the FMECA fault library (config.FAULTS).
"""
import numpy as np
import pandas as pd

import config as C
from envmodels import Orbit, shadow, station_ecef, elevation

DAY = 86400.0


def epoch_unix():
    return pd.Timestamp(C.EPOCH, tz="UTC").timestamp()


class Timeline:
    def __init__(self, name, spec, seed, days=None):
        self.name, self.spec, self.seed = name, spec, seed
        self.days = days or spec.get("days", C.DAYS)
        self.T = self.days * DAY
        self.rng = np.random.default_rng(seed)
        self.t0u = epoch_unix()
        self.orbit = Orbit(spec["orbit"], self.t0u)
        self.tcs = []          # (t, TC name)
        self.phys = []         # (t, kind, data)   physics events (Basilisk run)
        self.modes = []        # (t, mode code)    0 nadir 1 target 2 cal 3 inertial 4 safe
        self.loads = []        # (t0, t1, kind)    payload / tx windows for the power model
        self.rare = []         # dict: cls, start, end, tc, commanded
        self.faults = []       # dict: code, cls, start, end, component, params ...
        self.busy = []         # (t0, t1) blocks that routine ops must avoid
        self.wear = None

    # ------------------------------------------------------------ helpers
    def free(self, t0, t1, pad=300.0):
        return all(t1 + pad <= a or t0 - pad >= b for a, b in self.busy)

    def block(self, t0, t1):
        self.busy.append((t0, t1))

    def tc(self, t, name):
        self.tcs.append((float(t), name))

    def slew(self, t, a0, a1, code):
        """Profiled slew (agile EO guidance): minimum-jerk angle profile with rate feed-forward,
        guidance updated every 10 s; duration grows with the slew angle (120-240 s)."""
        a0, a1 = np.asarray(a0, float), np.asarray(a1, float)
        dA = a1 - a0
        T = float(np.clip(120.0 + 3.0 * np.abs(dA).max(), 120.0, 240.0))
        self.modes.append((float(t), code))
        for tau in np.arange(0.0, T, 10.0) / T:
            s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
            ds = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / T
            self.phys.append((float(t + tau * T), "mode", dict(type="slew_pt", angles=tuple(a0 + dA * s),
                                                                  rates=tuple(dA * ds))))
        self.phys.append((float(t + T), "mode", dict(type="nadir" if code == 0 else "offset", angles=tuple(a1))))
        return T

    def mode(self, t, kind, code, **kw):
        self.phys.append((float(t), "mode", dict(type=kind, **kw)))
        self.modes.append((float(t), code))

    # ------------------------------------------------------- geometry ----
    def geometry(self):
        g = np.arange(0.0, self.T, 30.0)
        tu = self.t0u + g
        r = self.orbit.r_P(tu)
        s, rau = self.orbit.sun_P(tu)
        self.g30 = g
        self.sunlit = shadow(r, s, rau) > 0.5
        passes = []
        for st, (la, lo) in C.STATIONS.items():
            el = elevation(self.orbit, tu, r, station_ecef(la, lo))
            vis = el > C.EL_MASK
            d = np.diff(np.r_[0, vis.astype(int), 0])
            for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                if (b - a) * 30 >= 240:
                    passes.append((g[a], g[b - 1], st))
        self.passes = sorted(passes)

    def sunlit_at(self, t):
        return bool(self.sunlit[min(int(t // 30), len(self.sunlit) - 1)])

    # --------------------------------------------------------- rare ops ----
    def rare_ops(self):
        rng, P = self.rng, self.orbit.period
        n_g, n_i, str_pm, n_dr = C.RARE_RATE[self.spec["profile"]]
        yr = self.days / 365.0
        # safe-mode drills (2 orbits in sun-safe), stratified over the mission
        k_dr = int(round(n_dr * yr)) if self.days >= 300 else 0
        drills = [(k + rng.uniform(0.15, 0.85)) * self.days / k_dr for k in range(k_dr)]
        for d in drills:
            t = np.floor(d * DAY / 60) * 60
            t1 = t + 2 * P
            self.tc(t, "SAFE_ENTER"); self.mode(t, "sunsafe", 4)
            self.tc(t1, "SAFE_EXIT"); self.mode(t1, "nadir", 0)
            self.block(t - 600, t1 + 1800)
            self.rare.append(dict(cls="safe_mode_drill", key="SAFE", start=t, end=t1 + 1200, tc="SAFE_ENTER", commanded=True))
        def slots(n):
            n = max(int(round(n * yr)), 0)
            return [(k * self.days / n, self.days / n) for k in range(n)]
        for m0, span in slots(n_g):
            # gyro calibration: +-20 deg about each axis, 3.5 min each, then nadir
            for _ in range(50):
                t = np.floor((m0 + rng.uniform(0.1, 0.9) * span) * DAY / 60) * 60
                if t + 1500 < self.T - DAY and self.free(t, t + 1500, 1800):
                    break
            else:
                continue
            self.tc(t, "GYRO_CAL")
            seq = [(0, 0, 20), (0, 0, -20), (0, 20, 0), (0, -20, 0), (20, 0, 0), (-20, 0, 0)]
            for i, ang in enumerate(seq):
                self.mode(t + i * 210, "offset", 2, angles=ang)
            self.mode(t + 6 * 210, "nadir", 0)
            self.block(t, t + 6 * 210 + 600)
            self.rare.append(dict(cls="gyro_calibration", key="GYRO_CAL", start=t, end=t + 6 * 210 + 600, tc="GYRO_CAL", commanded=True))
        for m0, span in slots(n_i):
            # inertial-hold calibration: 20 min fixed inertial attitude, then back to nadir
            for _ in range(50):
                t = np.floor((m0 + rng.uniform(0.1, 0.9) * span) * DAY / 60) * 60
                if t + 2400 < self.T - DAY and self.free(t, t + 2400, 1800):
                    break
            else:
                continue
            self.tc(t, "INERTIAL_CAL"); self.mode(t, "inertial_hold", 3)
            self.mode(t + 1200, "nadir", 0)
            self.block(t, t + 1200 + 900)
            self.rare.append(dict(cls="inertial_calibration", key="INERTIAL_CAL", start=t, end=t + 1200 + 900, tc="INERTIAL_CAL", commanded=True))
        # star tracker resets (~1.5 per month): 3 min without attitude fixes
        n_str = int(round(self.days / 30.4 * str_pm))
        for _ in range(n_str):
            for _ in range(50):
                t = np.floor(rng.uniform(2, self.days - 2) * DAY / 30) * 30
                if self.free(t, t + 180, 600):
                    break
            self.tc(t, "STR_RESET")
            self.block(t, t + 180)
            self.rare.append(dict(cls="star_tracker_reset", key="STR_RESET", start=t, end=t + 180, tc="STR_RESET", commanded=True,
                                  outage=180.0))

    # -------------------------------------------------------- routine ----
    def routine(self):
        rng = self.rng
        # commanded momentum dumps: ~03:00 and ~15:00 UTC, 30 min
        for d in range(1, self.days):
            for h in (3.0, 15.0):
                t = np.floor((d * DAY + (h + rng.uniform(-0.5, 0.5)) * 3600) / 60) * 60
                for _ in range(6):
                    if self.free(t, t + 1800, 120):
                        break
                    t += 3600
                else:
                    continue
                self.tc(t, "MTQ_DUMP"); self.phys.append((t, "dump", dict(dur=1800.0)))
                self.block(t, t + 1800)
        # payload windows
        imaging = self.spec["role"] == "imaging"
        for d in range(1, self.days):
            n = (max(4, rng.poisson(7)) if self.spec["profile"] == "ops" else max(3, rng.poisson(5))) if imaging \
                else rng.integers(2, 4)
            n_slew = n if imaging else int(rng.random() < 0.5)
            for i in range(n):
                for _ in range(40):
                    t = np.floor((d * DAY + rng.uniform(0, DAY - 3600)) / 30) * 30
                    hold = rng.uniform(360, 720) if (imaging or i < n_slew) else rng.uniform(900, 1800)
                    if self.sunlit_at(t) and self.free(t - 180, t + hold + 420, 600):
                        break
                else:
                    continue
                self.tc(t - 120, "PAYLOAD_ON")
                if i < n_slew:
                    ang = (0.0, rng.uniform(-15, 15), rng.choice([-1, 1]) * rng.uniform(8, 30))
                    self.tc(t, "SLEW_TARGET"); self.slew(t, (0.0, 0.0, 0.0), ang, 1)
                    self.tc(t + hold, "POINT_NADIR"); self.slew(t + hold, ang, (0.0, 0.0, 0.0), 0)
                self.tc(t + hold + 60, "PAYLOAD_OFF")
                self.loads.append((t - 120, t + hold + 60, "payload"))
                self.block(t - 180, t + hold + 420)
        # ground passes: transmitter on/off
        for a, b, st in self.passes:
            if b - a < 240:
                continue
            self.tc(a + 30, "TX_ON"); self.tc(b - 30, "TX_OFF")
            self.loads.append((a + 30, b - 30, "tx"))

    # ---------------------------------------------------------- faults ----
    def faults_sched(self):
        rng = self.rng
        prof = C.PROFILE_FAULTS[self.spec["profile"]]
        fid = 0
        placed = []
        split = C.SPLIT_DAY * DAY
        pay = sorted([w for w in self.loads if w[2] == "payload"])
        for code, (n_tr, n_te) in prof.items():
            f = C.FAULTS[code]
            for part, n in (("train", n_tr), ("test", n_te)):
                LO, HI = (3 * DAY, split - DAY) if part == "train" else (split + DAY, self.T - 3 * DAY)
                if part == "test" and self.spec["profile"] == "wheel":
                    HI = min(HI, C.SAT5_WEAR["t_fail"] * DAY - 5 * DAY)
                for kk in range(n):
                    # stratified: the kk-th fault of this type in the kk-th slice (random offset)
                    if part == "train" and kk == 0:
                        # first one of each trained type inside the validation window (settings are
                        # chosen there); the rest stratified over the remaining training period
                        lo, hi = split - C.VAL_DAYS * DAY, HI
                    else:
                        m_ = n - 1 if part == "train" else n
                        k_ = kk - 1 if part == "train" else kk
                        H_ = split - C.VAL_DAYS * DAY if part == "train" else HI
                        w_ = (H_ - LO) / m_
                        lo, hi = max(LO, LO + (k_ - 0.25) * w_), min(H_, LO + (k_ + 1.25) * w_)
                    for _ in range(200):
                        if code == "C4":
                            cand = [w for w in pay if lo < w[0] < hi]
                            w = cand[rng.integers(len(cand))]
                            t0, t1 = w[0], w[1]
                        else:
                            d0, d1 = f["dur"]
                            dur = np.exp(rng.uniform(np.log(d0), np.log(d1))) * 3600
                            t0 = np.floor(rng.uniform(lo, hi - dur) / 30) * 30
                            t1 = t0 + dur
                        same = [p for p in placed if p["family"] == f["family"]]
                        if all(t1 + 6 * 3600 < p["start"] or t0 > p["end"] + 6 * 3600 for p in same) and \
                                all(t1 + 600 < a or t0 > b + 600 for a, b in self._rare_blocks()):
                            break
                    fid += 1
                    flt = dict(code=code, cls=f["cls"], family=f["family"], kind=f["kind"], subsystem=f["subsystem"],
                               severity=f["sev"], mode=f["mode"], start=float(t0), end=float(t1), split=part)
                    self._params(flt, rng)
                    placed.append(flt)
        if self.spec["profile"] == "wheel":
            self._overlap(placed, rng)
        self.faults = sorted(placed, key=lambda x: x["start"])
        for f in self.faults:
            if f["kind"] == "physics":
                self._phys_fault(f)

    def _rare_blocks(self):
        return [(r["start"] - 600, r["end"]) for r in self.rare]

    def _params(self, f, rng):
        c = f["code"]
        if c in ("A1", "A2", "A3"):
            f["component"] = f"rw{rng.integers(4) + 1}"
        if c == "A1":
            f["mult"] = float(rng.uniform(4, 12))
        if c == "A3":
            f["frac"] = float(rng.uniform(0.25, 0.40))
        if c == "A4":
            f["component"] = "mtq_" + "xyz"[rng.integers(3)]
        if c in ("B1", "B2"):
            f["component"] = "gyro_" + "xyz"[rng.integers(3)]
        if c == "B1":
            f["amp"] = float(rng.uniform(0.004, 0.012)) * rng.choice([-1, 1])
        if c == "B2":
            f["n"] = int(rng.integers(3, 9))
            f["amp"] = float(rng.uniform(0.05, 0.5))
        if c == "B3":
            f["component"] = "mag_" + "xyz"[rng.integers(3)]
        if c == "B4":
            f["component"] = "star_tracker"
        if c == "C1":
            f["component"] = "solar_array"; f["strings"] = int(rng.choice([1, 1, 2]))
        if c == "C2":
            f["component"] = "battery"; f["factor"] = float(rng.uniform(2.5, 4.0))
        if c == "C3":
            f["component"] = "power_bus"; f["watts"] = float(rng.uniform(50, 90))
        if c == "C4":
            f["component"] = "payload"
        if c == "D1":
            f["component"] = "battery_heater"
        if c == "D2":
            f["component"] = rng.choice(["rw1_temp", "rw2_temp", "rw3_temp", "rw4_temp", "panel_temp", "bat_temp"])

    def _overlap(self, placed, rng):
        """SAT_4: make some faults simultaneous (D during A) and some back-to-back."""
        for part in ("test",):          # SAT_2 test: some F2 faults during an F1 fault
            A = [p for p in placed if p["family"] == "A" and p["split"] == part and p["code"] in ("A1", "A3")]
            D = [p for p in placed if p["family"] == "B" and p["code"] in ("B1", "B4") and p["split"] == part]
            for a, d in zip(A[:2], D[:2]):
                dur = d["end"] - d["start"]
                d["start"] = float(np.floor((a["start"] + 0.3 * (a["end"] - a["start"])) / 30) * 30)
                d["end"] = d["start"] + dur
                d["overlap_with"] = a["code"]

    def _phys_fault(self, f):
        k = int(f["component"][-1]) - 1 if f["component"].startswith("rw") else None
        if f["code"] == "A1":
            self.phys.append((f["start"], "fric", dict(wheel=k, mult=f["mult"], key=f"A1_{f['start']}")))
            self.phys.append((f["end"], "fric_end", dict(wheel=k, key=f"A1_{f['start']}")))
        elif f["code"] == "A2":
            self.phys.append((f["start"], "umax", dict(wheel=k, frac=0.0)))
            self.phys.append((f["end"], "umax_end", dict(wheel=k)))
        elif f["code"] == "A3":
            self.phys.append((f["start"], "umax", dict(wheel=k, frac=f["frac"])))
            self.phys.append((f["end"], "umax_end", dict(wheel=k)))
        elif f["code"] == "A4":
            ax = "xyz".index(f["component"][-1])
            self.phys.append((f["start"], "mtq_fail", dict(axis=ax)))
            self.phys.append((f["end"], "mtq_fail_end", dict(axis=ax)))

    # ------------------------------------------------------------ wear ----
    def wear_process(self, wspec):
        """Exponential mean wear path with gamma-distributed daily increments (monotone)."""
        rng = self.rng
        t_on, t2, tf = wspec["t_on"], wspec["t_phase2"], wspec["t_fail"]
        W2, WF = C.WEAR_W["phase2"], C.WEAR_W["fail"]
        # solve m(t) = a (exp(lam (t - t_on)) - 1): m(t2) = W2, m(tf) = WF
        lo, hi = 1e-4, 1.0
        for _ in range(100):
            lam = 0.5 * (lo + hi)
            ratio = (np.exp(lam * (tf - t_on)) - 1) / (np.exp(lam * (t2 - t_on)) - 1)
            lo, hi = (lam, hi) if ratio < WF / W2 else (lo, lam)
        a = W2 / (np.exp(lam * (t2 - t_on)) - 1)
        days = np.arange(0, self.days + 1)
        mean = np.where(days >= t_on, a * (np.exp(lam * np.clip(days - t_on, 0, None)) - 1), 0.0)
        inc = np.diff(mean, prepend=0.0)
        shape = 8.0                                            # cv of a daily increment = 0.35
        w = np.cumsum(np.where(inc > 0, rng.gamma(shape, np.maximum(inc, 1e-12) / shape), 0.0))
        k = wspec.get("wheel", 1)
        t_fail = None
        for d in range(1, self.days):
            if w[d] >= WF:
                t_fail = float(d * DAY + np.floor(rng.uniform(0, DAY) / 60) * 60)
                break
            self.phys.append((d * DAY, "wear", dict(wheel=k, w=float(w[d]))))
        t_spk = next((d * DAY for d in range(self.days) if w[d] >= C.WEAR_W["spikes"]), None)
        t_p2 = next((d * DAY for d in range(self.days) if w[d] >= W2), None)
        # friction spikes: Poisson, rate grows with wear; amplitude x(3-8) on top of the wear level
        spikes = []
        if t_spk is not None:
            d = t_spk / DAY
            end = (t_fail / DAY) if t_fail else self.days
            while d < end - 0.2:
                wd = w[int(d)]
                rate = min(2.0, 0.08 * np.sqrt(max(wd, C.WEAR_W["spikes"]) / C.WEAR_W["spikes"]))
                d += rng.exponential(1.0 / rate)
                if d >= end - 0.2:
                    break
                t0 = np.floor(d * DAY / 30) * 30
                dur = np.exp(rng.uniform(np.log(0.3), np.log(3.0))) * 3600
                mult = float(rng.uniform(3, 8))
                key = f"W_{t0}"
                spikes.append(dict(start=float(t0), end=float(t0 + dur), mult=mult))
                self.phys.append((float(t0), "fric", dict(wheel=k, mult=mult, key=key)))
                self.phys.append((float(t0 + dur), "fric_end", dict(wheel=k, key=key)))
        if t_fail is not None:
            self.phys.append((t_fail, "fail", dict(wheel=k)))
        self.wear = dict(wheel=k, w_daily=w.tolist(), t_spikes=t_spk, t_phase2=t_p2, t_fail=t_fail,
                         spikes=spikes, lam=float(lam), a=float(a), t_on=t_on)
        # labelled events: phase-1 spikes (each its own anomaly), phase 2 trend, phase 3 failure
        comp = f"rw{k + 1}"
        for s in spikes:
            if t_p2 is None or s["start"] < t_p2:
                self.faults.append(dict(code="W1", cls="rw_wear_spike", family="W", kind="physics", subsystem="AOCS",
                                        severity=2, mode=C.FAULTS["W1"]["mode"], start=s["start"], end=s["end"],
                                        component=comp, mult=s["mult"], phase=1,
                                        split="train" if s["start"] < C.SPLIT_DAY * DAY else "test"))
        if t_p2 is not None:
            e2 = t_fail if t_fail else self.T
            self.faults.append(dict(code="W2", cls="rw_degradation", family="W", kind="physics", subsystem="AOCS",
                                    severity=2, mode=C.FAULTS["W2"]["mode"], start=float(t_p2), end=float(e2),
                                    component=comp, phase=2, split="train" if t_p2 < C.SPLIT_DAY * DAY else "test"))
        if t_fail is not None:
            self.faults.append(dict(code="W3", cls="rw_failure", family="W", kind="physics", subsystem="AOCS",
                                    severity=3, mode=C.FAULTS["W3"]["mode"], start=float(t_fail), end=float(self.T),
                                    component=comp, phase=3, split="train" if t_fail < C.SPLIT_DAY * DAY else "test"))
        self.faults.sort(key=lambda x: x["start"])

    # ----------------------------------------------------------- build ----
    def build(self):
        self.geometry()
        self.rare_ops()
        self.routine()
        self.faults_sched()
        if self.spec["profile"] == "wheel":
            self.wear_process(C.SAT5_WEAR)
        elif self.spec["profile"] == "archive":
            w = self.spec["wear"]
            span = w["fail"] - w["t_on"]
            self.wear_process(dict(wheel=1, t_on=w["t_on"], t_phase2=w["t_on"] + 0.575 * span, t_fail=w["fail"]))
        self.tcs.sort()
        self.phys.sort(key=lambda e: e[0])
        self.modes.sort()
        self.loads.sort()
        self.rare.sort(key=lambda r: r["start"])
        return self

    def summary(self):
        from collections import Counter
        tc = Counter(n for _, n in self.tcs)
        fl = Counter(f["code"] for f in self.faults)
        return dict(name=self.name, tcs=dict(tc), faults=dict(fl), rare=len(self.rare), passes=len(self.passes),
                    wear=None if not self.wear else {k: self.wear[k] for k in ("t_spikes", "t_phase2", "t_fail")})
