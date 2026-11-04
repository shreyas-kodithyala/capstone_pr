"""One scored period (a selection fold or the test split) + the detector stack on top of it.

A Period holds a model's forecast errors for rows [0, e1): rows [0, fit_end) are the model's
own training period (used only for thresholds and as history), rows [e0, e1) are scored.
Every statistic is computed from the past only (see detect.py).

Detector setting (cfg) keys and what they do:
  thr     Detector B threshold: "q<quantile>/<global|perchan>" of training z, "a<z>" absolute,
          or "off"
  W       Detector B baseline window (1d .. 1y, all; "...x" = skip already-flagged rows)
  spans   Detector B smoothing: "multi" (max over 1/3/15/60/240) or "s15"
  tele    Detector A, ESA sliding Telemanom "p/span/min_error/back_reach" (M1 only)
  L       Detector C: silence >= L rows (0 = off)
  std     Global STD-k on raw values (0 = off)
  drift   slow-drift detector "d<k>" (1 day vs prior 30 days) / "w<k>" (1 week vs 90 days)
  hyst    two-tier: an alarm extends over the stretch above hyst x threshold (1.0 = off)
  cross   two-tier across channels: other channels above cross x threshold join (0 = off)
  onset   drop leading blips shorter than this many rows (0 = off)
  close   merge alarms closer than this many rows;  minlen  drop alarms shorter than this
"""
import numpy as np

from .config import MISSIONS
from . import detect as D

BLOCK_HOURS = 6
WINDOWS_DAYS = {"1d": 1, "1w": 7, "1m": 30, "3m": 91, "6m": 182, "1y": 365, "all": None}
SPANS = (1, 3, 15, 60, 240)
QS = ("q0.999", "q0.9999", "q0.99995", "q0.99999")
DRIFT = {"d": (1, 30), "w": (7, 90)}           # (short days, long days)
TELE_PS = (0.05, 0.13, 0.3)
MINERRS = (0.0, 0.007)                          # ESA min_error_value, raw units
BACKS = (100, 50, 25, 0)                        # Telemanom buffer reach backwards (ESA: 100)


def tele_key(s):
    p, sp, me, bk = s.split("/")
    return float(p), int(sp), float(me), int(bk)


def spans_of(spans):
    return SPANS if spans == "multi" else (15,)


class Period:
    def __init__(self, mk, name, err, vals, count, ts, mu, sd, rf, fit_end, e0, e1):
        self.mk, self.name = mk, name
        self.rf, self.fit_end, self.e0, self.e1 = rf, fit_end, e0, e1
        self.R = BLOCK_HOURS * 3600 // MISSIONS[mk]["grid"]
        self.sd = np.asarray(sd, np.float64)
        count = np.asarray(count[:e1])
        vals = np.asarray(vals[:e1], np.float32)
        err = np.asarray(err[:e1], np.float32).copy()
        err[:rf] = 0.0
        self.err, zeroed = D.clean(err, count)
        self.valid = ~zeroed
        self.valid[:rf] = False
        self.ts = ts[:e1]
        self.y = ((vals[e0:e1] - np.asarray(mu, np.float32)) / np.asarray(sd, np.float32))
        self.cnt, self.val = count[e0:e1], vals[e0:e1]
        self.n, self.C = e1 - e0, vals.shape[1]
        ok = count[:fit_end] > 0
        self.vmu = np.array([vals[:fit_end, c][ok[:, c]].mean() for c in range(self.C)], np.float32)
        self.vsd = np.array([vals[:fit_end, c][ok[:, c]].std() + 1e-8 for c in range(self.C)], np.float32)
        if MISSIONS[mk].get("std_floor_fix"):     # no training spread -> STD-k cannot judge it
            self.vsd[self.vsd < MISSIONS[mk]["sd_floor"]] = np.inf
        self._vals_all, self._count_all = vals, count          # needed by the drift statistic
        # full set: a channel sampled every ~15 min is "silent" most of the time; its silence
        # bar becomes mult x its own normal gap (fitted on the training rows only)
        mult = MISSIONS[mk].get("silence_mult") or 0
        self.cadence = D.zero_run_cadence(count[:fit_end]) if mult else None
        self.silence_mult = mult
        self.TZ, self.zs, self.tele, self.drift, self._sil = {}, {}, {}, {}, {}

    # ---- expensive building blocks (precomputed, then shared by forked workers) ----
    def build_tz(self, spans=SPANS):
        for s in spans:
            if s not in self.TZ:
                self.TZ[s] = D.TrailingZ(D.ewma(self.err, s), self.valid, self.R)

    def z_and_thresholds(self, W, spans, lazy=False):
        """Full-length Z for one (window, spans) -> (Z[e0:e1], {thr name: value}).
        lazy: build each span's baseline here and drop it after use (one evaluation, no reuse:
        same numbers, a fraction of the memory)."""
        ex = W.endswith("x")
        wd = WINDOWS_DAYS[W[:-1] if ex else W]
        wb = None if wd is None else max(int(round(wd * 24 / BLOCK_HOURS)), 1)
        Z = None
        for s in spans_of(spans):
            tz = self.TZ[s] if not lazy else (self.TZ.get(s) or D.TrailingZ(D.ewma(self.err, s), self.valid, self.R))
            zs = tz.z_excl(wb, self.fit_end) if ex else tz.z(wb, self.fit_end)
            del tz
            Z = zs if Z is None else np.maximum(Z, zs, out=Z)
        Zf = Z[self.rf:self.fit_end]
        qv = [float(q[1:]) for q in QS]
        g = np.quantile(Zf, qv)
        pc = np.quantile(Zf, qv, axis=0)
        thr = {}
        for i, q in enumerate(QS):
            thr[f"{q}/global"] = np.float32(g[i])
            thr[f"{q}/perchan"] = pc[i].astype(np.float32)
        return np.ascontiguousarray(Z[self.e0:self.e1]), thr

    def telemanom(self, c, span, combos):
        """ESA sliding Telemanom on channel c -> {(p, span, minerr, back): (n,) uint8}."""
        es = D.ewma(self.err[:, c], span)[self.e0:self.e1].astype(np.float64)
        y = self.y[:, c].astype(np.float64)
        st = D.telemanom_prepare(es, y)
        out = {}
        for me, bk in sorted({(me, bk) for (_, sp, me, bk) in combos if sp == span}):
            r = D.telemanom_sliding(es, y, TELE_PS, st=st, min_err=me / self.sd[c], back=bk)
            for p in TELE_PS:
                if (p, span, me, bk) in combos:
                    out[(p, span, me, bk)] = r[p]
        return out

    def build_drift(self, keys=tuple(DRIFT)):
        rpd = 24 // BLOCK_HOURS
        for k in keys:
            if k not in self.drift:
                s, L = DRIFT[k]
                self.drift[k] = np.ascontiguousarray(       # copy: frees the full-length array
                    D.drift_stat(self._vals_all, self._count_all, self.R, s * rpd, L * rpd,
                                 self.fit_end)[self.e0:self.e1])

    def release_history(self):
        """Once every Z / Telemanom / drift array the search needs is built: drop what only
        building them needed (errors, baselines, full-length histories). Detector settings are
        then assembled from the stored arrays alone, so search workers fork from a smaller
        process. Values are unchanged."""
        self.val, self.cnt = np.ascontiguousarray(self.val), np.ascontiguousarray(self.cnt)
        self.err = self._vals_all = self._count_all = None
        self.TZ = {}

    def silence(self, L):
        if L not in self._sil:
            Lc = L if self.cadence is None else np.maximum(L, np.ceil(self.silence_mult * self.cadence))
            self._sil[L] = D.silence(self.cnt, Lc)
        return self._sil[L]


def assemble(P, cfg, parts=False):
    """Detector setting -> (n, C) uint8 alarms on the scored rows (and the raw parts)."""
    core = np.zeros((P.n, P.C), np.uint8)
    comp = {}
    useB = cfg["thr"] != "off"
    if useB:
        Z, thrs = P.zs[(cfg["W"], cfg["spans"])]
        th = np.float32(float(cfg["thr"][1:])) if cfg["thr"].startswith("a") else thrs[cfg["thr"]]
        comp["B"] = (Z > th).astype(np.uint8)
    if cfg.get("tele"):
        comp["A"] = P.tele[tele_key(cfg["tele"])]
    if cfg.get("L"):
        comp["C"] = P.silence(cfg["L"])
    if cfg.get("std"):
        comp["STD"] = D.std_k(P.val, P.cnt, P.vmu, P.vsd, cfg["std"])
    if cfg.get("drift"):
        comp["drift"] = ((P.drift[cfg["drift"][0]] > float(cfg["drift"][1:])) & (P.cnt > 0)).astype(np.uint8)
    for v in comp.values():
        core |= v
    out = core.copy()
    if useB and cfg.get("hyst", 1.0) < 1.0:
        low = Z > cfg["hyst"] * th
        for c in range(P.C):
            out[:, c] |= D.hysteresis(core[:, c].astype(bool), low[:, c])
    if useB and cfg.get("cross", 0.0) > 0:
        g = core.any(1)
        lowx = Z > cfg["cross"] * th
        for c in range(P.C):
            out[:, c] |= D.hysteresis(g, lowx[:, c])
    for c in range(P.C):
        col = D.onset_refine(out[:, c], cfg["close"], cfg["onset"])
        out[:, c] = D.open_runs(D.close_runs(col, cfg["close"]), cfg["minlen"])
    return (out, comp) if parts else out
