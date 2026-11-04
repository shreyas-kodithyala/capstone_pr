"""v2 detection stack. Every piece is causal up to LATENCY (70) rows of retroactive labelling.

  clean          zero errors where every channel is silent, plus 30 rows after (as published)
  TrailingZ      Detector B: robust z of smoothed errors against a PAST-ONLY baseline
                 (median / IQR of the previous W, refreshed every 6 h, from block histograms)
  telemanom_sliding
                 Detector A: ESA's Telemanom-ESA errors.py, faithfully (windows of 30 batches
                 x 70 rows, decide only the newest 70, inverse errors, pruning, history of
                 earlier anomalies), with the epsilon search vectorised
  silence        Detector C: no telemetry for >= L rows (L <= 70)
  std_k          Global STD-k on the raw values vs the fit-period mean / std
  hysteresis     two-tier thresholds: a strict alarm may extend over the connected stretch
                 above a lower threshold (forward freely, backward <= 70 rows), in its own
                 channel and, optionally, in other channels
  onset_refine   a merged alarm may not START with a short blip (<= 70 rows look-ahead)
  close_runs / open_runs   as published, gap and length capped at 70
  typed          alarm typing: an alarm that starts within T after a command is a
                 "commanded change" notice; the rest are anomaly alarms
"""
import numpy as np, pandas as pd

from .config import LATENCY


# ------------------------------------------------------------------ basic utilities
def ewma(e, span):
    if span <= 1:
        return np.asarray(e, np.float32)
    out = pd.DataFrame(e).ewm(span=span, adjust=False).mean().to_numpy(np.float32)
    return out[:, 0] if np.ndim(e) == 1 else out


def runs_of(mask):
    d = np.diff(np.concatenate([[0], np.asarray(mask).astype(np.int8), [0]]))
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], 1)


def clean(err, count, margin=30):
    """-> (cleaned errors, zeroed-row mask)."""
    gap = (count == 0).all(1)
    z = np.zeros(len(err), bool)
    for s, e in runs_of(gap):
        z[s:min(e + margin, len(err))] = True
    out = err.copy()
    out[z] = 0.0
    return out, z


def close_runs(p, gap):
    if gap <= 0 or not p.any():
        return p
    out = p.copy()
    r = runs_of(p)
    if len(r) > 1:
        g = r[1:, 0] - r[:-1, 1]
        for k in np.flatnonzero(g <= gap):
            out[r[k, 1]:r[k + 1, 0]] = 1
    return out


def open_runs(p, m):
    if m <= 1 or not p.any():
        return p
    out = np.zeros_like(p)
    r = runs_of(p)
    for s, e in r[(r[:, 1] - r[:, 0]) >= m]:
        out[s:e] = 1
    return out


# ------------------------------------------------------------------ Detector B
class TrailingZ:
    """Past-only robust z. For rows of block k (R rows each) the baseline median / IQR come
    from the valid rows of blocks k-W .. k-1 only (W = window in blocks, None = all past).
    Percentiles are read from 512 log-spaced histogram bins (1e-5 .. 1e3) with log-linear
    interpolation inside a bin (< 0.2 % error on the quantiles)."""

    NB, LO, HI = 512, 1e-5, 1e3

    def __init__(self, E, valid, R):
        self.E, self.R, self._valid = E, R, valid
        T, C = E.shape
        self.T, self.C = T, C
        self.nb = int(np.ceil(T / R))
        le = np.linspace(np.log(self.LO), np.log(self.HI), self.NB + 1)
        self.logedges = le
        lb = np.clip(((np.log(np.maximum(E, self.LO)) - le[0]) / (le[1] - le[0])).astype(np.int32),
                     0, self.NB - 1)
        blk = (np.arange(T) // R).astype(np.int64)
        self.cum = np.zeros((self.nb + 1, C, self.NB), np.int32)
        for c in range(C):
            v = valid[:, c] if valid.ndim == 2 else valid
            h = np.bincount(blk[v] * self.NB + lb[v, c], minlength=self.nb * self.NB)
            self.cum[1:, c] = np.cumsum(h.reshape(self.nb, self.NB), axis=0)

    def _quantiles(self, H, qs):
        CH = np.cumsum(H, axis=-1)
        n = CH[..., -1]
        out = []
        for q in qs:
            r = q * n
            j = np.argmax(CH >= np.maximum(r, 1e-9)[..., None], axis=-1)
            prev = np.where(j > 0, np.take_along_axis(CH, np.maximum(j - 1, 0)[..., None], -1)[..., 0], 0)
            cnt = np.take_along_axis(H, j[..., None], -1)[..., 0]
            f = np.clip((r - prev) / np.maximum(cnt, 1), 0, 1)
            out.append(np.exp(self.logedges[j] + f * (self.logedges[1] - self.logedges[0])))
        return out, n

    def z_excl(self, W, fit_end, zx=10.0, floor_q=0.25, min_n=None):
        """As z(), but rows already scoring z > zx are left OUT of every later baseline, so a
        long event cannot turn into its own "normal". Sequential over 6-hour blocks,
        vectorised over channels."""
        k_all = np.arange(self.nb)
        lo = np.zeros(self.nb, int) if W is None else np.maximum(k_all - W, 0)
        min_n = self.R // 2 if min_n is None else min_n
        kf = max(min(fit_end // self.R, self.nb), 1)
        le, C, NB = self.logedges, self.C, self.NB
        fl = np.zeros(C)
        for c in range(C):                       # IQR floor from the plain baseline (as z())
            H0 = self.cum[k_all, c] - self.cum[lo, c]
            (p1, _, p3), n0 = self._quantiles(H0, (0.25, 0.5, 0.75))
            iq0 = np.maximum(p3 - p1, 1e-6)
            ok0 = n0 >= min_n
            fl[c] = np.quantile(iq0[:kf][ok0[:kf]], floor_q) if ok0[:kf].any() else 1e-6
        vm = self._valid if self._valid.ndim == 2 else np.repeat(self._valid[:, None], C, 1)
        Z = np.zeros((self.T, C), np.float32)
        blk_h = np.zeros((self.nb, C, NB), np.int32)
        run = np.zeros((C, NB), np.int64)
        off = (np.arange(C) * NB)[None]
        for k in range(self.nb):
            if W is not None and k - W - 1 >= 0:
                run -= blk_h[k - W - 1]
            a, b = k * self.R, min((k + 1) * self.R, self.T)
            e = self.E[a:b]
            ok = run.sum(1) >= min_n
            keep = vm[a:b].copy()
            if ok.any():
                (q1, q2, q3), _ = self._quantiles(run, (0.25, 0.5, 0.75))
                iqr = np.maximum(np.maximum(q3 - q1, 1e-6), fl)
                zz = (e - q2[None]) / iqr[None]
                Z[a:b] = np.where(ok[None], zz, 0.0)
                keep &= ~(ok[None] & (zz > zx))
            lb = np.clip(((np.log(np.maximum(e, self.LO)) - le[0]) / (le[1] - le[0])).astype(np.int64),
                         0, NB - 1)
            blk_h[k] = np.bincount((lb + off)[keep], minlength=C * NB).reshape(C, NB)
            run += blk_h[k]
        return Z

    def z(self, W, fit_end, floor_q=0.25, min_n=None):
        """W in blocks (None = all past). Returns (T, C) float32; rows without a baseline -> 0."""
        k = np.arange(self.nb)
        lo = np.zeros(self.nb, int) if W is None else np.maximum(k - W, 0)
        Z = np.zeros((self.T, self.C), np.float32)
        min_n = self.R // 2 if min_n is None else min_n
        kf = max(min(fit_end // self.R, self.nb), 1)
        for c in range(self.C):
            H = self.cum[k, c] - self.cum[lo, c]                      # blocks lo..k-1
            (q1, q2, q3), n = self._quantiles(H, (0.25, 0.5, 0.75))
            iqr = np.maximum(q3 - q1, 1e-6)
            ok = n >= min_n
            fl = np.quantile(iqr[:kf][ok[:kf]], floor_q) if ok[:kf].any() else 1e-6
            iqr = np.maximum(iqr, fl)
            rows_med = np.repeat(q2, self.R)[:self.T].astype(np.float32)
            rows_iqr = np.repeat(iqr, self.R)[:self.T].astype(np.float32)
            rows_ok = np.repeat(ok, self.R)[:self.T]
            Z[:, c] = np.where(rows_ok, (self.E[:, c] - rows_med) / rows_iqr, 0.0)
        return Z


# ------------------------------------------------------------------ Detector A (ESA Telemanom)
def _dilate(hit, k, axis=-1):
    """True wherever a hit lies within k positions (both sides), along the last axis."""
    L = hit.shape[axis]
    c = np.cumsum(hit, axis=axis, dtype=np.int32)
    c = np.concatenate([np.zeros(hit.shape[:-1] + (1,), np.int32), c], axis=-1)
    j = np.arange(L)
    return (c[..., np.minimum(j + k + 1, L)] - c[..., np.maximum(j - k, 0)]) > 0


def _dilate_asym(hit, back, fwd):
    """True wherever a hit lies within `fwd` rows before or `back` rows after position j,
    i.e. each hit h marks rows [h - back, h + fwd]."""
    L = len(hit)
    c = np.concatenate([[0], np.cumsum(hit, dtype=np.int32)])
    j = np.arange(L)
    return (c[np.minimum(j + back + 1, L)] - c[np.maximum(j - fwd, 0)]) > 0


def _eps_search(Ew, mean, sd, buf, inverse):
    """Telemanom find_epsilon for a batch of windows. Ew (B, L) -> eps (B,)."""
    zs = np.arange(2.5, 12.0, 0.5)
    e = (2 * mean[:, None] - Ew) if inverse else Ew
    eps = mean[:, None] + sd[:, None] * zs[None]                        # (B, Z)
    hit = e[:, None, :] >= eps[:, :, None]                              # (B, Z, L)
    below = ~hit
    nb = below.sum(-1)
    s1 = np.where(below, e[:, None, :], 0).sum(-1)
    s2 = np.where(below, (e * e)[:, None, :], 0).sum(-1)
    with np.errstate(all="ignore"):
        pm = s1 / nb
        pv = np.maximum(s2 / nb - pm * pm, 0)
        mpd = (mean[:, None] - pm) / mean[:, None]
        spd = (sd[:, None] - np.sqrt(pv)) / sd[:, None]
    dil = _dilate(hit, buf - 1)
    n_i = dil.sum(-1)
    prev = np.concatenate([np.zeros(dil.shape[:-1] + (1,), bool), dil[..., :-1]], -1)
    nxt = np.concatenate([dil[..., 1:], np.zeros(dil.shape[:-1] + (1,), bool)], -1)
    n_runs = (dil & ~prev).sum(-1)
    n_single = (dil & ~prev & ~nxt).sum(-1)
    n_seq = n_runs - n_single
    with np.errstate(all="ignore"):
        score = (mpd + spd) / (n_seq ** 2 + n_i)
    L = Ew.shape[1]
    anyhit = hit.any(-1)
    # the reference loop stops at the first z without hits; later z cannot have hits either
    reach = np.cumprod(anyhit, axis=1).astype(bool)
    valid = reach & (n_seq <= 5) & (n_i < L * 0.5) & np.isfinite(score)
    sv = np.where(valid, score, -np.inf)
    best = sv.max(1)
    # ties -> the later z (reference uses >=)
    idx = (sv.shape[1] - 1) - np.argmax((sv == best[:, None])[:, ::-1], axis=1)
    out = mean + 12.0 * sd
    has = np.isfinite(best)
    out[has] = eps[np.arange(len(eps)), idx][has]
    out[sd == 0] = mean[sd == 0] + 12.0 * sd[sd == 0]
    return out


def _count_gt(X, nlev):
    """#(X > k) for k = 0..nlev-1, per row. X (B, L) small non-negative ints -> (B, nlev)."""
    B = X.shape[0]
    h = np.bincount((np.arange(B)[:, None] * (nlev + 1) + X).ravel(),
                    minlength=B * (nlev + 1)).reshape(B, nlev + 1)
    ge = np.cumsum(h[:, ::-1], axis=1)[:, ::-1]                 # #(X >= k)
    return ge[:, 1:]                                             # #(X > k) = #(X >= k+1)


def _eps_search_fast(Ew, mean, sd, buf, inverse):
    """Same result as _eps_search, computed from per-point threshold levels.

    lvl[j] = how many of the 19 thresholds point j reaches; "hit at z_k" == lvl > k.
    The +/-(buf-1) dilation of every hit mask is one running max of lvl, and the counts of
    flagged points, runs and single-point runs for all 19 thresholds come from bincounts."""
    from scipy.ndimage import maximum_filter1d
    zs = np.arange(2.5, 12.0, 0.5)
    nz = len(zs)
    e = (2 * mean[:, None] - Ew) if inverse else Ew
    eps = mean[:, None] + sd[:, None] * zs[None]                          # (B, Z) as reference
    lvl = np.zeros(e.shape, np.int8)
    for k in range(nz):
        lvl += (e >= eps[:, k:k + 1])
    B, L = e.shape
    # pruned (below-threshold) sums for every z: points with lvl <= k
    w1 = np.bincount((np.arange(B)[:, None] * (nz + 1) + lvl).ravel(), weights=e.ravel(),
                     minlength=B * (nz + 1)).reshape(B, nz + 1)
    w2 = np.bincount((np.arange(B)[:, None] * (nz + 1) + lvl).ravel(), weights=(e * e).ravel(),
                     minlength=B * (nz + 1)).reshape(B, nz + 1)
    cnt = np.bincount((np.arange(B)[:, None] * (nz + 1) + lvl).ravel(),
                      minlength=B * (nz + 1)).reshape(B, nz + 1)
    s1, s2, nb = np.cumsum(w1, 1)[:, :nz], np.cumsum(w2, 1)[:, :nz], np.cumsum(cnt, 1)[:, :nz]
    with np.errstate(all="ignore"):
        pm = s1 / nb
        pv = np.maximum(s2 / nb - pm * pm, 0)
        mpd = (mean[:, None] - pm) / mean[:, None]
        spd = (sd[:, None] - np.sqrt(pv)) / sd[:, None]
    M = maximum_filter1d(lvl, size=2 * (buf - 1) + 1, axis=1, mode="constant", cval=0)
    Mp = np.concatenate([np.zeros((B, 1), np.int8), M[:, :-1]], 1)
    Mn = np.concatenate([M[:, 1:], np.zeros((B, 1), np.int8)], 1)
    n_i = _count_gt(M, nz)
    n_runs = n_i - _count_gt(np.minimum(Mp, M), nz)
    n_single = n_i - _count_gt(np.minimum(M, np.maximum(Mp, Mn)), nz)
    n_seq = n_runs - n_single
    anyhit = _count_gt(lvl, nz) > 0
    with np.errstate(all="ignore"):
        score = (mpd + spd) / (n_seq ** 2 + n_i)
    reach = np.cumprod(anyhit, axis=1).astype(bool)
    valid = reach & (n_seq <= 5) & (n_i < L * 0.5) & np.isfinite(score)
    sv = np.where(valid, score, -np.inf)
    best = sv.max(1)
    idx = (nz - 1) - np.argmax((sv == best[:, None])[:, ::-1], axis=1)
    out = mean + 12.0 * sd
    has = np.isfinite(best)
    out[has] = eps[np.arange(B), idx][has]
    out[sd == 0] = mean[sd == 0] + 12.0 * sd[sd == 0]
    return out


def telemanom_prepare(e_s, y, batch=70, nwin=30, buf=100, chunk=400):
    """Per-window statistics and epsilons for one channel (independent of p)."""
    T = len(e_s)
    n = int(np.ceil(T / batch))
    starts = np.arange(n) * batch
    ends = np.minimum(starts + nwin * batch, T)
    ends[-1] = T
    W = nwin * batch
    st = dict(starts=starts, ends=ends, mean=np.zeros(n), sd=np.zeros(n), eps=np.zeros(n),
              eps_inv=np.zeros(n), sdv=np.zeros(n), ir=np.zeros(n), emax=np.zeros(n))
    pad = np.concatenate([e_s, np.full(W, np.nan)])
    ypad = np.concatenate([y, np.full(W, np.nan)])
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        idx = starts[a:b, None] + np.arange(W)[None]
        Ew = pad[idx]
        Yw = ypad[idx]
        lens = ends[a:b] - starts[a:b]
        full = lens == W
        for sub, rows in ((full, np.flatnonzero(full)), (~full, np.flatnonzero(~full))):
            if not len(rows):
                continue
            if sub is full:
                E, Y = Ew[rows], Yw[rows]
                mean, sd = E.mean(1), E.std(1)
                ep = _eps_search_fast(E, mean, sd, buf, False)
                ei = _eps_search_fast(E, mean, sd, buf, True)
                ph, pl = np.percentile(Y, [95, 5], axis=1)
                sdv, emax = Y.std(1), E.max(1)
            else:                                                         # ragged tail windows
                mean, sd, ep, ei, ph, pl, sdv, emax = (np.zeros(len(rows)) for _ in range(8))
                for t, r in enumerate(rows):
                    L = lens[r]
                    E1, Y1 = Ew[r:r + 1, :L], Yw[r:r + 1, :L]
                    mean[t], sd[t] = E1.mean(), E1.std()
                    ep[t] = _eps_search(E1, mean[t:t + 1], sd[t:t + 1], buf, False)[0]
                    ei[t] = _eps_search(E1, mean[t:t + 1], sd[t:t + 1], buf, True)[0]
                    ph[t], pl[t] = np.percentile(Y1, [95, 5])
                    sdv[t], emax[t] = Y1.std(), E1.max()
            g = a + rows
            st["mean"][g], st["sd"][g], st["eps"][g], st["eps_inv"][g] = mean, sd, ep, ei
            st["sdv"][g], st["ir"][g], st["emax"][g] = sdv, ph - pl, emax
    return st


def _groups(i):
    if len(i) == 0:
        return []
    cut = np.flatnonzero(np.diff(i) != 1) + 1
    return [(g[0], g[-1]) for g in np.split(i, cut)]


def _prune(E_seq, e, non_anom_max, i_anom, p):
    if len(E_seq) == 0:
        return i_anom
    mx = np.array([e[a:b + 1].max() for a, b in E_seq])
    srt = np.append(np.sort(mx)[::-1], [non_anom_max])
    rm = []
    for k in range(len(srt) - 1):
        if (srt[k] - srt[k + 1]) / srt[k] < p:
            rm.extend(np.flatnonzero(mx == srt[k]).tolist())
        else:
            rm = []
    keep = [s for j, s in enumerate(E_seq) if j not in set(rm)]
    if not keep:
        return np.array([], int)
    kidx = np.concatenate([np.arange(a, b + 1) for a, b in keep])
    return i_anom[np.isin(i_anom, kidx)]


def telemanom_sliding(e_s, y, ps, st=None, batch=70, nwin=30, buf=100, min_err=0.0, ignore=500,
                      back=None):
    """Detector A for one channel of the evaluated period. Returns {p: (T,) uint8}.
    back: how far the buffer reaches BACK from each exceedance (ESA: back = buf = 100, i.e.
    symmetric). The smoothed error lags the real onset, which the backward reach compensates,
    but 100 rows (50 min on M1) also makes alarms start well before many events."""
    back = buf if back is None else back
    T = len(e_s)
    if st is None:
        st = telemanom_prepare(e_s, y, batch, nwin, buf)
    n = len(st["starts"])
    lens = st["ends"] - st["starts"]
    guard = ((st["sd"] > 0.05 * st["sdv"]) | (st["emax"] > 0.05 * st["ir"])) & \
            (lens >= buf) & (st["emax"] > min_err)
    # windows that could flag anything: a hit inside the decided region (+/- buffer)
    cand = []
    for w in np.flatnonzero(guard):
        a, b = st["starts"][w], st["ends"][w]
        lo = (ignore if w == 0 else (b - a) - batch) - buf
        seg = e_s[a + max(lo, 0):b]
        thr = 0.05 * st["ir"][w]
        if ((seg >= st["eps"][w]) & (seg > thr)).any():
            cand.append(w); continue
        inv = 2 * st["mean"][w] - seg
        if ((inv >= st["eps_inv"][w]) & (inv > thr)).any():
            cand.append(w)
    out = {}
    for p in ps:
        acc = np.zeros(T, bool)
        for w in cand:
            a, b = st["starts"][w], st["ends"][w]
            e = e_s[a:b].astype(np.float64)
            L = len(e)
            thr = 0.05 * st["ir"][w]
            res = []
            for inv, eps in ((False, st["eps"][w]), (True, st["eps_inv"][w])):
                ee = (2 * st["mean"][w] - e) if inv else e
                hit = np.flatnonzero((ee >= eps) & (ee > thr))
                if len(hit) == 0:
                    res.append((np.array([], int), [], -np.inf, ee)); continue
                hm = np.zeros(L, bool); hm[hit] = True
                i = np.flatnonzero(_dilate(hm, buf) if back == buf else _dilate_asym(hm, back, buf))
                i = i[i >= ignore] if w == 0 else i[i >= L - batch]
                if len(i) == 0:
                    res.append((np.array([], int), [], -np.inf, ee)); continue
                excl = np.zeros(L, bool)
                excl[acc[a:b]] = True
                excl[i] = True
                cand_idx = np.flatnonzero(~excl)
                nam = ee[cand_idx].max() if len(cand_idx) else -np.inf
                E_seq = [g for g in _groups(i) if g[0] != g[1]]
                res.append((i, E_seq, nam, ee))
            if len(res[0][0]) == 0 and len(res[1][0]) == 0:
                continue
            kept = [_prune(E_seq, ee, nam, i, p) for (i, E_seq, nam, ee) in res]
            if len(kept[0]) == 0 and len(kept[1]) == 0:
                continue
            ia = np.unique(np.concatenate(kept)).astype(int)
            acc[a + ia] = True
        o = np.zeros(T, np.uint8)
        for s, e in _groups(np.flatnonzero(acc)):
            if s != e:
                o[s:e + 1] = 1
        out[p] = o
    return out


# ------------------------------------------------------------------ Detector C and STD-k
def silence(count, L):
    """No data for >= L rows. L: scalar or one value per channel. The alarm may only reach back
    LATENCY rows before the moment the silence is confirmed, so for L > 70 the first L - 70
    rows of the gap stay unflagged (identical to before for L <= 70)."""
    C = count.shape[1]
    Ls = np.broadcast_to(np.asarray(L), (C,))
    out = np.zeros(count.shape, np.uint8)
    for c in range(C):
        r = runs_of(count[:, c] == 0)
        for s, e in r[(r[:, 1] - r[:, 0]) >= Ls[c]]:
            out[s + max(int(Ls[c]) - LATENCY, 0):e, c] = 1
    return out


def zero_run_cadence(count):
    """Median length of each channel's no-data runs (its normal sampling gap), >= 1."""
    C = count.shape[1]
    out = np.ones(C)
    for c in range(C):
        r = runs_of(count[:, c] == 0)
        if len(r):
            out[c] = max(float(np.median(r[:, 1] - r[:, 0])), 1.0)
    return out


def std_k(vals, count, mu, sd, k):
    return ((np.abs(vals - mu) / sd > k) & (count > 0)).astype(np.uint8)


# ------------------------------------------------------------------ post-processing
def hysteresis(core, low, back=LATENCY):
    """Keep every run of (low | core) that contains a core point; its start may reach back
    at most `back` rows before the first core point."""
    m = low | core
    out = np.zeros(core.shape, np.uint8)
    ci = np.flatnonzero(core)
    if not len(ci):
        return out
    r = runs_of(m)
    k = np.searchsorted(ci, r[:, 0])
    has = (k < len(ci))
    fc = np.where(has, ci[np.minimum(k, len(ci) - 1)], -1)
    has &= fc < r[:, 1]
    for (s, e), f in zip(r[has], fc[has]):
        out[max(s, f - back):e] = 1
    return out


def onset_refine(p, close, m_on, back=LATENCY):
    """Drop short leading pieces of a merged alarm (pieces < m_on rows that are followed
    within `close` rows by more alarm), but only within `back` rows of the main piece."""
    if m_on <= 1 or not p.any():
        return p
    r = runs_of(p)
    out = p.copy()
    L = r[:, 1] - r[:, 0]
    i = 0
    while i < len(r):
        j = i
        while j + 1 < len(r) and r[j + 1, 0] - r[j, 1] <= close:
            j += 1
        main = next((k for k in range(i, j + 1) if L[k] >= m_on), None)
        if main is not None and main > i:
            lim = r[main, 0] + m_on - back
            for k in range(i, main):
                if r[k, 0] >= lim:
                    out[r[k, 0]:r[k, 1]] = 0
        i = j + 1
    return out


def typed(P, ts_sec, fire_sec, window_s):
    """Remove alarm events (global OR runs) that start within window_s after a command."""
    g = P.max(1)
    out = P.copy()
    if not g.any() or len(fire_sec) == 0:
        return out, 0
    r = runs_of(g)
    t0 = ts_sec[r[:, 0]]
    j = np.searchsorted(fire_sec, t0, "right")
    last = np.where(j > 0, fire_sec[np.maximum(j - 1, 0)], -np.inf)
    cmdd = (t0 - last) <= window_s
    for s, e in r[cmdd]:
        out[s:e] = 0
    return out, int(cmdd.sum())


# ------------------------------------------------------------------ drift (new)
def drift_stat(vals, count, R, short_blocks, long_blocks, fit_end, min_frac=0.25):
    """Slow-drift / level-change statistic per channel, strictly causal.

    Works on 6-hour blocks (R rows). For rows of block k:
        short = mean channel value over blocks k-s .. k-1      (e.g. the last day)
        long  = mean over the L blocks before that              (e.g. the 30 days before)
        D     = |short - long| / robust scale of (short - long) over the fit period
    Only completed blocks before k are used (lag <= 6 h, no look-ahead). A slow drift that a
    one-step forecaster tracks (small error) still grows D. Rows with no data -> 0.
    """
    T, C = vals.shape
    nb = int(np.ceil(T / R))
    pad = nb * R - T
    ok = np.asarray(count) > 0
    v = np.where(ok, np.asarray(vals, np.float64), 0.0)
    if pad:
        v = np.vstack([v, np.zeros((pad, C))]); okp = np.vstack([ok, np.zeros((pad, C), bool)])
    else:
        okp = ok
    S = v.reshape(nb, R, C).sum(1)
    N = okp.reshape(nb, R, C).sum(1)
    m = np.where(N >= max(int(R * min_frac), 1), S / np.maximum(N, 1), np.nan)
    for c in range(C):                                   # carry the last valid block mean
        col = pd.Series(m[:, c]).ffill().bfill().to_numpy()
        m[:, c] = col if np.isfinite(col).any() else 0.0
    cs = np.vstack([np.zeros((1, C)), np.cumsum(m, axis=0)])
    k = np.arange(nb)
    s, L = short_blocks, long_blocks
    valid = k - s - L >= 0
    a, b = np.clip(k - s, 0, nb), k
    lo_a, lo_b = np.clip(k - s - L, 0, nb), np.clip(k - s, 0, nb)
    short = (cs[b] - cs[a]) / max(s, 1)
    long_ = (cs[lo_b] - cs[lo_a]) / max(L, 1)
    diff = np.where(valid[:, None], short - long_, np.nan)
    kf = max(min(fit_end // R, nb), 1)
    ref = diff[:kf]
    med = np.nanmedian(ref, axis=0) if np.isfinite(ref).any() else np.zeros(C)
    mad = np.nanmedian(np.abs(ref - med), axis=0) if np.isfinite(ref).any() else np.ones(C)
    scale = np.maximum(1.4826 * np.nan_to_num(mad, nan=1.0), 1e-9)
    Db = np.nan_to_num(np.abs(diff - np.nan_to_num(med)) / scale, nan=0.0).astype(np.float32)
    D = np.repeat(Db, R, axis=0)[:T]
    D[~ok] = 0.0
    return D


# ------------------------------------------------------------------ out-of-limit baseline (new)
def ool_limits(vals, count, nominal, q=None):
    """Fixed per-channel limits from NOMINAL training data (rows not labelled anomalous), as
    an operator would set them from normal operation. q=None -> min / max; q -> (q, 1-q)."""
    C = vals.shape[1]
    lo, hi = np.zeros(C), np.zeros(C)
    for c in range(C):
        x = np.asarray(vals[:, c])[(np.asarray(count[:, c]) > 0) & nominal[:, c]]
        if not len(x):
            lo[c], hi[c] = -np.inf, np.inf
        elif q is None:
            lo[c], hi[c] = x.min(), x.max()
        else:
            lo[c], hi[c] = np.quantile(x, [q, 1 - q])
    return lo, hi


def ool_mask(vals, count, lo, hi):
    v = np.asarray(vals)
    return (((v < lo) | (v > hi)) & (np.asarray(count) > 0)).astype(np.uint8)
