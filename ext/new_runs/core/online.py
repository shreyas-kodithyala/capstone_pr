"""Deployment with an operator in the loop (lightweight M1): past-only periodic fine-tuning.

A model trained on the training split is deployed at d0 and monitors month by month. At the
end of each period:
  1. the frozen detector stack (frozen.json, chosen on the folds) turns that period's forecast
     errors into alarms -- computed from data up to the period end (+ LATENCY rows) only;
  2. a simulated operator reviews every alarm raised (alarms only: missed events stay unseen):
       anomaly / rare_event  the alarm overlaps a labelled event  -> confirmed
       comm_gap              it overlaps only a communication-gap label
       commanded             no label, starts <= 1 h after a priority-3 telecommand (typing rule)
       false_alarm           no label
  3. the model is fine-tuned on everything seen so far (training years + deployment to date):
       variant nofb   no feedback: every deployment row is "normal"
               alarm  confirmed events' rows left out of the loss; false alarms kept as normal
                      (optionally up-weighted) -- the realistic main variant
               full   every labelled row seen so far left out (the mission's anomaly log)
     warm start, Adam (fresh), low lr, batches half from recent data / half replay of all
     history, a 7-day held-out tail as validation; the update is kept only if it forecasts
     that tail better than the current model (and the old training validation not >10 % worse);
  4. the (possibly) updated model forecasts the next period, which it has never seen.
Labels never enter the model as inputs; they only remove rows from the loss.

Rehearsal: the same loop inside the training years ('foldA' / 'foldB': the fold's model
deployed over the fold) chooses the fine-tuning setting without touching the test years.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .config import ROOT, MISSIONS, LATENCY
from .data import load_split, label_mask, events, row_of, row_end, fire_seconds, target_idx, \
    command_times
from .features import channel_tensor, CmdFeats
from .model import GDN2
from .train import seg_bs, make_batch, loss_fn, to, series_errors
from .period import Period, assemble, tele_key
from .score import Scorer, SEL, SEL_ANOM
from . import detect as D
from . import train as _T

_budget0 = _T._budget


def _budget(dev):
    """Laptop: smaller error-pass chunks on Apple GPUs (unified memory is the laptop's RAM)."""
    return int(2.5e8) if dev.type == "mps" else _budget0(dev)


_T._budget = _budget


def free_gpu(dev):
    if dev.type == "mps":
        torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()

MK = "M1"
BASE = os.environ.get("NR_BASE", os.path.join(os.path.dirname(ROOT), "out 2"))   # trained light models
OUTO = os.environ.get("NR_OUT_ONLINE", f"{ROOT}/out_online")
BITS = {"A": 1, "B": 2, "C": 4, "STD": 8, "drift": 16}          # which detector fired, per row/channel
FT = dict(lr=1e-4, steps=300, every=50, recent_days=90, recent_frac=0.5, hold_days=7,
          fa_weight=1, sp=0.0, replay_tol=0.10, cadence=1, warm_days=3)
CONFIRMED = ("anomaly", "rare_event")


def f16(e):
    """As the training pipeline stores errors (clip: 1e8 once overflowed float16)."""
    assert np.isfinite(e).all(), f"{int((~np.isfinite(e)).any(1).sum())} rows of non-finite forecast errors"
    return np.clip(e, 0, 6.0e4).astype(np.float16)


class Deployment:
    """Rows, model inputs, labels and the base model of one deployment.
    test : final model (training split), deployed 2007-2014 (train + test rows concatenated,
           the shared boundary timestamp kept once, as core.evaluate does)
    foldA / foldB : that fold's model, deployed over the fold (training years only)."""

    def __init__(self, name, mk=MK):
        self.name, self.mk = name, mk
        m = MISSIONS[mk]
        tr = load_split(mk, "train")
        if name == "test":
            te = load_split(mk, "test")
            cat = lambda a, b: np.concatenate([np.asarray(a[:-1]), np.asarray(b)])   # noqa: E731
            self.ts = cat(tr["ts"], te["ts"])
            vals, count, tsince = (cat(tr[k], te[k]) for k in ("vals", "count", "tsince"))
            self.d0, self.d1, job = len(tr["ts"]) - 1, len(self.ts), "final"
        else:
            s, e = m["folds"][name[-1]]
            self.d0, self.d1, job = row_of(tr["ts"], s), row_end(tr["ts"], e), name
            self.ts = tr["ts"][:self.d1]
            vals, count, tsince = (np.asarray(tr[k][:self.d1]) for k in ("vals", "count", "tsince"))
        self.base = f"{BASE}/{mk}/{job}_s0"
        self.meta = json.load(open(f"{self.base}/meta.json"))
        assert self.meta["fit_end"] == (len(tr["ts"]) if name == "test" else self.d0), "base model boundary"
        self.frz = json.load(open(f"{BASE}/{mk}/frozen.json"))
        self.chs, self.tix = m["chs"], target_idx(mk)
        self.rf = self.meta["rf"]
        self.mu, self.sd = np.asarray(self.meta["mu"]), np.asarray(self.meta["sd"])
        self.X = channel_tensor(vals, count, tsince, self.mu, self.sd)
        del tsince
        self.cmds = command_times(mk, self.meta["train"]["tc_min_priority"])
        self.cmd = CmdFeats(self.ts, self.cmds)
        allc = list(self.tix) == list(range(len(self.chs)))     # lightweight: every input is scored
        self.vals_t = np.asarray(vals, np.float32) if allc else np.ascontiguousarray(np.asarray(vals)[:, self.tix], np.float32)
        self.count_t = np.asarray(count) if allc else np.ascontiguousarray(np.asarray(count)[:, self.tix])
        del vals, count
        self.lab = label_mask(mk, self.ts)                       # (T, C) Anomaly / Rare Event rows
        self.grid = m["grid"]
        tw, tp = map(int, self.frz["typing"].split("_"))
        self.type_window, self.fire = tw, fire_seconds(mk, tp)
        self.sec = self.ts.astype("datetime64[ns]").astype(np.int64) / 1e9
        # every labelled event touching these channels (gaps included: an alarm on one is
        # explained, neither a hit nor a false alarm -- as in ESA's scoring)
        ev = events(mk, ("Anomaly", "Rare Event", "Communication Gap"))
        col = {c: i for i, c in enumerate(self.chs)}
        self.ev = {}
        for eid, g in ev.groupby("ID"):
            rows = []
            for _, r in g.iterrows():
                a = max(int(np.searchsorted(self.ts, np.datetime64(r.StartTime), "right")) - 1, 0)
                b = int(np.searchsorted(self.ts, np.datetime64(r.EndTime), "right"))
                rows.append((col[r.Channel], a, max(b, a + 1)))
            self.ev[eid] = dict(cat=g.Category.iloc[0], start=np.datetime64(g.StartTime.min()),
                                end=np.datetime64(g.EndTime.max()), rows=rows)
        self._ev_ids = list(self.ev)
        self._ev_s = np.array([self.ev[i]["start"] for i in self._ev_ids], "datetime64[ns]")
        self._ev_e = np.array([self.ev[i]["end"] for i in self._ev_ids], "datetime64[ns]")

    def model(self, path=None):
        mdl = GDN2(len(self.chs), 0, self.cmd.n, **self.meta["model"])
        mdl.load_state_dict(torch.load(path or f"{self.base}/model.pt", map_location="cpu"))
        return mdl

    def periods(self, cadence):
        """Row boundaries of the update periods (calendar months x cadence)."""
        t0, t1 = pd.Timestamp(self.ts[self.d0]), pd.Timestamp(self.ts[self.d1 - 1])
        first = (t0 + pd.offsets.MonthBegin(0)).normalize()
        dates = pd.date_range(first, t1, freq=f"{cadence}MS")
        rows = np.searchsorted(self.ts, dates.values.astype("datetime64[ns]"))
        rows = rows[(rows > self.d0) & (rows < self.d1 - self.rows(1))]     # no stub period at the end
        return np.unique(np.concatenate([[self.d0], rows, [self.d1]]))

    def rows(self, days):
        return int(days * 86400 / self.grid)

    def overlapping(self, a, b):
        """Event IDs whose [start, end] overlaps rows [a, b)."""
        ta, tb = self.ts[a], self.ts[min(b, self.d1) - 1]
        hit = (self._ev_s <= tb) & (self._ev_e >= ta)
        return [self._ev_ids[i] for i in np.flatnonzero(hit)]

    def last_command(self, a):
        """(seconds since the last typing-priority command before row a) or None."""
        j = np.searchsorted(self.fire, self.sec[a], "right")
        return None if j == 0 else float(self.sec[a] - self.fire[j - 1])


# ------------------------------------------------------------------ errors and detection
def base_stream(dep, dev):
    """The base model's errors exactly as the frozen pipeline computed them: the training
    split and the test split each in their own pass (first RF rows of a pass are 0)."""
    mdl = dep.model().to(dev)
    if dep.name != "test":
        return f16(series_errors(mdl, dep.X, dep.cmd, dev, 0, dep.d1)[:, dep.tix])
    etr = series_errors(mdl, dep.X[:dep.d0 + 1], dep.cmd, dev, 0, dep.d0 + 1)[:dep.d0, dep.tix]
    cmd_te = CmdFeats(dep.ts[dep.d0:], dep.cmds)
    ete = series_errors(mdl, dep.X[dep.d0:], cmd_te, dev)[:, dep.tix]
    free_gpu(dev)
    return f16(np.concatenate([etr, ete]))


_G = {}


def _tele(a):
    c, k = a
    return c, _G["P"].telemanom(c, k[1], {k})[k]


def detect(dep, err, e0, e1, workers=1):
    """Frozen detector stack on rows [e0, e1) using errors / values of rows < e1 only.
    -> (alarms (n, C) uint8, detector bits (n, C) uint8)."""
    cfg = dep.frz["detector"]
    P = Period(dep.mk, "online", err[:e1], dep.vals_t[:e1], dep.count_t[:e1], dep.ts[:e1],
               dep.mu[dep.tix], dep.sd[dep.tix], dep.rf, dep.d0, e0, e1)
    if cfg.get("drift"):
        P.build_drift((cfg["drift"][0],))
    if cfg["thr"] != "off":
        P.zs[(cfg["W"], cfg["spans"])] = P.z_and_thresholds(cfg["W"], cfg["spans"], lazy=True)
    if cfg.get("tele"):
        k = tele_key(cfg["tele"])
        T = np.zeros((P.n, P.C), np.uint8)
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import get_context
            _G["P"] = P
            with ProcessPoolExecutor(min(workers, P.C), mp_context=get_context("fork")) as pool:
                for c, r in pool.map(_tele, [(c, k) for c in range(P.C)]):
                    T[:, c] = r
            _G.clear()
        else:
            for c in range(P.C):
                T[:, c] = P.telemanom(c, k[1], {k})[k]
        P.tele[k] = T
    P.release_history()
    X, parts = assemble(P, cfg, parts=True)
    bits = np.zeros_like(X)
    for name, v in parts.items():
        bits |= (v.astype(np.uint8) * BITS[name])
    return X, bits


def warm_start(dep, s, warm):
    """Detection start for a period starting at row s: `warm` rows earlier, on the 70-row
    batch grid of a full-deployment run (Telemanom windows line up with it)."""
    if s - warm <= dep.d0:
        return dep.d0
    return dep.d0 + ((s - warm - dep.d0) // LATENCY) * LATENCY


# ------------------------------------------------------------------ the simulated operator
def review(dep, alarms, bits, upto, since, recs):
    """Verdicts for alarm events (global runs over channels) that end after row `since`
    (all rows absolute; alarms indexed from d0). Updates recs {start row: record}."""
    g = alarms[:upto - dep.d0].max(1)
    for s, e in D.runs_of(g):
        s, e = s + dep.d0, e + dep.d0
        if e <= since:
            continue
        ids = dep.overlapping(s, e)
        cats = {dep.ev[i]["cat"] for i in ids}
        dt = dep.last_command(s)
        commanded = dt is not None and dt <= dep.type_window
        if "Anomaly" in cats:
            v = "anomaly"
        elif "Rare Event" in cats:
            v = "rare_event"
        elif cats:
            v = "comm_gap"
        elif commanded:
            v = "commanded"
        else:
            v = "false_alarm"
        a, b = s - dep.d0, e - dep.d0
        chans = [dep.chs[dep.tix[c]] for c in range(alarms.shape[1]) if alarms[a:b, c].any()]
        bb = int(np.bitwise_or.reduce(bits[a:b].ravel()))
        recs[int(s)] = dict(start=int(s), end=int(e), t_start=str(dep.ts[s])[:19], t_end=str(dep.ts[e - 1])[:19],
                            minutes=round((e - s) * dep.grid / 60, 1), channels=chans,
                            detectors=[k for k, m in BITS.items() if bb & m], commanded=bool(commanded),
                            since_command_s=None if dt is None else round(dt), verdict=v,
                            events=[i for i in ids if dep.ev[i]["cat"] != "Communication Gap"],
                            gap_events=[i for i in ids if dep.ev[i]["cat"] == "Communication Gap"])
    return recs


def loss_weights(dep, variant, recs, upto, fa_weight):
    """(T, C) uint8 loss weights for fine-tuning on rows < upto."""
    W = np.ones((dep.d1, len(dep.chs)), np.uint8)
    W[:dep.d0] = ~dep.lab[:dep.d0]                       # the base model's own training rule
    if variant == "full":
        W[dep.d0:upto] = ~dep.lab[dep.d0:upto]
    elif variant == "alarm":
        if fa_weight != 1:                               # false alarms: "this is normal", louder
            for r in recs.values():
                if r["verdict"] in ("false_alarm", "commanded"):
                    W[r["start"]:min(r["end"], upto)] = fa_weight
        for r in recs.values():                          # confirmed events: out of the loss
            if r["verdict"] in CONFIRMED:
                for eid in r["events"]:
                    for c, a, b in dep.ev[eid]["rows"]:
                        W[a:b, c] = 0
    W[upto:] = 0                                         # never used (training stops at upto)
    return W


# ------------------------------------------------------------------ fine-tuning
def _val(model, dep, W, starts, seg, bs, dev):
    if not len(starts):
        return float("nan")
    model.eval()
    tot = n = 0.0
    with torch.no_grad():
        for i in range(0, len(starts), bs):
            xs, ys, ws, cf = make_batch(dep.X, dep.cmd, W, starts[i:i + bs], seg, model.rf, len(dep.chs))
            t = [None if a is None else torch.from_numpy(a) for a in (xs, ys, ws, cf)]
            tot += loss_fn(model.forward_segments(to(dev, t[0]), to(dev, t[3])), to(dev, t[1]),
                           to(dev, t[2])).item() * len(xs)
            n += len(xs)
    return tot / n


def finetune(model, dep, W, upto, ft, dev, seed, anchor=None):
    """Warm-start fine-tune on rows < upto. Returns (model, info). The kept weights are the
    checkpoint (step 0 = no change included) with the lowest loss on the held-out last
    hold_days, among those whose loss on the base model's own validation stretch is at most
    replay_tol worse than before."""
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rf = model.rf
    seg, bs = seg_bs(rf)
    tr_end = upto - dep.rows(ft["hold_days"])
    hi = tr_end - seg - 1                                   # window starts: [0, hi) history,
    rlo = max(0, tr_end - dep.rows(ft["recent_days"]))      # [rlo, hi) recent
    vr = np.linspace(tr_end, upto - seg - 1, 24).astype(int)
    es = int(0.85 * dep.d0)                                 # the base model's own validation stretch
    vo = np.linspace(es, dep.d0 - seg - 1, 24).astype(int)
    nr = int(round(bs * ft["recent_frac"]))
    p0 = {k: v.detach().clone() for k, v in (anchor or model).state_dict().items()
          if v.dtype.is_floating_point} if ft["sp"] > 0 else None
    model.to(dev)
    v_rec0, v_old0 = _val(model, dep, W, vr, seg, bs, dev), _val(model, dep, W, vo, seg, bs, dev)
    best = (v_rec0, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, v_old0)
    opt = torch.optim.Adam(model.parameters(), lr=ft["lr"])
    trace = [(0, v_rec0, v_old0)]
    tl = 0.0
    for step in range(1, ft["steps"] + 1):
        model.train()
        starts = np.concatenate([rng.integers(rlo, hi, nr), rng.integers(0, hi, bs - nr)])
        xs, ys, ws, cf = make_batch(dep.X, dep.cmd, W, starts, seg, rf, len(dep.chs))
        t = [None if a is None else torch.from_numpy(a) for a in (xs, ys, ws, cf)]
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model.forward_segments(to(dev, t[0]), to(dev, t[3])), to(dev, t[1]), to(dev, t[2]))
        if p0 is not None:
            sp = sum(((p - p0[k].to(dev)) ** 2).sum() for k, p in model.named_parameters() if k in p0)
            loss = loss + ft["sp"] * sp
        if not torch.isfinite(loss):                      # never let one bad batch poison the weights
            opt.zero_grad(set_to_none=True)
            continue
        loss.backward()
        opt.step()
        tl += loss.item()
        if step % ft["every"] == 0 or step == ft["steps"]:
            vrec, vold = _val(model, dep, W, vr, seg, bs, dev), _val(model, dep, W, vo, seg, bs, dev)
            trace.append((step, vrec, vold))
            if vrec < best[0] and vold <= v_old0 * (1 + ft["replay_tol"]):
                best = (vrec, step, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, vold)
    model.load_state_dict(best[2])
    model.eval()
    del opt
    free_gpu(dev)
    return model, dict(val_recent_before=v_rec0, val_recent_after=best[0], val_old_before=v_old0,
                       val_old_after=best[3], kept_step=best[1], accepted=best[1] > 0,
                       train_loss=tl / ft["steps"], trace=trace, secs=time.time() - t0)


# ------------------------------------------------------------------ scoring
def score_global(dep, alarms, upto):
    """Event-wise metrics (ESA headline selection) on deployment rows [d0, upto): events fully
    inside that range only. alarms indexed from d0."""
    S = Scorer(dep.mk, dep.ts[dep.d0:upto])
    if not S.n_events["sel"]:
        return dict(n_events=0)
    r = S.score(alarms[:upto - dep.d0], SEL, full=False)
    r["n_events"] = S.n_events["sel"]
    return r


def score_full(dep, alarms, a=None, b=None):
    """All five aspects (headline, anomaly-only, uncommanded anomaly-only) on rows [a, b)."""
    a, b = a or dep.d0, b or dep.d1
    X = alarms[a - dep.d0:b - dep.d0]
    S = Scorer(dep.mk, dep.ts[a:b])
    Xt, n_cmd = D.typed(X, dep.sec[a:b], dep.fire, dep.type_window)
    return dict(headline=S.score(X, SEL), anom=S.score(X, SEL_ANOM), anom_typed=S.score(Xt, SEL_ANOM),
                n_events=S.n_events, n_commanded=n_cmd, alarm_time=float(X.max(1).mean()))


def caught(dep, alarms, a, b):
    """{event ID: first alarm row or None} for Anomaly / Rare Event events fully in [a, b)."""
    out = {}
    for eid, ev in dep.ev.items():
        if ev["cat"] == "Communication Gap":
            continue
        r0, r1 = min(r[1] for r in ev["rows"]), max(r[2] for r in ev["rows"])
        if r0 < a or r1 > b or ev["start"] < dep.ts[a] or ev["end"] > dep.ts[b - 1]:
            continue
        first = None
        for c, s, e in ev["rows"]:
            if c not in dep.tix:
                continue
            h = np.flatnonzero(alarms[s - dep.d0:e - dep.d0, dep.tix.index(c)])
            if len(h):
                first = s + int(h[0]) if first is None else min(first, s + int(h[0]))
        out[eid] = first
    return out
