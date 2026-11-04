"""Pre-flight check for one subsystem model on THIS machine (about a minute).

  NR_SET=sub NR_SUB=M1_S6 python -m core.subcheck

1. imports, data files present and consistent (shapes, timestamps, channels)
2. ESA's scorer gives a perfect score to a perfect prediction on this model's targets
3. a few training steps on this device -> seconds per step -> time per training job
"""
import json
import os
import time

import numpy as np
import torch

from .config import MISSIONS, MODEL, TRAIN, SUB, OUT
from .data import load_split, label_mask, row_of, row_end, command_times
from .features import CmdFeats, channel_tensor
from .model import GDN2, n_params
from .score import Scorer, fmt
from .train import pick_device, seg_bs, make_batch, loss_fn, to


def main():
    dev = pick_device()
    name = torch.cuda.get_device_name(0) if dev.type == "cuda" else str(dev)
    print(f"torch {torch.__version__} | numpy {np.__version__} | device {name}", flush=True)
    mk = next(iter(MISSIONS))
    m = MISSIONS[mk]
    for sp in ("train", "test"):
        d = load_split(mk, sp)
        assert d["vals"].shape == d["count"].shape == d["tsince"].shape == (len(d["ts"]), len(m["chs"])), sp
        assert np.all(np.diff(d["ts"][:10_000].astype(np.int64)) > 0)
        v = np.asarray(d["vals"][-1000:])
        assert np.isfinite(v).all(), f"{sp}: non-finite values"
    cmds = command_times(mk, TRAIN["tc_min_priority"])
    print(f"  {SUB} ({mk}): data ok, {len(m['chs'])} inputs ({len(m['targets'])} targets), "
          f"{len(cmds)} telecommands (priority >= {TRAIN['tc_min_priority']})", flush=True)
    d = load_split(mk, "train")
    ev = {}                                   # labelled events per period (anomalies + rare events)
    for f in ("A", "B"):
        s0, e0 = m["folds"][f]
        ev[f] = Scorer(mk, d["ts"][row_of(d["ts"], s0):row_end(d["ts"], e0)]).n_events
    ev["test"] = Scorer(mk, load_split(mk, "test")["ts"]).n_events
    os.makedirs(f"{OUT}/{mk}", exist_ok=True)
    json.dump(ev, open(f"{OUT}/{mk}/fold_events.json", "w"))
    print(f"  labelled events (anomalies + rare events / anomalies): fold A {ev['A']['sel']}/{ev['A']['anom']}, "
          f"fold B {ev['B']['sel']}/{ev['B']['anom']}, test {ev['test']['sel']}/{ev['test']['anom']}", flush=True)
    a = row_of(d["ts"], m["folds"]["B"][0])
    ts = d["ts"][a:]
    lm = label_mask(mk, ts, chs=m["targets"])
    S = Scorer(mk, ts)
    if S.n_events["sel"]:
        r = S.score(lm.astype(np.uint8))
        assert r["global_EW_recall"] == 1.0 and r["chan_channel_F0.50"] == 1.0, r
        print(f"  scorer ok: perfect prediction -> "
              f"{fmt(r, ['global_EW_recall', 'chan_channel_F0.50', 'global_alarming_precision'])}", flush=True)
    else:
        print("  scorer: no labelled events on these targets in fold B (nothing to check there)", flush=True)
    n = 300_000
    v = np.asarray(d["vals"][:n])
    X = channel_tensor(v, d["count"][:n], d["tsince"][:n], v.mean(0), np.maximum(v.std(0), m["sd_floor"]) + 1e-8)
    cmd = CmdFeats(d["ts"][:n], cmds)
    model = GDN2(len(m["chs"]), 0, cmd.n, **MODEL).to(dev)
    seg, bs = seg_bs(model.rf)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    rng = np.random.default_rng(0)
    ts_ = []
    for i in range(6):
        xs, ys, ws, cf = make_batch(X, cmd, None, rng.choice(n - seg - 1, bs, replace=False), seg, model.rf, X.shape[1])
        t = [None if q is None else torch.from_numpy(q) for q in (xs, ys, ws, cf)]
        t0 = time.time()
        opt.zero_grad()
        loss = loss_fn(model.forward_segments(to(dev, t[0]), to(dev, t[3])), to(dev, t[1]), None)
        loss.backward()
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()
        ts_.append(time.time() - t0)
        assert np.isfinite(loss.item())
    s = float(np.median(ts_[2:]))
    full = s * TRAIN["epochs"] * TRAIN["batches"] / 60
    print(f"  train step ok: {n_params(model):,} params, {s:.3f} s/step -> at most {full:.0f} min per "
          f"training job ({TRAIN['epochs']} x {TRAIN['batches']} steps; early stopping can end it sooner)",
          flush=True)
    print("CHECK OK", flush=True)


if __name__ == "__main__":
    main()
