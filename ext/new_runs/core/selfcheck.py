"""Pre-flight check + speed benchmark on THIS machine (about 1-2 minutes).

  python -m core.selfcheck

1. every module and ESA's metric package import; data files present and consistent
2. the scorer gives a perfect score to a perfect prediction
3. the combined model trains on this GPU for both missions (forward + backward)
4. measured seconds per training step -> estimated time per training job and for the run
"""
import json
import os
import time

import numpy as np
import torch

from .config import DATA, MISSIONS, MODEL, TRAIN, OUT
from .data import load_split, label_mask, row_of, row_end, command_times
from .features import CmdFeats, channel_tensor
from .model import GDN2, n_params
from .score import Scorer, fmt
from .train import pick_device, seg_bs, make_batch, loss_fn, to, series_errors


def main():
    dev = pick_device()
    name = torch.cuda.get_device_name(0) if dev.type == "cuda" else str(dev)
    print(f"torch {torch.__version__} | device {name} | numpy {np.__version__}")
    for mk in MISSIONS:
        for sp in ("train", "test"):
            d = load_split(mk, sp)
            assert d["vals"].shape == d["count"].shape == d["tsince"].shape
            assert len(d["ts"]) == len(d["vals"]) and np.all(np.diff(d["ts"].astype(np.int64)[:10_000]) > 0)
        print(f"  {mk}: data ok ({len(d['ts']):,} test rows, {len(d['chs'])} input / "
              f"{len(MISSIONS[mk]['targets'])} scored channels), {len(command_times(mk, TRAIN['tc_min_priority']))} commands")
    # scorer sanity: labels as predictions on M1's fold B -> recall 1, precision ~1
    d = load_split("M1", "train")
    a = row_of(d["ts"], MISSIONS["M1"]["folds"]["B"][0])
    ts = d["ts"][a:]
    r = Scorer("M1", ts).score(label_mask("M1", ts, chs=MISSIONS["M1"]["targets"]).astype(np.uint8))
    assert r["global_EW_recall"] == 1.0 and r["chan_channel_F0.50"] == 1.0, r
    print(f"  scorer ok: perfect prediction -> {fmt(r, ['global_EW_recall', 'chan_channel_F0.50', 'global_alarming_precision'])}")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    est, bench = {}, {}
    for mk in MISSIONS:
        d = load_split(mk, "train")
        n = 400_000
        v = np.asarray(d["vals"][:n])
        X = channel_tensor(v, d["count"][:n], d["tsince"][:n], v.mean(0), v.std(0) + 1e-8)
        cmd = CmdFeats(d["ts"][:n], command_times(mk, TRAIN["tc_min_priority"]))
        cmd.cache = cmd.at(np.arange(n), np.float16)       # as in the real run (cached features)
        model = GDN2(len(d["chs"]), 0, cmd.n, **MODEL).to(dev)
        seg, bs = seg_bs(model.rf)
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        rng = np.random.default_rng(0)
        batches = [make_batch(X, cmd, None, rng.choice(n - seg - 1, bs, replace=False), seg, model.rf, X.shape[1])
                   for _ in range(4)]
        tt = []
        for i in range(14):
            xs, ys, ws, cf = (None if b is None else torch.from_numpy(b) for b in batches[i % 4])
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            opt.zero_grad()
            loss = loss_fn(model.forward_segments(to(dev, xs), to(dev, cf)), to(dev, ys), None)
            loss.backward(); opt.step()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            elif dev.type == "mps":
                torch.mps.synchronize()
            tt.append(time.time() - t0)
        assert np.isfinite(loss.item())
        step = float(np.median(tt[4:]))
        t0 = time.time()
        series_errors(model, X, cmd, dev, 0, n)                 # error-pass throughput
        rps = (n - model.rf) / (time.time() - t0)
        tr_rows = len(d["ts"]); te_rows = len(load_split(mk, "test")["ts"])
        f = MISSIONS[mk]["folds"]
        err_rows = row_end(d["ts"], f["A"][1]) + row_end(d["ts"], f["B"][1]) + tr_rows + te_rows
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else float("nan")
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        bench[mk] = dict(step_ms=step * 1000, peak_gb=mem if mem == mem else 4.0, err_rows_per_s=rps,
                         n_inputs=len(d["chs"]), n_scored=len(MISSIONS[mk]["targets"]), sparse=bool(model.sparse))
        steps = TRAIN["epochs"] * TRAIN["batches"]
        est[mk] = step * steps / 60
        print(f"  {mk}: {n_params(model):,} params, rf {model.rf}, batch {bs}x{seg}: "
              f"{step * 1000:.0f} ms/step (alone), peak GPU mem {mem:.1f} GB "
              f"-> <= {est[mk]:.0f} min per training job if it runs alone; error pass "
              f"{rps / 1000:.0f}k rows/s -> {err_rows / rps / 60:.0f} min over its 3 jobs")
        del model, opt
    os.makedirs(OUT, exist_ok=True)
    json.dump(bench, open(f"{OUT}/bench.json", "w"), indent=1)
    print(f"\nall checks passed. Per training job (30 epochs, alone on a GPU): "
          + ", ".join(f"{k} <= {v:.0f} min" for k, v in est.items())
          + ". Several jobs share a GPU at somewhat lower speed each; run_all prints live ETAs.")


if __name__ == "__main__":
    main()
