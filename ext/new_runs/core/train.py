"""Train one model and write its forecast errors.

  python -m core.train --mission M1 --job foldA|foldB|final --seed 0 [--smoke]

  foldA / foldB : trained on everything before the fold's start date; errors are written up to
                  the fold's end date. Those unseen rows are where detector settings get chosen.
  final         : trained on the whole training split; errors for train AND test.

Recipe (identical for every job): Adam 1e-3, masked MSE (labelled rows weight 0), 300
batches/epoch, up to 30 epochs, early stopping on the last 15 % of the training period with
patience 8. Segment length 2 x RF; batch size keeps ~6,200 supervised predictions per step.

Speed: batches are built by a background thread into pinned memory and copied asynchronously;
TF32 is on for training (Ada / Ampere tensor cores), OFF for the error pass so the errors are
full fp32; the error pass uses chunk sizes derived from free GPU memory.
"""
import argparse
import json
import os
import queue
import threading
import time

import numpy as np
import torch

from .config import MISSIONS, MODEL, TRAIN, SMOKE_TRAIN, INIT, TRAIN_DAYS, TRAIN_FROM_END, FREEZE, job_dir
from .data import load_split, row_of, row_end, label_mask, target_idx
from .features import build_inputs
from .model import GDN2, n_params
from . import graph as G


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seg_bs(rf):
    seg = max(512, 2 * rf)
    return seg, max(2, int(round(24 * 259 / (seg - rf))))


def make_batch(X, cmd, W, starts, seg, rf, n_ch, cmd_dtype=np.float32):
    idx = starts[:, None] + np.arange(seg)[None]
    xs = np.ascontiguousarray(np.transpose(X[idx], (0, 2, 1, 3)))          # (B, N, seg, 3)
    ys = X[idx[:, rf:], :n_ch, 0]                                            # (B, seg-rf, C)
    ws = None if W is None else W[idx[:, rf:]].astype(np.float32)
    cf = cmd.at(idx[:, rf - 1:seg - 1], cmd_dtype) if cmd is not None else None
    return xs, ys, ws, cf


class Loader:
    """Background producer: random training windows -> pinned CPU tensors."""

    def __init__(self, X, cmd, W, starts, bs, seg, rf, n_ch, n_batches, seed, pin, depth=6):
        self.q = queue.Queue(maxsize=depth)
        rng = np.random.default_rng(seed)
        cdt = np.float16 if (cmd is not None and cmd.cache is not None) else np.float32

        def run():
            for _ in range(n_batches):
                arrs = make_batch(X, cmd, W, rng.choice(starts, bs, replace=False), seg, rf, n_ch, cdt)
                ts = [None if a is None else torch.from_numpy(a) for a in arrs]
                if pin:
                    ts = [None if t is None else t.pin_memory() for t in ts]
                self.q.put(ts)

        self.n = n_batches
        self.th = threading.Thread(target=run, daemon=True)
        self.th.start()

    def __iter__(self):
        for _ in range(self.n):
            yield self.q.get()


def to(dev, t):
    # non_blocking only on CUDA (pinned batches): on Apple GPUs an asynchronous copy from a
    # temporary CPU buffer can read it after it is freed -> NaN / garbage forecast errors
    return None if t is None else t.to(dev, non_blocking=dev.type == "cuda").float()


def loss_fn(pred, y, w):
    se = (pred - y) ** 2
    return se.mean() if w is None else (se * w).sum() / w.sum().clamp_min(1.0)


def fit(X, cmd, W, n_ch, n_cmd, fit_end, seed, tr, dev, log, progress):
    torch.manual_seed(seed); np.random.seed(seed)
    model = GDN2(n_ch, 0, n_cmd, **MODEL).to(dev)
    if INIT:     # transfer: shared weights from another model; channel / command embeddings new
        src = torch.load(INIT, map_location="cpu")
        shared = {k: v for k, v in src.items() if k in model.shared_state()}
        miss = model.load_state_dict(shared, strict=False)
        assert all(k.startswith(("embedding.", "cmd_emb.")) for k in miss.missing_keys), miss
        log(f"  init from {INIT}: {len(shared)} shared tensors loaded, embeddings fresh")
    if FREEZE == "tcn":      # transfer with the temporal encoder fixed: only graph / heads / embeddings learn
        for p in model.tcn.parameters():
            p.requires_grad_(False)
        log(f"  TCN frozen: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")
    rf = model.rf
    seg, bs = seg_bs(rf)
    es = int(0.85 * fit_end)
    starts = np.arange(0, es - seg - 1)
    va = np.linspace(es, fit_end - seg - 1, 8 if tr.get("smoke") else 80).astype(int)
    val = [make_batch(X, cmd, W, va[i:i + bs], seg, rf, n_ch) for i in range(0, len(va), bs)]
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=tr["lr"])
    log(f"  params={n_params(model):,} rf={rf} seg={seg} bs={bs} train_rows<{es:,} "
        f"es_val=[{es:,},{fit_end:,}) device={dev}")
    best, bad, best_state, hist = float("inf"), 0, None, []
    for ep in range(tr["epochs"]):
        t0 = time.time()
        model.train()
        tl = 0.0
        for xs, ys, ws, cf in Loader(X, cmd, W, starts, bs, seg, rf, n_ch, tr["batches"],
                                     seed * 1000 + ep, dev.type == "cuda"):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model.forward_segments(to(dev, xs), to(dev, cf)), to(dev, ys), to(dev, ws))
            loss.backward()
            opt.step()
            tl += loss.item()
        model.eval()
        vl = vn = 0.0
        with torch.no_grad():
            for xs, ys, ws, cf in val:
                t = [None if a is None else torch.from_numpy(a) for a in (xs, ys, ws, cf)]
                vl += loss_fn(model.forward_segments(to(dev, t[0]), to(dev, t[3])),
                              to(dev, t[1]), to(dev, t[2])).item() * len(xs)
                vn += len(xs)
        vl /= vn
        dt = time.time() - t0
        hist.append((ep + 1, tl / tr["batches"], vl, dt))
        log(f"  ep{ep + 1}: train={tl / tr['batches']:.5f} val={vl:.5f} ({dt:.0f}s)")
        progress(phase="train", epoch=ep + 1, epochs=tr["epochs"], sec_per_epoch=dt, best_val=min(best, vl))
        if not np.isfinite(vl):
            break
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tr["patience"]:
                break
    if best_state is None:
        raise RuntimeError("training produced no finite validation loss")
    model.load_state_dict(best_state)
    return model, dict(best_val=best, epochs=len(hist), hist=hist, seg=seg, bs=bs)


def _budget(dev):
    if dev.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        return int(min(free * 0.3, 6e9))
    return int(2.5e8 if dev.type == "mps" else 2e9)     # mps: unified memory = the laptop RAM


@torch.no_grad()
def series_errors(model, X, cmd, dev, start=0, end=None):
    """|prediction - actual| for rows [start, end) (rows < RF are 0). Full fp32."""
    model.eval()
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    T, N, _ = X.shape
    end = T if end is None else end
    rf, n_ch, c = model.rf, model.n_ch, model.cfg
    budget = _budget(dev)
    enc = int(np.clip(budget / (N * c["width"] * 4 * 6) - rf, 20_000, 400_000))
    nb = (min(model.topk, N - 1) + 1) if model.sparse else N           # neighbours per node
    per_pos = 4 * (N * nb * (2 * c["da"] + model.d + 12) + N * N * 6 * c["dynamic"]
                   + N * 2 * model.d * 6 + model.n_cmd * (5 + 4 * c["dc"]) + 3 * N * max(model.n_cmd, 1))
    att = int(np.clip(budget / per_pos, 256, 65_536))
    errs = np.zeros((end - start, n_ch), np.float32)
    for a in range(max(start, rf), end, enc):
        b = min(a + enc, end)
        seg = torch.from_numpy(np.ascontiguousarray(np.transpose(X[a - rf:b], (1, 0, 2))[None])).to(dev)
        hp = model.encode(seg)[0][:, rf - 1:rf - 1 + (b - a)].permute(1, 0, 2)  # h[t-1] -> t
        preds = torch.empty(b - a, n_ch, device=dev)
        for s in range(0, b - a, att):
            e = min(s + att, b - a)
            cf = None if cmd is None else to(dev, torch.from_numpy(cmd.at(np.arange(a - 1 + s, a - 1 + e))))
            preds[s:e] = model.predict_positions(hp[s:e], cf)
        tgt = torch.from_numpy(np.ascontiguousarray(X[a:b, :n_ch, 0])).to(dev)
        errs[a - start:b - start] = (preds - tgt).abs().cpu().numpy()
        del seg, hp, preds, tgt
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", required=True, choices=list(MISSIONS))
    ap.add_argument("--job", required=True, choices=["foldA", "foldB", "final"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    mk = a.mission
    out = job_dir(mk, a.job, a.seed)
    if os.path.exists(f"{out}/meta.json"):
        print(f"done already: {out}"); return
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(a.threads)
    dev = pick_device()
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    tr = dict(TRAIN, **(SMOKE_TRAIN if a.smoke else {}), smoke=a.smoke)
    logf = open(f"{out}/train.log", "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    def progress(**kw):
        json.dump(dict(kw, t=time.time()), open(f"{out}/progress.tmp", "w"))
        os.replace(f"{out}/progress.tmp", f"{out}/progress.json")

    t0 = time.time()
    d = load_split(mk, "train")
    T = len(d["ts"])
    if a.job == "final":
        fit_end = err_end = T
    else:
        s, e = MISSIONS[mk]["folds"][a.job[-1]]
        fit_end, err_end = row_of(d["ts"], s), row_end(d["ts"], e)
    vfit = np.asarray(d["vals"][:fit_end])
    # near-constant channels (full set: M1 8, M2 20 with sd < 1e-6): an sd floor keeps a 1e-4
    # wobble from becoming a 1e4-sigma forecast error. 0.0 on the lightweight subsets.
    mu, sd = vfit.mean(0), np.maximum(vfit.std(0), MISSIONS[mk]["sd_floor"]) + 1e-8
    del vfit
    X, cmd = build_inputs(mk, d, mu, sd)
    W = None
    if tr["mask_labels"]:
        W = (~label_mask(mk, d["ts"])).astype(np.uint8)
        W[fit_end:] = 1                       # labels after the training period are never used
    if MISSIONS[mk].get("loss_targets_only"):   # subsystems: context channels are inputs only
        ctx = np.setdiff1d(np.arange(len(d["chs"])), target_idx(mk))
        if len(ctx):
            W = np.ones((T, len(d["chs"])), np.uint8) if W is None else W
            W[:, ctx] = 0
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else str(dev)
    log(f"[{mk} {a.job} seed={a.seed}] T={T:,} fit_end={fit_end:,} err_end={err_end:,} "
        f"n_cmd={cmd.n} cmd_cache={'yes' if cmd.cache is not None else 'no'} on {gpu} "
        f"(load {time.time() - t0:.0f}s)")
    progress(phase="train", epoch=0, epochs=tr["epochs"])
    model, info = fit(X, cmd, W, len(d["chs"]), cmd.n, fit_end, a.seed, tr, dev, log, progress)
    t1 = time.time()
    torch.save(model.state_dict(), f"{out}/model.pt")
    if a.job == "final":                     # the learned graph, as used over the training period
        try:
            G.save(f"{out}/graph_train", G.record(model, X, cmd, dev, 0, fit_end), mk, d["chs"],
                   MISSIONS[mk]["targets"])
            log(f"  graph (training period) -> {out}/graph_train.txt")
        except Exception as e:               # never lose a trained model to a summary
            log(f"  graph (training period) FAILED: {e!r}")
    progress(phase="errors", epoch=info["epochs"], epochs=tr["epochs"])
    tix = target_idx(mk)                     # only the scored channels are kept on disk

    def f16(e):                              # clip: one 1e8 error once overflowed float16 to inf
        return np.clip(e[:, tix], 0, 6.0e4).astype(np.float16)
    np.save(f"{out}/err_train.npy", f16(series_errors(model, X, cmd, dev, 0, err_end)))
    if a.job == "final":
        del X, cmd
        dt = load_split(mk, "test")
        Xt, cmdt = build_inputs(mk, dt, mu, sd)
        np.save(f"{out}/err_test.npy", f16(series_errors(model, Xt, cmdt, dev)))
        try:
            G.save(f"{out}/graph_test", G.record(model, Xt, cmdt, dev, 0, len(dt["ts"])), mk, d["chs"],
                   MISSIONS[mk]["targets"])
            log(f"  graph (test period) -> {out}/graph_test.txt")
        except Exception as e:
            log(f"  graph (test period) FAILED: {e!r}")
    meta = dict(mission=mk, job=a.job, seed=a.seed, model=MODEL, train={k: v for k, v in tr.items()},
                train_days=TRAIN_DAYS, train_from_end=TRAIN_FROM_END, init=INIT, freeze=FREEZE,
                targets=MISSIONS[mk]["targets"], n_inputs=len(MISSIONS[mk]["chs"]), chs=MISSIONS[mk]["chs"],
                sparse=bool(model.sparse),
                n_params=n_params(model), rf=model.rf, fit_end=fit_end, err_end=err_end,
                mu=mu.tolist(), sd=sd.tolist(), device=gpu, train_s=t1 - t0,
                err_s=time.time() - t1, **info)
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    with open(f"{out}/history.csv", "w") as fh:
        fh.write("epoch,train_loss,val_loss,seconds\n")
        for row in info["hist"]:
            fh.write(",".join(f"{x:.6g}" for x in row) + "\n")
    progress(phase="done", epoch=info["epochs"], epochs=tr["epochs"])
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if os.uname().sysname == "Darwin" else 1024)
    log(f"done: train {t1 - t0:.0f}s, errors {time.time() - t1:.0f}s, peak memory {rss / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
