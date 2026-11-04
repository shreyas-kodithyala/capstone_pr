"""Record the graph a trained model actually uses: which channels each channel listens to.

The graph is dynamic (re-picked at every time step), so it is summarised over sampled time
steps of one period:
  freq[i, j]   share of time steps at which channel i takes input from channel j (a top-k edge)
  att[i, j]    mean attention weight of j in i's update (0 at steps where j is not a neighbour)
  static[i, j] edge weight from the channel embeddings alone (the dynamic part removed)
  cmd[i, c]    mean attention of channel i on telecommand c; gate[i] = how much it uses them
Written as graph_<period>.npz (arrays + channel / command names) and graph_<period>.txt
(each channel's strongest neighbours and commands, and how much attention stays inside its
own subsystem).
"""
import csv

import numpy as np
import torch

from .config import MISSIONS, TRAIN
from .data import command_times


def _static(model):
    c, n = model.cfg, model.n_ch
    if c["graph"] == "cosine" or not hasattr(model, "G1"):
        return None
    V = model.embedding.weight[:n]
    M1, M2 = torch.tanh(3.0 * model.G1(V)), torch.tanh(3.0 * model.G2(V))
    S = M1 @ M2.t() - M2 @ M1.t() if c["graph"] == "dir_anti" else M1 @ M2.t()
    return torch.relu(torch.tanh(3.0 * S)).fill_diagonal_(0).cpu().numpy()


@torch.no_grad()
def record(model, X, cmd, dev, lo, hi, n_chunks=8, chunk=256):
    """Summaries over n_chunks x chunk time steps spread evenly over rows [lo, hi) of X."""
    model.eval()
    rf, n = model.rf, model.n_ch
    lo = max(lo, rf + 1)
    starts = np.unique(np.linspace(lo, max(hi - chunk, lo), n_chunks).astype(int))
    freq, att = np.zeros((n, n)), np.zeros((n, n))
    cm = np.zeros((n, model.n_cmd)) if model.n_cmd else None
    gate = np.zeros(n)
    npos = 0
    eye = torch.eye(n, dtype=torch.bool, device=dev)
    for a in starts:
        b = min(a + chunk, hi)
        if b <= a:
            continue
        P = b - a
        seg = torch.from_numpy(np.ascontiguousarray(np.transpose(X[a - rf:b], (1, 0, 2))[None])).to(dev).float()
        hp = model.encode(seg)[0][:, rf - 1:rf - 1 + P].permute(1, 0, 2)     # h[t-1] -> t
        cf = None
        if cmd is not None and model.n_cmd:
            cf = torch.from_numpy(cmd.at(np.arange(a - 1, b - 1))).to(dev).float()
        model._rec = []
        try:
            model.predict_positions(hp, cf)
            rec = model._rec
        finally:
            model._rec = None
        for kind, x, y in rec:
            if kind == "dense":                                    # alpha (P, N, N)
                al = x[:, :n, :n].masked_fill(eye, 0)
                att += al.sum(0).cpu().numpy()
                freq += (al > 0).float().sum(0).cpu().numpy()
            elif kind == "sparse":                                 # alpha (P, N, K) over idx
                al, idx = x[:, :n], y.expand(P, -1, -1)[:, :n]
                A = torch.zeros(P, n, n, device=dev).scatter_add_(2, idx, al)
                Fm = torch.zeros(P, n, n, device=dev).scatter_add_(2, idx, (al > 0).float())
                att += A.masked_fill(eye, 0).sum(0).cpu().numpy()
                freq += (Fm.masked_fill(eye, 0) > 0).float().sum(0).cpu().numpy()
            elif kind == "cmd":                                    # beta (P, n, n_cmd), gate (P, n)
                cm += x.sum(0).cpu().numpy()
                gate += y.sum(0).cpu().numpy()
        npos += P
    out = dict(freq=freq / max(npos, 1), att=att / max(npos, 1), n_positions=np.int64(npos))
    if cm is not None:
        out.update(cmd=cm / max(npos, 1), gate=gate / max(npos, 1))
    st = _static(model)
    if st is not None:
        out["static"] = st
    return out


def save(path, rec, mk, chs, targets):
    """path without extension -> path.npz + path.txt"""
    meta = {r["Channel"]: r for r in csv.DictReader(open(f"{MISSIONS[mk]['raw']}/channels.csv"))}
    sub = [meta[c]["Subsystem"].replace("subsystem_", "S") if c in meta else "?" for c in chs]
    cmds = [c[0] for c in command_times(mk, TRAIN["tc_min_priority"])]
    np.savez_compressed(f"{path}.npz", chs=np.array(chs), subsystem=np.array(sub),
                        target=np.array([c in targets for c in chs]), commands=np.array(cmds), **rec)
    short = [c.replace("channel_", "ch") for c in chs]
    att, freq = rec["att"], rec["freq"]
    L = [f"# learned graph over {int(rec['n_positions'])} sampled time steps",
         "# per channel: strongest inputs (mean attention, share of steps the edge is used); T = target, c = context",
         ""]
    own = []
    for i, c in enumerate(chs):
        o = np.argsort(-att[i])[:5]
        tag = "T" if c in targets else "c"
        nb = ", ".join(f"{short[j]}({sub[j]}) {att[i, j]:.2f}/{freq[i, j]:.0%}" for j in o if att[i, j] > 0)
        line = f"{short[i]:>6} ({sub[i]},{tag}) <- {nb or '(self only)'}"
        if "cmd" in rec and len(cmds):
            k = np.argsort(-rec["cmd"][i])[:3]
            line += f" | commands (gate {rec['gate'][i]:.2f}): " + ", ".join(
                f"{cmds[j]} {rec['cmd'][i, j]:.2f}" for j in k)
        L.append(line)
        tot = att[i].sum()
        if c in targets and tot > 0:
            own.append(att[i, [j for j in range(len(chs)) if sub[j] == sub[i]]].sum() / tot)
    if own:
        L += ["", f"targets: mean share of attention on channels of their own subsystem {np.mean(own):.0%}"
                  f" (rest: context channels from other subsystems)"]
    open(f"{path}.txt", "w").write("\n".join(L) + "\n")
