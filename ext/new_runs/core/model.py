"""Configurable forecaster for the v2 experiments.

Default settings reproduce the published GDNForecaster exactly (38,049 parameters on M1
lightweight): causal TCN (6 layers, kernel 5, 32 filters, 1x1 projection to 64), cosine top-k
graph over learned embeddings, GAT-v1 attention, priority-3 telecommands as graph nodes.

Options (cfg keys):
  width      TCN filters                       32 | 64
  layers     TCN depth, dilations 1..2^(L-1)   6 (RF 253) .. 10 (RF 4093)
  act        activation everywhere             relu | gelu | silu
  dropout    TCN layers, attention weights, head
  att        gat (v1: e_ij = LReLU(a1.g_i + a2.g_j)) | gatv2 (e_ij = a.LReLU(Ws g_i + Wt g_j))
  graph      cosine (symmetric, static) | dir_anti (STGLR/MTGNN one-way) | dir_asym (two-way)
  dynamic    graph scores also depend on the current TCN state (re-picked every position)
  topk       channel neighbours per channel (None = min(n_ch-1, 10))
  tc_mode    nodes (commands are graph nodes, as published) | path (separate command path)

Command path (tc_mode=path): per command c, hand-made temporal features f_c(t) (recency,
log time since last firing, firings in the last 1 h / 6 h / 24 h) go through a small shared
encoder plus a local command embedding. Each channel attends over the commands with its OWN
softmax (plus a "nothing" slot), with a learned channel-command relevance bias, and the
result is added to the channel's sensor context through a learned gate that starts near 0.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULTS = dict(width=32, d=64, layers=6, kernel=5, act="relu", dropout=0.0, att="gat",
                graph="cosine", dynamic=False, topk=None, tc_mode="nodes", dc=32, da=32,
                cmd_feat=5, sparse=None)          # sparse=None: neighbour-only attention when N > 24


def make_act(name):
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[name]


def act_fn(name):
    return {"relu": F.relu, "gelu": F.gelu, "silu": F.silu}[name]


class TCN2(nn.Module):
    """Causal dilated TCN. (B, F, T) -> (B, d, T); output[t] sees inputs t-RF+1 .. t."""

    def __init__(self, feat, width, d, layers, k, act, dropout):
        super().__init__()
        dil = [2 ** i for i in range(layers)]
        self.pads = [(k - 1) * dd for dd in dil]
        self.convs = nn.ModuleList(
            [nn.Conv1d(feat if i == 0 else width, width, k, dilation=dd) for i, dd in enumerate(dil)])
        self.proj = nn.Conv1d(width, d, 1)
        self.act = act_fn(act)
        self.drop = nn.Dropout(dropout) if dropout > 0 else None
        self.rf = 1 + (k - 1) * sum(dil)

    def forward(self, x):
        h = x
        for conv, p in zip(self.convs, self.pads):
            out = self.act(conv(F.pad(h, (p, 0))))
            if self.drop is not None:
                out = self.drop(out)
            h = out + (h if h.shape == out.shape else 0)
        return self.proj(h)


class GDN2(nn.Module):
    def __init__(self, n_ch, n_tc_nodes=0, n_cmd=0, **cfg):
        super().__init__()
        c = dict(DEFAULTS); c.update(cfg); self.cfg = c
        self.n_ch, self.n_tcn, self.n_cmd = n_ch, n_tc_nodes, n_cmd
        self.N = n_ch + n_tc_nodes
        d = self.d = c["d"]
        self.topk = c["topk"] if c["topk"] is not None else max(min(n_ch - 1, 10), 1)
        self.embedding = nn.Embedding(self.N, d)                              # LOCAL
        self.tcn = TCN2(3, c["width"], d, c["layers"], c["kernel"], c["act"], c["dropout"])
        self.rf = self.tcn.rf
        self.act = act_fn(c["act"])
        if c["att"] == "gat":
            self.a1 = nn.Linear(2 * d, 1, bias=False)
            self.a2 = nn.Linear(2 * d, 1, bias=False)
        else:
            self.Ws = nn.Linear(2 * d, c["da"], bias=False)
            self.Wt = nn.Linear(2 * d, c["da"], bias=True)
            self.av = nn.Linear(c["da"], 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        self.att_drop = nn.Dropout(c["dropout"]) if c["dropout"] > 0 else None
        if c["graph"] != "cosine" or c["dynamic"]:
            self.G1 = nn.Linear(d, d, bias=False)                            # shared
            self.G2 = nn.Linear(d, d, bias=False)
            self.gbeta = nn.Parameter(torch.tensor(1.0))                     # edge-strength bias
            if c["dynamic"]:
                self.D1 = nn.Linear(d, d, bias=False)
                self.D2 = nn.Linear(d, d, bias=False)
        if n_cmd:
            dc = c["dc"]
            self.cmd_emb = nn.Embedding(n_cmd, dc)                           # LOCAL
            self.cmd_enc = nn.Sequential(nn.Linear(c["cmd_feat"], dc), make_act(c["act"]),
                                         nn.Linear(dc, dc))
            self.cq = nn.Linear(2 * d, dc, bias=False)
            self.ck = nn.Linear(dc, dc, bias=False)
            self.cv = nn.Linear(dc, d, bias=False)
            self.crel = nn.Linear(d, dc, bias=False)
            self.cgate = nn.Linear(2 * d, d)
            nn.init.constant_(self.cgate.bias, -3.0)                         # gate starts ~0.05
        layers = [nn.Linear(2 * d, d), make_act(c["act"])]
        if c["dropout"] > 0:
            layers.append(nn.Dropout(c["dropout"]))
        layers.append(nn.Linear(d, 1))
        self.head = nn.Sequential(*layers)
        self._adj = None
        self._rec = None                # list -> attention weights are appended (core/graph.py)
        sp = c["sparse"]
        self.sparse = (self.N > 24) if sp is None else bool(sp)
        if self.sparse:
            assert n_tc_nodes == 0, "neighbour-only attention needs tc_mode='path'"

    # ---- federated split: everything except the per-mission embedding tables ----
    def shared_state(self):
        return {k: v for k, v in self.state_dict().items()
                if not (k.startswith("embedding.") or k.startswith("cmd_emb."))}

    def train(self, mode=True):
        self._adj = None
        return super().train(mode)

    # ---- graph ----
    def _static_cosine(self):
        if not self.training and self._adj is not None:
            return self._adj
        v = self.embedding.weight
        A = torch.zeros(self.N, self.N, device=v.device)
        n = self.n_ch
        if n > 1:
            vc = F.normalize(v[:n], dim=1)
            sim = vc @ vc.t() - torch.eye(n, device=v.device) * 9e15
            k = min(self.topk, n - 1)
            A[:n, :n].scatter_(1, sim.topk(k, 1).indices, 1.0)
        if self.n_tcn:
            A[:n, n:] = 1.0
        A = ((A + torch.eye(self.N, device=v.device)) > 0).float()
        if not self.training:
            self._adj = A
        return A

    def _learned(self, h_pos):
        """Directed / dynamic graph. Returns (mask, bias) shaped (P|1, N, N)."""
        c, n, N = self.cfg, self.n_ch, self.N
        V = self.embedding.weight[:n]                                        # (n, d)
        a = 3.0
        if c["dynamic"]:
            hc = h_pos[:, :n]                                                # (P, n, d)
            M1 = torch.tanh(a * (self.G1(V)[None] + self.D1(hc)))
            M2 = torch.tanh(a * (self.G2(V)[None] + self.D2(hc)))
        else:
            M1 = torch.tanh(a * self.G1(V))[None]
            M2 = torch.tanh(a * self.G2(V))[None]
        if c["graph"] == "dir_anti":
            S = M1 @ M2.transpose(1, 2) - M2 @ M1.transpose(1, 2)
        elif c["graph"] == "dir_asym":
            S = M1 @ M2.transpose(1, 2)
        else:                                                                # cosine (dynamic)
            Mn = F.normalize(M1, dim=2)
            S = Mn @ Mn.transpose(1, 2)
        W = torch.relu(torch.tanh(a * S))                                    # (P|1, n, n)
        eye = torch.eye(n, device=V.device, dtype=torch.bool)[None]
        Sm = S.masked_fill(eye, -9e15)
        k = min(self.topk, n - 1)
        keep = torch.zeros_like(W)
        if k > 0:
            keep.scatter_(2, Sm.topk(k, 2).indices, 1.0)
        if c["graph"] != "cosine":
            keep = keep * (W > 0).float()                                    # one-way edges only
        P0 = keep.size(0)
        mask = torch.zeros(P0, N, N, device=V.device)
        mask[:, :n, :n] = keep
        if self.n_tcn:
            mask[:, :n, n:] = 1.0
        mask = ((mask + torch.eye(N, device=V.device)[None]) > 0).float()
        bias = torch.zeros(P0, N, N, device=V.device)
        bias[:, :n, :n] = self.gbeta * W * keep
        return mask, bias

    def encode(self, x):
        """x (B, N, T, 3) -> (B, N, T, d)."""
        B, N, T, Fd = x.shape
        h = self.tcn(x.reshape(B * N, T, Fd).transpose(1, 2))
        return h.transpose(1, 2).reshape(B, N, T, self.d)

    # ---- neighbour-only attention (full channel sets) ----
    @staticmethod
    def _gather(x, idx):
        """x (P, N, F), idx (P|1, N, K) -> (P, N, K, F) with out[p,i,k] = x[p, idx[p,i,k]]."""
        P, N, F = x.shape
        K = idx.shape[2]
        flat = idx.expand(P, -1, -1).reshape(P, N * K, 1).expand(-1, -1, F)
        return torch.gather(x, 1, flat).reshape(P, N, K, F)

    def _neighbours(self, h_pos):
        """-> idx (P|1, N, K), valid (P|1, N, K) bool, bias (P|1, N, K) or None.
        Same neighbour sets as the dense masks: top-k channels (excluding self) plus self."""
        c, n = self.cfg, self.n_ch
        dev = h_pos.device
        k = min(self.topk, n - 1)
        self_idx = torch.arange(n, device=dev).view(1, n, 1)
        if c["graph"] == "cosine" and not c["dynamic"]:
            v = self.embedding.weight[:n]
            vc = F.normalize(v, dim=1)
            sim = vc @ vc.t() - torch.eye(n, device=dev) * 9e15
            ti = sim.topk(k, 1).indices[None] if k > 0 else torch.zeros(1, n, 0, dtype=torch.long, device=dev)
            idx = torch.cat([ti, self_idx], 2)
            return idx, torch.ones_like(idx, dtype=torch.bool), None
        V = self.embedding.weight[:n]
        a = 3.0
        if c["dynamic"]:
            hc = h_pos[:, :n]
            M1 = torch.tanh(a * (self.G1(V)[None] + self.D1(hc)))
            M2 = torch.tanh(a * (self.G2(V)[None] + self.D2(hc)))
        else:
            M1 = torch.tanh(a * self.G1(V))[None]
            M2 = torch.tanh(a * self.G2(V))[None]
        if c["graph"] == "dir_anti":
            S = M1 @ M2.transpose(1, 2) - M2 @ M1.transpose(1, 2)
        elif c["graph"] == "dir_asym":
            S = M1 @ M2.transpose(1, 2)
        else:
            Mn = F.normalize(M1, dim=2)
            S = Mn @ Mn.transpose(1, 2)
        W = torch.relu(torch.tanh(a * S))
        eye = torch.eye(n, device=dev, dtype=torch.bool)[None]
        P0 = S.size(0)
        if k > 0:
            ti = S.masked_fill(eye, -9e15).topk(k, 2).indices                  # (P0, n, k)
            Wk = torch.gather(W, 2, ti)
            vk = (Wk > 0) if c["graph"] != "cosine" else torch.ones_like(Wk, dtype=torch.bool)
            bk = self.gbeta * Wk * vk
        else:
            ti = torch.zeros(P0, n, 0, dtype=torch.long, device=dev)
            vk = torch.zeros(P0, n, 0, dtype=torch.bool, device=dev)
            bk = torch.zeros(P0, n, 0, device=dev)
        idx = torch.cat([ti, self_idx.expand(P0, -1, -1)], 2)
        valid = torch.cat([vk, torch.ones(P0, n, 1, dtype=torch.bool, device=dev)], 2)
        bias = torch.cat([bk, torch.zeros(P0, n, 1, device=dev)], 2)
        return idx, valid, bias

    def _predict_sparse(self, h_pos, cmdf):
        c = self.cfg
        P = h_pos.size(0)
        v = self.embedding.weight.unsqueeze(0).expand(P, -1, -1)
        g = torch.cat([v, h_pos], dim=2)                                     # (P, N, 2d)
        idx, valid, bias = self._neighbours(h_pos)
        if c["att"] == "gat":
            e = self.leaky(self.a1(g) + self._gather(self.a2(g), idx).squeeze(-1))      # (P, N, K)
        else:
            e = self.av(self.leaky(self.Ws(g)[:, :, None, :] + self._gather(self.Wt(g), idx))).squeeze(-1)
        if bias is not None:
            e = e + bias
        e = e.masked_fill(~valid, float("-inf"))
        alpha = torch.softmax(e, dim=2)
        if self._rec is not None:
            self._rec.append(("sparse", alpha.detach(), idx.detach()))
        if self.att_drop is not None:
            alpha = self.att_drop(alpha)
        z = self.act((alpha.unsqueeze(-1) * self._gather(h_pos, idx)).sum(2))  # (P, N, d)
        if self.n_cmd and cmdf is not None:
            z = z + self._cmd_context(g, h_pos, cmdf)
        gz = torch.cat([v * z, z], dim=2)
        return self.head(gz).squeeze(-1)[:, :self.n_ch]

    def predict_positions(self, h_pos, cmdf=None):
        """h_pos (P, N, d) [+ cmdf (P, n_cmd, cmd_feat)] -> (P, n_ch) next-step predictions."""
        if self.sparse:
            return self._predict_sparse(h_pos, cmdf)
        c = self.cfg
        P = h_pos.size(0)
        v = self.embedding.weight.unsqueeze(0).expand(P, -1, -1)
        g = torch.cat([v, h_pos], dim=2)                                     # (P, N, 2d)
        if c["att"] == "gat":
            e = self.leaky(self.a1(g) + self.a2(g).transpose(1, 2))         # (P, N, N)
        else:
            e = self.av(self.leaky(self.Ws(g)[:, :, None, :] + self.Wt(g)[:, None, :, :])).squeeze(-1)
        if c["graph"] == "cosine" and not c["dynamic"]:
            A = self._static_cosine()[None]
        else:
            A, bias = self._learned(h_pos)
            e = e + bias
        e = e.masked_fill(A == 0, float("-inf"))
        alpha = torch.softmax(e, dim=2)
        if self._rec is not None:
            self._rec.append(("dense", alpha.detach(), None))
        if self.att_drop is not None:
            alpha = self.att_drop(alpha)
        z = self.act(torch.bmm(alpha, h_pos))                                # (P, N, d)
        if self.n_cmd and cmdf is not None:
            z = z.clone()
            z[:, :self.n_ch] = z[:, :self.n_ch] + self._cmd_context(g[:, :self.n_ch], h_pos[:, :self.n_ch], cmdf)
        gz = torch.cat([v * z, z], dim=2)
        return self.head(gz).squeeze(-1)[:, :self.n_ch]

    def _cmd_context(self, g_ch, h_ch, cmdf):
        dc = self.cfg["dc"]
        u = self.act(self.cmd_enc(cmdf) + self.cmd_emb.weight[None])         # (P, n_cmd, dc)
        q = self.cq(g_ch)                                                    # (P, n_ch, dc)
        s = q @ self.ck(u).transpose(1, 2) / math.sqrt(dc)                   # (P, n_ch, n_cmd)
        rel = self.crel(self.embedding.weight[:self.n_ch]) @ self.cmd_emb.weight.t()
        s = s + rel[None]
        s = torch.cat([s, torch.zeros_like(s[..., :1])], dim=2)              # "nothing" slot
        beta = torch.softmax(s, dim=2)[..., :-1]
        m = beta @ self.cv(u)                                                # (P, n_ch, d)
        gate = torch.sigmoid(self.cgate(torch.cat([h_ch, m], dim=2)))
        if self._rec is not None:
            self._rec.append(("cmd", beta.detach(), gate.detach().mean(-1)))
        return gate * m

    def forward_segments(self, x, cmdf=None):
        """x (B, N, L, 3) -> (B, L-RF, n_ch); cmdf (B, L-RF, n_cmd, F) at the same positions."""
        B, N, L, Fd = x.shape
        h = self.encode(x)
        hp = h[:, :, self.rf - 1:L - 1]
        Lp = hp.size(2)
        hp = hp.permute(0, 2, 1, 3).reshape(B * Lp, N, self.d)
        cf = None if cmdf is None else cmdf.reshape(B * Lp, *cmdf.shape[2:])
        return self.predict_positions(hp, cf).reshape(B, Lp, self.n_ch)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
