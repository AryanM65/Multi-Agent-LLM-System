"""Relational message-passing model over the turn graph (Phase 5), validation only.

Per trajectory: turn nodes (frozen MiniLM vectors) + a query node, 4 edge types (graph.py).
Layer (2 of them): h_i' = LN(h_i + Drop(ReLU(W_self h_i + sum_r mean_{j in N_r(i)} W_r h_j)))
Agent vector = mean of that agent's turn nodes after the last layer. Head input = [agent vector | metadata], where metadata is
exactly the non-embedding part of the flat-MLP features (framework one-hot, agent-name one-hot, 4 count scalars), so the only
difference from the flat baseline is the graph encoder over turns. Output: 14 mode logits + 1 faulty-agent logit, BCE.
Decoder, early stopping and threshold tuning are identical to flat_mlp.py. Validation only; test is not touched.

Run: python -m aegis.src.gnn.turn_gnn [--seeds 3] [--layers 2]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch import nn

from ..baselines.flat_mlp import make_targets, tune
from ..embed import get_embeddings
from ..features import EmbStore, agent_vocab, build
from ..graph import N_REL, collate, make_items
from ..loaders import load_grouped_splits, load_paper_splits
from ..schema import ERROR_MODES
from ..utils import log_run, set_seed


class RelLayer(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.self_lin = nn.Linear(d, d)
        self.rel_lin = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(N_REL)])
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, src, dst, rel):
        out = self.self_lin(h)
        for r, lin in enumerate(self.rel_lin):
            m = rel == r
            if not m.any():
                continue
            s, d_ = src[m], dst[m]
            agg = torch.zeros_like(h).index_add_(0, d_, lin(h)[s])
            deg = torch.zeros(h.shape[0], device=h.device).index_add_(0, d_, torch.ones(len(d_), device=h.device))
            out = out + agg / deg.clamp(min=1).unsqueeze(1)
        return self.norm(h + self.drop(torch.relu(out)))


class TurnGNN(nn.Module):
    def __init__(self, d_meta: int, d: int = 128, layers: int = 2, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(384, d), nn.ReLU(), nn.Dropout(dropout))
        self.layers = nn.ModuleList([RelLayer(d, dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(d + d_meta, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, len(ERROR_MODES) + 1))

    def forward(self, b):
        h = self.inp(b["x"])
        for layer in self.layers:
            h = layer(h, b["src"], b["dst"], b["rel"])
        pool, n_rows = b["pool"], b["n_rows"]
        keep = pool >= 0
        agg = torch.zeros(n_rows, h.shape[1], device=h.device).index_add_(0, pool[keep], h[keep])
        cnt = torch.zeros(n_rows, device=h.device).index_add_(0, pool[keep], torch.ones(int(keep.sum()), device=h.device))
        return self.head(torch.cat([agg / cnt.clamp(min=1).unsqueeze(1), b["meta"]], dim=1))


def to_torch(b: dict, mu: np.ndarray, sd: np.ndarray) -> dict:
    t = {k: torch.as_tensor(v) for k, v in b.items() if k != "n_rows"}
    t["meta"] = torch.as_tensor((b["meta"] - mu) / sd, dtype=torch.float32)
    t["y"] = torch.as_tensor(make_targets(b["y"]))
    t["n_rows"] = b["n_rows"]
    return t


def predict(model, items, mu, sd, bs=64):
    model.eval()
    probs, loss_sum, n = [], 0.0, 0
    lossf = nn.BCEWithLogitsLoss(reduction="sum")
    with torch.no_grad():
        for i in range(0, len(items), bs):
            b = to_torch(collate(items[i: i + bs]), mu, sd)
            lg = model(b)
            loss_sum += lossf(lg, b["y"]).item()
            n += b["y"].numel()
            probs.append(torch.sigmoid(lg).numpy())
    return np.concatenate(probs), loss_sum / n


def train_gnn(tr_items, va_items, seed: int, layers: int, epochs: int = 40, patience: int = 6, bs: int = 32):
    set_seed(seed)
    allm = np.concatenate([it["meta"] for it in tr_items])
    mu, sd = allm.mean(0), allm.std(0) + 1e-6
    model = TurnGNN(allm.shape[1], layers=layers)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad = 1e9, None, 0
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(tr_items))
        for i in range(0, len(perm), bs):
            b = to_torch(collate([tr_items[j] for j in perm[i: i + bs]]), mu, sd)
            opt.zero_grad()
            lossf(model(b), b["y"]).backward()
            opt.step()
        _, vl = predict(model, va_items, mu, sd)
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    probs, _ = predict(model, va_items, mu, sd)
    return probs, ep + 1, sum(p.numel() for p in model.parameters())


def run(name: str, splits: dict, store: EmbStore, seeds: list, layers: int) -> None:
    vocab = agent_vocab(splits["train"])
    queries = sorted({t.query for k in ("train", "val") for t in splits[k]})
    qvec = dict(zip(queries, get_embeddings(queries, verbose=False)))
    data = {}
    for k in ("train", "val"):
        X, Y, meta = build(splits[k], store, vocab)
        data[k] = (make_items(splits[k], store, qvec, X, Y, meta), meta)
    tr_items, va_items, meta_va = data["train"][0], data["val"][0], data["val"][1]
    print(f"[{name}] train traj {len(tr_items)}  val traj {len(va_items)}  layers {layers}", flush=True)
    results = []
    for s in seeds:
        t0 = time.time()
        probs, epochs, n_params = train_gnn(tr_items, va_items, s, layers)
        _, ta, tm, res = tune(probs, meta_va, splits["val"])
        results.append(res)
        print(f"[{name}] seed {s}: epochs {epochs} params {n_params} t_a {ta} t_m {tm}  "
              f"pair {100*res['pair']['micro_f1']:.2f}/{100*res['pair']['macro_f1']:.2f} "
              f"agent {100*res['agent']['micro_f1']:.2f}/{100*res['agent']['macro_f1']:.2f} "
              f"error {100*res['error']['micro_f1']:.2f}/{100*res['error']['macro_f1']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
        log_run(phase="5", model=f"turn_gnn_L{layers}", seed=s, split=f"val_{name}", n_params=n_params,
                train_seconds=time.time() - t0,
                config={"split": name, "layers": layers, "dim": 128, "edges": "seq+same-agent+query", "t_a": ta, "t_m": tm,
                        "epochs_run": epochs},
                metrics={"val": {l: res[l] for l in ("pair", "agent", "error")}},
                notes="relational GNN over turn graph + flat metadata; val only; no test eval")
    for lvl in ("pair", "agent", "error"):
        for k in ("micro_f1", "macro_f1"):
            v = [100 * r[lvl][k] for r in results]
            print(f"[{name}] {lvl:5s} {k}: {np.mean(v):.2f} +- {np.std(v):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--splits", default="grouped,paper")
    args = ap.parse_args()
    store = EmbStore()
    loaders = {"paper": load_paper_splits, "grouped": load_grouped_splits}
    for name in args.splits.split(","):
        run(name, loaders[name](), store, list(range(args.seeds)), args.layers)
