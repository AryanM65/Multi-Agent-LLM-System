"""Flat (no-graph) baseline (Phase 4): per-agent mean-pooled MiniLM turn embeddings -> MLP -> multi-label modes.

Each row is one (trajectory, agent). Heads: 14 error-mode logits + 1 faulty-agent logit (any mode), BCE loss.
Decision rule: predict pair (agent, mode) iff p_agent >= t_a and p_mode >= t_m; if a trajectory ends with no pair,
add its single highest-scoring (agent, mode) (every record in the data has at least one fault). t_a, t_m are
tuned on VALIDATION only (grid, maximising the mean of pair/agent/error micro-F1). Early stopping on val loss.

Run: python -m aegis.src.baselines.flat_mlp [--seeds 3]
Only validation is evaluated; the test set is not touched in Phase 4.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch import nn

from ..features import EmbStore, agent_vocab, build
from ..loaders import load_grouped_splits, load_paper_splits
from ..metrics import evaluate
from ..schema import ERROR_MODES
from ..utils import log_run, set_seed

GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, len(ERROR_MODES) + 1))

    def forward(self, x):
        return self.net(x)


def make_targets(Y: np.ndarray) -> np.ndarray:
    agent = (Y.sum(1, keepdims=True) > 0).astype(np.float32)
    return np.concatenate([Y, agent], axis=1)


def decode(probs: np.ndarray, meta: list, n_traj: int, ta: float, tm: float) -> list:
    """probs: rows x 15 (14 modes, agent). Returns a list of pair sets per trajectory."""
    pred = [set() for _ in range(n_traj)]
    best = [(-1.0, None)] * n_traj
    for (ti, a), p in zip(meta, probs):
        pa, pm = p[-1], p[:-1]
        for mi in range(len(ERROR_MODES)):
            score = pa * pm[mi]
            if score > best[ti][0]:
                best[ti] = (score, (a, ERROR_MODES[mi]))
            if pa >= ta and pm[mi] >= tm:
                pred[ti].add((a, ERROR_MODES[mi]))
    for ti in range(n_traj):
        if not pred[ti] and best[ti][1] is not None:
            pred[ti].add(best[ti][1])
    return pred


def tune(probs, meta, trajs):
    truth = [t.labels for t in trajs]
    best = None
    for ta in GRID:
        for tm in GRID:
            res = evaluate(truth, decode(probs, meta, len(trajs), ta, tm))
            score = sum(res[l]["micro_f1"] for l in ("pair", "agent", "error")) / 3
            if best is None or score > best[0]:
                best = (score, ta, tm, res)
    return best


def train_one(Xtr, Ytr, Xva, Yva, seed: int, epochs: int = 60, patience: int = 8):
    set_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    f = lambda X: torch.tensor((X - mu) / sd, dtype=torch.float32)
    xtr, xva = f(Xtr), f(Xva)
    ytr, yva = torch.tensor(make_targets(Ytr)), torch.tensor(make_targets(Yva))
    model = MLP(xtr.shape[1])
    model.mu, model.sd = mu, sd   # kept so the same model can score other splits later
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    best_loss, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), 256):
            idx = perm[i: i + 256]
            opt.zero_grad()
            lossf(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(xva), yva).item()
        if vl < best_loss - 1e-4:
            best_loss, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(xva)).numpy()
    return model, probs, ep + 1, sum(p.numel() for p in model.parameters())


def run(name: str, splits: dict, store: EmbStore, seeds: list) -> None:
    vocab = agent_vocab(splits["train"])
    Xtr, Ytr, _ = build(splits["train"], store, vocab)
    Xva, Yva, meta_va = build(splits["val"], store, vocab)
    print(f"[{name}] train rows {len(Xtr)}  val rows {len(Xva)}  dim {Xtr.shape[1]}", flush=True)
    results = []
    for s in seeds:
        t0 = time.time()
        _, probs, epochs, n_params = train_one(Xtr, Ytr, Xva, Yva, s)
        score, ta, tm, res = tune(probs, meta_va, splits["val"])
        results.append(res)
        print(f"[{name}] seed {s}: epochs {epochs} t_a {ta} t_m {tm}  pair {100*res['pair']['micro_f1']:.2f}/{100*res['pair']['macro_f1']:.2f} "
              f"agent {100*res['agent']['micro_f1']:.2f}/{100*res['agent']['macro_f1']:.2f} "
              f"error {100*res['error']['micro_f1']:.2f}/{100*res['error']['macro_f1']:.2f}", flush=True)
        log_run(phase="4", model="flat_mlp", seed=s, split=f"val_{name}", n_params=n_params,
                train_seconds=time.time() - t0,
                config={"split": name, "hidden": 256, "dropout": 0.3, "lr": 1e-3, "t_a": ta, "t_m": tm, "epochs_run": epochs},
                metrics={"val": {l: res[l] for l in ("pair", "agent", "error")}},
                notes="mean-pooled MiniLM per agent + framework + agent-name one-hot; thresholds tuned on val; no test eval")
    for lvl in ("pair", "agent", "error"):
        for k in ("micro_f1", "macro_f1"):
            v = [100 * r[lvl][k] for r in results]
            print(f"[{name}] {lvl:5s} {k}: {np.mean(v):.2f} +- {np.std(v):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    store = EmbStore()
    for name, splits in (("paper", load_paper_splits()), ("grouped", load_grouped_splits())):
        run(name, splits, store, list(range(args.seeds)))
