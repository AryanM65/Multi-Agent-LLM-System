"""Relational-signal probe (Phase 4c), validation only.

Question: does telling the flat MLP how each agent's text relates to the query and to the other agents help?
Adds 4 features per (trajectory, agent) row to the flat_mlp inputs, built from cached MiniLM vectors:
  cos(agent mean, query) | max over the agent's turns of cos(turn, query)
  cos(agent mean, mean of the other agents' turns) | min over the agent's turns of cos(turn, other-agent mean)
(peer features are 0 when the agent is alone). Everything else (model, decoder, tuning) is identical to flat_mlp.py.
If the probe helps, that supports a model that compares turns with each other (the graph); if not, it does not.

Run: python -m aegis.src.baselines.flat_probe [--seeds 3]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ..embed import get_embeddings
from ..features import EmbStore, agent_vocab, build
from ..loaders import load_grouped_splits, load_paper_splits
from ..utils import log_run
from .flat_mlp import train_one, tune


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def relational_features(trajs: list, store: EmbStore) -> np.ndarray:
    """Rows follow features.build order: trajectories in order, agents by first appearance."""
    qvec = dict(zip(*(lambda qs: (qs, get_embeddings(qs, verbose=False)))(sorted({t.query for t in trajs}))))
    out = []
    for t in trajs:
        q = _unit(qvec[t.query])
        by_agent: dict = {}
        for u in t.turns:
            by_agent.setdefault(u.agent, []).append(store.lookup(u.text))
        for a, vs in by_agent.items():
            V = _unit(np.stack(vs))
            m = _unit(V.mean(0))
            others = [v for b, w in by_agent.items() if b != a for v in w]
            if others:
                o = _unit(_unit(np.stack(others)).mean(0))
                cp, cmin = float(m @ o), float((V @ o).min())
            else:
                cp = cmin = 0.0
            out.append([float(m @ q), float((V @ q).max()), cp, cmin])
    return np.asarray(out, dtype=np.float32)


def run(name: str, splits: dict, store: EmbStore, seeds: list) -> None:
    vocab = agent_vocab(splits["train"])
    Xtr, Ytr, _ = build(splits["train"], store, vocab)
    Xva, Yva, meta_va = build(splits["val"], store, vocab)
    Rtr, Rva = relational_features(splits["train"], store), relational_features(splits["val"], store)
    assert len(Rtr) == len(Xtr) and len(Rva) == len(Xva)
    variants = {"base": (Xtr, Xva), "probe": (np.hstack([Xtr, Rtr]), np.hstack([Xva, Rva]))}
    results = {k: [] for k in variants}
    for s in seeds:
        for vname, (A, B) in variants.items():
            t0 = time.time()
            _, probs, epochs, n_params = train_one(A, Ytr, B, Yva, s)
            _, ta, tm, res = tune(probs, meta_va, splits["val"])
            results[vname].append(res)
            print(f"[{name}] seed {s} {vname:5s}: pair {100*res['pair']['micro_f1']:.2f}/{100*res['pair']['macro_f1']:.2f} "
                  f"agent {100*res['agent']['micro_f1']:.2f}/{100*res['agent']['macro_f1']:.2f} "
                  f"error {100*res['error']['micro_f1']:.2f}/{100*res['error']['macro_f1']:.2f}", flush=True)
            if vname == "probe":
                log_run(phase="4c", model="flat_mlp_relational_probe", seed=s, split=f"val_{name}", n_params=n_params,
                        train_seconds=time.time() - t0,
                        config={"split": name, "extra_features": 4, "t_a": ta, "t_m": tm, "epochs_run": epochs},
                        metrics={"val": {l: res[l] for l in ("pair", "agent", "error")}},
                        notes="flat MLP + cosine-to-query and cosine-to-peers features; val only; no test eval")
    for vname, rs in results.items():
        for lvl in ("pair", "agent", "error"):
            for k in ("micro_f1", "macro_f1"):
                v = [100 * r[lvl][k] for r in rs]
                print(f"[{name}] {vname:5s} {lvl:5s} {k}: {np.mean(v):.2f} +- {np.std(v):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    store = EmbStore()
    for name, splits in (("paper", load_paper_splits()), ("grouped", load_grouped_splits())):
        run(name, splits, store, list(range(args.seeds)))
