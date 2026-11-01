"""Per-(trajectory, agent) features from CACHED MiniLM embeddings (never recomputes embeddings).

Row = one agent that speaks in one trajectory. Features:
  mean-pooled turn embedding (384) | framework one-hot | agent-name one-hot (train vocab + UNK)
  | scalars: log(1+turns by agent), share of trajectory turns, log(1+total turns), first-appearance rank (0..1)
Targets: 14-d multi-hot of error modes this agent was labelled with.

Leakage policy: only turn text, agent names, framework and turn counts are used (see schema.py).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from .embed import CACHE_DIR, text_key
from .schema import ERROR_MODES, MODE_INDEX

FRAMEWORKS = ["agentverse", "dylan", "llm_debate", "macnet", "magentic_one", "smoagents", "who_when"]


class EmbStore:
    """All cached vectors as one array plus a key -> row index."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        keys: list = []
        vecs: list = []
        for p in sorted(glob.glob(str(Path(cache_dir) / "minilm_shard_*.npz"))):
            z = np.load(p)
            keys.extend(z["keys"].tolist())
            vecs.append(z["vecs"])
        self.vecs = np.concatenate(vecs).astype(np.float32)
        self.index = {k: i for i, k in enumerate(keys)}

    def lookup(self, text: str) -> np.ndarray:
        return self.vecs[self.index[text_key(text)]]


def agent_vocab(trajs: list) -> list:
    """Agent names seen in the given (training) trajectories' turns, most frequent first."""
    counts: dict = {}
    for t in trajs:
        for a in t.agents:
            counts[a] = counts.get(a, 0) + 1
    return [a for a, _ in sorted(counts.items(), key=lambda kv: -kv[1])]


def build(trajs: list, store: EmbStore, vocab: list):
    """Return X (rows x D), Y (rows x 14), meta list of (traj_index, agent_name)."""
    vidx = {a: i for i, a in enumerate(vocab)}
    fw_idx = {f: i for i, f in enumerate(FRAMEWORKS)}
    D = 384 + len(FRAMEWORKS) + len(vocab) + 1 + 4
    rows, ys, meta = [], [], []
    for ti, t in enumerate(trajs):
        n_total = len(t.turns)
        by_agent: dict = {}
        for u in t.turns:
            by_agent.setdefault(u.agent, []).append(store.lookup(u.text))
        order = list(by_agent)
        for rank, a in enumerate(order):
            vs = by_agent[a]
            x = np.zeros(D, dtype=np.float32)
            x[:384] = np.mean(vs, axis=0)
            x[384 + fw_idx[t.framework]] = 1.0
            off = 384 + len(FRAMEWORKS)
            x[off + vidx.get(a, len(vocab))] = 1.0          # last slot = UNK
            off += len(vocab) + 1
            x[off] = np.log1p(len(vs))
            x[off + 1] = len(vs) / max(n_total, 1)
            x[off + 2] = np.log1p(n_total)
            x[off + 3] = rank / max(len(order) - 1, 1)
            y = np.zeros(len(ERROR_MODES), dtype=np.float32)
            for (la, lm) in t.labels:
                if la == a:
                    y[MODE_INDEX[lm]] = 1.0
            rows.append(x)
            ys.append(y)
            meta.append((ti, a))
    return np.stack(rows), np.stack(ys), meta
