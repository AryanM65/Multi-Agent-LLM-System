"""Turn graph per trajectory (decision D3, plus a query node), built from cached MiniLM vectors.

Nodes: one per turn (embedding of the turn text) and one query node (embedding of the query), placed last.
Relations (directed edges, src -> dst):
  0 seq    : turn i <-> turn i+1 (both directions)
  1 same   : consecutive turns of the same agent <-> each other (a chain, not a clique)
  2 q2t    : query node -> every turn
  3 t2q    : every turn -> query node
No communication edges are stored in the dataset, so the graph is a modelling choice, not ground truth.
Only turn text, agent names, order and the query are used (leakage policy in schema.py).
"""
from __future__ import annotations

import numpy as np

N_REL = 4


def edges_for(agent_of_turn: list) -> np.ndarray:
    """agent_of_turn: local agent index per turn. Returns int array (E, 3) of (src, dst, rel); query node id = n."""
    n = len(agent_of_turn)
    q = n
    e = []
    for i in range(n - 1):
        e += [(i, i + 1, 0), (i + 1, i, 0)]
    last: dict = {}
    for i, a in enumerate(agent_of_turn):
        if a in last:
            j = last[a]
            e += [(j, i, 1), (i, j, 1)]
        last[a] = i
    for i in range(n):
        e += [(q, i, 2), (i, q, 3)]
    return np.asarray(e, dtype=np.int64).reshape(-1, 3)


def make_items(trajs: list, store, qvec: dict, X: np.ndarray, Y: np.ndarray, meta: list) -> list:
    """One item per trajectory. X, Y, meta come from features.build (same trajectory/agent order)."""
    items, row = [], 0
    for ti, t in enumerate(trajs):
        order: dict = {}
        for u in t.turns:
            order.setdefault(u.agent, len(order))
        n_agents = len(order)
        assert all(meta[row + k][0] == ti for k in range(n_agents)), "row order mismatch"
        agent_of_turn = [order[u.agent] for u in t.turns]
        items.append({
            "emb": np.stack([store.lookup(u.text) for u in t.turns]).astype(np.float32),
            "q": qvec[t.query].astype(np.float32),
            "agent_of_turn": agent_of_turn,
            "edges": edges_for(agent_of_turn),
            "meta": X[row: row + n_agents, 384:],
            "y": Y[row: row + n_agents],
            "n_agents": n_agents,
        })
        row += n_agents
    assert row == len(X)
    return items


def collate(items: list):
    """Merge items into one batch of numpy arrays (nodes concatenated, edges offset)."""
    xs, src, dst, rel, pool, metas, ys = [], [], [], [], [], [], []
    off = row = 0
    for it in items:
        n = len(it["emb"])
        xs += [it["emb"], it["q"][None]]
        ed = it["edges"]
        src.append(ed[:, 0] + off)
        dst.append(ed[:, 1] + off)
        rel.append(ed[:, 2])
        # pooling index: global agent-row id for each turn node; the query node gets -1
        pool.append(np.asarray([a + row for a in it["agent_of_turn"]] + [-1]))
        metas.append(it["meta"])
        ys.append(it["y"])
        off += n + 1
        row += it["n_agents"]
    return {
        "x": np.concatenate(xs), "src": np.concatenate(src), "dst": np.concatenate(dst), "rel": np.concatenate(rel),
        "pool": np.concatenate(pool), "meta": np.concatenate(metas), "y": np.concatenate(ys), "n_rows": row,
    }
