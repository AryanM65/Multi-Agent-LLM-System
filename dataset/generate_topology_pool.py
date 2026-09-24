"""generate_topology_pool.py — build and freeze the train/OOD topology pool.

Per plan.md Section 2. This is a pure CPU/Python task, no GPU or model calls
needed. Generates:
  - a ~14-topology TRAIN pool (hand-authored variants covering fan-in,
    fan-out, deep chains, and named canonical shapes)
  - a ~6-topology OOD (out-of-distribution) test pool, generated via
    random-DAG construction (MOC-style: role-before-edges, backbone chain +
    random forward edges), deduplicated against both itself and the train pool

Output: topology_pool.json, frozen and committed BEFORE any trial generation
starts (the train/OOD split must be fixed in advance, per the master plan's
explicit requirement — regenerating this file after generation has begun
would silently invalidate the split guarantee the whole dataset design
depends on).

Usage:
    python generate_topology_pool.py
    python generate_topology_pool.py --seed 42 --ood-count 6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

# btp-pipeline is a sibling directory of this dataset/ folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BTP_PIPELINE_ROOT = os.path.abspath(os.path.join(_HERE, "..", "btp-pipeline"))
if _BTP_PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _BTP_PIPELINE_ROOT)

from src.topology import NodeSpec, Topology, validate_topology, ROLE_RANK
from src.pipeline import RETRIEVER_INSTRUCTION, REASONER_INSTRUCTION, WRITER_INSTRUCTION
from src.topologies import (
    chain_topology,
    dual_retriever_fanin_topology,
    parallel_reasoner_topology,
    deep_chain_topology,
)

_INSTR = {
    "retriever": RETRIEVER_INSTRUCTION,
    "reasoner": REASONER_INSTRUCTION,
    "writer": WRITER_INSTRUCTION,
}

_REFINE_INSTR = (
    "You are a Refining Reasoning agent. You will receive a first-pass chain-of-thought "
    "produced by a previous Reasoning agent. Your task is to critically review it, "
    "identify any logical gaps or errors, and produce a corrected, more concise reasoning "
    "trace.\n\nEnd your response with a final line in exactly this format:\n"
    "FINAL ANSWER: <your one-sentence conclusion>"
)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def topology_to_dict(topo: Topology, split: str) -> dict:
    return {
        "topology_id": topo.topology_id,
        "split": split,
        "nodes": {
            nid: {"role": spec.role, "instruction": spec.instruction}
            for nid, spec in topo.nodes.items()
        },
        "edges": [[src, dst] for src, dst in topo.edges],
    }


def topology_from_dict(d: dict) -> Topology:
    nodes = {
        nid: NodeSpec(nid, info["role"], info["instruction"])
        for nid, info in d["nodes"].items()
    }
    edges = [(e[0], e[1]) for e in d["edges"]]
    return Topology(nodes=nodes, edges=edges, topology_id=d["topology_id"])


def load_topology_pool(path: str) -> list:
    """Load a frozen topology_pool.json back into (Topology, split) pairs."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return [(topology_from_dict(r), r["split"]) for r in records]


# ---------------------------------------------------------------------------
# Canonical signature (for dedup)
# ---------------------------------------------------------------------------

def _canonical_signature(topo: Topology) -> tuple:
    """A (role-sequence, edge-structure) signature invariant to node_id naming,
    used to detect structurally-identical topologies (e.g. same shape, renamed
    nodes) so they aren't counted twice in the pool.
    """
    order = sorted(topo.nodes.keys())
    index = {nid: i for i, nid in enumerate(order)}
    roles = tuple(topo.nodes[nid].role for nid in order)
    edges = tuple(sorted((index[s], index[d]) for s, d in topo.edges))
    return (roles, edges)


# ---------------------------------------------------------------------------
# Train pool — hand-authored variants (~14 topologies)
# ---------------------------------------------------------------------------

def _mk(nodes_roles: dict, edges: list, topology_id: str, overrides: dict = None) -> Topology:
    overrides = overrides or {}
    nodes = {
        nid: NodeSpec(nid, role, overrides.get(nid, _INSTR[role]))
        for nid, role in nodes_roles.items()
    }
    return Topology(nodes=nodes, edges=edges, topology_id=topology_id)


def triple_retriever_fanin_topology() -> Topology:
    return _mk(
        {"retriever_a": "retriever", "retriever_b": "retriever", "retriever_c": "retriever",
         "reasoner": "reasoner", "writer": "writer"},
        [("retriever_a", "reasoner"), ("retriever_b", "reasoner"), ("retriever_c", "reasoner"),
         ("reasoner", "writer")],
        "triple_retriever_fanin",
    )


def dual_retriever_fanin_deep_topology() -> Topology:
    return _mk(
        {"retriever_a": "retriever", "retriever_b": "retriever",
         "reasoner_1": "reasoner", "reasoner_2": "reasoner", "writer": "writer"},
        [("retriever_a", "reasoner_1"), ("retriever_b", "reasoner_1"),
         ("reasoner_1", "reasoner_2"), ("reasoner_2", "writer")],
        "dual_retriever_fanin_deep",
        overrides={"reasoner_2": _REFINE_INSTR},
    )


def triple_parallel_reasoner_topology() -> Topology:
    return _mk(
        {"retriever": "retriever", "reasoner_a": "reasoner", "reasoner_b": "reasoner",
         "reasoner_c": "reasoner", "writer": "writer"},
        [("retriever", "reasoner_a"), ("retriever", "reasoner_b"), ("retriever", "reasoner_c"),
         ("reasoner_a", "writer"), ("reasoner_b", "writer"), ("reasoner_c", "writer")],
        "triple_parallel_reasoner",
    )


def crossed_fanin_fanout_topology() -> Topology:
    """Two retrievers each feed two reasoners (full fan-in), both reasoners feed one writer."""
    return _mk(
        {"retriever_a": "retriever", "retriever_b": "retriever",
         "reasoner_a": "reasoner", "reasoner_b": "reasoner", "writer": "writer"},
        [("retriever_a", "reasoner_a"), ("retriever_b", "reasoner_a"),
         ("retriever_a", "reasoner_b"), ("retriever_b", "reasoner_b"),
         ("reasoner_a", "writer"), ("reasoner_b", "writer")],
        "crossed_fanin_fanout",
    )


def deep_chain_5node_topology() -> Topology:
    return _mk(
        {"retriever": "retriever", "reasoner_1": "reasoner", "reasoner_2": "reasoner",
         "reasoner_3": "reasoner", "writer": "writer"},
        [("retriever", "reasoner_1"), ("reasoner_1", "reasoner_2"),
         ("reasoner_2", "reasoner_3"), ("reasoner_3", "writer")],
        "deep_chain_5node",
        overrides={"reasoner_2": _REFINE_INSTR, "reasoner_3": _REFINE_INSTR},
    )


def star_topology() -> Topology:
    """Multiple retrievers -> one central reasoner -> multiple writers."""
    return _mk(
        {"retriever_a": "retriever", "retriever_b": "retriever", "retriever_c": "retriever",
         "reasoner": "reasoner", "writer_a": "writer", "writer_b": "writer"},
        [("retriever_a", "reasoner"), ("retriever_b", "reasoner"), ("retriever_c", "reasoner"),
         ("reasoner", "writer_a"), ("reasoner", "writer_b")],
        "star",
    )


def tree_topology() -> Topology:
    """One retriever branches into two full independent reasoner->writer paths."""
    return _mk(
        {"retriever": "retriever", "reasoner_a": "reasoner", "reasoner_b": "reasoner",
         "writer_a": "writer", "writer_b": "writer"},
        [("retriever", "reasoner_a"), ("retriever", "reasoner_b"),
         ("reasoner_a", "writer_a"), ("reasoner_b", "writer_b")],
        "tree",
    )


def wide_fanin_topology() -> Topology:
    return _mk(
        {"retriever_a": "retriever", "retriever_b": "retriever",
         "retriever_c": "retriever", "retriever_d": "retriever",
         "reasoner": "reasoner", "writer": "writer"},
        [("retriever_a", "reasoner"), ("retriever_b", "reasoner"),
         ("retriever_c", "reasoner"), ("retriever_d", "reasoner"), ("reasoner", "writer")],
        "wide_fanin",
    )


def wide_fanout_reasoner_topology() -> Topology:
    return _mk(
        {"retriever": "retriever", "reasoner_a": "reasoner", "reasoner_b": "reasoner",
         "reasoner_c": "reasoner", "reasoner_d": "reasoner", "writer": "writer"},
        [("retriever", "reasoner_a"), ("retriever", "reasoner_b"),
         ("retriever", "reasoner_c"), ("retriever", "reasoner_d"),
         ("reasoner_a", "writer"), ("reasoner_b", "writer"),
         ("reasoner_c", "writer"), ("reasoner_d", "writer")],
        "wide_fanout_reasoner",
    )


def mixed_asymmetric_topology() -> Topology:
    """Asymmetric fan-in: reasoner_a sees both retrievers, reasoner_b sees only one."""
    return _mk(
        {"retriever_a": "retriever", "retriever_b": "retriever",
         "reasoner_a": "reasoner", "reasoner_b": "reasoner", "writer": "writer"},
        [("retriever_a", "reasoner_a"), ("retriever_b", "reasoner_a"),
         ("retriever_b", "reasoner_b"),
         ("reasoner_a", "writer"), ("reasoner_b", "writer")],
        "mixed_asymmetric",
    )


TRAIN_FACTORIES = [
    chain_topology,
    dual_retriever_fanin_topology,
    triple_retriever_fanin_topology,
    dual_retriever_fanin_deep_topology,
    parallel_reasoner_topology,
    triple_parallel_reasoner_topology,
    crossed_fanin_fanout_topology,
    deep_chain_topology,
    deep_chain_5node_topology,
    star_topology,
    tree_topology,
    wide_fanin_topology,
    wide_fanout_reasoner_topology,
    mixed_asymmetric_topology,
]


def build_train_pool() -> list:
    topos = [f() for f in TRAIN_FACTORIES]
    for t in topos:
        validate_topology(t)
    return topos


# ---------------------------------------------------------------------------
# OOD pool — random-DAG construction (MOC-style)
# ---------------------------------------------------------------------------

def _random_topology(rng: random.Random, node_count: int, topology_id: str,
                      extra_edge_prob: float = 0.3) -> Topology:
    """Role-before-edges random DAG: valid by construction, no rejection sampling.

    1. Assign roles to a fixed linear order, non-decreasing in ROLE_RANK
       (all retrievers first, then reasoners, then writers) -- any edge drawn
       from an earlier position to a later position is therefore automatically
       role-valid (ROLE_RANK[src] <= ROLE_RANK[dst]).
    2. Force a connected backbone chain across that order (no isolated nodes).
    3. Randomly add extra forward-only edges for fan-in/fan-out structure
       beyond the plain backbone.
    """
    n_retriever = rng.randint(1, max(1, node_count // 3))
    n_writer = rng.randint(1, max(1, node_count // 3))
    n_reasoner = max(1, node_count - n_retriever - n_writer)
    roles_ordered = (["retriever"] * n_retriever + ["reasoner"] * n_reasoner
                     + ["writer"] * n_writer)

    node_ids = [f"n{i}_{role[0]}" for i, role in enumerate(roles_ordered)]
    instr_overrides = {}
    # Vary Reasoner instructions a bit for reasoners past the first (refine-style),
    # matching the flavor of the hand-authored deep_chain variants.
    reasoner_seen = 0
    for nid, role in zip(node_ids, roles_ordered):
        if role == "reasoner":
            reasoner_seen += 1
            if reasoner_seen > 1:
                instr_overrides[nid] = _REFINE_INSTR

    nodes = {
        nid: NodeSpec(nid, role, instr_overrides.get(nid, _INSTR[role]))
        for nid, role in zip(node_ids, roles_ordered)
    }

    edges = [(node_ids[i], node_ids[i + 1]) for i in range(len(node_ids) - 1)]
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            if (node_ids[i], node_ids[j]) in edges:
                continue
            if rng.random() < extra_edge_prob:
                edges.append((node_ids[i], node_ids[j]))

    return Topology(nodes=nodes, edges=edges, topology_id=topology_id)


def build_ood_pool(seed: int, count: int, train_signatures: set,
                    node_count_range: tuple = (5, 7)) -> list:
    """Generate `count` deduplicated, validated, role-valid random topologies.

    Deliberately larger node counts than most of the train pool (5-7 vs.
    mostly 3-6) to actually test structural generalization, not just a
    same-size shuffle. Deduplicated against both the OOD pool itself and the
    train pool signatures, by canonical (role-sequence, edge-structure).
    """
    rng = random.Random(seed)
    seen = set(train_signatures)
    ood = []
    attempts = 0
    while len(ood) < count and attempts < count * 50:
        attempts += 1
        node_count = rng.randint(*node_count_range)
        topo = _random_topology(rng, node_count, f"ood_random_{len(ood)}_{attempts}")
        try:
            validate_topology(topo)
        except ValueError:
            continue
        sig = _canonical_signature(topo)
        if sig in seen:
            continue
        seen.add(sig)
        topo.topology_id = f"ood_random_{len(ood)}"
        ood.append(topo)
    if len(ood) < count:
        raise RuntimeError(
            f"Only generated {len(ood)}/{count} unique OOD topologies after "
            f"{attempts} attempts -- widen node_count_range or extra_edge_prob."
        )
    return ood


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ood-count", type=int, default=6)
    parser.add_argument("--out", type=str, default=os.path.join(_HERE, "topology_pool.json"))
    args = parser.parse_args()

    train = build_train_pool()
    train_signatures = {_canonical_signature(t) for t in train}
    print(f"Train pool: {len(train)} topologies")
    for t in train:
        print(f"  {t.topology_id:28s} nodes={len(t.nodes)} edges={len(t.edges)}")

    ood = build_ood_pool(args.seed, args.ood_count, train_signatures)
    print(f"\nOOD pool: {len(ood)} topologies")
    for t in ood:
        roles = [t.nodes[n].role for n in sorted(t.nodes)]
        print(f"  {t.topology_id:20s} nodes={len(t.nodes)} edges={len(t.edges)} roles={roles}")

    records = (
        [topology_to_dict(t, "train") for t in train]
        + [topology_to_dict(t, "ood_test") for t in ood]
    )

    # Final sanity: every record round-trips through topology_from_dict + validate_topology.
    for r in records:
        validate_topology(topology_from_dict(r))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote {len(records)} topologies ({len(train)} train + {len(ood)} ood_test) to {args.out}")


if __name__ == "__main__":
    main()
