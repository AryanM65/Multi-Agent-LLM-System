"""
topologies.py — Named multi-agent topology factory functions.

All topologies here are valid DAGs expressible with the NodeSpec / Topology
data structures defined in src/topology.py.  Role ordering is non-decreasing
along every edge (ROLE_RANK: retriever=0, reasoner=1, writer=2).

Available topologies
--------------------
  "chain"                 — default 3-node linear chain  (Retriever→Reasoner→Writer)
  "dual_retriever_fanin"  — two independent Retrievers fan into one Reasoner
  "parallel_reasoner"     — one Retriever fans out to two independent Reasoners
                            that both feed one Writer  (shared Retriever parent)
  "deep_chain"            — extra Reasoner refines the first one's CoT

Use get_topology(name) to look up a topology by string name (for --topology CLI flag).

Research notes (why these shapes)
----------------------------------
dual_retriever_fanin:
  Hypothesis: if the two Retrievers disagree on evidence selection, the Reasoner
  uncertainty will rise even on clean inputs.  The Jaccard uncertainty of the two
  Retriever outputs becomes a natural leading indicator of downstream trouble.

parallel_reasoner:
  Hypothesis: two independent reasoning paths from the same evidence should
  converge on the same conclusion, yielding low Writer uncertainty.  A fault at
  Reasoner-A that does NOT affect Reasoner-B creates a distinctive asymmetric
  uncertainty signature at the Writer (partial contamination).

deep_chain:
  Hypothesis: a second Reasoner refining the first CoT should reduce Reasoner-1
  uncertainty propagation into the Writer (error correction).  If Reasoner-1 is
  faulted, Reasoner-2 may partially recover — useful for studying fault attenuation.
"""

from __future__ import annotations

from src.pipeline import (
    RETRIEVER_INSTRUCTION,
    REASONER_INSTRUCTION,
    WRITER_INSTRUCTION,
)
from src.topology import NodeSpec, Topology


# ---------------------------------------------------------------------------
# Shared instruction variants
# ---------------------------------------------------------------------------

# For topologies with multiple Retrievers / Reasoners, we use numbered
# variants so the node_id is unique but the role is the same.
# The instruction itself is identical — differentiation comes only from the
# random seed at sampling time (different prompt hash → different mock output).

_RETRIEVER_B_INSTRUCTION = RETRIEVER_INSTRUCTION  # identical role, unique node_id

_REASONER_A_INSTRUCTION = REASONER_INSTRUCTION
_REASONER_B_INSTRUCTION = REASONER_INSTRUCTION  # same instruction, independent samples

# For Reasoner-2 in the deep chain: refines the first Reasoner's output.
_REASONER_REFINE_INSTRUCTION = (
    "You are a Refining Reasoning agent. You will receive a first-pass chain-of-thought "
    "produced by a previous Reasoning agent.  Your task is to critically review it, "
    "identify any logical gaps or errors, and produce a corrected, more concise reasoning "
    "trace.\n\n"
    "End your response with a final line in exactly this format:\n"
    "FINAL ANSWER: <your one-sentence conclusion>"
)


# ---------------------------------------------------------------------------
# Topology A — default chain (alias for pipeline.default_topology())
# ---------------------------------------------------------------------------

def chain_topology() -> Topology:
    """Standard 3-node linear chain: Retriever → Reasoner → Writer.

    This is identical to pipeline.default_topology() — included here so
    callers can use get_topology("chain") without importing from pipeline.py.
    """
    return Topology(
        nodes={
            "retriever": NodeSpec("retriever", "retriever", RETRIEVER_INSTRUCTION),
            "reasoner":  NodeSpec("reasoner",  "reasoner",  REASONER_INSTRUCTION),
            "writer":    NodeSpec("writer",     "writer",    WRITER_INSTRUCTION),
        },
        edges=[
            ("retriever", "reasoner"),
            ("reasoner",  "writer"),
        ],
        topology_id="chain",
    )


# ---------------------------------------------------------------------------
# Topology B — dual_retriever_fanin
# ---------------------------------------------------------------------------
#
#  Retriever-A ─┐
#               ├──→ Reasoner ──→ Writer
#  Retriever-B ─┘
#
# Two independent Retrievers read the same raw context.  The Reasoner sees
# both outputs as separate [Input from retriever_a] / [Input from retriever_b]
# blocks in its prompt, then synthesises across them.

def dual_retriever_fanin_topology() -> Topology:
    """Two independent Retrievers fan into a single Reasoner.

    Research goal: test how evidence disagreement between Retriever-A and
    Retriever-B propagates into Reasoner uncertainty.  The Jaccard distance
    between the two Retrievers' item sets is a direct leading indicator.

    Node IDs:
        retriever_a, retriever_b → reasoner → writer

    Both Retriever nodes are source nodes (no incoming edges — they read raw
    question + context directly).
    """
    return Topology(
        nodes={
            "retriever_a": NodeSpec("retriever_a", "retriever", RETRIEVER_INSTRUCTION),
            "retriever_b": NodeSpec("retriever_b", "retriever", _RETRIEVER_B_INSTRUCTION),
            "reasoner":    NodeSpec("reasoner",    "reasoner",  REASONER_INSTRUCTION),
            "writer":      NodeSpec("writer",       "writer",    WRITER_INSTRUCTION),
        },
        edges=[
            ("retriever_a", "reasoner"),
            ("retriever_b", "reasoner"),
            ("reasoner",    "writer"),
        ],
        topology_id="dual_retriever_fanin",
    )


# ---------------------------------------------------------------------------
# Topology C — parallel_reasoner  (shared-Retriever fan-out)
# ---------------------------------------------------------------------------
#
#              ┌──→ Reasoner-A ──┐
#  Retriever ──┤                  ├──→ Writer
#              └──→ Reasoner-B ──┘
#
# One Retriever fans out to two independent Reasoners.  Both Reasoners
# receive the same Retriever output (as separate prompt inputs) and produce
# independent reasoning chains.  The Writer synthesises both chains.
#
# Research goal: asymmetric fault detection.  If only Reasoner-A is faulted,
# the Writer sees one clean + one noisy input — producing a characteristic
# partial-contamination uncertainty signature that differs from a fault on
# the sole Reasoner in the chain topology.

def parallel_reasoner_topology() -> Topology:
    """Single Retriever fans out to two independent Reasoners, both feeding Writer.

    Node IDs:
        retriever → reasoner_a → writer
        retriever → reasoner_b → writer

    The Writer prompt will contain two blocks:
        [Input from reasoner_a]: <CoT from Reasoner A>
        [Input from reasoner_b]: <CoT from Reasoner B>
    """
    return Topology(
        nodes={
            "retriever":  NodeSpec("retriever",  "retriever", RETRIEVER_INSTRUCTION),
            "reasoner_a": NodeSpec("reasoner_a", "reasoner",  _REASONER_A_INSTRUCTION),
            "reasoner_b": NodeSpec("reasoner_b", "reasoner",  _REASONER_B_INSTRUCTION),
            "writer":     NodeSpec("writer",      "writer",    WRITER_INSTRUCTION),
        },
        edges=[
            ("retriever",  "reasoner_a"),
            ("retriever",  "reasoner_b"),
            ("reasoner_a", "writer"),
            ("reasoner_b", "writer"),
        ],
        topology_id="parallel_reasoner",
    )


# ---------------------------------------------------------------------------
# Topology D — deep_chain
# ---------------------------------------------------------------------------
#
#  Retriever → Reasoner-1 → Reasoner-2 → Writer
#
# A second Reasoner explicitly refines the first one's chain-of-thought.
# Research goal: study fault attenuation — does a refining Reasoner reduce
# Reasoner-1 fault propagation into the Writer?

def deep_chain_topology() -> Topology:
    """Linear 4-node chain with two sequential Reasoners.

    Node IDs:
        retriever → reasoner_1 → reasoner_2 → writer

    Reasoner-2 uses the _REASONER_REFINE_INSTRUCTION (review + correct the
    first CoT) rather than the standard REASONER_INSTRUCTION.  It receives
    Reasoner-1's full chain-of-thought output as its input block.

    ROLE_RANK validity: reasoner(1) → reasoner(1) is non-decreasing → valid.
    """
    return Topology(
        nodes={
            "retriever":  NodeSpec("retriever",  "retriever", RETRIEVER_INSTRUCTION),
            "reasoner_1": NodeSpec("reasoner_1", "reasoner",  REASONER_INSTRUCTION),
            "reasoner_2": NodeSpec("reasoner_2", "reasoner",  _REASONER_REFINE_INSTRUCTION),
            "writer":     NodeSpec("writer",      "writer",    WRITER_INSTRUCTION),
        },
        edges=[
            ("retriever",  "reasoner_1"),
            ("reasoner_1", "reasoner_2"),
            ("reasoner_2", "writer"),
        ],
        topology_id="deep_chain",
    )


# ---------------------------------------------------------------------------
# Registry — get_topology(name) → Topology
# ---------------------------------------------------------------------------

_REGISTRY = {
    "chain":                chain_topology,
    "dual_retriever_fanin": dual_retriever_fanin_topology,
    "parallel_reasoner":    parallel_reasoner_topology,
    "deep_chain":           deep_chain_topology,
}


def get_topology(name: str) -> Topology:
    """Look up and instantiate a named topology.

    Args:
        name: One of "chain", "dual_retriever_fanin", "parallel_reasoner", "deep_chain".

    Returns:
        A freshly instantiated Topology object.

    Raises:
        ValueError if name is not in the registry.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown topology '{name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]()


def all_topologies() -> list:
    """Return a list of all registered topology names."""
    return list(_REGISTRY.keys())
