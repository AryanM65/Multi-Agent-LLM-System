"""
topology.py — Pipeline topology as data, not as hardcoded control flow.

Defines the NodeSpec / Topology data structures and all structural validation
functions.  The existing Retriever→Reasoner→Writer chain is expressed as a
specific Topology instance via default_chain_topology(), not as a special code
path in pipeline.py.

Design principles:
  - A Topology is a DAG (directed acyclic graph) of NodeSpec objects.
  - Edges encode data-flow direction: (src, dst) means src's output feeds dst's
    prompt as input.
  - Role ordering is enforced: retriever < reasoner < writer along every edge.
    This ensures the research's conceptual pipeline model is never violated by
    topology configuration mistakes.
  - validate_topology() must be called before executing any topology.  All
    run_pipeline() callers call it automatically.

NOTE (for future random-topology generator — dataset-generation phase, out of
scope here): any random topology generator must assign roles *before* generating
edges, and must only draw edges where ROLE_RANK[src.role] <= ROLE_RANK[dst.role].
This makes generated topologies valid by construction rather than requiring
post-hoc rejection sampling.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Role ordering (determines valid edge directions)
# ---------------------------------------------------------------------------

ROLE_RANK: Dict[str, int] = {
    "retriever": 0,
    "reasoner":  1,
    "writer":    2,
}

VALID_ROLES = set(ROLE_RANK.keys())


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class NodeSpec:
    """Specification for a single node in a topology.

    Attributes
    ----------
    node_id:
        Unique identifier within a topology (e.g. "retriever", "R1", "W2").
    role:
        Functional role — one of "retriever", "reasoner", "writer".
        Determines which uncertainty metric is used and which prompt format
        is applied.
    instruction:
        Role-specific system/task instruction string. This is the text
        injected at the top of every prompt built for this node.
    """
    node_id: str
    role: str
    instruction: str

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(
                f"NodeSpec '{self.node_id}': unknown role '{self.role}'. "
                f"Must be one of {sorted(VALID_ROLES)}."
            )


@dataclass
class Topology:
    """A DAG of NodeSpecs representing a multi-agent pipeline topology.

    Attributes
    ----------
    nodes:
        Mapping from node_id to NodeSpec.
    edges:
        List of (from_node_id, to_node_id) pairs.  'from' feeds into 'to'.
    topology_id:
        Human-readable / auto-generated identifier.  Written into every
        PipelineTrace and trial log record produced under this topology.
    """
    nodes: Dict[str, NodeSpec]
    edges: List[Tuple[str, str]]
    topology_id: str = ""


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

def validate_role_order(topo: Topology) -> None:
    """Raise ValueError if any edge violates retriever→reasoner→writer ordering.

    The role ordering must be non-decreasing along every directed edge.
    """
    for src, dst in topo.edges:
        src_role = topo.nodes[src].role
        dst_role = topo.nodes[dst].role
        if ROLE_RANK[src_role] > ROLE_RANK[dst_role]:
            raise ValueError(
                f"Invalid edge {src}->{dst}: a '{src_role}' node cannot feed "
                f"into a '{dst_role}' node (role order must be non-decreasing "
                f"along every directed edge)."
            )


def validate_topology(topo: Topology) -> None:
    """Full structural validation — call before executing any topology.

    Checks (in order):
      1. Every node has a valid role.
      2. Every edge references real node IDs.
      3. Role ordering is non-decreasing along every edge.
      4. The graph is a DAG (no cycles), verified via topological sort.
      5. No fully isolated node (every node must connect to at least one edge
         when there are multiple nodes — single-node topologies are allowed).

    Raises ValueError with a descriptive message on any violation.
    """
    # 1. Valid roles
    for nid, spec in topo.nodes.items():
        if spec.role not in VALID_ROLES:
            raise ValueError(
                f"Node '{nid}' has unknown role '{spec.role}'. "
                f"Must be one of {sorted(VALID_ROLES)}."
            )

    # 2. Every edge endpoint must reference a real node
    for src, dst in topo.edges:
        if src not in topo.nodes:
            raise ValueError(
                f"Edge ({src}, {dst}) references undefined node '{src}'."
            )
        if dst not in topo.nodes:
            raise ValueError(
                f"Edge ({src}, {dst}) references undefined node '{dst}'."
            )

    # 3. Role ordering
    validate_role_order(topo)

    # 4. Must be a DAG — detect via attempted topological sort
    topological_order(topo)  # raises ValueError if a cycle exists

    # 5. No fully isolated node (when multi-node topology)
    if len(topo.nodes) > 1:
        connected = {n for edge in topo.edges for n in edge}
        isolated = set(topo.nodes) - connected
        if isolated:
            raise ValueError(
                f"Isolated node(s) with no edges in a multi-node topology: "
                f"{isolated}.  Every node must participate in at least one edge."
            )


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def topological_order(topo: Topology) -> List[str]:
    """Return node IDs in topological (execution) order via Kahn's algorithm.

    Raises ValueError if the graph contains a cycle (not a valid DAG).
    """
    indegree: Dict[str, int] = {nid: 0 for nid in topo.nodes}
    children: Dict[str, List[str]] = {nid: [] for nid in topo.nodes}

    for src, dst in topo.edges:
        children[src].append(dst)
        indegree[dst] += 1

    queue: deque[str] = deque(nid for nid, d in indegree.items() if d == 0)
    order: List[str] = []

    while queue:
        nid = queue.popleft()
        order.append(nid)
        for child in children[nid]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if len(order) != len(topo.nodes):
        raise ValueError(
            "Topology contains a cycle — not a valid DAG. "
            "Check your edge definitions for circular dependencies."
        )
    return order


def parents_of(topo: Topology, node_id: str) -> List[str]:
    """Return the list of node IDs whose output feeds into node_id."""
    return [src for src, dst in topo.edges if dst == node_id]


def children_of(topo: Topology, node_id: str) -> List[str]:
    """Return the list of node IDs that consume node_id's output."""
    return [dst for src, dst in topo.edges if src == node_id]


def source_nodes(topo: Topology) -> List[str]:
    """Return node IDs with no incoming edges (they read raw question + context)."""
    has_parent = {dst for _, dst in topo.edges}
    return [nid for nid in topo.nodes if nid not in has_parent]


def sink_nodes(topo: Topology) -> List[str]:
    """Return node IDs with no outgoing edges (final answer producers)."""
    has_child = {src for src, _ in topo.edges}
    return [nid for nid in topo.nodes if nid not in has_child]


# ---------------------------------------------------------------------------
# Built-in topology factory
# ---------------------------------------------------------------------------

def default_chain_topology(
    retriever_instruction: str,
    reasoner_instruction: str,
    writer_instruction: str,
) -> Topology:
    """The existing 3-node chain expressed in the new topology format.

    Reproduces the current Retriever→Reasoner→Writer behavior exactly when
    used with run_pipeline().  This is the canonical topology for all existing
    experiments (Study 1, debug scripts, standalone agent CLIs).
    """
    return Topology(
        nodes={
            "retriever": NodeSpec("retriever", "retriever", retriever_instruction),
            "reasoner":  NodeSpec("reasoner",  "reasoner",  reasoner_instruction),
            "writer":    NodeSpec("writer",     "writer",    writer_instruction),
        },
        edges=[
            ("retriever", "reasoner"),
            ("reasoner",  "writer"),
        ],
        topology_id="chain_3node_default",
    )
