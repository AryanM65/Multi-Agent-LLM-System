"""
pipeline.py — Topology-agnostic multi-agent pipeline engine.

Phase 1 redesign: replaces the hardcoded Retriever→Reasoner→Writer call
sequence with a generic engine that executes any valid DAG of NodeSpecs
defined by a Topology object.

The existing 3-node chain is preserved exactly via default_chain_topology()
(defined in topology.py) — it is now one specific Topology instance, not a
special code path.

Key public interface:
  RETRIEVER_INSTRUCTION   — module-level constant (unchanged text)
  REASONER_INSTRUCTION    — updated in Phase 2 with FINAL ANSWER: marker
  WRITER_INSTRUCTION      — module-level constant (unchanged text)
  build_prompt(...)       — builds the prompt for any node given parent outputs
  apply_fault_to_prompt() — node-scoped fault hook (Phase 3)
  run_pipeline(...)       — executes a Topology over question + context

Invariants (enforced here and throughout the codebase):
  - Per-node uncertainties are NEVER collapsed to a scalar.
  - All model calls are sequential (no threading / asyncio).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from src.config import DEFAULT_K, DEFAULT_TEMPERATURE, NOISE_TEMPERATURE
from src.nodes import NodeResult, PipelineTrace, get_normalizer, sample_node
from src.topology import Topology, parents_of, topological_order, validate_topology

# ---------------------------------------------------------------------------
# Prompt instruction strings (module-level constants)
# ---------------------------------------------------------------------------
# These become the NodeSpec.instruction field values when building topologies.
# Kept here (not in topology.py) so that call-sites that import these
# strings directly (e.g. for building node_prompts in diagnose logic) keep
# working without importing topology.

RETRIEVER_INSTRUCTION = (
    "You are a Retriever agent. Given the question and candidate paragraphs, "
    "select and return only the sentences that are directly relevant to answering "
    "the question. Do not add any commentary or explanation — only return the "
    "relevant sentences."
)

# Phase 2 update: added FINAL ANSWER: marker so extract_conclusion() can
# reliably parse the conclusion without heuristic fallback.
REASONER_INSTRUCTION = (
    "You are a Reasoning agent. Given the evidence below, reason step by step "
    "to derive the answer to the question. Show your full chain of thought.\n\n"
    "End your response with a final line in exactly this format:\n"
    "FINAL ANSWER: <your one-sentence conclusion>"
)

WRITER_INSTRUCTION = (
    "You are a Writer agent. Given the reasoning trace below, produce a final, "
    "concise answer to the question. Output only the answer — no explanation, "
    "no preamble."
)


# ---------------------------------------------------------------------------
# Backward-compatible prompt builder functions
# ---------------------------------------------------------------------------
# These are preserved so that run_study1.py and diagnose.py callers that
# build node_prompts manually (using retriever_prompt / reasoner_prompt /
# writer_prompt) continue to work without modification.

def retriever_prompt(question: str, context: str) -> str:
    return (
        f"{RETRIEVER_INSTRUCTION}\n\n"
        f"Question: {question}\n"
        f"Paragraphs:\n{context}\n\n"
        "Relevant sentences:"
    )


def reasoner_prompt(question: str, evidence: str) -> str:
    return (
        f"{REASONER_INSTRUCTION}\n\n"
        f"Question: {question}\n"
        f"Evidence: {evidence}\n\n"
        "Reasoning:"
    )


def writer_prompt(question: str, reasoning: str) -> str:
    return (
        f"{WRITER_INSTRUCTION}\n\n"
        f"Question: {question}\n"
        f"Reasoning: {reasoning}\n\n"
        "Final answer:"
    )


# ---------------------------------------------------------------------------
# Topology engine: prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    topo: Topology,
    node_id: str,
    outputs: Dict[str, str],
    question: str,
    context: str,
    fault_config: Optional[dict] = None,
) -> str:
    """Build the full prompt for node_id given parent outputs produced so far.

    Source nodes (no incoming edges) receive the raw question + context block,
    just like the original Retriever node.

    Non-source nodes receive a block of "[Input from <parent_id>]: <output>"
    lines, one per parent (in edge-definition order).

    Phase 3 hook: when fault_config specifies contamination or ceiling at a
    downstream (non-source) node, apply corrupt_downstream_input to the
    parent output before embedding it in this node's prompt.  This is
    handled here (not in apply_fault_to_prompt) because the corruption
    must modify what this node *sees* from its parent, not this node's
    own generated content.
    """
    node = topo.nodes[node_id]
    parent_ids = parents_of(topo, node_id)

    if not parent_ids:
        # Source node: Retriever pattern — reads question + context directly.
        input_block = f"Paragraphs:\n{context}\n\nQuestion: {question}"
    else:
        # Downstream node: assemble parent outputs.
        lines = []
        for pid in parent_ids:
            parent_output = outputs[pid]
            # Phase 3: contamination / ceiling targeting this node — corrupt the
            # parent's output before it enters this node's prompt.
            if (
                fault_config is not None
                and fault_config.get("type") in ("contamination", "ceiling")
                and fault_config.get("target_node") == node_id
                and fault_config.get("_corrupted_input") is not None
            ):
                # The corrupted replacement was pre-computed and stored in
                # fault_config["_corrupted_input"] by run_pipeline before
                # calling build_prompt — use it here.
                parent_output = fault_config["_corrupted_input"]
            lines.append(f"[Input from {pid}]: {parent_output}")
        input_block = "\n".join(lines) + f"\n\nQuestion: {question}"

    return f"{node.instruction}\n\n{input_block}"


# ---------------------------------------------------------------------------
# Topology engine: fault hook
# ---------------------------------------------------------------------------

def apply_fault_to_prompt(
    prompt: str,
    temperature: float,
    fault_config: dict,
) -> Tuple[str, float]:
    """Apply a node-scoped fault to the prompt and/or sampling temperature.

    Called inside run_pipeline immediately before sample_node() for the
    targeted node.  The fault_config["target_node"] must equal the current
    node_id for this hook to activate.

    Fault semantics (Phase 3):
      noise         → only this node's sampling temperature changes to
                      NOISE_TEMPERATURE.  The prompt is unchanged.
                      This explicitly fixes the pre-Phase-3 ambiguity where
                      noise scope (pipeline-wide vs node-scoped) was unclear.
      contamination → prompt corruption handled in build_prompt via
                      fault_config["_corrupted_input"]; no-op here.
      ceiling       → same as contamination — input stripping handled upstream.

    Returns:
        (possibly modified prompt, possibly modified temperature)
    """
    fault_type = fault_config.get("type", "")

    if fault_type == "noise":
        # Node-scoped noise: only this node's temperature is elevated.
        return prompt, NOISE_TEMPERATURE

    # contamination and ceiling: prompt was already modified in build_prompt.
    return prompt, temperature


# ---------------------------------------------------------------------------
# Topology engine: main pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    topo: Topology,
    question: str,
    context: str,
    k: int = DEFAULT_K,
    temperature: float = DEFAULT_TEMPERATURE,
    fault_config: Optional[dict] = None,
) -> PipelineTrace:
    """Execute a Topology over a question + context and return a PipelineTrace.

    Runs nodes in topological order (Kahn's algorithm).  For each node:
      1. Build the prompt from parent outputs (via build_prompt).
      2. Apply fault hook if this node is the fault target (via apply_fault_to_prompt).
      3. Sample self-consistently (via sample_node).
      4. Store NodeResult in outputs and in the trace.

    Phase 3 fault_config structure:
      None                                     — no fault (clean control run)
      {"type": "noise",         "target_node": node_id}
      {"type": "contamination", "target_node": node_id,
       "_corrupted_input": <pre-computed corrupted text>}
      {"type": "ceiling",       "target_node": node_id,
       "_corrupted_input": <pre-computed stripped text>}

    The "_corrupted_input" key is populated by the caller (run_study1.py)
    before passing fault_config here, so run_pipeline itself does not need
    to know how contamination or ceiling inputs are generated.

    Returns:
        PipelineTrace with all NodeResults and topology_id set.
    """
    validate_topology(topo)
    order = topological_order(topo)

    outputs: Dict[str, str] = {}
    trace = PipelineTrace(question=question, topology_id=topo.topology_id)

    for node_id in order:
        role = topo.nodes[node_id].role

        # Build this node's prompt from parent outputs (fault-aware for downstream targets).
        prompt = build_prompt(topo, node_id, outputs, question, context, fault_config)
        node_temperature = temperature

        # Apply fault to this specific node if it's the target.
        if fault_config is not None and fault_config.get("target_node") == node_id:
            prompt, node_temperature = apply_fault_to_prompt(prompt, node_temperature, fault_config)

        result = sample_node(
            node_name=node_id,
            prompt=prompt,
            k=k,
            temperature=node_temperature,
            normalize_fn=get_normalizer(role),
            role=role,
        )
        outputs[node_id] = result.output
        trace.add(result)

    return trace


# ---------------------------------------------------------------------------
# Convenience factory: the default 3-node chain topology
# ---------------------------------------------------------------------------

def default_topology() -> Topology:
    """Return the canonical Retriever→Reasoner→Writer topology.

    Wraps topology.default_chain_topology() with the module-level instruction
    strings.  This is the one-stop call for any script that wants to run the
    standard pipeline without importing from topology.py directly.
    """
    from src.topology import default_chain_topology
    return default_chain_topology(
        RETRIEVER_INSTRUCTION,
        REASONER_INSTRUCTION,
        WRITER_INSTRUCTION,
    )
