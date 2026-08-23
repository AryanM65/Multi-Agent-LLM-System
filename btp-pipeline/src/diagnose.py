"""
diagnose.py — Retry-then-reprobe fault diagnostic protocol.

Protocol (cheapest explanation first — §5.4 of the brief):
  1. Retry with the SAME input (multiple times) → if uncertainty drops: NOISE
  2. Retry with a CLEAN (known-good) input     → if uncertainty drops: CONTAMINATION
  3. Neither retry helped                       → CEILING

Multi-node flagging strategy (chosen by user, §10 item 4):
  When multiple nodes exceed the uncertainty threshold, ALL are diagnosed.
  The highest-uncertainty node's diagnosis is stored as PipelineTrace.diagnosed_label
  (primary result).  Full per-node diagnoses are stored in PipelineTrace.per_node_diagnoses
  and also written to the JSONL for post-hoc analysis.

Inference Gap (semantic drift):
  Implemented as an *optional* experimental metric.  It is computed alongside
  self-consistency but does NOT drive the diagnostic decision in Study 1.
  See compute_inference_gap() and the TODO note for Study 2.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

from src.config import DEFAULT_K, DEFAULT_RETRIES, UNCERTAINTY_THRESHOLD
from src.nodes import NodeResult, PipelineTrace, sample_node

if TYPE_CHECKING:
    # Avoid circular import at runtime; only needed for type hints.
    pass


# ---------------------------------------------------------------------------
# Threshold check
# ---------------------------------------------------------------------------

def needs_diagnosis(trace: PipelineTrace, node_name: str) -> bool:
    """Return True if the given node's uncertainty exceeds the threshold.

    UNCERTAINTY_THRESHOLD is a placeholder (0.3) — recalibrate from Phase 1-2
    baseline distributions before trusting Study 1 confusion matrix numbers.
    """
    return trace.node_results[node_name].uncertainty > UNCERTAINTY_THRESHOLD


# ---------------------------------------------------------------------------
# Inference Gap (optional / experimental — Study 2)
# ---------------------------------------------------------------------------

def compute_inference_gap(
    node_result: NodeResult,
    input_text: str,
    *,
    enabled: bool = True,
) -> Optional[float]:
    """Compute the Inference Gap (semantic drift) between the node's input and
    output using cosine distance in sentence-embedding space.

    This is an EXPERIMENTAL metric that is NOT used to drive the diagnostic
    decision in Study 1.  It is logged alongside self-consistency uncertainty
    for exploratory analysis.

    # TODO (Study 2): evaluate whether Inference Gap improves contamination
    #   detection over self-consistency alone (e.g., contaminated output may
    #   drift semantically from the faulted input in a characteristic way).

    Set enabled=False to skip computation (saves ~200ms per call).
    """
    if not enabled:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        # Load model lazily so processes that don't call this incur no cost.
        # The 'all-MiniLM-L6-v2' model is fast and fits comfortably in RAM.
        if not hasattr(compute_inference_gap, "_model"):
            compute_inference_gap._model = SentenceTransformer("all-MiniLM-L6-v2")

        model = compute_inference_gap._model
        embs = model.encode([input_text, node_result.output], normalize_embeddings=True)
        # Cosine distance = 1 - cosine similarity; higher → more semantic drift.
        gap = float(1.0 - float(np.dot(embs[0], embs[1])))
        return round(gap, 4)
    except ImportError:
        # sentence-transformers not installed — silently skip.
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core diagnostic protocol
# ---------------------------------------------------------------------------

def diagnose(
    node_name: str,
    same_input_prompt: str,
    clean_input_prompt: str,
    k: int = DEFAULT_K,
    retries: int = DEFAULT_RETRIES,
) -> str:
    """Classify the cause of high uncertainty into 'noise', 'contamination',
    or 'ceiling' using the retry-then-reprobe protocol.

    Args:
        node_name:          Name of the node being diagnosed (e.g. 'retriever').
        same_input_prompt:  The prompt with the *original* (possibly faulted) input.
        clean_input_prompt: The prompt with known-good input.
        k:                  Self-consistency samples per retry run.
        retries:            Number of same-input retry runs (>= 2 recommended;
                            a single retry is weak evidence — soft ceilings can
                            intermittently succeed by chance).

    Returns:
        One of: 'noise', 'contamination', 'ceiling'.

    NOTE (retries=2 validation): after local Phase 1-2 runs, check whether
    any ceiling-fault examples "fake recover" across all 2 retries.  If so,
    raise DEFAULT_RETRIES to 3 in config.py.
    """
    # Step 1 — Retry same input (checking for noise: instability, not bad input)
    retry_results: List[NodeResult] = [
        sample_node(node_name, same_input_prompt, k=k)
        for _ in range(retries)
    ]
    avg_retry_uncertainty = sum(r.uncertainty for r in retry_results) / len(retry_results)
    if avg_retry_uncertainty < UNCERTAINTY_THRESHOLD:
        return "noise"

    # Step 2 — Retry with clean/known-good input (checking for contamination)
    clean_result = sample_node(node_name, clean_input_prompt, k=k)
    if clean_result.uncertainty < UNCERTAINTY_THRESHOLD:
        return "contamination"

    # Step 3 — Neither retry helped → capability ceiling
    return "ceiling"


# ---------------------------------------------------------------------------
# Batch diagnosis: all nodes above threshold for one trace
# ---------------------------------------------------------------------------

def diagnose_trace(
    trace: PipelineTrace,
    node_prompts: Dict[str, Dict[str, str]],
    k: int = DEFAULT_K,
    retries: int = DEFAULT_RETRIES,
    compute_gap: bool = True,
) -> None:
    """Diagnose all flagged nodes in a trace and update it in-place.

    DESIGN CHOICE (multi-node flagging, §10 item 4):
      - All nodes exceeding UNCERTAINTY_THRESHOLD are diagnosed.
      - trace.diagnosed_label is set to the diagnosis of the *highest*
        uncertainty node (most likely root cause).
      - trace.per_node_diagnoses stores the full {node_name: label} dict.

    Args:
        trace:        PipelineTrace produced by run_pipeline().
        node_prompts: {node_name: {"same": prompt, "clean": prompt}} for every
                      node that might be diagnosed.  If a node is not in this
                      dict but exceeds threshold, diagnosis is skipped with a
                      warning.
        k:            Self-consistency samples per retry run.
        retries:      Same-input retry count.
        compute_gap:  Whether to compute Inference Gap for flagged nodes.
    """
    flagged_nodes = [
        name for name in trace.node_results
        if needs_diagnosis(trace, name)
    ]

    if not flagged_nodes:
        trace.diagnosed_label = "no_fault_detected"
        return

    per_node: Dict[str, str] = {}
    for node_name in flagged_nodes:
        if node_name not in node_prompts:
            import warnings
            warnings.warn(
                f"Node '{node_name}' exceeded threshold but no prompts provided "
                f"for diagnosis — skipping.",
                stacklevel=2,
            )
            continue

        prompts = node_prompts[node_name]
        label = diagnose(
            node_name,
            same_input_prompt=prompts["same"],
            clean_input_prompt=prompts["clean"],
            k=k,
            retries=retries,
        )
        per_node[node_name] = label

        # Optional: compute Inference Gap for this node.
        if compute_gap:
            trace.node_results[node_name].inference_gap = compute_inference_gap(
                trace.node_results[node_name],
                input_text=prompts["same"],  # input to this node
                enabled=True,
            )

    trace.per_node_diagnoses = per_node

    if per_node:
        # Primary label = diagnosis of the highest-uncertainty diagnosed node.
        highest_node = max(
            per_node.keys(),
            key=lambda n: trace.node_results[n].uncertainty,
        )
        trace.diagnosed_label = per_node[highest_node]
    else:
        trace.diagnosed_label = "no_fault_detected"
