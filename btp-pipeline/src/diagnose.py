"""
diagnose.py — Retry-then-reprobe fault diagnostic protocol.

Protocol (cheapest explanation first):
  1. Retry with the SAME input (multiple times) → if uncertainty drops: NOISE
  2. Retry with a CLEAN (known-good) input     → if uncertainty drops: CONTAMINATION
  3. Neither retry helped                       → CEILING

Multi-node flagging strategy:
  When multiple nodes exceed the uncertainty threshold, ALL are diagnosed.
  The highest-uncertainty node's diagnosis is stored as PipelineTrace.diagnosed_label
  (primary result).  Full per-node diagnoses are stored in per_node_diagnoses.

Phase 3 addition: verify_against_baseline() computes a z-score comparing an
observed (faulted) uncertainty against the per-question clean baseline, providing
statistical confirmation that the injection actually caused a deviation.

Inference Gap (semantic drift):
  Implemented as an optional experimental metric.  It does NOT drive the
  diagnostic decision in Study 1.  Logged alongside self-consistency for Study 2.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

from src.config import DEFAULT_K, DEFAULT_RETRIES, UNCERTAINTY_THRESHOLD
from src.nodes import NodeResult, PipelineTrace, sample_node

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Threshold check
# ---------------------------------------------------------------------------

def needs_diagnosis(trace: PipelineTrace, node_name: str) -> bool:
    """Return True if the given node's lexical uncertainty exceeds the threshold."""
    return trace.node_results[node_name].uncertainty > UNCERTAINTY_THRESHOLD


# ---------------------------------------------------------------------------
# Phase 3 addition: baseline verification
# ---------------------------------------------------------------------------

def verify_against_baseline(
    observed_uncertainty: float,
    baseline_uncertainty: float,
    fault_config: dict,
    assumed_std: float = 0.10,
    z_thresh: float = 1.0,
) -> Dict:
    """Compute a z-score comparing observed (faulted) vs. clean baseline uncertainty.

    Provides statistical confirmation that the fault injection actually caused a
    measurable deviation at the target node, rather than being swallowed by the
    model's parametric memory or natural variance.

    Args:
        observed_uncertainty:  Uncertainty at the target node after fault injection.
        baseline_uncertainty:  Uncertainty at the same node from the clean control run.
        fault_config:          The fault condition dict (for context in the record).
        assumed_std:           Assumed per-node standard deviation (population estimate).
                               Replace with empirically estimated std once ~15-20 control
                               trials have accumulated (see implementation plan §3.7).
        z_thresh:              Z-score threshold to declare the target deviated.

    Returns:
        Dict with keys:
          "target_deviated": bool   — True if z > z_thresh (fault caused a real signal)
          "z_score": float          — raw z-score
          "observed": float         — passed-through for logging convenience
          "baseline": float         — passed-through for logging convenience
    """
    z = (observed_uncertainty - baseline_uncertainty) / (assumed_std + 1e-9)
    return {
        "target_deviated": z > z_thresh,
        "z_score": round(z, 4),
        "observed": round(observed_uncertainty, 4),
        "baseline": round(baseline_uncertainty, 4),
    }


# ---------------------------------------------------------------------------
# Inference Gap (optional / experimental — Study 2)
# ---------------------------------------------------------------------------

def compute_inference_gap(
    node_result: NodeResult,
    input_text: str,
    *,
    enabled: bool = True,
) -> Optional[float]:
    """Compute the Inference Gap (semantic drift) between node input and output.

    Uses cosine distance in sentence-embedding space.  Experimental metric —
    NOT used to drive diagnostic decisions in Study 1.

    Set enabled=False to skip computation (~200ms per call).
    """
    if not enabled:
        return None
    try:
        from src.uncertainty import get_embedder
        import numpy as np
        embedder = get_embedder()
        embs = embedder.encode([input_text, node_result.output], normalize_embeddings=True)
        gap = float(1.0 - float(np.dot(embs[0], embs[1])))
        return round(gap, 4)
    except ImportError:
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
    role: str = "",
) -> str:
    """Classify the cause of high uncertainty: 'noise', 'contamination', or 'ceiling'.

    Uses the retry-then-reprobe protocol.

    Args:
        node_name:          Node identifier.
        same_input_prompt:  Prompt with the original (possibly faulted) input.
        clean_input_prompt: Prompt with known-good input.
        k:                  Self-consistency samples per retry run.
        retries:            Number of same-input retry runs.
        role:               Node role passed through to sample_node for Phase 2
                            metrics (if omitted, lexical uncertainty is used for
                            the diagnostic decision regardless).

    Returns:
        One of: 'noise', 'contamination', 'ceiling'.
    """
    # Step 1 — Retry same input
    retry_results: List[NodeResult] = [
        sample_node(node_name, same_input_prompt, k=k, role=role)
        for _ in range(retries)
    ]
    avg_retry_uncertainty = sum(r.uncertainty for r in retry_results) / len(retry_results)
    if avg_retry_uncertainty < UNCERTAINTY_THRESHOLD:
        return "noise"

    # Step 2 — Retry with clean/known-good input
    clean_result = sample_node(node_name, clean_input_prompt, k=k, role=role)
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
    node_roles: Optional[Dict[str, str]] = None,
) -> None:
    """Diagnose all flagged nodes in a trace and update it in-place.

    Design choice (multi-node flagging):
      - All nodes exceeding UNCERTAINTY_THRESHOLD are diagnosed.
      - trace.diagnosed_label = diagnosis of the highest-uncertainty diagnosed node.
      - trace.per_node_diagnoses = full {node_name: label} dict.

    Phase 2 addition: node_roles dict allows passing role information through
    to diagnose() → sample_node() so that Phase 2 uncertainty metrics are
    also computed during retry runs (optional enhancement; diagnostic decisions
    still use lexical uncertainty for consistency with Study 1).

    Args:
        trace:        PipelineTrace produced by run_pipeline().
        node_prompts: {node_name: {"same": prompt, "clean": prompt}}.
        k:            Self-consistency samples per retry.
        retries:      Same-input retry count.
        compute_gap:  Whether to compute Inference Gap for flagged nodes.
        node_roles:   Optional {node_name: role} for Phase 2 metric computation.
    """
    flagged_nodes = [
        name for name in trace.node_results
        if needs_diagnosis(trace, name)
    ]

    if not flagged_nodes:
        trace.diagnosed_label = "no_fault_detected"
        return

    node_roles = node_roles or {}
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
        role = node_roles.get(node_name, "")
        label = diagnose(
            node_name,
            same_input_prompt=prompts["same"],
            clean_input_prompt=prompts["clean"],
            k=k,
            retries=retries,
            role=role,
        )
        per_node[node_name] = label

        # Optional: compute Inference Gap for this node.
        if compute_gap:
            trace.node_results[node_name].inference_gap = compute_inference_gap(
                trace.node_results[node_name],
                input_text=prompts["same"],
                enabled=True,
            )

    trace.per_node_diagnoses = per_node

    if per_node:
        highest_node = max(
            per_node.keys(),
            key=lambda n: trace.node_results[n].uncertainty,
        )
        trace.diagnosed_label = per_node[highest_node]
    else:
        trace.diagnosed_label = "no_fault_detected"
