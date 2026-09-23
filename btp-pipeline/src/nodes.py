"""
nodes.py — Self-consistency sampling for a single pipeline node.

Core design rules:
  - NodeResult and PipelineTrace store per-node uncertainties separately;
    they are NEVER merged into a scalar aggregate anywhere in this file.
  - All model calls are sequential — no threading, no asyncio.
  - hotpotqa_normalize() is the official SQuAD-style normalisation used by
    the HotpotQA evaluation script.

Phase 2 additions:
  - NodeResult gains optional fields for semantic/Jaccard uncertainty and
    Reasoner extracted conclusions.  All new fields default to None for
    backward compatibility with any existing deserialization code.
  - sample_node() gains a `role` parameter and branches to compute the
    additional role-appropriate uncertainty metric alongside the existing
    lexical metric.
  - get_normalizer(role) provides the normalizer function for each role,
    keeping role dispatch in one place.

Backend: mlx_lm (Apple Silicon / Metal).
The model is loaded ONCE as a module-level singleton via _get_model() to
avoid reloading 9×k times per pipeline run (loading the 8B model takes ~3s).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.config import DEFAULT_K, DEFAULT_TEMPERATURE, MODEL, MAX_TOKENS, MAX_TOKENS_REASONER

# ---------------------------------------------------------------------------
# MLX model singleton loader
# ---------------------------------------------------------------------------
_mlx_model = None
_mlx_tokenizer = None


def _get_model():
    """Lazy-load and cache the MLX model + tokenizer (once per process)."""
    global _mlx_model, _mlx_tokenizer
    if _mlx_model is None:
        from mlx_lm import load
        print(f"[nodes] Loading model '{MODEL}' via mlx_lm (once per process)...")
        _mlx_model, _mlx_tokenizer = load(MODEL)
        print("[nodes] Model loaded.")
    return _mlx_model, _mlx_tokenizer


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def hotpotqa_normalize(text: str) -> str:
    """Official SQuAD-style normalisation used by the HotpotQA evaluation script.

    Lowercase → strip punctuation → remove articles → collapse whitespace.

    This replaces naive strip().lower() which would cause false disagreements
    on 'the X' vs 'X', inflating uncertainty scores.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Keep the old alias so existing call-sites that pass normalize_fn=default_normalize still work.
default_normalize = hotpotqa_normalize


def get_normalizer(role: str) -> Callable[[str], str]:
    """Return the normalizer function appropriate for a given node role.

    All roles currently use hotpotqa_normalize for lexical scoring (the primary
    uncertainty measure).  The semantic / Jaccard measures live in uncertainty.py
    and are branched on in sample_node() — they are not normalizer functions.

    This dispatch point exists so Phase 2's role-specific sampling (via
    sample_node's role= parameter) has a clean configuration hook.
    """
    if role in ("retriever", "reasoner", "writer"):
        return hotpotqa_normalize
    raise ValueError(
        f"get_normalizer: unknown role '{role}'. "
        f"Must be one of 'retriever', 'reasoner', 'writer'."
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NodeResult:
    """Stores raw samples + derived uncertainty for one pipeline node.

    Primary metric (always populated):
      uncertainty: float        lexical exact-match uncertainty = 1 - agreement_rate
                                0.0 = all k samples agreed; 1.0 = no two agreed

    Phase 2 extended metrics (populated based on role, None otherwise):
      uncertainty_semantic:     semantic-clustering-based uncertainty (Reasoner, Writer)
      uncertainty_jaccard:      Jaccard set-overlap uncertainty (Retriever multi-item)
      conclusions:              extracted FINAL ANSWER strings from Reasoner samples
      item_frequencies:         per-item inclusion rate across k Retriever samples

    Experimental (Study 2):
      inference_gap:            cosine distance (input→output in embedding space)
    """
    node_name: str
    output: str               # The sample that matches the modal normalised answer
    uncertainty: float        # Primary lexical metric ∈ [0, 1]
    samples: List[str] = field(default_factory=list)

    # Phase 2: role-specific additional uncertainty metrics
    uncertainty_semantic: Optional[float] = None     # Reasoner, Writer
    uncertainty_jaccard: Optional[float] = None       # Retriever (multi-item case)
    conclusions: Optional[List[str]] = None           # Reasoner: extracted conclusions
    item_frequencies: Optional[Dict[str, float]] = None  # Retriever: per-item inclusion

    # Study 2 / experimental
    inference_gap: Optional[float] = None


@dataclass
class PipelineTrace:
    """Collects NodeResults from an entire pipeline run for one question.

    CRITICAL: node_results stores per-node uncertainty separately.
    Never pass trace.uncertainties() as a scalar to any function.

    Phase 1 addition: topology_id records which topology was used to produce
    this trace (written to every trial log record for reproducibility).
    """
    question: str
    node_results: Dict[str, NodeResult] = field(default_factory=dict)
    true_label: Optional[str] = None          # set at fault-injection time
    diagnosed_label: Optional[str] = None     # set by diagnose.py
    gold_answer: Optional[str] = None
    topology_id: str = ""                     # Phase 1: from Topology.topology_id

    # DESIGN CHOICE (multi-node flagging):
    # All nodes above threshold are diagnosed; diagnosed_label = highest-
    # uncertainty node's diagnosis.  Full per-node map kept for post-hoc work.
    per_node_diagnoses: Dict[str, str] = field(default_factory=dict)

    def add(self, result: NodeResult) -> None:
        """Register a NodeResult.  Preserves insertion order (Python 3.7+)."""
        self.node_results[result.node_name] = result

    def uncertainties(self) -> Dict[str, float]:
        """Return per-node LEXICAL uncertainty dict.  NEVER collapse to a scalar."""
        return {name: r.uncertainty for name, r in self.node_results.items()}

    def semantic_uncertainties(self) -> Dict[str, Optional[float]]:
        """Return per-node semantic uncertainty (None if not computed for that role)."""
        return {name: r.uncertainty_semantic for name, r in self.node_results.items()}

    def inference_gaps(self) -> Dict[str, Optional[float]]:
        """Return per-node inference gap (None if not computed)."""
        return {name: r.inference_gap for name, r in self.node_results.items()}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_node(
    node_name: str,
    prompt: str,
    k: int = DEFAULT_K,
    temperature: float = DEFAULT_TEMPERATURE,
    normalize_fn: Callable[[str], str] = hotpotqa_normalize,
    role: str = "",
) -> NodeResult:
    """Run one pipeline node with self-consistency sampling.

    Calls mlx_lm.generate() k times sequentially (never concurrent —
    hardware constraint).  Computes:
      1. Lexical uncertainty (always): agreement rate over normalised outputs.
      2. Role-specific additional uncertainty (Phase 2 — when role is provided):
           role='reasoner' → semantic clustering over extracted conclusions
           role='writer'   → semantic clustering over raw outputs
           role='retriever'→ Jaccard set-overlap (multi-item outputs only)

    Returns a NodeResult with the modal (most-agreed-on) raw sample as .output.

    Args:
        node_name:    Node identifier (used in error messages and NodeResult).
        prompt:       Full prompt string (after topology build_prompt applied).
        k:            Number of self-consistency samples.
        temperature:  Sampling temperature for this node (may differ from
                      pipeline-level default when a noise fault is targeted here).
        normalize_fn: Normalisation function for lexical scoring.
        role:         Node role ('retriever', 'reasoner', 'writer').  Drives
                      which Phase 2 uncertainty metric is computed.  Empty string
                      disables Phase 2 metrics (backward-compatible default).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    samples: List[str] = []
    # Use Reasoner's token budget for the reasoner role; default otherwise.
    token_limit = MAX_TOKENS_REASONER if role == "reasoner" else MAX_TOKENS

    for i in range(k):
        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
            model, tokenizer = _get_model()

            messages = [{"role": "user", "content": prompt}]
            # enable_thinking=False suppresses Qwen3's internal <think> scratchpad.
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            sampler = make_sampler(temp=temperature)
            resp = generate(
                model,
                tokenizer,
                prompt=formatted,
                max_tokens=token_limit,
                sampler=sampler,
                verbose=False,
            )
            samples.append(resp)
        except Exception as exc:
            raise RuntimeError(
                f"mlx_lm generation failed on sample {i + 1}/{k} "
                f"for node '{node_name}' (role='{role}'): {exc}"
            ) from exc

    # --- Lexical uncertainty (primary metric, always computed) ---
    normed = [normalize_fn(s) for s in samples]
    counts = Counter(normed)
    top_normed, top_count = counts.most_common(1)[0]
    agreement_rate = top_count / len(normed)
    uncertainty_lexical = round(1.0 - agreement_rate, 4)
    best_output = next(s for s, n in zip(samples, normed) if n == top_normed)

    result_kwargs: dict = dict(
        node_name=node_name,
        output=best_output,
        uncertainty=uncertainty_lexical,
        samples=samples,
    )

    # --- Phase 2: role-specific additional metrics ---
    if role == "reasoner":
        from src.uncertainty import extract_conclusion, semantic_uncertainty
        conclusions = [extract_conclusion(s) for s in samples]
        result_kwargs["conclusions"] = conclusions
        result_kwargs["uncertainty_semantic"] = semantic_uncertainty(conclusions)

    elif role == "writer":
        from src.uncertainty import semantic_uncertainty
        result_kwargs["uncertainty_semantic"] = semantic_uncertainty(samples)

    elif role == "retriever":
        from src.uncertainty import (
            parse_retrieved_items,
            jaccard_uncertainty,
            per_item_inclusion_frequency,
        )
        item_sets = [parse_retrieved_items(s) for s in samples]
        # Jaccard is meaningfully different from lexical only when the Retriever
        # returns multiple items (multi-item set comparison vs whole-string match).
        if any(len(s) > 1 for s in item_sets):
            result_kwargs["uncertainty_jaccard"] = jaccard_uncertainty(item_sets)
            result_kwargs["item_frequencies"] = per_item_inclusion_frequency(item_sets)

    return NodeResult(**result_kwargs)
