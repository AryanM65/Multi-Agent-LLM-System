"""
nodes.py — Self-consistency sampling for a single pipeline node.

Core design rules (from the project brief):
  - NodeResult and PipelineTrace store per-node uncertainties separately;
    they are NEVER merged into a scalar aggregate anywhere in this file.
  - All model calls are sequential — no threading, no asyncio.
  - hotpotqa_normalize() replaces the placeholder default_normalize()
    specified in the brief (§5.1 action item).

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
# The model is loaded once on first call and cached.  This avoids the ~3s
# reload on every sample_node() invocation across a pipeline run.
_mlx_model = None
_mlx_tokenizer = None

def _get_model():
    """Lazy-load and cache the MLX model + tokenizer."""
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
    """Official SQuAD-style normalisation used by the HotpotQA evaluation
    script.  Lowercase → strip punctuation → remove articles → collapse
    whitespace.

    This replaces the placeholder default_normalize(text.strip().lower())
    mentioned in the brief (§5.1 action item).  Using naive strip().lower()
    would undercount agreement on Writer output where superficial phrasing
    differences (e.g. leading 'the') cause false disagreements.
    """
    text = text.lower()
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


# Keep the brief's original name as an alias so existing call-sites that
# pass normalize_fn=default_normalize still work without changes.
default_normalize = hotpotqa_normalize

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NodeResult:
    """Stores raw samples + derived uncertainty for one pipeline node.

    uncertainty = 1 - agreement_rate
        0.0 → all k samples agreed (fully self-consistent)
        1.0 → no two samples agreed (fully inconsistent)
    """
    node_name: str
    output: str               # The sample that matches the modal normalised answer
    uncertainty: float        # ∈ [0, 1]; higher = less self-consistent
    samples: List[str] = field(default_factory=list)

    # Optional: semantic-drift score (Inference Gap) — Study 2 / experimental.
    # Populated by compute_inference_gap() in diagnose.py if enabled.
    inference_gap: Optional[float] = None


@dataclass
class PipelineTrace:
    """Collects NodeResults from an entire pipeline run for one question.

    CRITICAL: node_results stores per-node uncertainty separately.
    Never pass trace.uncertainties() as a scalar to any function.
    """
    question: str
    node_results: Dict[str, NodeResult] = field(default_factory=dict)
    true_label: Optional[str] = None          # set at fault-injection time
    diagnosed_label: Optional[str] = None     # set by diagnose.py
    gold_answer: Optional[str] = None

    # DESIGN CHOICE (multi-node flagging):
    # When multiple nodes exceed threshold we diagnose ALL of them and store
    # the full per-node diagnoses here.  diagnosed_label is set to the
    # diagnosis for the HIGHEST-uncertainty node (most likely root cause).
    # See run_study1.py for details.
    per_node_diagnoses: Dict[str, str] = field(default_factory=dict)

    def add(self, result: NodeResult) -> None:
        """Register a NodeResult.  Preserves insertion order (Python 3.7+)."""
        self.node_results[result.node_name] = result

    def uncertainties(self) -> Dict[str, float]:
        """Return per-node uncertainty dict.  NEVER collapse this to a scalar."""
        return {name: r.uncertainty for name, r in self.node_results.items()}

    def inference_gaps(self) -> Dict[str, Optional[float]]:
        """Return per-node inference gap dict (None if not computed)."""
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
) -> NodeResult:
    """Run one pipeline node with self-consistency sampling.

    Calls mlx_lm.generate() k times sequentially (never concurrent —
    hardware constraint, §2.2 of the brief).  Computes agreement rate
    over normalised outputs; uncertainty = 1 - agreement_rate.

    Returns the NodeResult whose .output is the modal (most-agreed-on)
    raw sample before normalisation.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    samples: List[str] = []
    for i in range(k):
        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
            model, tokenizer = _get_model()

            # Apply the Qwen3 instruct chat template so the model sees
            # proper <|im_start|>system / user / assistant tokens.
            messages = [{"role": "user", "content": prompt}]
            # enable_thinking=False suppresses Qwen3's internal chain-of-thought
            # scratchpad, which would confuse self-consistency comparison.
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            # Give Reasoner a bit more token budget for chain-of-thought.
            token_limit = MAX_TOKENS_REASONER if node_name == "reasoner" else MAX_TOKENS
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
                f"mlx_lm generation failed on sample {i+1}/{k} for node '{node_name}': {exc}"
            ) from exc

    normed = [normalize_fn(s) for s in samples]
    counts = Counter(normed)
    top_normed, top_count = counts.most_common(1)[0]

    agreement_rate = top_count / len(normed)
    uncertainty = round(1.0 - agreement_rate, 4)

    # Pick the first raw sample whose normalised form matches the modal answer.
    best_output = next(s for s, n in zip(samples, normed) if n == top_normed)

    return NodeResult(
        node_name=node_name,
        output=best_output,
        uncertainty=uncertainty,
        samples=samples,
    )
