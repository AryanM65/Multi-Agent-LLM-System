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
  - get_normalizer(role) provides the normalizer function for each role.

Work-stream 1 — Mock / CPU backend:
  - When MOCK_MODE is True (BTP_MOCK=1 env-var), _generate_once() returns
    a deterministic mock response instead of calling mlx_lm.
  - Mock responses are seeded by (node_name, sample_index, prompt_hash) so
    they are reproducible and vary enough across k samples to produce
    non-trivial uncertainty values for full pipeline testing on Windows.
"""

from __future__ import annotations

import hashlib
import random
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from src.config import (
    DEFAULT_K,
    DEFAULT_TEMPERATURE,
    MAX_TOKENS,
    MAX_TOKENS_REASONER,
    MOCK_MODE,
    MODEL,
    OLLAMA_MODEL,
    BACKEND,
    VLLM_MODEL,
    VLLM_QUANTIZATION,
    VLLM_DTYPE,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_MAX_MODEL_LEN,
)

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
# vLLM model singleton loader
# ---------------------------------------------------------------------------
_vllm_llm = None


def _get_vllm_llm():
    """Lazy-load and cache the vLLM LLM instance (once per process).

    GPU-only. This is a single-prompt-at-a-time loader used by sample_node's
    existing k-sample-loop structure (see _vllm_generate below) -- it does NOT
    do vLLM's real batching-across-trials optimization (see plan.md Section
    4.3 / scripts/run_study_vllm.py for the batched generation path used for
    actual bulk dataset generation). This loader exists so the same
    backend-agnostic run_pipeline()/sample_node() code can be used directly
    for calibration runs (scripts/calibrate_vllm_model.py) without writing a
    separate calibration-only script.
    """
    global _vllm_llm
    if _vllm_llm is None:
        from vllm import LLM
        print(f"[nodes] Loading model '{VLLM_MODEL}' via vLLM "
              f"(quantization={VLLM_QUANTIZATION}, dtype={VLLM_DTYPE}, once per process)...")
        _vllm_llm = LLM(
            model=VLLM_MODEL,
            quantization=VLLM_QUANTIZATION,
            dtype=VLLM_DTYPE,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len=VLLM_MAX_MODEL_LEN,
        )
        print("[nodes] Model loaded.")
    return _vllm_llm


def _vllm_generate(prompt: str, temperature: float, token_limit: int) -> Tuple[str, bool]:
    """Generate one sample using vLLM.

    Qwen2.5-Instruct is a standard instruction-tuned model (no hidden
    reasoning pass like gpt-oss), so there is no separate "thinking" field
    and no thinking-fallback path needed here -- the fallback flag is always
    False for this backend. If output ever comes back empty for a reason
    other than hidden reasoning (e.g. the model choosing to emit nothing),
    that's a real data-quality signal worth investigating directly rather
    than papering over with a fallback, unlike the gpt-oss case.

    Uses the model's chat template via vLLM's .chat() convenience method so
    prompts are formatted the same way (with special tokens etc.) that the
    model was instruction-tuned to expect.
    """
    from vllm import SamplingParams

    llm = _get_vllm_llm()
    params = SamplingParams(temperature=temperature, max_tokens=token_limit)
    outputs = llm.chat([{"role": "user", "content": prompt}], params, use_tqdm=False)
    text = outputs[0].outputs[0].text.strip()
    return text, False


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def _ollama_generate(prompt: str, temperature: float, token_limit: int) -> Tuple[str, bool]:
    """Generate one sample using the Ollama server.

    Strips Qwen3's <think>...</think> scratchpad from the response if present
    (only relevant if OLLAMA_MODEL is ever switched back to a Qwen3-style model
    that inlines thinking in `content`).

    gpt-oss models (the current OLLAMA_MODEL) reason by default and return that
    reasoning in a separate `message["thinking"]` field, not inlined in
    `content`. This reasoning pass is NON-DETERMINISTIC in length and appears
    budget-seeking: raising num_predict does not reliably fix empty content —
    measured examples ranged from ~570 to ~7600 chars of thinking across
    different calls/budgets for the same prompt, sometimes still leaving
    content="" even at num_predict=1200. Neither `think=False` nor
    `think="low"` reliably suppresses this for the gpt-oss:20b-cloud proxy
    (both confirmed via scripts/debug_gptoss_thinking.py to still produce
    hundreds-to-thousands of chars of thinking and, in some calls, empty
    content). Raising num_predict alone is therefore necessary but NOT
    sufficient.

    As a result, this function falls back to extracting an answer from
    `thinking` whenever `content` comes back empty: first by looking for a
    "final answer:" marker inside `thinking` (works well for the Reasoner,
    whose prompt asks for that marker); otherwise by taking the last
    non-empty line of `thinking` as a best-effort answer, since the model's
    reasoning usually converges on the answer near the end even when it never
    emits it into `content`. This guarantees a non-empty sample in nearly all
    cases instead of silently returning "" (the root cause of ~95% empty
    Retriever samples, and in one pilot question 100% empty Reasoner/Writer
    samples, seen before this fallback was added).

    Args:
        prompt:      Full prompt string to send.
        temperature: Sampling temperature.
        token_limit: Maximum tokens to generate (must cover reasoning + answer).

    Returns:
        (text, used_fallback) — text falls back to a thinking-derived answer
        if `content` is empty ("" only if `thinking` is also empty/absent);
        used_fallback is True whenever that fallback path fired, so callers
        can flag/filter degraded samples instead of treating them as
        equivalent to a native FINAL ANSWER extraction.
    """
    import ollama
    import re as _re

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": temperature,
            "num_predict": token_limit,
        },
    )
    message = response["message"]
    text = message.get("content", "") or ""
    # Strip Qwen3 thinking block if present
    text = _re.sub(r"<think>[\s\S]*?</think>", "", text, flags=_re.IGNORECASE).strip()

    used_fallback = False
    if not text:
        thinking = message.get("thinking") or ""
        if thinking:
            match = _re.search(r"final answer:?\s*(.*)", thinking, _re.IGNORECASE | _re.DOTALL)
            if match:
                text = match.group(1).strip().strip("*_").split("\n")[0].strip()
            if not text:
                lines = [ln.strip() for ln in thinking.strip().split("\n") if ln.strip()]
                text = lines[-1] if lines else ""
            if text:
                used_fallback = True

    return text, used_fallback


# ---------------------------------------------------------------------------
# Mock backend (Work-stream 1)
# ---------------------------------------------------------------------------

# Mock output banks keyed by role.  Enough variety to produce non-trivial k=3
# self-consistency uncertainty (some samples agree, some differ slightly).
_MOCK_OUTPUTS: Dict[str, List[str]] = {
    "retriever": [
        "Scott Derrickson is an American director, screenwriter, and producer.",
        "Ed Wood was an American filmmaker, actor, and author.",
        "Scott Derrickson was born in Denver, Colorado, United States.",
        "Ed Wood is the subject of the 1994 Tim Burton biographical film.",
        "Both Scott Derrickson and Ed Wood are American.",
    ],
    "reasoner": [
        "Scott Derrickson is American. Ed Wood is also American. "
        "FINAL ANSWER: Yes, they were of the same nationality — both American.",
        "The evidence shows Derrickson is American and Ed Wood is American. "
        "FINAL ANSWER: Yes, both Scott Derrickson and Ed Wood are American.",
        "Derrickson hails from Denver, Colorado (USA). Ed Wood was born in the USA too. "
        "FINAL ANSWER: Yes, they share American nationality.",
        "Based on the evidence, both are from the United States. "
        "FINAL ANSWER: Yes, both are American.",
        "Scott Derrickson is American. Ed Wood is American too. "
        "FINAL ANSWER: Yes, same nationality.",
    ],
    "writer": [
        "Yes, they were both American.",
        "Yes.",
        "Yes, both were American.",
        "Yes, they share the same nationality (American).",
        "Yes, both Scott Derrickson and Ed Wood are American.",
    ],
}


def _mock_generate(node_name: str, sample_index: int, prompt: str, role: str) -> str:
    """Return a deterministic mock response for the given node and sample index.

    The response is drawn from _MOCK_OUTPUTS[role] using a seed derived from
    (node_name, sample_index, prompt_hash).  Two different prompts will produce
    different but consistently sampled outputs across re-runs.

    Args:
        node_name:    Node identifier (for seeding).
        sample_index: Which of the k samples this is (0-indexed).
        prompt:       The full prompt string (only its hash is used for seeding).
        role:         Node role — determines which bank to draw from.
    """
    prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % (2 ** 32)
    seed = (prompt_hash + sample_index * 1337 + hash(node_name)) % (2 ** 32)
    rng = random.Random(seed)
    bank = _MOCK_OUTPUTS.get(role, _MOCK_OUTPUTS["writer"])
    return rng.choice(bank)


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

    # Per-sample flag: True if this sample's text came from the thinking-field
    # fallback (src/nodes.py:_ollama_generate) rather than a native model
    # `content` response. Fallback-derived samples are guaranteed non-empty
    # but may be lower-quality truncated-reasoning fragments rather than
    # clean answers — downstream consumers should filter/down-weight on this
    # rather than treating all samples as equivalent. Always all-False for
    # mock/MLX backends. Same length as `samples`.
    used_thinking_fallback: List[bool] = field(default_factory=list)

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

def _generate_once(
    node_name: str,
    sample_index: int,
    prompt: str,
    role: str,
    temperature: float,
    token_limit: int,
) -> Tuple[str, bool]:
    """Generate one sample — dispatches to mock, Ollama, or MLX backend.

    Returns (text, used_thinking_fallback). The fallback flag is always False
    for mock/MLX backends; only the Ollama path (gpt-oss reasoning models)
    can set it True — see _ollama_generate's docstring.

    Backend selection priority:
      1. MOCK_MODE=True  → deterministic mock stubs (no LLM needed)
      2. BACKEND='ollama' → local/cloud Ollama server (works on Windows/CPU/GPU)
      3. BACKEND='vllm'  → vLLM, GPU-only (Kaggle T4 etc.)
      4. BACKEND='mlx'   → Apple Silicon MLX (macOS only)
    """
    if MOCK_MODE:
        return _mock_generate(node_name, sample_index, prompt, role), False

    if BACKEND == "ollama":
        return _ollama_generate(prompt, temperature, token_limit)

    if BACKEND == "vllm":
        return _vllm_generate(prompt, temperature, token_limit)

    # Real MLX generation (Apple Silicon)
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
    return generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=token_limit,
        sampler=sampler,
        verbose=False,
    ), False


def sample_node(
    node_name: str,
    prompt: str,
    k: int = DEFAULT_K,
    temperature: float = DEFAULT_TEMPERATURE,
    normalize_fn: Callable[[str], str] = hotpotqa_normalize,
    role: str = "",
) -> NodeResult:
    """Run one pipeline node with self-consistency sampling.

    Calls _generate_once() k times sequentially (never concurrent —
    hardware constraint).  Computes:
      1. Lexical uncertainty (always): agreement rate over normalised outputs.
      2. Role-specific additional uncertainty (Phase 2 — when role is provided):
           role='reasoner' → semantic clustering over extracted conclusions
           role='writer'   → semantic clustering over raw outputs
           role='retriever'→ Jaccard set-overlap (multi-item outputs only)

    Mock mode (BTP_MOCK=1): calls _mock_generate() instead of the real model.
    All downstream pipeline logic (uncertainty, fault injection, diagnosis) is
    exercised identically — only the generation step is stubbed.

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
    used_thinking_fallback: List[bool] = []
    token_limit = MAX_TOKENS_REASONER if role == "reasoner" else MAX_TOKENS

    for i in range(k):
        try:
            out, fell_back = _generate_once(node_name, i, prompt, role, temperature, token_limit)
            samples.append(out)
            used_thinking_fallback.append(fell_back)
        except Exception as exc:
            raise RuntimeError(
                f"Generation failed on sample {i + 1}/{k} "
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
        used_thinking_fallback=used_thinking_fallback,
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
