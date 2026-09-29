"""
faults.py — Fault-injection library (Phase 3 redesign).

Phase 3 changes from the original:
  1. All fault conditions are now (fault_type × target_node) pairs.
     No fault implicitly targets only the Retriever anymore.
  2. Reproducibility: every injector uses a seeded random.Random (make_rng).
     No direct calls to the global random module anywhere in this file.
  3. Noise: explicitly node-scoped (only that node's temperature changes).
     Handled entirely in pipeline.apply_fault_to_prompt — inject_noise here
     only constructs the fault_config dict describing what to do.
  4. Contamination (Retriever target): proportional swap + optional
     plausibility-ranked distractor selection.
  5. Contamination (Reasoner / Writer target): corrupt the upstream parent's
     output with a plausible-but-wrong substitute from a pre-built bank.
  6. Ceiling: answer-survival verification is mandatory before logging.
     Trials where the answer survived stripping are discarded + logged to
     skipped_trials.jsonl (not silently dropped).
  7. Per-question no-fault control trial always runs first (handled in
     run_study1.py, but the fault_config=None convention is documented here).

Original helper functions (get_gold_context, get_distractor_context,
format_context) are preserved unchanged — they are used throughout the codebase.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Dict, List, Optional, Tuple

from src.config import DEFAULT_TEMPERATURE, NOISE_TEMPERATURE

# ---------------------------------------------------------------------------
# Context helpers (unchanged from original)
# ---------------------------------------------------------------------------

def get_gold_context(example: dict) -> List[Tuple[str, List[str]]]:
    """Return list of (title, sentences) for gold (supporting) paragraphs."""
    gold_titles = set(example["supporting_facts"]["title"])
    return [
        (title, sents)
        for title, sents in zip(
            example["context"]["title"], example["context"]["sentences"]
        )
        if title in gold_titles
    ]


def get_distractor_context(example: dict) -> List[Tuple[str, List[str]]]:
    """Return list of (title, sentences) for distractor (irrelevant) paragraphs."""
    gold_titles = set(example["supporting_facts"]["title"])
    return [
        (title, sents)
        for title, sents in zip(
            example["context"]["title"], example["context"]["sentences"]
        )
        if title not in gold_titles
    ]


def format_context(paragraphs: List[Tuple[str, List[str]]]) -> str:
    """Render paragraphs as a single string with bracketed title headers."""
    return "\n".join(
        f"[{title}] " + " ".join(sents)
        for title, sents in paragraphs
    )


# ---------------------------------------------------------------------------
# Reproducibility: seeded RNG keyed on (question_id, fault_type, target_node)
# ---------------------------------------------------------------------------

def make_rng(question_id: str, fault_type: str, target_node: str) -> random.Random:
    """Return a deterministic, isolated Random instance for this trial.

    The seed is derived from the MD5 hash of the concatenated key string,
    ensuring identical corrupted inputs for any re-run of the same trial.
    Using a per-trial instance (not seeding the global random state) keeps
    different trials independent of each other's execution order.

    Args:
        question_id:  The HotpotQA example's '_id' field.
        fault_type:   One of "noise", "contamination", "ceiling".
        target_node:  The node_id being faulted (e.g. "retriever").

    Returns:
        A seeded random.Random instance.
    """
    seed_str = f"{question_id}_{fault_type}_{target_node}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2 ** 32)
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Sentence splitting helper (used by inject_ceiling)
# ---------------------------------------------------------------------------

def split_sentences(paragraph_text: str) -> List[str]:
    """Split a paragraph string into individual sentences.

    Used to strip a fraction of sentences for ceiling fault injection.
    Simple regex split — adequate for the Wikipedia-style sentences in HotpotQA.
    """
    if not paragraph_text:
        return []
    # Split on sentence-ending punctuation followed by whitespace.
    sents = re.split(r"(?<=[.!?])\s+", paragraph_text.strip())
    return [s.strip() for s in sents if s.strip()]


# ---------------------------------------------------------------------------
# Fault 1: Noise — node-scoped temperature elevation
# ---------------------------------------------------------------------------

def inject_noise(
    example: dict,
    target_node: str,
    rng: random.Random,  # accepted for API consistency; not used (no random choice needed)
) -> Optional[Dict]:
    """Noise fault config: elevates temperature for a single targeted node.

    The fault_config dict returned here is interpreted by
    pipeline.apply_fault_to_prompt, which raises only that node's sampling
    temperature to NOISE_TEMPERATURE.  All other nodes run at DEFAULT_TEMPERATURE.

    This explicitly fixes the pre-Phase-3 ambiguity where noise scope
    (pipeline-wide vs. node-scoped) was unclear.

    Diagnostic expectation: high uncertainty at target node → recovers on
    same-input retry at normal temperature → classified as NOISE.

    Args:
        example:     HotpotQA dataset example dict.
        target_node: Which node to apply noise to ("retriever", "reasoner", "writer").
        rng:         Seeded RNG (unused here, kept for consistent injector signature).

    Returns:
        fault_config dict (never None for noise faults).
    """
    return {
        "type": "noise",
        "target_node": target_node,
        "true_label": "noise",
        "context": format_context(get_gold_context(example)),  # gold context unchanged
        "answer": example["answer"],
        # node temperature override applied in pipeline.apply_fault_to_prompt
        "sampling_override": {"temperature": NOISE_TEMPERATURE},
    }


# ---------------------------------------------------------------------------
# Fault 2: Contamination — bad input injection
# ---------------------------------------------------------------------------

def _pick_plausible_distractor(
    gold_para_text: str,
    distractors: List[str],
    embedder,
    top_k: int,
    rng: random.Random,
) -> str:
    """Pick a plausible distractor ranked by embedding similarity to the gold paragraph.

    Higher-similarity distractors are topically related but still wrong —
    a more realistic contamination scenario than a random off-topic paragraph.
    """
    import numpy as np
    gold_emb = embedder.encode([gold_para_text], normalize_embeddings=True)[0]
    dist_embs = embedder.encode(distractors, normalize_embeddings=True)
    sims = dist_embs @ gold_emb  # cosine (normalized)
    top_idx = list(np.argsort(sims)[-top_k:])
    return distractors[rng.choice(top_idx)]


def inject_contamination_retriever(
    example: dict,
    rng: random.Random,
    swap_fraction: float = 0.5,
    embedder=None,
    top_k_plausible: int = 3,
    target_node: str = "retriever",
) -> Optional[Dict]:
    """Contamination fault targeting the Retriever.

    Replaces a proportional fraction of gold paragraphs with distractor
    paragraphs.  If embedder is provided, distractors are plausibility-ranked
    (topically closest to the gold they replace); otherwise, chosen at random.

    Returns None if there are not enough distractor paragraphs to perform
    the swap.  Callers must log this skip explicitly (not silently drop it).

    Diagnostic expectation: high Retriever (and cascading Writer) uncertainty
    that recovers on clean-input retry → CONTAMINATION.

    Args:
        example:         HotpotQA dataset example dict.
        rng:             Seeded random.Random for reproducibility.
        swap_fraction:   Fraction of gold paragraphs to replace (default 0.5).
        embedder:        Optional SentenceTransformer for plausibility ranking.
        top_k_plausible: Top-k candidates to randomly sample from when ranking.
        target_node:     The actual retriever-role node's ID (e.g. "retriever_a").
                         Defaults to the literal "retriever" for backward
                         compatibility. NOTE (hygiene fix, 2026-09-25): this
                         field used to be hardcoded to "retriever" regardless
                         of what the caller passed, which was harmless in
                         practice only because every caller (run_study1/2/vllm.py)
                         discards this function's own "target_node" and rebuilds
                         it fresh with the real node ID -- but the dead field
                         was misleading and a latent risk if anything ever
                         trusted it directly. Now honors the real node ID.
    """
    gold = get_gold_context(example)
    distractors_raw = get_distractor_context(example)
    distractor_texts = [format_context([d]) for d in distractors_raw]

    n_swap = max(1, round(len(gold) * swap_fraction))
    if len(distractors_raw) < n_swap:
        return None  # not enough distractors — skip, log reason externally

    swap_idx = rng.sample(range(len(gold)), n_swap)
    corrupted = list(gold)
    for i in swap_idx:
        if embedder is not None and len(distractor_texts) >= top_k_plausible:
            gold_text = format_context([gold[i]])
            chosen_text = _pick_plausible_distractor(
                gold_text, distractor_texts, embedder, top_k_plausible, rng
            )
            chosen_idx = distractor_texts.index(chosen_text)
            corrupted[i] = distractors_raw[chosen_idx]
        else:
            corrupted[i] = rng.choice(distractors_raw)

    return {
        "type": "contamination",
        "target_node": target_node,
        "true_label": "contamination",
        "context": format_context(corrupted),
        "answer": example["answer"],
        "sampling_override": {"temperature": DEFAULT_TEMPERATURE},
    }


def inject_contamination_downstream(
    example: dict,
    target_node: str,
    clean_parent_output: str,
    substitute_bank: List[str],
    rng: random.Random,
    embedder=None,
    top_k: int = 5,
) -> Optional[Dict]:
    """Contamination fault targeting a downstream node (Reasoner or Writer).

    Corrupts the upstream parent's output after the parent ran cleanly, so the
    targeted node's own input is bad but its upstream neighbor was never touched.
    The corruption is stored in fault_config["_corrupted_input"] and applied
    inside pipeline.build_prompt when this node's prompt is assembled.

    The substitute is drawn from substitute_bank (a pool of clean outputs from
    other examples' runs of the same parent node).  If embedder is provided,
    the most-similar substitute is selected (a plausible-but-wrong alternative).

    Args:
        example:            HotpotQA dataset example dict.
        target_node:        "reasoner" or "writer".
        clean_parent_output: The clean output from the targeted node's parent.
        substitute_bank:    List of output strings from other examples (same parent node).
        rng:                Seeded random.Random for reproducibility.
        embedder:           Optional SentenceTransformer for similarity ranking.
        top_k:              Number of top candidates to sample from when ranking.

    Returns:
        fault_config dict with "_corrupted_input" field, or None if bank is empty.
    """
    # BUG FOUND AND FIXED (2026-09-25): substitute_bank is built once per
    # topology from ALL assigned questions' clean outputs (see
    # scripts/run_study2.py's build_substitute_bank), including the very
    # question currently being contaminated. Nothing previously excluded a
    # question's own clean output from being drawn as its own "wrong"
    # substitute -- with only 2-4 questions per topology (see plan.md's
    # coverage strategy), this was a real, unguarded risk of a
    # "contamination" trial whose corrupted input was actually still
    # correct. Exact-match self-exclusion below closes this.
    substitute_bank = [s for s in substitute_bank if s != clean_parent_output]

    if not substitute_bank:
        return None  # bank had no usable (non-self) substitute -- caller logs this skip

    if embedder is not None and len(substitute_bank) >= top_k:
        import numpy as np
        clean_emb = embedder.encode([clean_parent_output], normalize_embeddings=True)[0]
        bank_embs = embedder.encode(substitute_bank, normalize_embeddings=True)
        sims = bank_embs @ clean_emb
        top_idx = list(np.argsort(sims)[-top_k:])
        chosen = substitute_bank[rng.choice(top_idx)]
    else:
        chosen = rng.choice(substitute_bank)

    return {
        "type": "contamination",
        "target_node": target_node,
        "true_label": "contamination",
        "context": format_context(get_gold_context(example)),  # gold context passed normally
        "answer": example["answer"],
        "sampling_override": {"temperature": DEFAULT_TEMPERATURE},
        # Consumed by pipeline.build_prompt to replace the parent's output in
        # this node's prompt input block.
        "_corrupted_input": chosen,
    }


# ---------------------------------------------------------------------------
# Fault 3: Ceiling — answer-survival verification
# ---------------------------------------------------------------------------

def inject_ceiling(
    example: dict,
    target_node: str,
    rng: random.Random,
    # Raised from 0.5 -> 0.75 (2026-09-29): only 8% of ceiling trials
    # registered target_deviated=True against the old (buggy, assumed_std=0.10)
    # verifier, and node_len_z's oracle hit rate (0.716) shows length IS the
    # right signal -- the fault itself was just too mild to reliably shrink
    # output. Stripping more of the gold context forces a bigger length drop.
    strip_fraction: float = 0.75,
    role: Optional[str] = None,
) -> Optional[Dict]:
    """Ceiling fault — strips sentences from gold paragraphs then verifies the
    answer-bearing content was actually removed.

    REQUIRED: If the gold answer string survives in the stripped context,
    this trial is discarded (returns None).  The caller must log this skip
    to skipped_trials.jsonl with reason "ceiling_answer_survived" rather than
    silently dropping it.

    Phase 3 change: random sentence selection uses the provided seeded rng
    (no global random calls); and the answer-survival check is enforced before
    returning, making it impossible to accidentally log a mislabeled trial.

    For Retriever targets: strips sentences from the gold context.
    For Reasoner / Writer targets: applies a stricter constraint in the
    instruction prompt (word-limit hardening) rather than context stripping,
    since these nodes don't own the gold context.  The same answer-survival
    verification principle applies — we verify the instruction actually degrades
    output relative to the clean baseline (verified externally in the sanity
    checks; not checkable statically here).

    Args:
        example:       HotpotQA dataset example dict.
        target_node:   Which node to apply the ceiling fault to (node ID,
                       e.g. "retriever_a", "n0_r" -- NOT necessarily the
                       literal string "retriever").
        rng:           Seeded random.Random for reproducibility.
        strip_fraction: Fraction of sentences to remove (default 0.5).
        role:          The target node's ROLE ("retriever"/"reasoner"/"writer").
                       REQUIRED for correct dispatch on any topology other than
                       the plain 3-node chain -- see note below. If omitted,
                       falls back to the old (broken-for-custom-node-IDs)
                       target_node=="retriever" string check, kept only for
                       backward compatibility with old call sites.

    Returns:
        fault_config dict, or None if the answer survived (discard this trial).

    BUG FOUND AND FIXED (2026-09-25): this function used to dispatch on
    `target_node == "retriever"` (a literal node-ID string match) rather than
    the node's actual role. That only works for the plain default_chain
    topology's canonical node IDs. Every other topology in the pool uses
    custom node IDs (retriever_a, reasoner_1, n0_r, writer_b, ...), for which
    the check silently failed and fell through to _inject_ceiling_downstream,
    which ALSO checked by literal string ("reasoner"/"writer" only) and
    returned None (an "injection failed" skip) for anything else. Net effect:
    ceiling faults on any non-chain topology were being skipped almost every
    time, regardless of whether the fault was actually valid -- discovered by
    noticing ceiling ended up with only 20/118 successful trials (~17%) in
    the first full generation run, then tracing the skip reasons back to
    node IDs like "n2_r", "reasoner_1", "retriever_a" that should have worked.
    """
    is_retriever = (role == "retriever") if role is not None else (target_node == "retriever")
    if is_retriever:
        return _inject_ceiling_retriever(example, rng, strip_fraction)
    else:
        return _inject_ceiling_downstream(example, target_node, rng, role=role)


def _inject_ceiling_retriever(
    example: dict,
    rng: random.Random,
    strip_fraction: float,
) -> Optional[Dict]:
    """Ceiling for Retriever target: strip sentences from gold context."""
    gold = get_gold_context(example)
    answer_span = example["answer"].lower().strip()
    stripped_paras: List[Tuple[str, List[str]]] = []

    for title, sents in gold:
        n_total = len(sents)
        n_keep = max(1, round(n_total * (1 - strip_fraction)))
        # Randomly choose which indices to keep (seeded, reproducible).
        keep_idx = sorted(rng.sample(range(n_total), min(n_keep, n_total)))
        kept_sents = [sents[i] for i in keep_idx]
        stripped_paras.append((title, kept_sents))

    stripped_context = format_context(stripped_paras)

    # Verify the answer-bearing content was actually removed.
    if answer_span and answer_span in stripped_context.lower():
        # Answer survived stripping — this trial would not create a genuine
        # capability gap.  Discard; caller logs to skipped_trials.jsonl.
        return None

    return {
        "type": "ceiling",
        "target_node": "retriever",
        "true_label": "ceiling",
        "context": stripped_context,
        "answer": example["answer"],
        "sampling_override": {"temperature": DEFAULT_TEMPERATURE},
        "skip_reason": None,  # not skipped
    }


def _inject_ceiling_downstream(
    example: dict,
    target_node: str,
    rng: random.Random,
    role: Optional[str] = None,
) -> Optional[Dict]:
    """Ceiling for Reasoner / Writer target: harder-subtask instruction variant.

    These nodes don't own a gold context to strip, so we induce difficulty via
    an instruction constraint rather than content removal.  The gold context is
    passed normally to the pipeline; only this node's instruction is hardened.

    For Reasoner: constrain to ≤2 reasoning steps (forces information loss).
    For Writer:   constrain to ≤5 words (forces truncation of evidence).

    IMPORTANT: after running, verify this actually caused degraded output
    relative to the clean baseline (sanity check 3.8d).  The verification
    that the instruction change degrades performance is empirical (run-time),
    not checkable statically here.

    role: the target node's ROLE ("reasoner"/"writer"), used to select which
    hardened instruction template to apply. Falls back to matching target_node
    directly against the template keys (old, broken-for-custom-IDs behavior)
    if role is not provided, for backward compatibility only.
    """
    # Caps tightened 2026-09-29 (reasoner 2->1 step, writer 5->2 words): the
    # 2-step/5-word caps only shrank output enough to register as deviated in
    # 8% of ceiling trials. See strip_fraction note above for the same
    # rationale -- the goal is a length drop big enough for node_len_z to
    # reliably fire, not just a nominally "harder" instruction.
    hardened_instructions = {
        "reasoner": (
            "You are a Reasoning agent. Given the evidence below, attempt to "
            "derive the answer to the question. You may use AT MOST 1 reasoning "
            "step. If you cannot determine the answer within this constraint, "
            "state your best guess.\n\n"
            "End your response with a final line in exactly this format:\n"
            "FINAL ANSWER: <your one-sentence conclusion>"
        ),
        "writer": (
            "You are a Writer agent. Given the reasoning trace below, produce "
            "a final answer to the question. Output AT MOST 2 WORDS — no explanation."
        ),
    }

    lookup_key = role if role is not None else target_node
    if lookup_key not in hardened_instructions:
        return None  # unsupported target for downstream ceiling

    return {
        "type": "ceiling",
        "target_node": target_node,
        "true_label": "ceiling",
        "context": format_context(get_gold_context(example)),  # gold context unchanged
        "answer": example["answer"],
        "sampling_override": {"temperature": DEFAULT_TEMPERATURE},
        # The hardened instruction is stored here and must be applied by the
        # caller (run_study1) by overriding the NodeSpec.instruction for this
        # node when constructing the fault_config's topology.
        "_hardened_instruction": hardened_instructions[lookup_key],
    }


# ---------------------------------------------------------------------------
# Legacy single-argument interface (backward compatibility for old call-sites)
# ---------------------------------------------------------------------------
# The original inject_noise / inject_contamination / inject_ceiling functions
# accepted only (example,) or (example, n_distractors).  These wrappers
# preserve that interface so that run_local_debug.py (Phase 1-3) continues
# to work without changes for the simple no-target-node case.

def _legacy_rng(example: dict, fault_type: str) -> random.Random:
    """Deterministic RNG for legacy single-arg injectors."""
    qid = example.get("_id", example.get("question", "unknown"))
    return make_rng(str(qid), fault_type, "retriever")
