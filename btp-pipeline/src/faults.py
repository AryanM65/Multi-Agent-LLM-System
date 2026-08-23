"""
faults.py — Fault-injection library for Study 1.

Three fault types (matching §1.3 of the brief):

  noise          — gold context, elevated temperature (sampling instability)
  contamination  — distractor paragraphs replace gold context (bad input)
  ceiling        — gold context stripped of half its sentences (capability gap)

Each inject_* function returns a dict with keys:
  question, context, answer, true_label, sampling_override

Returns None when the example lacks sufficient distractors for contamination
injection — the caller (run_labeled_trial) must skip such examples.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from src.config import DEFAULT_TEMPERATURE, NOISE_TEMPERATURE

# ---------------------------------------------------------------------------
# Context helpers
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
# Fault injectors
# ---------------------------------------------------------------------------

def inject_noise(example: dict) -> Dict:
    """Noise fault — gold context is intact; fault is purely sampling noise.

    The elevated temperature (1.2 vs 0.7) makes the model's sampling
    distribution flatter, causing higher expected disagreement across k
    samples even though the *input* is perfectly good.

    Diagnostic expectation: high initial uncertainty, recovers on same-input
    retry at normal temperature.
    """
    return {
        "question": example["question"],
        "context": format_context(get_gold_context(example)),
        "answer": example["answer"],
        "true_label": "noise",
        "sampling_override": {"temperature": NOISE_TEMPERATURE},
    }


def inject_contamination(example: dict, n_distractors: int = 2) -> Optional[Dict]:
    """Contamination fault — distractor paragraphs replace gold context.

    Uses HotpotQA's shipped distractor paragraphs (§3.1 of the brief), so
    this is a naturalistic bad-input scenario, not an adversarially crafted one.

    Returns None if the example has fewer than n_distractors distractor
    paragraphs — the caller must skip such examples.

    Diagnostic expectation: high uncertainty that does NOT recover on
    same-input retry, but DOES recover on clean (gold) input retry.
    """
    distractors = get_distractor_context(example)
    if len(distractors) < n_distractors:
        return None
    chosen = random.sample(distractors, n_distractors)
    return {
        "question": example["question"],
        "context": format_context(chosen),
        "answer": example["answer"],
        "true_label": "contamination",
        "sampling_override": {"temperature": DEFAULT_TEMPERATURE},
    }


def inject_ceiling(example: dict, strip_fraction: float = 0.5) -> Dict:
    """Ceiling fault — strip back half of each gold paragraph's sentences.

    Removes the evidence required for a complete multi-hop answer, creating
    a genuine reasoning gap.  Using gold context (not distractors) keeps the
    topic on-point so the model isn't confused by off-topic text — it just
    lacks the specific cross-paragraph evidence it needs.

    Diagnostic expectation: high uncertainty that persists through BOTH
    same-input retry AND clean-input retry (because 'clean' is the same
    incomplete gold here — there's no good input to fall back on).

    NOTE: The 'clean_input_prompt' passed to diagnose() for ceiling faults
    should use the FULL gold context, not the stripped version.  This is
    handled in run_study1.py by always passing get_gold_context(example) as
    the clean reference, regardless of the injected fault type.
    """
    gold = get_gold_context(example)
    weakened = []
    for title, sents in gold:
        keep_n = max(1, int(len(sents) * (1 - strip_fraction)))
        weakened.append((title, sents[:keep_n]))
    return {
        "question": example["question"],
        "context": format_context(weakened),
        "answer": example["answer"],
        "true_label": "ceiling",
        "sampling_override": {"temperature": DEFAULT_TEMPERATURE},
    }
