"""
uncertainty.py — Role-specific uncertainty metrics for multi-agent LLM nodes.

Design principles (per §2.1 of the implementation plan):
  - The existing lexical uncertainty metric (exact-match after normalization) is
    NEVER deleted.  Every NodeResult still stores it in the .uncertainty field.
  - New metrics are stored in additional optional fields alongside the lexical
    metric, enabling before/after comparison against the existing Study 1 log.
  - Each role gets the metric best suited to its failure mode:
      Reasoner → semantic clustering over extracted FINAL ANSWER conclusions
      Writer   → semantic clustering over raw (already concise) outputs
      Retriever → Jaccard similarity over parsed item sets (multi-item outputs)

All heavy imports (sentence_transformers, numpy) are deferred until first use;
a process that never calls the semantic/jaccard functions pays zero loading cost.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Embedding model singleton (lazy-loaded)
# ---------------------------------------------------------------------------

_embedder = None


def get_embedder():
    """Lazy-load and cache the sentence embedding model (once per process)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print("[uncertainty] Loading SentenceTransformer 'all-MiniLM-L6-v2'...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("[uncertainty] Embedder loaded.")
    return _embedder


# ---------------------------------------------------------------------------
# Reasoner: FINAL ANSWER extraction
# ---------------------------------------------------------------------------

def extract_conclusion(raw_output: str) -> str:
    """Extract the conclusion sentence from a Reasoner output.

    Looks for a line matching 'FINAL ANSWER: ...' (case-insensitive) placed
    there by the updated REASONER_INSTRUCTION prompt.

    Fallback: if the model didn't include the marker, returns the last non-empty
    line of the output.  Every caller should track how often the fallback is
    reached (via check_conclusion_marker()) to monitor prompt compliance.
    """
    match = re.search(
        r"final\s+answer\s*:?\s*(.*)",
        raw_output,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        extracted = match.group(1).strip().strip("*_").split("\n")[0].strip()
        if extracted:
            return extracted

    # Fallback: last non-empty line
    lines = [ln.strip() for ln in raw_output.splitlines() if ln.strip()]
    return lines[-1] if lines else raw_output.strip()


def check_conclusion_marker(raw_output: str) -> bool:
    """Return True if the model included the FINAL ANSWER: marker correctly."""
    return bool(re.search(r"final\s+answer\s*:", raw_output, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Semantic uncertainty (Reasoner + Writer)
# ---------------------------------------------------------------------------

def semantic_uncertainty(
    texts: List[str],
    sim_threshold: float = 0.85,
) -> float:
    """Cluster texts by embedding similarity; return normalized entropy.

    Algorithm:
      1. Embed all texts.
      2. Greedily assign each text to the first existing cluster whose
         representative has cosine similarity >= sim_threshold; otherwise
         open a new cluster.
      3. Compute cluster-size distribution.
      4. Return H(distribution) / log(n) — normalized entropy ∈ [0, 1].
         0 = all texts in one cluster (fully consistent).
         1 = each text in its own cluster (maximally inconsistent).

    Args:
        texts:          List of text strings to compare (k samples).
        sim_threshold:  Minimum cosine similarity to merge two texts into
                        the same cluster.  0.85 is a reasonable default for
                        short answer strings; lower it (e.g. 0.75) for longer
                        reasoning chains.

    Returns:
        float in [0, 1].  Returns 0.0 for a single-element list.
    """
    if len(texts) <= 1:
        return 0.0

    embedder = get_embedder()
    embs = embedder.encode(texts, normalize_embeddings=True)  # L2-normalized → dot = cosine
    n = len(texts)

    cluster_of: List[int] = [-1] * n
    cluster_reps: List[int] = []

    for i in range(n):
        assigned = False
        for c_idx, rep_idx in enumerate(cluster_reps):
            sim = float(np.dot(embs[i], embs[rep_idx]))  # cosine (both normalized)
            if sim >= sim_threshold:
                cluster_of[i] = c_idx
                assigned = True
                break
        if not assigned:
            cluster_reps.append(i)
            cluster_of[i] = len(cluster_reps) - 1

    counts = np.bincount(cluster_of, minlength=len(cluster_reps))
    probs = counts / n
    # Shannon entropy; clip to avoid log(0)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    max_entropy = float(np.log(n))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


# ---------------------------------------------------------------------------
# Jaccard uncertainty (Retriever multi-item outputs)
# ---------------------------------------------------------------------------

def jaccard_uncertainty(item_sets: List[List[str]]) -> float:
    """Compute uncertainty for multi-item Retriever outputs via Jaccard distance.

    Average pairwise Jaccard similarity over all (i, j) pairs →
    uncertainty = 1 - mean_pairwise_jaccard.

    When the Retriever returns a variable-size set of selected items, exact
    string match on the whole set-as-string treats "agree on 2/3 items" the
    same as "agree on 0/3 items".  Jaccard correctly gives partial credit for
    partial overlap.

    Args:
        item_sets:  List of k lists, where each inner list contains the
                    individual items returned in one sample.

    Returns:
        float in [0, 1].  0 = all samples returned identical sets.
                          1 = no overlap at all across any pair.
    """
    n = len(item_sets)
    if n <= 1:
        return 0.0

    sets = [set(s) for s in item_sets]
    sims: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            union = sets[i] | sets[j]
            inter = sets[i] & sets[j]
            sims.append(len(inter) / len(union) if union else 1.0)

    avg_sim = float(np.mean(sims)) if sims else 1.0
    return round(1.0 - avg_sim, 4)


def per_item_inclusion_frequency(item_sets: List[List[str]]) -> Dict[str, float]:
    """Compute how often each distinct item appeared across k Retriever samples.

    Returns a dict mapping item string → inclusion_rate ∈ [0, 1].
    Useful as an extra diagnostic feature: items with rate ≈ 1.0 were always
    selected; items with rate ≈ 0.5 were borderline; low-rate items may be
    noise or model hallucinations.
    """
    n = len(item_sets)
    if n == 0:
        return {}
    all_items = set(item for s in item_sets for item in s)
    return {
        item: round(sum(1 for s in item_sets if item in s) / n, 4)
        for item in all_items
    }


# ---------------------------------------------------------------------------
# Retriever: item-set parsing
# ---------------------------------------------------------------------------

def parse_retrieved_items(raw_output: str) -> List[str]:
    """Split a Retriever's raw text output into discrete selected items.

    The Retriever is prompted to return only the relevant sentences, one per
    line or as a comma-separated list (depending on what the model produces).
    This function handles both.

    IMPORTANT: The exact split logic was validated against real Retriever outputs
    in logs/trials.jsonl and results/retriever_output.jsonl.  The model tends to
    return one relevant sentence per line, occasionally comma-separated.  If the
    output format changes (e.g. via prompt edits), verify this function still
    splits correctly before running bulk studies.

    Returns empty list if raw_output is empty or whitespace-only.
    """
    if not raw_output or not raw_output.strip():
        return []

    # Try newline-first (the dominant format in existing Study 1 outputs).
    lines = [x.strip() for x in raw_output.splitlines() if x.strip()]
    if len(lines) > 1:
        return lines

    # Fallback: comma-separated (seen in some model outputs)
    items = [x.strip() for x in raw_output.split(",") if x.strip()]
    if len(items) > 1:
        return items

    # Single item (or unrecognized format) — treat whole output as one item
    stripped = raw_output.strip()
    return [stripped] if stripped else []


# ---------------------------------------------------------------------------
# Retroactive re-scoring helper (for existing Study 1 logs)
# ---------------------------------------------------------------------------

def retroactive_extract_conclusion(raw_output: str) -> str:
    """Extract a conclusion from an old Reasoner output that lacks the FINAL ANSWER: marker.

    Used only for the retroactive re-scoring pass on existing logs/trials.jsonl
    (where the old prompt was used).  Heuristic: look for a sentence containing
    'therefore', 'thus', 'so', 'answer is', or take the last sentence.

    This is intentionally kept separate from extract_conclusion() to avoid
    contaminating the new forward-looking extraction logic.
    """
    sentences = re.split(r"(?<=[.!?])\s+", raw_output.strip())
    conclusion_keywords = [
        r"\btherefore\b", r"\bthus\b", r"\bhence\b",
        r"\bthe answer is\b", r"\bin conclusion\b", r"\bso,?\s",
    ]
    for sent in reversed(sentences):
        for kw in conclusion_keywords:
            if re.search(kw, sent, re.IGNORECASE):
                return sent.strip()
    # Last sentence as final fallback
    return sentences[-1].strip() if sentences else raw_output.strip()
