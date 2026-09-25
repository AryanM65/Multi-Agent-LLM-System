"""sample_examples.py -- Extend the example question pool for dataset generation.

Keeps the existing 15 questions in data/study1_examples.json (for continuity
with prior pilot runs) and adds N new, randomly-sampled, distinct questions
from the full cached HotpotQA distractor set, writing the combined pool to a
new file. Pure data prep -- no GPU/model calls.

Usage:
    python scripts/sample_examples.py --add 15 --seed 42
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=str, default="data/study1_examples.json")
    parser.add_argument("--add", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/study30_examples.json")
    args = parser.parse_args()

    with open(args.existing, encoding="utf-8") as f:
        existing = json.load(f)
    existing_questions = {ex["question"] for ex in existing}
    print(f"[sample_examples] {len(existing)} existing questions loaded.")

    from datasets import load_from_disk
    print(f"[sample_examples] Loading cached dataset from {DATA_DIR}...")
    ds = load_from_disk(DATA_DIR)
    print(f"[sample_examples] Dataset size: {len(ds)}")

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    new_examples = []
    for i in indices:
        if len(new_examples) >= args.add:
            break
        item = ds[i]
        q = item.get("question", "")
        if not q or q in existing_questions:
            continue
        # Light quality filter: gold titles must actually be present in
        # context (get_gold_context requires this), and context not absurdly
        # long (keeps generation cost/time predictable).
        gold_titles = set(item["supporting_facts"]["title"])
        context_titles = set(item["context"]["title"])
        if not gold_titles.issubset(context_titles):
            continue
        total_sentences = sum(len(s) for s in item["context"]["sentences"])
        if total_sentences > 60:
            continue
        new_examples.append(dict(item))
        existing_questions.add(q)

    if len(new_examples) < args.add:
        print(f"[sample_examples] WARNING: only found {len(new_examples)}/{args.add} "
              f"qualifying new questions.")

    combined = existing + new_examples
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"[sample_examples] Added {len(new_examples)} new questions.")
    print(f"[sample_examples] Total: {len(combined)} questions -> {args.out}")


if __name__ == "__main__":
    main()
