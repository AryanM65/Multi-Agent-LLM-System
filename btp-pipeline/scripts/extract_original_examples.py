"""
extract_original_examples.py -- Extract the original 15 Study 1 questions from HotpotQA.

Finds the exact 15 examples used in the original Study 1 run by matching
question text, then saves them to data/study1_examples.json for reproducibility.

Usage:
    python scripts/extract_original_examples.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ORIGINAL_LOG = "logs/trials.jsonl"
OUTPUT_PATH = "data/study1_examples.json"
DATA_DIR = "./data/hotpotqa_distractor"


def main() -> None:
    # Load original questions
    with open(ORIGINAL_LOG, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    from collections import OrderedDict
    original_questions = list(OrderedDict.fromkeys(r["question"] for r in records))
    print(f"[extract] Found {len(original_questions)} original questions from Study 1.")

    # Load HotpotQA dataset
    from datasets import load_from_disk
    print(f"[extract] Loading dataset from {DATA_DIR}...")
    ds = load_from_disk(DATA_DIR)
    print(f"[extract] Dataset size: {len(ds)} examples")

    # Build lookup by question text
    q_to_example = {}
    for item in ds:
        q = item.get("question", "")
        if q not in q_to_example:
            q_to_example[q] = dict(item)

    # Match
    matched = []
    missing = []
    for q in original_questions:
        if q in q_to_example:
            matched.append(q_to_example[q])
        else:
            missing.append(q)

    print(f"[extract] Matched: {len(matched)}/{len(original_questions)}")
    if missing:
        print(f"[extract] WARNING: {len(missing)} questions not found in dataset:")
        for q in missing:
            print(f"  - {q[:80]}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(matched, f, ensure_ascii=False, indent=2)
    print(f"[extract] Saved {len(matched)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
