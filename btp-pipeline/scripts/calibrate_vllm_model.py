"""calibrate_vllm_model.py — Qwen2.5-7B-Instruct-AWQ calibration pilot.

GPU-only (run via a Kaggle GPU kernel, see plan.md Section 5). Exercises the
real pipeline (run_pipeline, sample_node, uncertainty.py) against vLLM/Qwen2.5
on real example questions, clean control only (no faults), and reports
everything needed for plan.md Section 0.1's recalibration checklist:

  1. Real output-token usage per role (for MAX_TOKENS / MAX_TOKENS_REASONER)
  2. FINAL ANSWER: marker compliance rate (Reasoner)
  3. parse_retrieved_items output on real Retriever samples (for manual audit)
  4. Achievable lexical/semantic uncertainty value set at k=5
  5. Per-node clean-baseline uncertainty stats (for NODE_THRESHOLDS)

Does NOT use vLLM's batching (Section 4.3's run_study_vllm.py handles that,
separately, for bulk generation) -- this reuses the existing sequential
run_pipeline() so the exact same code path that bulk generation will run is
what's being calibrated, not a parallel implementation.

Usage (on a GPU machine/kernel with vllm installed):
    BTP_BACKEND=vllm python scripts/calibrate_vllm_model.py --n-examples 5 --k 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("BTP_BACKEND", "vllm")

from src.config import DEFAULT_K, MAX_TOKENS, MAX_TOKENS_REASONER
from src.faults import get_gold_context, format_context
from src.pipeline import RETRIEVER_INSTRUCTION, REASONER_INSTRUCTION, WRITER_INSTRUCTION
from src.topology import default_chain_topology
from src.pipeline import run_pipeline
from src.uncertainty import parse_retrieved_items


def _word_count(text: str) -> int:
    return len(text.split())


def run_calibration(n_examples: int, k: int, examples_path: str):
    with open(examples_path, encoding="utf-8") as f:
        examples = json.load(f)[:n_examples]

    topo = default_chain_topology(RETRIEVER_INSTRUCTION, REASONER_INSTRUCTION, WRITER_INSTRUCTION)

    role_word_counts = {"retriever": [], "reasoner": [], "writer": []}
    role_uncertainties_lexical = {"retriever": [], "reasoner": [], "writer": []}
    role_uncertainties_semantic = {"retriever": [], "reasoner": [], "writer": []}
    marker_hits, marker_total = 0, 0
    retriever_parse_samples = []  # for manual audit
    empty_samples = {"retriever": 0, "reasoner": 0, "writer": 0}
    total_samples = {"retriever": 0, "reasoner": 0, "writer": 0}

    for i, example in enumerate(examples):
        question = example["question"]
        gold_ctx = format_context(get_gold_context(example))
        print(f"\n{'='*70}\n[{i+1}/{len(examples)}] {question}\n{'='*70}")

        trace = run_pipeline(topo, question, gold_ctx, k=k, fault_config=None)

        for node_id, result in trace.node_results.items():
            role = topo.nodes[node_id].role
            print(f"\n  -- {node_id} ({role}) --")
            print(f"     lexical_uncertainty={result.uncertainty}  semantic={result.uncertainty_semantic}")
            role_uncertainties_lexical[role].append(result.uncertainty)
            if result.uncertainty_semantic is not None:
                role_uncertainties_semantic[role].append(result.uncertainty_semantic)

            for s_idx, sample in enumerate(result.samples):
                total_samples[role] += 1
                wc = _word_count(sample)
                role_word_counts[role].append(wc)
                if not sample.strip():
                    empty_samples[role] += 1
                print(f"     [{s_idx}] ({wc}w) {sample[:150]!r}")

            if role == "reasoner":
                for conclusion in (result.conclusions or []):
                    marker_total += 1
                    # A real FINAL ANSWER extraction vs. the "last line" fallback
                    # can't be distinguished post-hoc from conclusions alone, so
                    # re-check raw samples directly for the marker.
                for sample in result.samples:
                    if re.search(r"final answer:?", sample, re.IGNORECASE):
                        marker_hits += 1

            if role == "retriever":
                for sample in result.samples:
                    items = parse_retrieved_items(sample)
                    retriever_parse_samples.append((sample, items))

    print(f"\n\n{'#'*70}\n# CALIBRATION SUMMARY\n{'#'*70}")

    print("\n== 1. Output word counts per role (proxy for token usage) ==")
    for role, counts in role_word_counts.items():
        if not counts:
            continue
        print(f"  {role:10s} n={len(counts)}  min={min(counts)}  "
              f"median={statistics.median(counts):.0f}  max={max(counts)}  "
              f"mean={statistics.mean(counts):.1f}")
    print(f"  Current MAX_TOKENS={MAX_TOKENS}, MAX_TOKENS_REASONER={MAX_TOKENS_REASONER} "
          f"(gpt-oss-era values -- compare against the word-count maxima above; "
          f"word count is a rough proxy for token count, add margin, then verify "
          f"empirically at whatever value is chosen)")

    print("\n== 2. Empty samples per role ==")
    for role in total_samples:
        n_empty = empty_samples[role]
        n_total = total_samples[role]
        print(f"  {role:10s} {n_empty}/{n_total} empty"
              + ("  <-- investigate, Qwen2.5 has no hidden-reasoning excuse for this" if n_empty else ""))

    print(f"\n== 3. FINAL ANSWER: marker compliance (Reasoner) ==")
    if marker_hits or (marker_total == 0 and role_word_counts.get("reasoner")):
        n_reasoner_samples = total_samples["reasoner"]
        print(f"  {marker_hits}/{n_reasoner_samples} Reasoner samples contained a "
              f"'final answer' marker ({100*marker_hits/n_reasoner_samples:.1f}%)"
              if n_reasoner_samples else "  no reasoner samples")

    print(f"\n== 4. Retriever parse_retrieved_items audit (first 15 samples) ==")
    print(f"  Manually review these -- confirm items look like sensible sentence-level splits,")
    print(f"  not fragments (this is the audit the original master plan required but was never done).")
    for sample, items in retriever_parse_samples[:15]:
        print(f"\n  RAW:  {sample[:200]!r}")
        print(f"  ITEMS ({len(items)}): {items}")

    print(f"\n== 5. Achievable uncertainty value set at k={k} ==")
    expected = {round(i / k, 4) for i in range(k + 1)}
    print(f"  Expected lexical value set (0..k)/k: {sorted(expected)}")
    for role, vals in role_uncertainties_lexical.items():
        if vals:
            print(f"  {role:10s} observed lexical values: {sorted(set(vals))}")

    print(f"\n== 6. Suggested NODE_THRESHOLDS (2x observed clean-baseline semantic mean) ==")
    for role, vals in role_uncertainties_semantic.items():
        if vals:
            mean = statistics.mean(vals)
            suggested = round(min(mean * 2, 0.9), 2)
            print(f"  {role:10s} clean semantic mean={mean:.4f}  std={statistics.pstdev(vals):.4f}  "
                  f"suggested_threshold~={suggested}")
        else:
            print(f"  {role:10s} no semantic uncertainty values recorded (role may not compute it)")

    print(f"\nDone. Copy relevant values into src/config.py MAX_TOKENS / "
          f"MAX_TOKENS_REASONER / NODE_THRESHOLDS manually after reviewing -- "
          f"this script does not write config.py automatically.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--examples-json", type=str, default="data/study1_examples.json")
    args = parser.parse_args()
    run_calibration(args.n_examples, args.k, args.examples_json)


if __name__ == "__main__":
    main()
