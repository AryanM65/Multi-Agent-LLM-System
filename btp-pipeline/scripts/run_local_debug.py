"""
run_local_debug.py — Phase 1-3 sanity checks (small N, verbose output).

Execution order from the brief (§6):

  Phase 1 — k=1, 3 examples, NO faults.
             Goal: confirm all three prompts produce sane, on-topic output.
             Catch prompt-formatting bugs here.

  Phase 2 — k=3, same 3 examples, NO faults.
             Goal: inspect trace.uncertainties() manually; easy questions
             should show low uncertainty (<0.35 adjusted for quantized model).

  Phase 3 — k=3, 3 examples, ALL 3 fault types each.
             Goal: manually verify that the three fault types produce visibly
             distinct uncertainty patterns before trusting the scoring loop.

Usage:
    python scripts/run_local_debug.py --phase 1
    python scripts/run_local_debug.py --phase 2
    python scripts/run_local_debug.py --phase 3
    python scripts/run_local_debug.py          # runs all phases
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_from_disk

from src.config import DATA_DIR
from src.faults import (
    format_context,
    get_gold_context,
    inject_ceiling,
    inject_contamination,
    inject_noise,
)
from src.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 72

def _section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def _print_trace(trace, *, show_samples: bool = False) -> None:
    for node_name, result in trace.node_results.items():
        print(f"\n  [{node_name.upper()}]")
        print(f"    uncertainty : {result.uncertainty:.4f}")
        print(f"    output      : {result.output[:200].strip()}{'...' if len(result.output) > 200 else ''}")
        if show_samples and len(result.samples) > 1:
            for i, s in enumerate(result.samples):
                print(f"    sample[{i}]  : {s[:120].strip()}")
    print(f"\n  Uncertainties dict (never collapsed): {trace.uncertainties()}")


# ---------------------------------------------------------------------------
# Phase 1 — No fault, k=1
# ---------------------------------------------------------------------------

def phase1(examples) -> None:
    _section("PHASE 1 — No fault, k=1  (sanity: prompt output quality)")
    for ex in examples:
        q = ex["question"]
        ctx = format_context(get_gold_context(ex))
        print(f"\n  Question : {q}")
        print(f"  Answer   : {ex['answer']}")
        trace = run_pipeline(q, ctx, k=1, temperature=0.7)
        _print_trace(trace)


# ---------------------------------------------------------------------------
# Phase 2 — No fault, k=3
# ---------------------------------------------------------------------------

def phase2(examples) -> None:
    _section("PHASE 2 — No fault, k=3  (sanity: uncertainty values on clean input)")
    for ex in examples:
        q = ex["question"]
        ctx = format_context(get_gold_context(ex))
        print(f"\n  Question : {q}")
        print(f"  Answer   : {ex['answer']}")
        trace = run_pipeline(q, ctx, k=3, temperature=0.7)
        _print_trace(trace, show_samples=True)
        u = trace.uncertainties()
        all_low = all(v < 0.35 for v in u.values())
        status = "✓ low uncertainty (expected)" if all_low else "⚠  high uncertainty — check prompts or normalization"
        print(f"\n  Status: {status}")


# ---------------------------------------------------------------------------
# Phase 3 — All 3 fault types, k=3
# ---------------------------------------------------------------------------

def phase3(examples) -> None:
    _section("PHASE 3 — Fault injection, k=3  (sanity: distinct uncertainty patterns)")

    fault_fns = [
        ("NOISE",          inject_noise),
        ("CONTAMINATION",  inject_contamination),
        ("CEILING",        inject_ceiling),
    ]

    for ex in examples:
        print(f"\n{'─' * 72}")
        print(f"  Question : {ex['question']}")
        print(f"  Answer   : {ex['answer']}")

        # Baseline for comparison
        baseline_trace = run_pipeline(
            ex["question"],
            format_context(get_gold_context(ex)),
            k=3,
            temperature=0.7,
        )
        print(f"\n  [BASELINE (gold context, temp=0.7)]")
        print(f"  Uncertainties: {baseline_trace.uncertainties()}")

        for fault_name, fault_fn in fault_fns:
            faulted = fault_fn(ex)
            if faulted is None:
                print(f"\n  [{fault_name}] Skipped — insufficient distractors.")
                continue

            temperature = faulted["sampling_override"]["temperature"]
            trace = run_pipeline(
                faulted["question"],
                faulted["context"],
                k=3,
                temperature=temperature,
            )
            print(f"\n  [{fault_name}] (temp={temperature})")
            print(f"  Uncertainties: {trace.uncertainties()}")

            # Expectation hints for manual inspection
            hints = {
                "NOISE":         "Expect: moderate ↑ vs baseline; should recover on same-input retry",
                "CONTAMINATION": "Expect: retriever uncertainty ↑↑; reasoner may follow",
                "CEILING":       "Expect: uncertainty ↑ across nodes, persists through retries",
            }
            print(f"  Hint: {hints[fault_name]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local debug phases 1-3 for the BTP pipeline."
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Which phase to run (1, 2, or 3). Omit to run all.",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=3,
        help="Number of HotpotQA examples to use (default: 3).",
    )
    args = parser.parse_args()

    print(f"[run_local_debug] Loading dataset from '{DATA_DIR}'...")
    ds = load_from_disk(DATA_DIR)
    examples = [ds[i] for i in range(min(args.n_examples, len(ds)))]
    print(f"[run_local_debug] Loaded {len(examples)} example(s).")

    run_phases = [args.phase] if args.phase else [1, 2, 3]

    for phase in run_phases:
        if phase == 1:
            phase1(examples)
        elif phase == 2:
            phase2(examples)
        elif phase == 3:
            phase3(examples)

    print(f"\n{SEPARATOR}")
    print("  Debug run complete.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
