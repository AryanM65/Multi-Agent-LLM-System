"""
run_local_debug.py — Phase 1–3 sanity checks (small N, verbose output).

Execution order:

  Phase 1 — k=1, 3 examples, NO faults.
             Goal: confirm all three prompts produce sane, on-topic output.

  Phase 2 — k=3, same 3 examples, NO faults.
             Goal: inspect trace.uncertainties() manually; verify baseline.
             Also prints new Phase 2 semantic uncertainty where available.

  Phase 3 — k=3, 3 examples, ALL 3 fault types each.
             Goal: distinct uncertainty patterns per fault type.

  Phase 3x — Extended Phase 3 sanity checks (§3.8 of implementation plan):
             2 examples, full fault grid (clean + 9 conditions).
             Checks: (a) noise@X changes only that node, (b) contamination
             at each target shows a real prompt change, (c) skip log populated,
             (d) re-run produces identical corrupted text (reproducibility).

Usage:
    python scripts/run_local_debug.py --phase 1
    python scripts/run_local_debug.py --phase 2
    python scripts/run_local_debug.py --phase 3
    python scripts/run_local_debug.py --phase 3x   # extended Phase 3 checks
    python scripts/run_local_debug.py              # runs phases 1, 2, 3
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_from_disk

from src.config import DATA_DIR, build_fault_conditions
from src.faults import (
    format_context,
    get_gold_context,
    inject_ceiling,
    inject_contamination_retriever,
    inject_noise,
    make_rng,
)
from src.pipeline import default_topology, run_pipeline

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
        print(f"    uncertainty (lexical) : {result.uncertainty:.4f}")
        if result.uncertainty_semantic is not None:
            print(f"    uncertainty (semantic): {result.uncertainty_semantic:.4f}")
        if result.uncertainty_jaccard is not None:
            print(f"    uncertainty (jaccard) : {result.uncertainty_jaccard:.4f}")
        print(f"    output : {result.output[:200].strip()}{'...' if len(result.output) > 200 else ''}")
        if show_samples and len(result.samples) > 1:
            for i, s in enumerate(result.samples):
                print(f"    sample[{i}]: {s[:120].strip()}")
        if result.conclusions:
            for i, c in enumerate(result.conclusions):
                print(f"    conclusion[{i}]: {c[:120].strip()}")
    print(f"\n  Uncertainties (lexical, never collapsed): {trace.uncertainties()}")


# ---------------------------------------------------------------------------
# Phase 1 — No fault, k=1
# ---------------------------------------------------------------------------

def phase1(examples) -> None:
    _section("PHASE 1 — No fault, k=1  (sanity: prompt output quality)")
    topo = default_topology()
    for ex in examples:
        q = ex["question"]
        ctx = format_context(get_gold_context(ex))
        print(f"\n  Question : {q}")
        print(f"  Answer   : {ex['answer']}")
        trace = run_pipeline(topo, q, ctx, k=1, temperature=0.7)
        _print_trace(trace)


# ---------------------------------------------------------------------------
# Phase 2 — No fault, k=3
# ---------------------------------------------------------------------------

def phase2(examples) -> None:
    _section("PHASE 2 — No fault, k=3  (sanity: uncertainty values on clean input)")
    topo = default_topology()
    for ex in examples:
        q = ex["question"]
        ctx = format_context(get_gold_context(ex))
        print(f"\n  Question : {q}")
        print(f"  Answer   : {ex['answer']}")
        trace = run_pipeline(topo, q, ctx, k=3, temperature=0.7)
        _print_trace(trace, show_samples=True)
        u = trace.uncertainties()
        # Post-Phase-2 calibration: Reasoner baseline is ~0.667; threshold is 0.75.
        # All nodes below 0.75 on clean input = expected healthy baseline.
        from src.config import UNCERTAINTY_THRESHOLD
        all_below_thresh = all(v < UNCERTAINTY_THRESHOLD for v in u.values())
        status = (
            "✓ all below UNCERTAINTY_THRESHOLD (expected clean baseline)"
            if all_below_thresh
            else "⚠  one or more nodes above threshold on CLEAN input — recheck prompts"
        )
        print(f"\n  Status: {status}")


# ---------------------------------------------------------------------------
# Phase 3 — All 3 fault types, k=3
# ---------------------------------------------------------------------------

def phase3(examples) -> None:
    _section("PHASE 3 — Fault injection, k=3  (sanity: distinct uncertainty patterns)")
    topo = default_topology()
    fault_fns = [
        ("NOISE",         lambda ex: inject_noise(ex, "retriever", make_rng(ex.get("_id", ex["question"]), "noise", "retriever"))),
        ("CONTAMINATION", lambda ex: inject_contamination_retriever(ex, make_rng(ex.get("_id", ex["question"]), "contamination", "retriever"))),
        ("CEILING",       lambda ex: inject_ceiling(ex, "retriever", make_rng(ex.get("_id", ex["question"]), "ceiling", "retriever"))),
    ]

    for ex in examples:
        print(f"\n{'─' * 72}")
        print(f"  Question : {ex['question']}")
        print(f"  Answer   : {ex['answer']}")

        # Baseline for comparison
        baseline_trace = run_pipeline(
            topo, ex["question"], format_context(get_gold_context(ex)),
            k=3, temperature=0.7,
        )
        print(f"\n  [BASELINE (gold context, temp=0.7)]")
        print(f"  Uncertainties (lexical): {baseline_trace.uncertainties()}")

        for fault_name, fault_fn in fault_fns:
            faulted = fault_fn(ex)
            if faulted is None:
                print(f"\n  [{fault_name}] Skipped — injection not applicable (answer survived / no distractors).")
                continue

            temperature = faulted["sampling_override"]["temperature"]
            fault_config = {
                "type": faulted["type"],
                "target_node": faulted["target_node"],
            }
            trace = run_pipeline(
                topo, faulted["question"], faulted["context"],
                k=3, temperature=temperature, fault_config=fault_config,
            )
            print(f"\n  [{fault_name}] target={faulted['target_node']}  temp={temperature}")
            print(f"  Uncertainties (lexical): {trace.uncertainties()}")

            hints = {
                "NOISE":         "Expect: moderate ↑ at retriever only; others unchanged",
                "CONTAMINATION": "Expect: retriever uncertainty ↑↑; writer may cascade",
                "CEILING":       "Expect: uncertainty ↑ across nodes, persists through retries",
            }
            print(f"  Hint: {hints[fault_name]}")


# ---------------------------------------------------------------------------
# Phase 3x — Extended sanity checks (§3.8 of implementation plan)
# ---------------------------------------------------------------------------

def phase3x(examples) -> None:
    """Extended Phase 3 checks: full 3×3 fault grid on 2 examples.

    Manually confirm:
      (a) noise@node changes ONLY that node's temperature / uncertainty
      (b) contamination@node shows real corrupted text in the prompt
      (c) ceiling skips appear in skip log when answer survives
      (d) re-running produces identical corrupted text (reproducibility)
    """
    _section("PHASE 3x — Extended sanity checks (2-example pilot, full fault grid)")
    topo = default_topology()
    two_examples = examples[:2]
    conditions = build_fault_conditions()

    print("\n  Fault grid: clean + 9 conditions (3 types × 3 target nodes)")
    print(f"  {'Question':<35} {'Condition':<30} {'U_retriever':>12} {'U_reasoner':>12} {'U_writer':>12} {'deviated?'}")
    print(f"  {'─'*35} {'─'*30} {'─'*12} {'─'*12} {'─'*12} {'─'*9}")

    for ex in two_examples:
        qid = ex.get("_id", ex["question"][:20])
        gold_ctx = format_context(get_gold_context(ex))

        # (a)/(d): run clean baseline first, store per-node uncertainties
        baseline_trace = run_pipeline(topo, ex["question"], gold_ctx, k=3, temperature=0.7)
        baseline_u = baseline_trace.uncertainties()
        cond_str = "CLEAN (control)"
        u = baseline_u
        print(f"  {ex['question'][:35]:<35} {cond_str:<30} "
              f"{u.get('retriever', 0):>12.4f} {u.get('reasoner', 0):>12.4f} "
              f"{u.get('writer', 0):>12.4f} {'—':>9}")

        for fc in conditions:
            if fc is None:
                continue  # already printed as clean

            ft = fc["type"]
            tn = fc["target_node"]
            rng = make_rng(str(qid), ft, tn)
            cond_str = f"{ft}@{tn}"

            if ft == "noise":
                faulted = inject_noise(ex, tn, rng)
            elif ft == "contamination":
                faulted = inject_contamination_retriever(ex, rng) if tn == "retriever" else None
            else:  # ceiling
                faulted = inject_ceiling(ex, tn, rng)

            if faulted is None:
                print(f"  {'':<35} {cond_str:<30} {'SKIP':>12} {'':>12} {'':>12} {'':>9}")
                continue

            fault_config = {"type": faulted["type"], "target_node": faulted["target_node"]}
            trace = run_pipeline(
                topo, faulted["question"], faulted["context"],
                k=3, temperature=faulted["sampling_override"]["temperature"],
                fault_config=fault_config,
            )
            u = trace.uncertainties()

            from src.diagnose import verify_against_baseline
            v = verify_against_baseline(
                u.get(tn, 0),
                baseline_u.get(tn, 0),
                fault_config,
            )
            print(f"  {'':<35} {cond_str:<30} "
                  f"{u.get('retriever', 0):>12.4f} {u.get('reasoner', 0):>12.4f} "
                  f"{u.get('writer', 0):>12.4f} "
                  f"{'YES' if v['target_deviated'] else 'no':>9}")

    # (d) Reproducibility check: re-run and compare
    print("\n  === Reproducibility check ===")
    ex = two_examples[0]
    qid = ex.get("_id", ex["question"])
    rng1 = make_rng(str(qid), "contamination", "retriever")
    rng2 = make_rng(str(qid), "contamination", "retriever")
    r1 = inject_contamination_retriever(ex, rng1)
    r2 = inject_contamination_retriever(ex, rng2)
    if r1 is not None and r2 is not None:
        match = r1["context"] == r2["context"]
        print(f"  Contamination context identical on re-run: {'✓ YES' if match else '✗ NO (BUG!)'}")
    else:
        print("  Contamination skipped (no distractors) — reproducibility check N/A.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local debug phases 1-3 for the BTP pipeline."
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "3", "3x"],
        default=None,
        help="Which phase to run (1, 2, 3, or 3x). Omit to run phases 1, 2, 3.",
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

    run_phases = [args.phase] if args.phase else ["1", "2", "3"]

    for phase in run_phases:
        if phase == "1":
            phase1(examples)
        elif phase == "2":
            phase2(examples)
        elif phase == "3":
            phase3(examples)
        elif phase == "3x":
            phase3x(examples)

    print(f"\n{SEPARATOR}")
    print("  Debug run complete.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
