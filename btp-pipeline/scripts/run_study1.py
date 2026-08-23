"""
run_study1.py — Phase 4+ entry point: scored batch run + confusion matrix.

Generates the primary Study 1 deliverable: a 3×3 confusion matrix with rows
= true fault type, columns = diagnosed fault type.

Features:
  - Append-only JSONL logging (crash-safe; completed trials are never lost).
  - tqdm progress bar.
  - Resumes from an existing log if run is interrupted and restarted
    (--resume flag).
  - Per-node diagnosis stored alongside primary diagnosed_label.
  - Inference Gap computed as an experimental metric (disable with
    --no-inference-gap).

Usage:
    python scripts/run_study1.py
    python scripts/run_study1.py --n-examples 15 --k 3
    python scripts/run_study1.py --n-examples 15 --k 5 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

from src.config import (
    DATA_DIR,
    DEFAULT_K,
    DEFAULT_LOG_PATH,
    DEFAULT_CONFUSION_MATRIX_PATH,
    RESULTS_DIR,
    LOG_DIR,
)
from src.diagnose import diagnose_trace, needs_diagnosis
from src.faults import (
    format_context,
    get_gold_context,
    inject_ceiling,
    inject_contamination,
    inject_noise,
)
from src.nodes import PipelineTrace
from src.pipeline import (
    retriever_prompt,
    reasoner_prompt,
    writer_prompt,
    run_pipeline,
)

# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_labeled_trial(
    example: dict,
    fault_fn: Callable,
    k: int = DEFAULT_K,
    compute_gap: bool = True,
) -> Optional[PipelineTrace]:
    """Run one fault-injected trial: pipeline → diagnosis → labeled PipelineTrace.

    Returns None if fault injection failed (e.g. insufficient distractors).

    DESIGN CHOICE (multi-node flagging):
      node_prompts is built for ALL three nodes so that diagnose_trace() can
      handle any combination of flagged nodes.  The clean reference for every
      node uses the FULL gold context (important for ceiling faults where
      the faulted context is a stripped subset of gold — the clean prompt
      uses the complete gold so the diagnostic can detect a genuine capability
      gap vs. a data-quality issue).
    """
    faulted = fault_fn(example)
    if faulted is None:
        return None

    temperature = faulted["sampling_override"]["temperature"]

    trace = run_pipeline(
        faulted["question"],
        faulted["context"],
        k=k,
        temperature=temperature,
    )
    trace.true_label = faulted["true_label"]
    trace.gold_answer = faulted["answer"]

    # Build per-node prompts for the diagnostic.
    # same_input_prompt: the prompt actually used in the faulted run.
    # clean_input_prompt: the prompt with known-good (gold) context.
    clean_context = format_context(get_gold_context(example))

    # The retriever output used in the faulted run (for downstream nodes).
    faulted_retriever_output = trace.node_results["retriever"].output
    faulted_reasoner_output  = trace.node_results["reasoner"].output

    node_prompts: Dict[str, Dict[str, str]] = {
        "retriever": {
            "same":  retriever_prompt(faulted["question"], faulted["context"]),
            "clean": retriever_prompt(faulted["question"], clean_context),
        },
        "reasoner": {
            # Same input for reasoner = the (possibly faulted) retriever output.
            "same":  reasoner_prompt(faulted["question"], faulted_retriever_output),
            # Clean input = reasoner given a fresh retrieval from gold context.
            # We use the faulted retriever output as a proxy here since we
            # can't re-run retriever inline; full clean re-run is a Study 2
            # enhancement.  Logged as a known approximation.
            "clean": reasoner_prompt(faulted["question"], clean_context),
        },
        "writer": {
            "same":  writer_prompt(faulted["question"], faulted_reasoner_output),
            "clean": writer_prompt(faulted["question"], clean_context),
        },
    }

    diagnose_trace(trace, node_prompts, k=k, compute_gap=compute_gap)

    return trace


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def trace_to_record(trace: PipelineTrace) -> Dict:
    """Convert a PipelineTrace to a JSON-serialisable dict for JSONL logging."""
    return {
        "question":          trace.question,
        "true_label":        trace.true_label,
        "diagnosed_label":   trace.diagnosed_label,
        "uncertainties":     trace.uncertainties(),
        "inference_gaps":    trace.inference_gaps(),
        "per_node_diagnoses": trace.per_node_diagnoses,
        "gold_answer":       trace.gold_answer,
        # Store raw samples for post-hoc analysis (e.g. manual spot-check of
        # Writer normalization, §8 of the brief).
        "samples": {
            name: r.samples for name, r in trace.node_results.items()
        },
    }


# ---------------------------------------------------------------------------
# Main study loop
# ---------------------------------------------------------------------------

def run_study1(
    n_examples: int = 15,
    k: int = DEFAULT_K,
    log_path: str = DEFAULT_LOG_PATH,
    compute_gap: bool = True,
    resume: bool = False,
) -> List[Dict]:
    """Run the full Study 1 batch: n_examples × 3 fault types.

    Logging is append-only (crash-safe).  If resume=True and log_path already
    contains records, those are loaded and used to skip already-completed
    trials (matched by question + true_label).

    Returns list of all records (loaded + newly computed).
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load dataset.
    print(f"[run_study1] Loading dataset from '{DATA_DIR}'...")
    ds = load_from_disk(DATA_DIR)
    examples = [ds[i] for i in range(min(n_examples, len(ds)))]
    print(f"[run_study1] {len(examples)} examples, k={k}, fault types=3")

    # Load existing records if resuming.
    existing_records: List[Dict] = []
    done_keys: set = set()
    if resume and os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    existing_records.append(rec)
                    done_keys.add((rec["question"], rec["true_label"]))
        print(f"[run_study1] Resuming — loaded {len(existing_records)} existing records.")

    fault_fns: List[Callable] = [inject_noise, inject_contamination, inject_ceiling]
    results: List[Dict] = list(existing_records)
    skipped = 0

    total = len(examples) * len(fault_fns)
    with open(log_path, "a") as log_file, tqdm(total=total, desc="Trials") as pbar:
        for example in examples:
            for fault_fn in fault_fns:
                pbar.update(1)

                faulted = fault_fn(example)
                if faulted is None:
                    skipped += 1
                    pbar.set_postfix(skipped=skipped)
                    continue

                key = (example["question"], faulted["true_label"])
                if key in done_keys:
                    skipped += 1
                    pbar.set_postfix(skipped=skipped)
                    continue

                trace = run_labeled_trial(example, fault_fn, k=k, compute_gap=compute_gap)
                if trace is None:
                    continue

                record = trace_to_record(trace)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()   # ensure write is durable
                results.append(record)
                done_keys.add(key)

    print(f"[run_study1] Done. {len(results)} records total, {skipped} skipped.")
    return results


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def build_confusion_matrix(
    results: List[Dict],
    save_path: str = DEFAULT_CONFUSION_MATRIX_PATH,
) -> pd.DataFrame:
    """Build and save a 3×3 confusion matrix (true × diagnosed).

    Rows = true fault type, columns = diagnosed fault type.
    """
    df = pd.DataFrame(results)
    # Filter out any records where diagnosed_label is "no_fault_detected".
    df_diag = df[df["diagnosed_label"] != "no_fault_detected"].copy()
    no_fault_count = len(df) - len(df_diag)
    if no_fault_count > 0:
        print(
            f"[confusion_matrix] Note: {no_fault_count} trial(s) had 'no_fault_detected' "
            f"(all nodes below threshold) — excluded from confusion matrix."
        )

    labels = ["noise", "contamination", "ceiling"]
    cm = pd.DataFrame(0, index=labels, columns=labels)
    cm.index.name = "true"
    cm.columns.name = "diagnosed"

    if not df_diag.empty:
        ct = pd.crosstab(
            df_diag["true_label"],
            df_diag["diagnosed_label"],
            rownames=["true"],
            colnames=["diagnosed"],
        )
        for r in ct.index:
            for c in ct.columns:
                if r in cm.index and c in cm.columns:
                    cm.loc[r, c] = ct.loc[r, c]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm.to_csv(save_path)
    print(f"[confusion_matrix] Saved to '{save_path}'.")
    print("\n=== Confusion Matrix ===")
    print(cm.to_string())
    print()

    # Quick diagonal dominance check.
    total = cm.values.sum()
    correct = sum(cm.loc[l, l] for l in ["noise", "contamination", "ceiling"] if l in cm.index and l in cm.columns)
    if total > 0:
        accuracy = correct / total
        print(f"Overall diagnostic accuracy: {correct}/{total} = {accuracy:.1%}")
        if accuracy > 0.5:
            print("✓ Diagonal dominates — diagnostic mechanism shows positive signal.")
        else:
            print("⚠  Accuracy ≤ 50% — review UNCERTAINTY_THRESHOLD and injection logic.")

    return cm


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Study 1 batch: fault injection + diagnosis + confusion matrix."
    )
    parser.add_argument(
        "--n-examples", type=int, default=15,
        help="Number of HotpotQA examples to use (default: 15).",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help=f"Self-consistency samples per node (default: {DEFAULT_K}).",
    )
    parser.add_argument(
        "--log-path", type=str, default=DEFAULT_LOG_PATH,
        help=f"Path to append-only trial log (default: {DEFAULT_LOG_PATH}).",
    )
    parser.add_argument(
        "--no-inference-gap", action="store_true",
        help="Disable Inference Gap computation (saves time; Study 2 feature).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing log file, skipping already-completed trials.",
    )
    args = parser.parse_args()

    results = run_study1(
        n_examples=args.n_examples,
        k=args.k,
        log_path=args.log_path,
        compute_gap=not args.no_inference_gap,
        resume=args.resume,
    )
    build_confusion_matrix(results)


if __name__ == "__main__":
    main()
