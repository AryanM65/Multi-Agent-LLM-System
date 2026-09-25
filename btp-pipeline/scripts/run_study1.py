"""
run_study1.py — Phase 4+ entry point: scored batch run + confusion matrix.

Phase 3 restructure:
  - Per-question control trial always runs first (clean baseline).
  - Full 3×3 fault grid (build_fault_conditions()) replaces the original 3-type list.
  - verify_against_baseline() is computed and stored for every fault trial.
  - Skipped trials (ceiling answer-survived, no distractors, etc.) logged to
    a separate skipped_trials.jsonl with explicit reason fields.
  - Reproducibility: every injector call uses make_rng(question_id, type, node).
  - Substitute bank for downstream contamination is built once before the main
    loop from a pilot set of clean Retriever/Reasoner outputs.

Features (unchanged from original):
  - Append-only JSONL logging (crash-safe).
  - tqdm progress bar.
  - --resume flag skips already-logged trials.
  - Confusion matrix generation + diagonal dominance check.

Usage:
    python scripts/run_study1.py --n-examples 15 --k 3 --no-inference-gap
    python scripts/run_study1.py --n-examples 15 --k 3 --no-inference-gap --resume
    python scripts/run_study1.py --n-examples 300 --k 5  # cloud scale-up
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure") and sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable
    tqdm.total = None
    tqdm.set_postfix = lambda **kw: None
    tqdm.update = lambda n=1: None

from src.config import (
    DATA_DIR,
    DEFAULT_K,
    DEFAULT_LOG_PATH,
    DEFAULT_SKIP_LOG_PATH,
    DEFAULT_CONFUSION_MATRIX_PATH,
    RESULTS_DIR,
    LOG_DIR,
    build_fault_conditions,
)
from src.diagnose import diagnose_trace, needs_diagnosis, verify_against_baseline
from src.faults import (
    format_context,
    get_gold_context,
    inject_ceiling,
    inject_contamination_downstream,
    inject_contamination_retriever,
    inject_noise,
    make_rng,
)
from src.nodes import PipelineTrace
from src.pipeline import (
    retriever_prompt,
    reasoner_prompt,
    writer_prompt,
    run_pipeline,
    default_topology,
)

# ---------------------------------------------------------------------------
# Substitute bank builder (for downstream contamination faults)
# ---------------------------------------------------------------------------

def build_substitute_bank(
    examples: list,
    topo,
    k: int,
    n_pilot: int = 10,
) -> Dict[str, List[str]]:
    """Build a bank of clean node outputs from pilot examples.

    Used as the replacement pool for contamination faults targeting Reasoner
    or Writer nodes.  Pre-built once before the main trial loop to avoid
    per-trial overhead.

    Args:
        examples:  Full example list.
        topo:      Pipeline topology.
        k:         Self-consistency samples (same as study k).
        n_pilot:   Number of examples to collect clean outputs from.

    Returns:
        Dict {"retriever": [outputs...], "reasoner": [outputs...], "writer": [outputs...]}
    """
    bank: Dict[str, List[str]] = {"retriever": [], "reasoner": [], "writer": []}
    pilot = examples[:min(n_pilot, len(examples))]
    print(f"[run_study1] Building substitute bank from {len(pilot)} pilot examples...")
    for ex in pilot:
        ctx = format_context(get_gold_context(ex))
        trace = run_pipeline(topo, ex["question"], ctx, k=k, fault_config=None)
        for node_id, result in trace.node_results.items():
            if node_id in bank:
                bank[node_id].append(result.output)
    print(f"[run_study1] Bank sizes: { {k: len(v) for k, v in bank.items()} }")
    return bank


# ---------------------------------------------------------------------------
# Per-trial runner
# ---------------------------------------------------------------------------

def run_trial(
    example: dict,
    fault_config: Optional[dict],
    topo,
    k: int,
    compute_gap: bool,
    baseline_uncertainties: Optional[Dict[str, float]],
    substitute_bank: Optional[Dict[str, List[str]]],
    diagnose_enabled: bool = True,
) -> Optional[Dict]:
    """Run one trial (control or fault) and return the logged record.

    Returns None if the fault injection was not applicable (logged externally
    as a skip with reason).  For control trials, fault_config is None.

    The returned dict is ready for JSON serialisation and appending to
    trials.jsonl or skipped_trials.jsonl.
    """
    question = example["question"]
    qid = str(example.get("_id", question))
    gold_ctx = format_context(get_gold_context(example))
    answer = example["answer"]

    # --- Determine actual context + fault_config to run with ---
    run_context = gold_ctx

    if fault_config is None:
        # Clean control trial
        trace = run_pipeline(topo, question, gold_ctx, k=k, fault_config=None)
        trace.true_label = "clean"
        trace.gold_answer = answer
        record = _trace_to_record(trace, fault_config=None, is_control=True)
        return record

    ft = fault_config["type"]
    tn = fault_config["target_node"]
    rng = make_rng(qid, ft, tn)
    skip_reason = None

    if ft == "noise":
        injected = inject_noise(example, tn, rng)
        run_fault_config = {"type": "noise", "target_node": tn}

    elif ft == "contamination":
        if tn == "retriever":
            injected = inject_contamination_retriever(example, rng, target_node=tn)
            if injected is None:
                skip_reason = "contamination_insufficient_distractors"
                return _skip_record(qid, question, fault_config, skip_reason)
            run_context = injected["context"]
            run_fault_config = {"type": "contamination", "target_node": tn}
        else:
            # Downstream contamination: run clean first to get parent's output,
            # then corrupt it.
            clean_trace = run_pipeline(topo, question, gold_ctx, k=k, fault_config=None)
            parent_map = {"reasoner": "retriever", "writer": "reasoner"}
            parent_id = parent_map.get(tn)
            if parent_id is None or parent_id not in clean_trace.node_results:
                skip_reason = f"contamination_no_parent_for_{tn}"
                return _skip_record(qid, question, fault_config, skip_reason)

            clean_parent_output = clean_trace.node_results[parent_id].output
            bank = (substitute_bank or {}).get(parent_id, [])
            injected = inject_contamination_downstream(
                example, tn, clean_parent_output, bank, rng
            )
            if injected is None:
                skip_reason = "contamination_empty_substitute_bank"
                return _skip_record(qid, question, fault_config, skip_reason)
            run_fault_config = {
                "type": "contamination",
                "target_node": tn,
                "_corrupted_input": injected["_corrupted_input"],
            }

    else:  # ceiling
        # run_study1.py only ever runs the canonical 3-node chain, where
        # node_id == role, so passing tn as role is correct here (unlike
        # run_study2.py/run_study_vllm.py, which must look up topo.nodes[tn].role
        # explicitly since arbitrary topologies use custom node IDs -- see the
        # ceiling role-dispatch bug fix in src/faults.py).
        injected = inject_ceiling(example, tn, rng, role=tn)
        if injected is None:
            skip_reason = "ceiling_answer_survived"
            return _skip_record(qid, question, fault_config, skip_reason)
        if tn == "retriever":
            run_context = injected["context"]
            run_fault_config = {"type": "ceiling", "target_node": tn}
        else:
            # Downstream ceiling: hardened instruction variant
            run_fault_config = {"type": "ceiling", "target_node": tn}
            # The hardened instruction is stored in injected["_hardened_instruction"]
            # For now, we signal it via the fault_config and let the topology
            # engine read it when building the prompt.
            run_fault_config["_hardened_instruction"] = injected.get("_hardened_instruction", "")

    # --- Run pipeline with fault_config ---
    trace = run_pipeline(topo, question, run_context, k=k, fault_config=run_fault_config)
    trace.true_label = ft
    trace.gold_answer = answer

    # --- Diagnosis ---
    faulted_retriever_output = trace.node_results.get("retriever", None)
    faulted_reasoner_output = trace.node_results.get("reasoner", None)

    node_prompts: Dict[str, Dict[str, str]] = {
        "retriever": {
            "same":  retriever_prompt(question, run_context),
            "clean": retriever_prompt(question, gold_ctx),
        },
        "reasoner": {
            "same":  reasoner_prompt(question, faulted_retriever_output.output if faulted_retriever_output else ""),
            # Known approximation: clean Reasoner gets gold ctx directly (not a re-run of Retriever).
            "clean": reasoner_prompt(question, gold_ctx),
        },
        "writer": {
            "same":  writer_prompt(question, faulted_reasoner_output.output if faulted_reasoner_output else ""),
            "clean": writer_prompt(question, gold_ctx),
        },
    }
    node_roles = {nid: topo.nodes[nid].role for nid in topo.nodes}

    if diagnose_enabled:
        diagnose_trace(
            trace, node_prompts, k=k, compute_gap=compute_gap,
            node_roles=node_roles,
        )
    else:
        # Diagnosis skipped (--no-diagnose): raw per-node uncertainties + true_label
        # are still recorded for dataset generation, but the retry-then-reprobe
        # protocol's extra k-sample batches (up to 3x per flagged node) are not
        # run. diagnosed_label/per_node_diagnoses stay at their defaults (None/{})
        # rather than being computed — distinct from "no_fault_detected", which
        # means diagnosis ran and found nothing.
        pass

    # --- Baseline verification ---
    verification = None
    if baseline_uncertainties is not None:
        observed_u = trace.node_results[tn].uncertainty if tn in trace.node_results else 0.0
        baseline_u = baseline_uncertainties.get(tn, 0.0)
        verification = verify_against_baseline(observed_u, baseline_u, fault_config)

    record = _trace_to_record(trace, fault_config=fault_config, is_control=False,
                               verification=verification)
    return record


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _trace_to_record(
    trace: PipelineTrace,
    fault_config: Optional[dict] = None,
    is_control: bool = False,
    verification: Optional[Dict] = None,
) -> Dict:
    """Serialise a PipelineTrace to a JSON-ready dict."""
    return {
        "question":           trace.question,
        "true_label":         trace.true_label,
        "diagnosed_label":    trace.diagnosed_label,
        "is_control":         is_control,
        "fault_config":       {k: v for k, v in (fault_config or {}).items()
                               if not k.startswith("_")},  # strip internal keys
        "topology_id":        trace.topology_id,
        "uncertainties":      trace.uncertainties(),
        "semantic_uncertainties": trace.semantic_uncertainties(),
        "inference_gaps":     trace.inference_gaps(),
        "per_node_diagnoses": trace.per_node_diagnoses,
        "gold_answer":        trace.gold_answer,
        "verification":       verification,
        "samples": {
            name: r.samples for name, r in trace.node_results.items()
        },
        "used_thinking_fallback": {
            name: r.used_thinking_fallback for name, r in trace.node_results.items()
            if any(r.used_thinking_fallback)
        },
        "conclusions": {
            name: r.conclusions for name, r in trace.node_results.items()
            if r.conclusions is not None
        },
    }


def _skip_record(
    question_id: str,
    question: str,
    fault_config: Dict,
    reason: str,
) -> Dict:
    """Create a skip record for skipped_trials.jsonl."""
    return {
        "_is_skip": True,
        "question_id": question_id,
        "question": question,
        "fault_config": {k: v for k, v in fault_config.items() if not k.startswith("_")},
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Main study loop
# ---------------------------------------------------------------------------

def run_study1(
    n_examples: int = 15,
    k: int = DEFAULT_K,
    log_path: str = DEFAULT_LOG_PATH,
    skip_log_path: str = DEFAULT_SKIP_LOG_PATH,
    compute_gap: bool = True,
    resume: bool = False,
    examples_json: Optional[str] = None,
    diagnose_enabled: bool = True,
) -> List[Dict]:
    """Run the full Study 1 batch: n_examples × (clean control + 9 fault conditions).

    Per-question order: control trial first, then all 9 fault conditions.
    Append-only JSONL logging (crash-safe).  --resume skips already-logged trials.

    diagnose_enabled: when False (--no-diagnose), skips the retry-then-reprobe
    diagnostic protocol entirely for every fault trial — raw per-node
    uncertainties + true_label are still recorded (this is what a dataset built
    for downstream GNN/belief-propagation training actually needs), but the up
    to 3 extra k-sample batches per flagged node that diagnose_trace() would
    otherwise trigger are skipped, cutting a meaningful fraction of the
    measured ~4.5-5x per-trial cost increase (see correct_project_context.md
    Section 9.1). diagnosed_label/per_node_diagnoses are left at their
    PipelineTrace defaults (None / {}) rather than computed.

    Returns list of all non-skip records (loaded + newly computed).
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    from src.config import MOCK_MODE, BACKEND
    mock_active = MOCK_MODE or (BACKEND == "mock")

    if mock_active:
        print("[run_study1] Mock mode ON -- generating synthetic examples.")
        examples = [
            {
                "question": f"Were Person-{i} and Person-{i+1} of the same nationality?",
                "answer": "yes",
                "_id": f"mock-{i}",
                "supporting_facts": {
                    "title": [f"Person-{i}", f"Person-{i+1}"],
                    "sent_id": [0, 0],
                },
                "context": {
                    "title": [f"Person-{i}", f"Person-{i+1}", f"Distractor-{i}"],
                    "sentences": [
                        [f"Person-{i} is an American artist born in Denver."],
                        [f"Person-{i+1} was an American filmmaker born in New York."],
                        [f"Distractor-{i} is an unrelated entity from France."],
                    ],
                },
            }
            for i in range(n_examples)
        ]
    elif examples_json and os.path.exists(examples_json):
        print(f"[run_study1] Loading examples from JSON: {examples_json}")
        with open(examples_json, encoding="utf-8") as f:
            all_ex = json.load(f)
        examples = all_ex[:n_examples]
        print(f"[run_study1] Loaded {len(examples)} examples from JSON.")
    else:
        try:
            from datasets import load_from_disk
            print(f"[run_study1] Loading dataset from '{DATA_DIR}'...")
            ds = load_from_disk(DATA_DIR)
            examples = [ds[i] for i in range(min(n_examples, len(ds)))]
        except Exception as e:
            print(f"[run_study1] ERROR: Could not load dataset: {e}")
            print(f"[run_study1] Run 'python scripts/download_data.py' first, or use --mock.")
            sys.exit(1)
    print(f"[run_study1] {len(examples)} examples, k={k}")

    topo = default_topology()
    conditions = build_fault_conditions()  # [None, {"type":..., "target_node":...}, ...]
    print(f"[run_study1] Fault conditions per question: {len(conditions)} "
          f"(1 clean control + {len(conditions) - 1} fault conditions)")

    # Build substitute bank for downstream contamination faults.
    substitute_bank = build_substitute_bank(examples, topo, k=k, n_pilot=10)

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
                    # Key: (question, true_label, fault_config_type, fault_config_target)
                    fc = rec.get("fault_config") or {}
                    done_keys.add((
                        rec["question"],
                        rec.get("true_label", ""),
                        fc.get("type", ""),
                        fc.get("target_node", ""),
                    ))
        print(f"[run_study1] Resuming — loaded {len(existing_records)} existing records.")

    results: List[Dict] = list(existing_records)
    skipped_count = 0

    total = len(examples) * len(conditions)
    with (
        open(log_path, "a", encoding="utf-8") as log_file,
        open(skip_log_path, "a", encoding="utf-8") as skip_file,
    ):
        with tqdm(total=total, desc="Trials") as pbar:
            for example in examples:
                qid = str(example.get("_id", example["question"]))
                gold_ctx = format_context(get_gold_context(example))

                # (1) Control trial -- always first to establish per-question baseline.
                ctrl_key = (example["question"], "clean", "", "")
                if ctrl_key in done_keys:
                    pbar.update(1)
                    skipped_count += 1
                    pbar.set_postfix(skipped=skipped_count)
                    # We still need the baseline uncertainties from the existing control record.
                    ctrl_rec = next(
                        (r for r in results if r["question"] == example["question"]
                         and r.get("true_label") == "clean"), None
                    )
                    baseline_u = ctrl_rec["uncertainties"] if ctrl_rec else None
                else:
                    ctrl_record = run_trial(
                        example, fault_config=None, topo=topo, k=k,
                        compute_gap=compute_gap, baseline_uncertainties=None,
                        substitute_bank=substitute_bank,
                    )
                    if ctrl_record and not ctrl_record.get("_is_skip"):
                        log_file.write(json.dumps(ctrl_record) + "\n")
                        log_file.flush()
                        results.append(ctrl_record)
                        done_keys.add(ctrl_key)
                    baseline_u = ctrl_record["uncertainties"] if ctrl_record else None
                    pbar.update(1)

                # (2) Each fault condition, verified against this question's baseline.
                for fault_config in conditions:
                    if fault_config is None:
                        continue  # control already done above

                    ft = fault_config["type"]
                    tn = fault_config["target_node"]
                    trial_key = (example["question"], ft, ft, tn)
                    pbar.update(1)

                    if trial_key in done_keys:
                        skipped_count += 1
                        pbar.set_postfix(skipped=skipped_count)
                        continue

                    record = run_trial(
                        example,
                        fault_config=dict(fault_config),  # copy to avoid mutation
                        topo=topo,
                        k=k,
                        compute_gap=compute_gap,
                        baseline_uncertainties=baseline_u,
                        substitute_bank=substitute_bank,
                        diagnose_enabled=diagnose_enabled,
                    )
                    if record is None:
                        continue

                    if record.get("_is_skip"):
                        skip_file.write(json.dumps(record) + "\n")
                        skip_file.flush()
                        skipped_count += 1
                        pbar.set_postfix(skipped=skipped_count)
                    else:
                        log_file.write(json.dumps(record) + "\n")
                        log_file.flush()
                        results.append(record)
                        done_keys.add(trial_key)

    print(f"[run_study1] Done. {len(results)} records total, {skipped_count} skipped.")
    print(f"[run_study1] Logs: {log_path}")
    print(f"[run_study1] Skip log: {skip_log_path}")
    return results


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def build_confusion_matrix(
    results: List[Dict],
    save_path: str = DEFAULT_CONFUSION_MATRIX_PATH,
) -> None:
    """Build and save a 3x3 confusion matrix using stdlib csv only.

    Excludes clean control trials and 'no_fault_detected' records.
    """
    import csv
    labels = ["noise", "contamination", "ceiling"]

    fault_records = [r for r in results if r.get("true_label") in labels]
    diagnosed = [r for r in fault_records if r.get("diagnosed_label") not in (None, "no_fault_detected")]
    no_fault_count = len(fault_records) - len(diagnosed)
    if no_fault_count > 0:
        print(f"[confusion_matrix] {no_fault_count} fault trial(s) had 'no_fault_detected' "
              f"-- excluded from confusion matrix.")

    counts = {r: {c: 0 for c in labels} for r in labels}
    for rec in diagnosed:
        tl = rec.get("true_label")
        dl = rec.get("diagnosed_label")
        if tl in counts and dl in counts.get(tl, {}):
            counts[tl][dl] += 1

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true"] + labels)
        for r in labels:
            writer.writerow([r] + [counts[r][c] for c in labels])
    print(f"[confusion_matrix] Saved to '{save_path}'.")

    total = sum(counts[r][c] for r in labels for c in labels)
    correct = sum(counts[l][l] for l in labels)

    print("\n=== Confusion Matrix ===")
    print(f"  {'':>15s}" + "".join(f"{c:>16s}" for c in labels))
    for r in labels:
        print(f"  {r:>15s}" + "".join(f"{counts[r][c]:>16d}" for c in labels))
    print()
    if total > 0:
        accuracy = correct / total
        print(f"Overall diagnostic accuracy: {correct}/{total} = {accuracy:.1%}")
        if accuracy > 0.5:
            print("Diagonal dominates -- diagnostic mechanism shows positive signal.")
        else:
            print("Accuracy <= 50% -- review NODE_THRESHOLDS and injection logic.")
    else:
        print("[confusion_matrix] No diagnosed fault records -- cannot compute accuracy.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Study 1 batch: fault injection + diagnosis + confusion matrix."
    )
    parser.add_argument(
        "--n-examples", type=int, default=15,
        help="Number of HotpotQA examples (default: 15).",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help=f"Self-consistency samples per node (default: {DEFAULT_K}).",
    )
    parser.add_argument(
        "--log-path", type=str, default=DEFAULT_LOG_PATH,
        help=f"Append-only trial log path (default: {DEFAULT_LOG_PATH}).",
    )
    parser.add_argument(
        "--skip-log-path", type=str, default=DEFAULT_SKIP_LOG_PATH,
        help=f"Skipped trials log path (default: {DEFAULT_SKIP_LOG_PATH}).",
    )
    parser.add_argument(
        "--no-inference-gap", action="store_true",
        help="Disable Inference Gap computation (saves time).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing log file, skipping already-completed trials.",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock backend instead of real LLM (no Ollama/MLX required).",
    )
    parser.add_argument(
        "--examples-json", type=str, default=None,
        help="Path to a JSON file of pre-extracted HotpotQA examples. "
             "Overrides --n-examples count (uses all examples in the file).",
    )
    parser.add_argument(
        "--no-diagnose", action="store_true",
        help="Skip the retry-then-reprobe diagnostic protocol (diagnose_trace) "
             "for every fault trial. Raw per-node uncertainties + true_label are "
             "still recorded (what a GNN-training dataset needs); diagnosed_label/"
             "per_node_diagnoses are left unset. Cuts up to 3x the k-sample calls "
             "per flagged node -- use this for bulk dataset generation where the "
             "retry-heuristic's own diagnostic accuracy isn't the thing being measured.",
    )
    args = parser.parse_args()

    if args.mock:
        os.environ["BTP_MOCK"] = "1"
        import src.config as _cfg; _cfg.MOCK_MODE = True
        import src.nodes as _nodes; _nodes.MOCK_MODE = True

    from src.config import MOCK_MODE, BACKEND, OLLAMA_MODEL
    backend_label = "mock" if MOCK_MODE else (f"Ollama ({OLLAMA_MODEL})" if BACKEND == "ollama" else "MLX")
    print(f"[run_study1] Backend: {backend_label}")

    results = run_study1(
        n_examples=args.n_examples,
        k=args.k,
        log_path=args.log_path,
        skip_log_path=args.skip_log_path,
        compute_gap=not args.no_inference_gap,
        resume=args.resume,
        examples_json=args.examples_json,
        diagnose_enabled=not args.no_diagnose,
    )
    # Only build confusion matrix over fault trials (not control trials).
    fault_results = [r for r in results if r.get("true_label") not in ("clean", None)]
    if args.no_diagnose:
        print("[run_study1] --no-diagnose was set: skipping confusion matrix "
              "(diagnosed_label was never computed).")
    else:
        build_confusion_matrix(fault_results)


if __name__ == "__main__":
    main()
