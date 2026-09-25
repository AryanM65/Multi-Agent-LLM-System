"""
run_study2.py — Multi-topology fault injection study (Study 2).

Key differences from run_study1.py
────────────────────────────────────
  Topologies  : All 4 named topologies (chain, dual_retriever_fanin,
                parallel_reasoner, deep_chain) — or specify one with --topology.
  Fault grid  : Full (fault_type × target_node_of_topology) — derived from
                build_fault_conditions(topo) so it adapts to any topology shape.
  Uncertainty : Lexical + Semantic + Jaccard all logged per node.
  Thresholds  : Per-node NODE_THRESHOLDS dict (replaces global UNCERTAINTY_THRESHOLD).
  Logs        : logs/study2_trials.jsonl + logs/study2_skipped.jsonl
  Confusion   : results/study2_confusion/<topology_id>.csv + aggregate.csv
  Mock mode   : BTP_MOCK=1 or --mock flag runs without MLX (for Windows testing).

Usage
─────
  # All topologies, mock mode (Windows / no MLX):
  python scripts/run_study2.py --n-examples 2 --k 3 --mock

  # Single topology, real model:
  python scripts/run_study2.py --n-examples 5 --k 3 --topology chain

  # All topologies, real model, resume:
  python scripts/run_study2.py --n-examples 10 --k 5 --topology all --resume

  # No inference-gap (faster):
  python scripts/run_study2.py --n-examples 5 --k 3 --no-inference-gap
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

from src.config import (
    DATA_DIR,
    DEFAULT_K,
    DEFAULT_STUDY2_LOG_PATH,
    DEFAULT_STUDY2_SKIP_LOG_PATH,
    DEFAULT_STUDY2_CM_DIR,
    LOG_DIR,
    MOCK_MODE,
    NODE_THRESHOLDS,
    RESULTS_DIR,
    build_fault_conditions,
)
from src.diagnose import diagnose_trace, verify_against_baseline
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
)
from src.topologies import all_topologies, get_topology


# ---------------------------------------------------------------------------
# Per-node threshold helper (Study 2 upgrade from global threshold)
# ---------------------------------------------------------------------------

def needs_diagnosis_per_node(result, node_id: str) -> bool:
    """Return True if this node's uncertainty exceeds its per-node threshold."""
    threshold = NODE_THRESHOLDS.get(node_id, NODE_THRESHOLDS.get("_default", 0.75))
    return result.uncertainty > threshold


# ---------------------------------------------------------------------------
# Substitute bank builder
# ---------------------------------------------------------------------------

def build_substitute_bank(
    examples: list,
    topo,
    k: int,
    n_pilot: int = 10,
) -> Dict[str, List[str]]:
    """Build a bank of clean node outputs from pilot examples.

    For non-chain topologies with multiple nodes of the same role (e.g.
    retriever_a, retriever_b), both are stored under their role name so the
    contamination injector can draw from the bank regardless of node_id.
    """
    bank: Dict[str, List[str]] = {}
    pilot = examples[:min(n_pilot, len(examples))]
    print(f"[study2] Building substitute bank from {len(pilot)} pilot examples "
          f"(topology: {topo.topology_id})...")
    for ex in pilot:
        ctx = format_context(get_gold_context(ex))
        trace = run_pipeline(topo, ex["question"], ctx, k=k, fault_config=None)
        for node_id, result in trace.node_results.items():
            role = topo.nodes[node_id].role
            bank.setdefault(role, []).append(result.output)
            bank.setdefault(node_id, []).append(result.output)
    print(f"[study2] Bank sizes: { {k: len(v) for k, v in bank.items()} }")
    return bank


# ---------------------------------------------------------------------------
# Trial runner
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
    """Run one trial (control or fault) and return the logged record (or skip dict)."""
    question  = example["question"]
    qid       = str(example.get("_id", question))
    gold_ctx  = format_context(get_gold_context(example))
    answer    = example["answer"]

    if fault_config is None:
        trace = run_pipeline(topo, question, gold_ctx, k=k, fault_config=None)
        trace.true_label  = "clean"
        trace.gold_answer = answer
        return _trace_to_record(trace, fault_config=None, is_control=True)

    ft = fault_config["type"]
    tn = fault_config["target_node"]
    rng = make_rng(qid, ft, tn)
    run_context = gold_ctx
    run_fault_config = None

    # ── Determine if this node actually exists in this topology ──────────────
    if tn not in topo.nodes:
        return _skip_record(qid, question, fault_config,
                            f"target_node_{tn}_not_in_topology_{topo.topology_id}")

    if ft == "noise":
        run_fault_config = {"type": "noise", "target_node": tn}

    elif ft == "contamination":
        if topo.nodes[tn].role == "retriever":
            injected = inject_contamination_retriever(example, rng, target_node=tn)
            if injected is None:
                return _skip_record(qid, question, fault_config,
                                    "contamination_insufficient_distractors")
            run_context = injected["context"]
            run_fault_config = {"type": "contamination", "target_node": tn}
        else:
            # Downstream contamination: find a parent of this node
            from src.topology import parents_of
            parent_ids = parents_of(topo, tn)
            if not parent_ids:
                return _skip_record(qid, question, fault_config,
                                    f"contamination_no_parent_for_{tn}")
            parent_id = parent_ids[0]
            parent_role = topo.nodes[parent_id].role

            # Run clean first to get the parent's actual output
            clean_trace = run_pipeline(topo, question, gold_ctx, k=k, fault_config=None)
            clean_parent_output = clean_trace.node_results.get(parent_id)
            if clean_parent_output is None:
                return _skip_record(qid, question, fault_config,
                                    f"contamination_parent_not_in_trace_{parent_id}")
            clean_parent_output = clean_parent_output.output

            bank = (substitute_bank or {}).get(parent_id,
                   (substitute_bank or {}).get(parent_role, []))
            injected = inject_contamination_downstream(
                example, tn, clean_parent_output, bank, rng)
            if injected is None:
                return _skip_record(qid, question, fault_config,
                                    "contamination_empty_substitute_bank")
            run_fault_config = {
                "type": "contamination",
                "target_node": tn,
                "_corrupted_input": injected["_corrupted_input"],
            }

    else:  # ceiling
        injected = inject_ceiling(example, tn, rng, role=topo.nodes[tn].role)
        if injected is None:
            return _skip_record(qid, question, fault_config, "ceiling_answer_survived")
        if topo.nodes[tn].role == "retriever":
            run_context = injected["context"]
            run_fault_config = {"type": "ceiling", "target_node": tn}
        else:
            run_fault_config = {
                "type": "ceiling",
                "target_node": tn,
                "_hardened_instruction": injected.get("_hardened_instruction", ""),
            }

    trace = run_pipeline(topo, question, run_context, k=k,
                         fault_config=run_fault_config)
    trace.true_label  = ft
    trace.gold_answer = answer

    # ── Diagnosis (per-node thresholds) ─────────────────────────────────────
    node_prompts = _build_node_prompts(topo, question, run_context, gold_ctx, trace)
    node_roles = {nid: topo.nodes[nid].role for nid in topo.nodes}
    if diagnose_enabled:
        diagnose_trace(trace, node_prompts, k=k, compute_gap=compute_gap,
                       node_roles=node_roles)
    # else: --no-diagnose — raw per-node uncertainties + true_label still
    # recorded; diagnosed_label/per_node_diagnoses left at defaults (None/{}).
    # See run_study1.py's run_trial() docstring-equivalent comment for why.

    # ── Baseline verification ────────────────────────────────────────────────
    verification = None
    if baseline_uncertainties is not None and tn in trace.node_results:
        verification = verify_against_baseline(
            trace.node_results[tn].uncertainty,
            baseline_uncertainties.get(tn, 0.0),
            fault_config,
        )

    return _trace_to_record(trace, fault_config=fault_config, is_control=False,
                            verification=verification)


# ---------------------------------------------------------------------------
# Node prompt builder helper
# ---------------------------------------------------------------------------

def _build_node_prompts(topo, question, run_context, gold_ctx, trace) -> Dict:
    """Build same/clean prompt pairs for every node in the trace."""
    node_prompts = {}
    from src.pipeline import build_prompt
    outputs_from_trace = {nid: r.output for nid, r in trace.node_results.items()}

    for node_id in topo.nodes:
        from src.topology import parents_of
        parents = parents_of(topo, node_id)
        if not parents:
            # Source node — uses raw context
            same_prompt  = build_prompt(topo, node_id, outputs_from_trace,
                                        question, run_context)
            clean_prompt = build_prompt(topo, node_id, outputs_from_trace,
                                        question, gold_ctx)
        else:
            same_prompt  = build_prompt(topo, node_id, outputs_from_trace,
                                        question, run_context)
            clean_prompt = build_prompt(topo, node_id, outputs_from_trace,
                                        question, gold_ctx)
        node_prompts[node_id] = {"same": same_prompt, "clean": clean_prompt}
    return node_prompts


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _trace_to_record(
    trace: PipelineTrace,
    fault_config: Optional[dict] = None,
    is_control: bool = False,
    verification: Optional[Dict] = None,
) -> Dict:
    return {
        "question":            trace.question,
        "true_label":          trace.true_label,
        "diagnosed_label":     trace.diagnosed_label,
        "is_control":          is_control,
        "fault_config":        {k: v for k, v in (fault_config or {}).items()
                                if not k.startswith("_")},
        "topology_id":         trace.topology_id,
        "uncertainties":       trace.uncertainties(),
        "semantic_uncertainties": trace.semantic_uncertainties(),
        "inference_gaps":      trace.inference_gaps(),
        "per_node_diagnoses":  trace.per_node_diagnoses,
        "gold_answer":         trace.gold_answer,
        "verification":        verification,
        "samples": {
            nid: r.samples for nid, r in trace.node_results.items()
        },
        "used_thinking_fallback": {
            nid: r.used_thinking_fallback for nid, r in trace.node_results.items()
            if any(r.used_thinking_fallback)
        },
        "conclusions": {
            nid: r.conclusions for nid, r in trace.node_results.items()
            if r.conclusions is not None
        },
        "jaccard_uncertainties": {
            nid: r.uncertainty_jaccard for nid, r in trace.node_results.items()
            if r.uncertainty_jaccard is not None
        },
    }


def _skip_record(qid, question, fault_config, reason) -> Dict:
    return {
        "_is_skip": True,
        "question_id": qid,
        "question": question,
        "fault_config": {k: v for k, v in fault_config.items()
                         if not k.startswith("_")},
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Per-topology study runner
# ---------------------------------------------------------------------------

def run_topology_study(
    topo,
    examples: list,
    k: int,
    log_file,
    skip_file,
    compute_gap: bool,
    resume: bool,
    done_keys: set,
    diagnose_enabled: bool = True,
) -> List[Dict]:
    """Run the full fault grid for one topology and return all new records."""
    conditions = build_fault_conditions(topo)
    n_conditions = len(conditions)
    total = len(examples) * n_conditions

    results: List[Dict] = []
    skipped = 0

    substitute_bank = build_substitute_bank(examples, topo, k=k, n_pilot=10)

    with tqdm(total=total, desc=f"  {topo.topology_id}", leave=False) as pbar:
        for example in examples:
            qid = str(example.get("_id", example["question"]))

            # -- Control trial (always first) --
            ctrl_key = (topo.topology_id, example["question"], "clean", "", "")
            baseline_u = None

            if ctrl_key in done_keys:
                # Find existing control record to recover baseline_u
                existing = next((r for r in results
                                 if r.get("question") == example["question"]
                                 and r.get("true_label") == "clean"
                                 and r.get("topology_id") == topo.topology_id), None)
                baseline_u = existing["uncertainties"] if existing else None
                pbar.update(1)
                skipped += 1
            else:
                ctrl_rec = run_trial(example, None, topo, k, compute_gap,
                                     None, substitute_bank)
                if ctrl_rec and not ctrl_rec.get("_is_skip"):
                    log_file.write(json.dumps(ctrl_rec) + "\n"); log_file.flush()
                    results.append(ctrl_rec)
                    done_keys.add(ctrl_key)
                    baseline_u = ctrl_rec.get("uncertainties")
                pbar.update(1)

            # -- Fault conditions --
            for fc in conditions:
                if fc is None:
                    continue
                ft, tn = fc["type"], fc["target_node"]
                trial_key = (topo.topology_id, example["question"], ft, ft, tn)
                pbar.update(1)

                if trial_key in done_keys:
                    skipped += 1; pbar.set_postfix(skip=skipped); continue

                record = run_trial(example, dict(fc), topo, k, compute_gap,
                                   baseline_u, substitute_bank,
                                   diagnose_enabled=diagnose_enabled)
                if record is None:
                    continue
                if record.get("_is_skip"):
                    skip_file.write(json.dumps(record) + "\n"); skip_file.flush()
                    skipped += 1; pbar.set_postfix(skip=skipped)
                else:
                    log_file.write(json.dumps(record) + "\n"); log_file.flush()
                    results.append(record)
                    done_keys.add(trial_key)

    return results


# ---------------------------------------------------------------------------
# Confusion matrix (per topology + aggregate)
# ---------------------------------------------------------------------------

def build_confusion_matrices(
    results: List[Dict],
    cm_dir: str,
) -> None:
    """Build and save per-topology + aggregate confusion matrices."""
    import csv
    os.makedirs(cm_dir, exist_ok=True)
    labels = ["noise", "contamination", "ceiling"]

    valid_records = [
        r for r in results
        if r.get("true_label") in labels and r.get("diagnosed_label") in labels
    ]

    topologies = sorted(list({r.get("topology_id", "") for r in valid_records if r.get("topology_id")}))

    summary_rows = []
    for topo_id in list(topologies) + ["__aggregate__"]:
        subset = valid_records if topo_id == "__aggregate__" else [r for r in valid_records if r.get("topology_id") == topo_id]
        if not subset:
            continue

        counts = {r: {c: 0 for c in labels} for r in labels}
        for item in subset:
            tl = item["true_label"]
            dl = item["diagnosed_label"]
            if tl in counts and dl in counts[tl]:
                counts[tl][dl] += 1

        path = os.path.join(cm_dir, f"{topo_id}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([""] + labels)
            for r in labels:
                writer.writerow([r] + [counts[r][c] for c in labels])

        total = sum(counts[r][c] for r in labels for c in labels)
        correct = sum(counts[l][l] for l in labels)
        acc = correct / total if total > 0 else 0.0

        summary_rows.append({
            "topology": topo_id,
            "total_diag": total,
            "correct": correct,
            "accuracy": round(acc, 4),
        })

        print(f"\n  [{topo_id}] acc={acc:.1%}  ({correct}/{total})")
        header = f"{'':15s}" + "".join(f"{c:>15s}" for c in labels)
        print(header)
        for r in labels:
            row_str = f"{r:15s}" + "".join(f"{counts[r][c]:>15d}" for c in labels)
            print(row_str)

    if summary_rows:
        sum_path = os.path.join(cm_dir, "summary.csv")
        with open(sum_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["topology", "total_diag", "correct", "accuracy"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[study2] Confusion matrices written to '{cm_dir}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Study 2: multi-topology fault injection + diagnosis."
    )
    parser.add_argument("--n-examples", type=int, default=5,
                        help="Number of HotpotQA examples (default: 5).")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help=f"Self-consistency samples (default: {DEFAULT_K}).")
    parser.add_argument("--topology", type=str, default="chain",
                        help=f"Topology name or 'all'. Choices: {all_topologies()} (default: chain).")
    parser.add_argument("--log-path", type=str, default=DEFAULT_STUDY2_LOG_PATH)
    parser.add_argument("--skip-log-path", type=str, default=DEFAULT_STUDY2_SKIP_LOG_PATH)
    parser.add_argument("--cm-dir", type=str, default=DEFAULT_STUDY2_CM_DIR)
    parser.add_argument("--no-inference-gap", action="store_true",
                        help="Disable Inference Gap computation.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-logged trials.")
    parser.add_argument("--mock", action="store_true",
                        help="Force mock backend (same as BTP_MOCK=1).")
    parser.add_argument("--no-diagnose", action="store_true",
                        help="Skip the retry-then-reprobe diagnostic protocol for every "
                             "fault trial (see run_study1.py --no-diagnose for rationale). "
                             "Use for bulk dataset generation.")
    args = parser.parse_args()

    # Allow --mock flag to override env-var
    if args.mock:
        os.environ["BTP_MOCK"] = "1"
        import src.config as _cfg
        _cfg.MOCK_MODE = True
        import src.nodes as _nodes
        _nodes.MOCK_MODE = True

    mock_active = os.environ.get("BTP_MOCK", "0") == "1"
    print(f"[study2] Mock mode: {'ON (no MLX needed)' if mock_active else 'OFF (MLX required)'}")

    topo_names = all_topologies() if args.topology == "all" else [args.topology]
    print(f"[study2] Topologies: {topo_names}")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(args.cm_dir, exist_ok=True)

    if not mock_active:
        print(f"[study2] Loading dataset from '{DATA_DIR}'...")
        ds = load_from_disk(DATA_DIR)
        examples = [ds[i] for i in range(min(args.n_examples, len(ds)))]
    else:
        # In mock mode, synthesise dummy examples so the pipeline can run
        # without the HotpotQA dataset being present on this machine.
        examples = _make_mock_examples(args.n_examples)
    print(f"[study2] {len(examples)} examples, k={args.k}")

    # Load existing records for --resume
    done_keys: set = set()
    all_results: List[Dict] = []
    if args.resume and os.path.exists(args.log_path):
        with open(args.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    all_results.append(rec)
                    fc = rec.get("fault_config") or {}
                    done_keys.add((
                        rec.get("topology_id", ""),
                        rec["question"],
                        rec.get("true_label", ""),
                        fc.get("type", ""),
                        fc.get("target_node", ""),
                    ))
        print(f"[study2] Resuming — {len(all_results)} existing records.")

    with (
        open(args.log_path, "a", encoding="utf-8") as log_file,
        open(args.skip_log_path, "a", encoding="utf-8") as skip_file,
    ):
        for topo_name in topo_names:
            topo = get_topology(topo_name)
            print(f"\n[study2] ═══ Topology: {topo.topology_id} "
                  f"| nodes={list(topo.nodes.keys())} ═══")
            new_records = run_topology_study(
                topo, examples, args.k,
                log_file, skip_file,
                compute_gap=not args.no_inference_gap,
                resume=args.resume,
                done_keys=done_keys,
                diagnose_enabled=not args.no_diagnose,
            )
            all_results.extend(new_records)

    print(f"\n[study2] Total records: {len(all_results)}")
    fault_results = [r for r in all_results if r.get("true_label") not in ("clean", None)]
    if args.no_diagnose:
        print("[study2] --no-diagnose was set: skipping confusion matrices "
              "(diagnosed_label was never computed).")
    else:
        build_confusion_matrices(fault_results, args.cm_dir)


# ---------------------------------------------------------------------------
# Mock example factory (for Windows / no-dataset testing)
# ---------------------------------------------------------------------------

def _make_mock_examples(n: int) -> list:
    """Generate n synthetic HotpotQA-shaped examples for mock-mode testing."""
    import hashlib
    base_examples = [
        {
            "_id": hashlib.md5(f"mock_{i}".encode()).hexdigest()[:16],
            "question": f"Were Person-{i} and Person-{i+1} of the same nationality?",
            "answer": "yes",
            "context": {
                "title": [f"Person-{i}", f"Person-{i+1}", f"Distractor-{i}"],
                "sentences": [
                    [f"Person-{i} is an American filmmaker.", f"Person-{i} was born in New York."],
                    [f"Person-{i+1} is also American.", f"Person-{i+1} worked in Hollywood."],
                    [f"Distractor-{i} is a British author.", f"Distractor-{i} was born in London."],
                ],
            },
            "supporting_facts": {
                "title": [f"Person-{i}", f"Person-{i+1}"],
                "sent_id": [0, 0],
            },
        }
        for i in range(n)
    ]
    return base_examples


if __name__ == "__main__":
    main()
