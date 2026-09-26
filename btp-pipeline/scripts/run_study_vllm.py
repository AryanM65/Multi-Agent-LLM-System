"""
run_study_vllm.py — Batched multi-topology dataset generation via vLLM.

The GPU-batched counterpart to run_study1.py/run_study2.py. Those scripts
call run_pipeline() once per trial, sequentially -- correct for Ollama's
one-request-at-a-time API, but wasteful for vLLM, whose entire value
proposition is batching many prompts into one generate()/chat() call. This
script reuses every piece of existing logic (fault injection, build_prompt,
uncertainty computation via compute_node_result_from_samples, verified
baseline labeling) -- ONLY the generation step is batched differently. See
plan.md Section 4.3 for the design this implements.

Per the decision recorded in correct_project_context.md Section 9.6/plan.md:
diagnosis (diagnose_trace, the retry-then-reprobe protocol) is NOT run here.
The dataset's purpose is raw per-node uncertainty features + a verified
true_label (via verify_against_baseline) for downstream belief-propagation/
GNN training -- diagnosed_label is not something that work consumes, and
skipping it avoids the extra 2-3x k-sample retry batches per flagged node
that diagnose_trace would otherwise trigger.

Batching strategy, per topology:
  Phase 1 -- run every question's CONTROL trial as one batch (topological
             order, one llm.chat() call per node across all questions in
             the batch) to establish per-question baselines.
  Phase 2 -- resolve every fault trial's injection (noise/contamination/
             ceiling; skip-and-log invalid ones, exactly as run_study2.py
             does) BEFORE generation, then run all resolved fault trials as
             one batch (or several, per --batch-size) through the same
             per-node batched execution, verifying each against its own
             question's Phase 1 baseline.

NOT YET SMOKE-TESTED (plan.md Section 5.5) -- this is new code that has not
been run against a real GPU. Before trusting it for a real generation run:
  1. Confirm the GPU is actually used (not silent CPU fallback).
  2. Confirm a safe --batch-size empirically (start small, e.g. 8-15).
  3. Confirm batched fault-config isolation: run a small batch mixing a
     noise-faulted trial with clean/other trials and verify only the
     targeted trial/node received NOISE_TEMPERATURE (the "two divergent
     code paths" risk class -- see compute_node_result_from_samples'
     docstring for why this matters).
  4. Confirm llm.chat() actually accepts a batch (List[List[message]]) with
     a matching list of SamplingParams the way this script assumes -- this
     was used for single prompts in scripts/calibrate_vllm_model.py's
     successful real run, but the batched call form is unverified.

Usage (Kaggle GPU kernel; local execution without vllm installed will fail
at import time):
    BTP_BACKEND=vllm python scripts/run_study_vllm.py \\
        --topology-pool ../dataset/topology_pool.json \\
        --examples-json data/study1_examples.json \\
        --k 5 --batch-size 15 \\
        --log-path logs/dataset_trials.jsonl --skip-log-path logs/dataset_skipped.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("BTP_BACKEND", "vllm")

if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.config import DEFAULT_K, MAX_TOKENS, MAX_TOKENS_REASONER, DEFAULT_TEMPERATURE, build_fault_conditions
from src.diagnose import verify_against_baseline
from src.faults import (
    format_context,
    get_gold_context,
    inject_ceiling,
    inject_contamination_downstream,
    inject_contamination_retriever,
    inject_noise,
    make_rng,
)
from src.nodes import PipelineTrace, compute_node_result_from_samples, get_normalizer
from src.pipeline import apply_fault_to_prompt, build_prompt, run_pipeline
from src.topology import Topology, NodeSpec, parents_of, topological_order, validate_topology

# Reuse serialization + substitute-bank helpers from run_study2.py rather
# than re-implementing them (avoids a second, divergent implementation of
# "what a trial record looks like").
sys.path.insert(0, os.path.dirname(__file__))
from run_study2 import _trace_to_record, _skip_record, build_substitute_bank


# ---------------------------------------------------------------------------
# Topology pool loading (mirrors dataset/generate_topology_pool.py's schema,
# reimplemented locally rather than importing across the btp-pipeline/dataset
# directory boundary, to keep this script self-contained under btp-pipeline/).
# ---------------------------------------------------------------------------

def load_topology_pool(path: str) -> List[tuple]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    pool = []
    for r in records:
        nodes = {
            nid: NodeSpec(nid, info["role"], info["instruction"])
            for nid, info in r["nodes"].items()
        }
        edges = [(e[0], e[1]) for e in r["edges"]]
        topo = Topology(nodes=nodes, edges=edges, topology_id=r["topology_id"])
        validate_topology(topo)
        pool.append((topo, r["split"]))
    return pool


# ---------------------------------------------------------------------------
# Per-trial fault resolution (mirrors run_study2.py's run_trial's injection
# logic exactly, but returns a resolved spec instead of calling run_pipeline
# -- the actual generation is deferred to the batched executor below).
# ---------------------------------------------------------------------------

def resolve_fault_trial(
    example: dict,
    fault_config: dict,
    topo: Topology,
    k: int,
    substitute_bank: Optional[Dict[str, List[str]]],
) -> Optional[dict]:
    """Resolve one fault trial's injection. Returns a dict with keys
    {question, run_context, fault_config, true_label, gold_answer, qid} ready
    for batched execution, or None if the injection was invalid (caller must
    log the skip via _skip_record, matching run_study2.py's behavior).
    """
    question = example["question"]
    qid = str(example.get("_id", question))
    gold_ctx = format_context(get_gold_context(example))
    answer = example["answer"]

    ft = fault_config["type"]
    tn = fault_config["target_node"]
    rng = make_rng(qid, ft, tn)
    run_context = gold_ctx
    run_fault_config = None

    if tn not in topo.nodes:
        return None  # caller logs target_node_not_in_topology

    if ft == "noise":
        run_fault_config = {"type": "noise", "target_node": tn}

    elif ft == "contamination":
        if topo.nodes[tn].role == "retriever":
            injected = inject_contamination_retriever(example, rng, target_node=tn)
            if injected is None:
                return None  # contamination_insufficient_distractors
            run_context = injected["context"]
            run_fault_config = {"type": "contamination", "target_node": tn}
        else:
            parent_ids = parents_of(topo, tn)
            if not parent_ids:
                return None  # contamination_no_parent_for_<tn>
            # Full multi-parent fix (2026-09-26): every parent feeding the
            # targeted node gets its OWN independent corruption, drawn from
            # its own role/node-id's substitute bank. Previously only
            # parent_ids[0] was corrupted, and (a worse bug found while
            # fixing this) pipeline.build_prompt's `_maybe_corrupt` applied
            # that single corrupted string to EVERY parent's displayed input
            # line regardless of which parent it was rendering -- so a
            # 2-parent contamination trial actually showed the identical
            # corrupted text twice, not "one corrupted + one clean" as
            # previously documented. `_corrupted_inputs` (plural, dict keyed
            # by parent_id) below fixes both: each parent is corrupted
            # independently and pipeline.py now looks up per-pid.
            clean_trace = run_pipeline(topo, question, gold_ctx, k=k, fault_config=None)
            corrupted_inputs = {}
            for parent_id in parent_ids:
                parent_role = topo.nodes[parent_id].role
                clean_parent_result = clean_trace.node_results.get(parent_id)
                if clean_parent_result is None:
                    return None  # contamination_parent_not_in_trace
                bank = (substitute_bank or {}).get(parent_id, (substitute_bank or {}).get(parent_role, []))
                injected = inject_contamination_downstream(
                    example, tn, clean_parent_result.output, bank, rng)
                if injected is None:
                    return None  # contamination_empty_substitute_bank
                corrupted_inputs[parent_id] = injected["_corrupted_input"]
            run_fault_config = {
                "type": "contamination",
                "target_node": tn,
                "_corrupted_inputs": corrupted_inputs,
            }

    else:  # ceiling
        injected = inject_ceiling(example, tn, rng, role=topo.nodes[tn].role)
        if injected is None:
            return None  # ceiling_answer_survived
        if topo.nodes[tn].role == "retriever":
            run_context = injected["context"]
            run_fault_config = {"type": "ceiling", "target_node": tn}
        else:
            # _hardened_instruction is now actually applied by build_prompt()
            # (fixed in src/pipeline.py, 2026-09-25) -- previously computed
            # here but silently ignored, meaning every downstream-targeted
            # ceiling trial ran with no real fault applied.
            run_fault_config = {
                "type": "ceiling",
                "target_node": tn,
                "_hardened_instruction": injected.get("_hardened_instruction", ""),
            }

    return {
        "qid": qid,
        "question": question,
        "run_context": run_context,
        "fault_config": run_fault_config,
        "true_label": ft,
        "gold_answer": answer,
        "orig_fault_config": fault_config,
    }


# ---------------------------------------------------------------------------
# Batched per-node execution (the core new logic this script adds)
# ---------------------------------------------------------------------------

def run_pipeline_batch(topo: Topology, specs: List[dict], k: int,
                        temperature: float = DEFAULT_TEMPERATURE) -> List[PipelineTrace]:
    """Execute `topo` over a batch of trial specs, one llm.chat() call per
    node across the WHOLE batch (vLLM's real batching), instead of one
    run_pipeline() call per trial.

    Each spec: {question, run_context, fault_config (resolved run_fault_config
    or None)}. Returns one PipelineTrace per spec, same order, with all
    node_results populated -- uncertainty computed via
    compute_node_result_from_samples, the exact same function the sequential
    path (run_pipeline -> sample_node) uses, so results are computed
    identically regardless of which path produced the raw samples.
    """
    from src.config import MOCK_MODE

    order = topological_order(topo)
    n = len(specs)
    outputs_per_trial: List[Dict[str, str]] = [dict() for _ in range(n)]
    traces = [PipelineTrace(question=s["question"], topology_id=topo.topology_id) for s in specs]

    llm = None
    if not MOCK_MODE:
        from vllm import SamplingParams
        from src.nodes import _get_vllm_llm  # reuses the lazy singleton loader
        llm = _get_vllm_llm()

    for node_id in order:
        role = topo.nodes[node_id].role
        token_limit = MAX_TOKENS_REASONER if role == "reasoner" else MAX_TOKENS

        conversations = []
        params_list = []
        for i, spec in enumerate(specs):
            fc = spec["fault_config"]
            prompt = build_prompt(topo, node_id, outputs_per_trial[i],
                                   spec["question"], spec["run_context"], fc)
            node_temp = temperature
            if fc is not None and fc.get("target_node") == node_id:
                prompt, node_temp = apply_fault_to_prompt(prompt, node_temp, fc)
            conversations.append([{"role": "user", "content": prompt}])
            if not MOCK_MODE:
                params_list.append(SamplingParams(temperature=node_temp, max_tokens=token_limit, n=k))

        if MOCK_MODE:
            # Deterministic mock "batch": same _mock_generate stand-in the rest
            # of the pipeline uses, just called in a loop here to exercise the
            # batching LOGIC (prompt building, fault-temp application, trace
            # assembly) without needing vllm installed or a GPU. This is what
            # was actually run to sanity-check this script locally -- the real
            # llm.chat() batched call below is unexercised until the Kaggle
            # smoke test (plan.md Section 5.5).
            from src.nodes import _mock_generate
            all_samples = [
                [_mock_generate(node_id, j, conversations[i][0]["content"], role) for j in range(k)]
                for i in range(n)
            ]
        else:
            # ONE batched call for this node across every trial in the batch --
            # this is the actual throughput win vLLM is being used for.
            # UNVERIFIED against a real GPU as of this writing (see module
            # docstring item 4) -- confirm llm.chat() accepts this batched
            # (List[List[message]], List[SamplingParams]) form during the
            # Kaggle smoke test before trusting it for a real run.
            outputs = llm.chat(conversations, params_list, use_tqdm=False)
            all_samples = [[c.text.strip() for c in output.outputs] for output in outputs]

        for i, samples in enumerate(all_samples):
            used_fallback = [False] * len(samples)  # Qwen2.5: no thinking-fallback path
            result = compute_node_result_from_samples(
                node_id, samples, used_fallback, get_normalizer(role), role
            )
            outputs_per_trial[i][node_id] = result.output
            traces[i].add(result)

    return traces


# ---------------------------------------------------------------------------
# Coverage strategy (plan.md Section 3): full grid on the simplest
# topologies, partial (>=1 fault type per node) on the rest.
# ---------------------------------------------------------------------------

def select_fault_conditions(topo: Topology, conditions: List[Optional[dict]],
                             full_grid: bool) -> List[Optional[dict]]:
    if full_grid:
        return conditions
    # Partial: keep control (None) + first occurrence of each (target_node)
    # across fault types, cycling fault types so every node gets >=1 fault
    # type covered without the full cross-product.
    seen_nodes = set()
    selected = [None]
    fault_types_cycle = ["noise", "contamination", "ceiling"]
    node_list = [n for n in topo.nodes if topo.nodes[n]]
    for i, node_id in enumerate(node_list):
        ft = fault_types_cycle[i % len(fault_types_cycle)]
        match = next((c for c in conditions
                      if c is not None and c["target_node"] == node_id and c["type"] == ft), None)
        if match:
            selected.append(match)
            seen_nodes.add(node_id)
    return selected


def fallback_fault_conditions_for_node(node_id: str, primary_type: str,
                                        conditions: List[Optional[dict]]) -> List[dict]:
    """Coverage-gap fix (2026-09-26): previously, if a node's single cyclically-
    assigned fault type (select_fault_conditions above) failed injection for
    EVERY example (observed for `retriever_c` in `star`/`triple_retriever_fanin`
    -- ceiling injection failed there 11/11 times, so those nodes ended up with
    ZERO fault trials at all despite the "every node gets >=1 fault type"
    coverage intent), there was no retry -- the node was silently left
    uncovered. Returns the OTHER fault-type conditions for this node, in cycle
    order, to try as a fallback if the primary type turns out to fail for
    every example in this topology.
    """
    fault_types_cycle = ["noise", "contamination", "ceiling"]
    start = fault_types_cycle.index(primary_type)
    ordered_types = fault_types_cycle[start + 1:] + fault_types_cycle[:start]
    fallbacks = []
    for ft in ordered_types:
        match = next((c for c in conditions
                      if c is not None and c["target_node"] == node_id and c["type"] == ft), None)
        if match:
            fallbacks.append(match)
    return fallbacks


# ---------------------------------------------------------------------------
# Per-topology driver
# ---------------------------------------------------------------------------

def run_topology(topo: Topology, examples: list, k: int, batch_size: int,
                  full_grid: bool, log_file, skip_file) -> int:
    conditions = select_fault_conditions(topo, build_fault_conditions(topo), full_grid)
    substitute_bank = build_substitute_bank(examples, topo, k=k, n_pilot=min(10, len(examples)))

    n_written = 0

    # --- Phase 1: control trials, batched, establishes per-question baselines ---
    control_specs = [
        {"question": ex["question"], "run_context": format_context(get_gold_context(ex)),
         "fault_config": None, "true_label": "clean", "gold_answer": ex["answer"]}
        for ex in examples
    ]
    baselines: Dict[str, Dict[str, float]] = {}
    for batch_start in range(0, len(control_specs), batch_size):
        batch = control_specs[batch_start:batch_start + batch_size]
        traces = run_pipeline_batch(topo, batch, k=k)
        for spec, trace in zip(batch, traces):
            trace.true_label = "clean"
            trace.gold_answer = spec["gold_answer"]
            baselines[spec["question"]] = trace.uncertainties()
            record = _trace_to_record(trace, fault_config=None, is_control=True)
            log_file.write(json.dumps(record) + "\n")
            n_written += 1
        log_file.flush()

    # --- Phase 2: resolve every fault trial's injection, then batch-execute ---
    # Coverage-gap fix (2026-09-26): if a node's assigned fault type fails
    # injection for EVERY example, retry with the next fault type in the
    # cycle for that node specifically, instead of leaving it with zero
    # fault trials -- see fallback_fault_conditions_for_node's docstring.
    resolved_specs = []
    pending_skips = []  # (ex, fc, reason) -- only written to skip_file once we
                         # know whether a fallback for that (ex, node) ever succeeded
    resolved_count_by_node: Dict[str, int] = {}

    def try_condition(ex, fc):
        resolved = resolve_fault_trial(ex, fc, topo, k, substitute_bank)
        if resolved is None:
            reason = f"{fc['type']}_injection_failed_{fc['target_node']}"
            pending_skips.append((ex, fc, reason))
            return False
        resolved_specs.append(resolved)
        resolved_count_by_node[fc["target_node"]] = resolved_count_by_node.get(fc["target_node"], 0) + 1
        return True

    for ex in examples:
        for fc in conditions:
            if fc is None:
                continue
            try_condition(ex, fc)

    if not full_grid:
        for fc in conditions:
            if fc is None:
                continue
            node_id, primary_type = fc["target_node"], fc["type"]
            if resolved_count_by_node.get(node_id, 0) > 0:
                continue  # this node already has at least one real fault trial
            for fallback_fc in fallback_fault_conditions_for_node(node_id, primary_type, conditions):
                for ex in examples:
                    try_condition(ex, fallback_fc)
                if resolved_count_by_node.get(node_id, 0) > 0:
                    break  # this fallback type worked for at least one example -- stop trying more types

    for ex, fc, reason in pending_skips:
        skip_file.write(json.dumps(_skip_record(
            str(ex.get("_id", ex["question"])), ex["question"], fc, reason)) + "\n")
    skip_file.flush()

    for batch_start in tqdm(range(0, len(resolved_specs), batch_size),
                             desc=f"  {topo.topology_id} faults"):
        batch = resolved_specs[batch_start:batch_start + batch_size]
        traces = run_pipeline_batch(topo, batch, k=k)
        for spec, trace in zip(batch, traces):
            trace.true_label = spec["true_label"]
            trace.gold_answer = spec["gold_answer"]
            tn = spec["fault_config"]["target_node"]
            baseline_u = baselines.get(spec["question"], {})
            verification = None
            if tn in trace.node_results:
                verification = verify_against_baseline(
                    trace.node_results[tn].uncertainty, baseline_u.get(tn, 0.0), spec["fault_config"]
                )
            record = _trace_to_record(trace, fault_config=spec["orig_fault_config"],
                                       is_control=False, verification=verification)
            log_file.write(json.dumps(record) + "\n")
            n_written += 1
        log_file.flush()

    return n_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-pool", type=str, default="../dataset/topology_pool.json")
    parser.add_argument("--examples-json", type=str, default="data/study1_examples.json")
    parser.add_argument("--n-examples", type=int, default=None,
                         help="Limit to the first N examples from --examples-json (default: all).")
    parser.add_argument("--topology-limit", type=int, default=None,
                         help="Limit to the first N topologies (after sorting by size) -- for testing.")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "ood_test"])
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--batch-size", type=int, default=15,
                         help="Trials per batched generate() call. Not yet empirically "
                              "verified safe -- see plan.md Section 5.5 smoke test.")
    parser.add_argument("--full-grid-count", type=int, default=3,
                         help="Number of smallest topologies (by node count) to run the "
                              "full fault grid on; the rest get partial coverage.")
    parser.add_argument("--log-path", type=str, default="logs/dataset_trials.jsonl")
    parser.add_argument("--skip-log-path", type=str, default="logs/dataset_skipped.jsonl")
    parser.add_argument("--mock", action="store_true",
                         help="Use mock backend (no vllm/GPU needed) -- for local dry-run "
                              "testing of the batching logic only, not a real generation run.")
    parser.add_argument("--examples-per-topology-full-grid", type=int, default=2,
                         help="Questions assigned to each full-grid topology (they already "
                              "get ~3x more fault conditions per question, so fewer questions "
                              "keeps total trial count from blowing up). Default 2.")
    parser.add_argument("--examples-per-topology-partial", type=int, default=4,
                         help="Questions assigned to each partial-coverage topology. Default 4.")
    parser.add_argument("--resume", action="store_true",
                         help="Skip topologies that already have records in --log-path "
                              "(topology-level granularity, not per-trial -- a killed/restarted "
                              "run redoes at most the one topology that was in progress, not "
                              "everything). Added after a real multi-hour run had to be killed "
                              "and restarted with no way to preserve partial progress.")
    args = parser.parse_args()

    if args.mock:
        os.environ["BTP_MOCK"] = "1"
        import src.config as _cfg; _cfg.MOCK_MODE = True
        import src.nodes as _nodes; _nodes.MOCK_MODE = True

    with open(args.examples_json, encoding="utf-8") as f:
        examples = json.load(f)
    if args.n_examples is not None:
        examples = examples[:args.n_examples]

    pool = load_topology_pool(args.topology_pool)
    if args.split != "all":
        pool = [(t, s) for t, s in pool if s == args.split]
    pool.sort(key=lambda ts: len(ts[0].nodes))
    full_grid_ids = {t.topology_id for t, _ in pool[:args.full_grid_count]}
    if args.topology_limit is not None:
        pool = pool[:args.topology_limit]

    print(f"[run_study_vllm] {len(pool)} topologies, {len(examples)} total questions available, "
          f"k={args.k}, batch_size={args.batch_size}")
    print(f"[run_study_vllm] Full-grid topologies ({args.examples_per_topology_full_grid} "
          f"questions each): {sorted(full_grid_ids)}")
    print(f"[run_study_vllm] Partial-coverage topologies ({args.examples_per_topology_partial} "
          f"questions each): the rest")

    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)

    done_topology_ids = set()
    if args.resume and os.path.exists(args.log_path):
        with open(args.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                done_topology_ids.add(json.loads(line)["topology_id"])
        print(f"[run_study_vllm] --resume: {len(done_topology_ids)} topologies already "
              f"complete in {args.log_path}, will skip: {sorted(done_topology_ids)}")

    total = 0
    n_examples_total = len(examples)
    rotation_cursor = 0  # advances across topologies so different topologies draw different
                          # questions (round-robin over the pool) rather than every topology
                          # reusing the same fixed first-N questions -- maximizes how much of
                          # the 30-question pool actually gets exercised across the run.
                          # Advanced for EVERY topology (even skipped ones on --resume) so the
                          # question assignment stays deterministic regardless of where a run
                          # was interrupted and restarted.
    with open(args.log_path, "a", encoding="utf-8") as log_file, \
         open(args.skip_log_path, "a", encoding="utf-8") as skip_file:
        for topo, split in pool:
            is_full_grid = topo.topology_id in full_grid_ids
            n_q = args.examples_per_topology_full_grid if is_full_grid else args.examples_per_topology_partial
            n_q = min(n_q, n_examples_total)
            topo_examples = [examples[(rotation_cursor + i) % n_examples_total] for i in range(n_q)]
            rotation_cursor += n_q

            if topo.topology_id in done_topology_ids:
                print(f"\n[run_study_vllm] === {topo.topology_id} (split={split}) "
                      f"SKIPPED (--resume, already complete) ===")
                continue

            print(f"\n[run_study_vllm] === {topo.topology_id} (split={split}, "
                  f"{n_q} questions) ===")
            n = run_topology(topo, topo_examples, k=args.k, batch_size=args.batch_size,
                              full_grid=is_full_grid,
                              log_file=log_file, skip_file=skip_file)
            total += n
            print(f"[run_study_vllm] {topo.topology_id}: {n} records written")

    print(f"\n[run_study_vllm] Done. {total} total records. Logs: {args.log_path}")
    print(f"[run_study_vllm] Now run: python scripts/verify_results.py {args.log_path} "
          f"--skip-log {args.skip_log_path}")


if __name__ == "__main__":
    main()
