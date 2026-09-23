"""
run_agent.py — Unified CLI to run Retriever, Reasoner, or Writer agents.

Dispatches to each agent's standalone pipeline module.  Also supports
running all three in sequence (Retriever → Reasoner → Writer) and printing
the per-node uncertainty comparison.

Usage
-----
Run a single agent:
    python scripts/run_agent.py retriever \\
        --question "What nationality is the director of Crocodile Dundee?" \\
        --context  "[Crocodile Dundee] Australian film directed by Peter Faiman."

    python scripts/run_agent.py reasoner \\
        --question "What nationality is the director of Crocodile Dundee?" \\
        --evidence "Australian film directed by Peter Faiman."

    python scripts/run_agent.py writer \\
        --question  "What nationality is the director of Crocodile Dundee?" \\
        --reasoning "Peter Faiman directed the film. It is Australian. ..."

Chain all three sequentially (equivalent to run_pipeline, but via individual modules):
    python scripts/run_agent.py all \\
        --question "What nationality is the director of Crocodile Dundee?" \\
        --context  "[Crocodile Dundee] Australian film directed by Peter Faiman."

Pipe between saved outputs:
    # Step 1: run retriever, save to results/retriever_output.jsonl
    python scripts/run_agent.py retriever --question "..." --context "..."
    # Step 2: run reasoner using saved retriever output
    python scripts/run_agent.py reasoner --question "..." \\
        --from-retriever-output results/retriever_output.jsonl
    # Step 3: run writer using saved reasoner output
    python scripts/run_agent.py writer   --question "..." \\
        --from-reasoner-output results/reasoner_output.jsonl

Options
-------
    --k             Number of self-consistency samples (default: 3)
    --temperature   Sampling temperature (default: 0.7)
    --no-save       Skip writing to results/*.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Ensure btp-pipeline/ is on sys.path when run as `python scripts/run_agent.py`
# ---------------------------------------------------------------------------
_SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, ".."))
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from src.config import DEFAULT_K, DEFAULT_TEMPERATURE


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------

def build_top_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/run_agent.py",
        description=(
            "Run the Retriever, Reasoner, or Writer agent in isolation, "
            "or chain all three with 'all'."
        ),
    )
    p.add_argument(
        "agent",
        choices=["retriever", "reasoner", "writer", "all"],
        help=(
            "Which agent to run. Use 'all' to chain Retriever → Reasoner → Writer "
            "and print a per-node uncertainty summary."
        ),
    )
    p.add_argument("--question", required=True, help="The question to process.")

    # Context / evidence / reasoning sources
    p.add_argument(
        "--context",
        default=None,
        help="Candidate paragraphs (required for 'retriever' and 'all').",
    )
    p.add_argument(
        "--context-file",
        default=None,
        metavar="FILE",
        help="Path to plain-text file used as context (alternative to --context).",
    )
    p.add_argument(
        "--evidence",
        default=None,
        help="Evidence string (required for 'reasoner' unless --from-retriever-output).",
    )
    p.add_argument(
        "--reasoning",
        default=None,
        help="Reasoning string (required for 'writer' unless --from-reasoner-output).",
    )
    p.add_argument(
        "--from-retriever-output",
        default=None,
        metavar="JSONL",
        help="Path to retriever_output.jsonl to pipe into reasoner.",
    )
    p.add_argument(
        "--from-reasoner-output",
        default=None,
        metavar="JSONL",
        help="Path to reasoner_output.jsonl to pipe into writer.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Self-consistency samples per node (default: {DEFAULT_K}).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write per-agent JSONL output files.",
    )
    return p


# ---------------------------------------------------------------------------
# Context resolver helper
# ---------------------------------------------------------------------------

def _resolve_context(args: argparse.Namespace) -> str:
    if args.context_file:
        with open(args.context_file, "r", encoding="utf-8") as f:
            return f.read()
    if args.context:
        return args.context
    raise ValueError(
        "No context provided. Use --context or --context-file for "
        "'retriever' and 'all' modes."
    )


# ---------------------------------------------------------------------------
# Agent dispatch helpers
# ---------------------------------------------------------------------------

def run_retriever_mode(args: argparse.Namespace):
    from src.pipelines.retriever_pipeline import main as retriever_main

    argv = [
        "--question", args.question,
        "--k", str(args.k),
        "--temperature", str(args.temperature),
    ]
    context = _resolve_context(args)
    argv += ["--context", context]
    if args.no_save:
        argv.append("--no-save")

    return retriever_main(argv)


def run_reasoner_mode(args: argparse.Namespace):
    from src.pipelines.reasoner_pipeline import main as reasoner_main

    argv = [
        "--question", args.question,
        "--k", str(args.k),
        "--temperature", str(args.temperature),
    ]
    if args.from_retriever_output:
        argv += ["--from-retriever-output", args.from_retriever_output]
    elif args.evidence:
        argv += ["--evidence", args.evidence]
    else:
        raise ValueError(
            "For 'reasoner' mode, provide --evidence or --from-retriever-output."
        )
    if args.no_save:
        argv.append("--no-save")

    return reasoner_main(argv)


def run_writer_mode(args: argparse.Namespace):
    from src.pipelines.writer_pipeline import main as writer_main

    argv = [
        "--question", args.question,
        "--k", str(args.k),
        "--temperature", str(args.temperature),
    ]
    if args.from_reasoner_output:
        argv += ["--from-reasoner-output", args.from_reasoner_output]
    elif args.reasoning:
        argv += ["--reasoning", args.reasoning]
    else:
        raise ValueError(
            "For 'writer' mode, provide --reasoning or --from-reasoner-output."
        )
    if args.no_save:
        argv.append("--no-save")

    return writer_main(argv)


# ---------------------------------------------------------------------------
# 'all' mode — chain all three agents and print uncertainty summary
# ---------------------------------------------------------------------------

def run_all_mode(args: argparse.Namespace):
    """Run all three nodes in sequence via the topology engine.

    Phase 1 update: uses run_pipeline(default_topology(), ...) instead of
    calling each standalone pipeline module separately.  The individual
    per-node pipeline modules (retriever_pipeline, reasoner_pipeline,
    writer_pipeline) still exist for single-node interactive testing.
    """
    from src.pipeline import run_pipeline, default_topology

    context = _resolve_context(args)
    question = args.question
    k = args.k
    temperature = args.temperature

    SEP = "═" * 68
    print(f"\n{SEP}")
    print(f"  FULL PIPELINE (topology engine): Retriever → Reasoner → Writer")
    print(f"  Question   : {question}")
    print(f"  k={k}  temperature={temperature}")
    print(SEP)

    topo = default_topology()
    trace = run_pipeline(topo, question, context, k=k, temperature=temperature)

    # --- Print per-node results ---
    for i, (node_id, result) in enumerate(trace.node_results.items(), 1):
        print(f"\n[{i}/{len(trace.node_results)}] {node_id.upper()}")
        print(f"  ✓ uncertainty (lexical)   = {result.uncertainty:.4f}")
        if result.uncertainty_semantic is not None:
            print(f"  ✓ uncertainty (semantic)  = {result.uncertainty_semantic:.4f}")
        print(f"  output: {result.output[:300].strip()}")

    # --- Summary ---
    print(f"\n{SEP}")
    print("  PER-NODE UNCERTAINTY SUMMARY  (never collapsed to a scalar)")
    print(f"  {'Node':<12}  {'Lex Unc':>10}  {'Sem Unc':>10}  {'Status'}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*20}")

    from src.config import UNCERTAINTY_THRESHOLD
    for node_id, result in trace.node_results.items():
        flag = "⚠  ABOVE THRESHOLD" if result.uncertainty > UNCERTAINTY_THRESHOLD else "✓  ok"
        sem_str = f"{result.uncertainty_semantic:.4f}" if result.uncertainty_semantic is not None else "  N/A  "
        print(f"  {node_id:<12}  {result.uncertainty:>10.4f}  {sem_str:>10}  {flag}")

    writer_result = trace.node_results.get("writer")
    final_answer = writer_result.output.strip() if writer_result else "(not available)"
    print(f"\n  Threshold (UNCERTAINTY_THRESHOLD): {UNCERTAINTY_THRESHOLD}")
    print(f"  Topology : {trace.topology_id}")
    print(f"  Final answer : {final_answer}")
    print(SEP + "\n")

    return trace


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    parser = build_top_parser()
    args = parser.parse_args()

    dispatch = {
        "retriever": run_retriever_mode,
        "reasoner":  run_reasoner_mode,
        "writer":    run_writer_mode,
        "all":       run_all_mode,
    }
    dispatch[args.agent](args)


if __name__ == "__main__":
    main()
