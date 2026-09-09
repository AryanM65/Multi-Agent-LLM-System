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
    from src.pipelines.retriever_pipeline import run_retriever_pipeline
    from src.pipelines.reasoner_pipeline  import run_reasoner_pipeline
    from src.pipelines.writer_pipeline    import run_writer_pipeline
    from src.pipelines.retriever_pipeline import _save_output as save_ret
    from src.pipelines.reasoner_pipeline  import _save_output as save_rea
    from src.pipelines.writer_pipeline    import _save_output as save_wri

    context = _resolve_context(args)
    question = args.question
    k = args.k
    temperature = args.temperature

    SEP = "═" * 68
    print(f"\n{SEP}")
    print(f"  FULL PIPELINE: Retriever → Reasoner → Writer")
    print(f"  Question   : {question}")
    print(f"  k={k}  temperature={temperature}")
    print(SEP)

    # --- Node 1: Retriever ---
    print("\n[1/3] Running RETRIEVER...")
    ret_out = run_retriever_pipeline(question=question, context=context, k=k, temperature=temperature)
    print(f"  ✓ Retriever  uncertainty = {ret_out.result.uncertainty:.4f}")
    print(f"  Selected evidence: {ret_out.selected_evidence[:200].strip()}")
    if not args.no_save:
        save_ret(ret_out)

    # --- Node 2: Reasoner ---
    print("\n[2/3] Running REASONER...")
    rea_out = run_reasoner_pipeline(question=question, evidence=ret_out.selected_evidence, k=k, temperature=temperature)
    print(f"  ✓ Reasoner   uncertainty = {rea_out.result.uncertainty:.4f}")
    print(f"  Reasoning (first 300 chars): {rea_out.reasoning_chain[:300].strip()}")
    if not args.no_save:
        save_rea(rea_out)

    # --- Node 3: Writer ---
    print("\n[3/3] Running WRITER...")
    wri_out = run_writer_pipeline(question=question, reasoning=rea_out.reasoning_chain, k=k, temperature=temperature)
    print(f"  ✓ Writer     uncertainty = {wri_out.result.uncertainty:.4f}")
    print(f"  Final answer: {wri_out.final_answer.strip()}")
    if not args.no_save:
        save_wri(wri_out)

    # --- Summary ---
    print(f"\n{SEP}")
    print("  PER-NODE UNCERTAINTY SUMMARY  (never collapsed to a scalar)")
    print(f"  {'Node':<12}  {'Uncertainty':>12}  {'Status'}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*20}")

    from src.config import UNCERTAINTY_THRESHOLD
    all_results = [
        ("retriever", ret_out.result.uncertainty),
        ("reasoner",  rea_out.result.uncertainty),
        ("writer",    wri_out.result.uncertainty),
    ]
    for name, u in all_results:
        flag = "⚠  ABOVE THRESHOLD" if u > UNCERTAINTY_THRESHOLD else "✓  ok"
        print(f"  {name:<12}  {u:>12.4f}  {flag}")

    print(f"\n  Threshold (UNCERTAINTY_THRESHOLD): {UNCERTAINTY_THRESHOLD}")
    print(f"  Final answer : {wri_out.final_answer.strip()}")
    print(SEP + "\n")

    return ret_out, rea_out, wri_out


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
