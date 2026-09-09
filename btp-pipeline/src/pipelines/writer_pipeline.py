"""
writer_pipeline.py — Standalone Writer agent pipeline.

The Writer receives a question and the chain-of-thought reasoning trace
produced by the Reasoner, then synthesises a final concise answer.

Self-consistency sampling (k samples) is used to compute an uncertainty
score.  Writer outputs tend to be short and highly consistent (low
uncertainty) on clean inputs — high uncertainty here flags a genuine
failure in the reasoning passed to it.

Usage — imported from other code:
    from src.pipelines.writer_pipeline import run_writer_pipeline
    out = run_writer_pipeline(question="...", reasoning="...")
    print(out.final_answer)        # the concise answer
    print(out.result.uncertainty)  # [0, 1]

Usage — standalone CLI:
    cd btp-pipeline/
    python -m src.pipelines.writer_pipeline \\
        --question  "What nationality is the director of Crocodile Dundee?" \\
        --reasoning "Peter Faiman directed it. It is Australian. Therefore Australian."

    # pipe from a saved reasoner output:
    python -m src.pipelines.writer_pipeline \\
        --question "..." \\
        --from-reasoner-output results/reasoner_output.jsonl

Output is printed to stdout AND appended to results/writer_output.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Make the package importable when run as `python -m src.pipelines.writer_pipeline`
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)

from src.config import DEFAULT_K, DEFAULT_TEMPERATURE, RESULTS_DIR
from src.nodes import NodeResult, sample_node


# ---------------------------------------------------------------------------
# Typed I/O dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WriterInput:
    """Input to the Writer pipeline."""
    question: str
    reasoning: str   # Should be the Reasoner's reasoning_chain output


@dataclass
class WriterOutput:
    """Output from the Writer pipeline.

    Attributes
    ----------
    question:
        The original question (echoed for traceability).
    reasoning:
        The reasoning chain given to the Writer (echoed for traceability).
    final_answer:
        The Writer's best output — a single, concise answer string.
    result:
        Full NodeResult including uncertainty score and all k raw samples.
        uncertainty ∈ [0, 1]; higher = less self-consistent.

    Note
    ----
    Writer uncertainty is expected to be very low (~0.0) on clean inputs
    because short factual answers converge quickly across samples.
    If writer uncertainty is high after clean retrieval and reasoning,
    it may indicate a genuine ambiguity in the question or a corrupted
    reasoning trace from the Reasoner.
    """
    question: str
    reasoning: str
    final_answer: str
    result: NodeResult

    def to_dict(self) -> dict:
        return {
            "node": "writer",
            "question": self.question,
            "reasoning": self.reasoning,
            "final_answer": self.final_answer,
            "uncertainty": self.result.uncertainty,
            "samples": self.result.samples,
        }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _writer_prompt(question: str, reasoning: str) -> str:
    """Build the Writer system + user prompt."""
    return (
        "You are a Writer agent. Given the reasoning trace below, produce a final, "
        "concise answer to the question. Output only the answer — no explanation, "
        "no preamble.\n\n"
        f"Question: {question}\n"
        f"Reasoning: {reasoning}\n\n"
        "Final answer:"
    )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_writer_pipeline(
    question: str,
    reasoning: str,
    k: int = DEFAULT_K,
    temperature: float = DEFAULT_TEMPERATURE,
) -> WriterOutput:
    """Run the Writer agent with self-consistency sampling.

    Parameters
    ----------
    question:
        The natural-language question to answer.
    reasoning:
        The chain-of-thought reasoning trace — typically the output from
        run_reasoner_pipeline().reasoning_chain.
    k:
        Number of independent samples for self-consistency.
    temperature:
        Sampling temperature.

    Returns
    -------
    WriterOutput
        Includes the concise final answer and the NodeResult (with
        per-node uncertainty and all k raw samples).
    """
    prompt = _writer_prompt(question, reasoning)
    result: NodeResult = sample_node(
        node_name="writer",
        prompt=prompt,
        k=k,
        temperature=temperature,
    )
    return WriterOutput(
        question=question,
        reasoning=reasoning,
        final_answer=result.output,
        result=result,
    )


# ---------------------------------------------------------------------------
# Output persistence helper
# ---------------------------------------------------------------------------

def _save_output(output: WriterOutput, results_dir: str = RESULTS_DIR) -> str:
    """Append the writer output as a JSONL line.  Returns the file path."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "writer_output.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(output.to_dict(), ensure_ascii=False) + "\n")
    return path


def _load_latest_reasoner_output(jsonl_path: str) -> dict:
    """Read the last line from a reasoner_output.jsonl file."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        raise ValueError(f"No records found in {jsonl_path}")
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.pipelines.writer_pipeline",
        description=(
            "Standalone Writer agent — synthesises a final answer from reasoning."
        ),
    )
    p.add_argument("--question", required=True, help="The question to answer.")

    reasoning_group = p.add_mutually_exclusive_group(required=True)
    reasoning_group.add_argument(
        "--reasoning",
        default=None,
        help="Chain-of-thought reasoning string.",
    )
    reasoning_group.add_argument(
        "--from-reasoner-output",
        default=None,
        metavar="JSONL",
        help=(
            "Path to a reasoner_output.jsonl file. "
            "The last record's 'reasoning_chain' is used."
        ),
    )

    p.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of self-consistency samples (default: {DEFAULT_K}).",
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
        help="Do not write output to results/writer_output.jsonl.",
    )
    return p


def main(argv: Optional[list] = None) -> WriterOutput:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Resolve reasoning source
    if args.from_reasoner_output:
        record = _load_latest_reasoner_output(args.from_reasoner_output)
        reasoning = record["reasoning_chain"]
        print(f"[writer_pipeline] Loaded reasoning from: {args.from_reasoner_output}")
    else:
        reasoning = args.reasoning

    print(f"\n[writer_pipeline] Question  : {args.question}")
    print(f"[writer_pipeline] Reasoning : {reasoning[:120].strip()}{'...' if len(reasoning) > 120 else ''}")
    print(f"[writer_pipeline] k={args.k}  temperature={args.temperature}")
    print("[writer_pipeline] Running Writer agent...\n")

    output = run_writer_pipeline(
        question=args.question,
        reasoning=reasoning,
        k=args.k,
        temperature=args.temperature,
    )

    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  [WRITER RESULT]")
    print(f"  Uncertainty  : {output.result.uncertainty:.4f}  (0=certain, 1=random)")
    print(f"  Final answer : {output.final_answer.strip()}")
    if len(output.result.samples) > 1:
        print(f"\n  All {len(output.result.samples)} samples:")
        for i, s in enumerate(output.result.samples):
            print(f"    [{i}] {s.strip()}")
    print(sep)

    if not args.no_save:
        path = _save_output(output)
        print(f"\n[writer_pipeline] Output saved → {path}")

    return output


if __name__ == "__main__":
    main()
