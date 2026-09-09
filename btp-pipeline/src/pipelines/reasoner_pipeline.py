"""
reasoner_pipeline.py — Standalone Reasoner agent pipeline.

The Reasoner receives a question and the evidence already filtered by the
Retriever, then produces a full chain-of-thought reasoning trace.

Self-consistency sampling (k samples) is used to compute an uncertainty
score for this node.  The Reasoner gets a slightly larger token budget
(MAX_TOKENS_REASONER) because chain-of-thought needs more space.

Usage — imported from other code:
    from src.pipelines.reasoner_pipeline import run_reasoner_pipeline
    out = run_reasoner_pipeline(question="...", evidence="...")
    print(out.reasoning_chain)     # the chain-of-thought trace
    print(out.result.uncertainty)  # [0, 1]

Usage — standalone CLI:
    cd btp-pipeline/
    python -m src.pipelines.reasoner_pipeline \\
        --question "What nationality is the director of Crocodile Dundee?" \\
        --evidence "Australian film directed by Peter Faiman."

    # pipe from a saved retriever output:
    python -m src.pipelines.reasoner_pipeline \\
        --question "..." \\
        --from-retriever-output results/retriever_output.jsonl

Output is printed to stdout AND appended to results/reasoner_output.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Make the package importable when run as `python -m src.pipelines.reasoner_pipeline`
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
class ReasonerInput:
    """Input to the Reasoner pipeline."""
    question: str
    evidence: str    # Should be the Retriever's selected_evidence output


@dataclass
class ReasonerOutput:
    """Output from the Reasoner pipeline.

    Attributes
    ----------
    question:
        The original question (echoed for downstream chaining).
    evidence:
        The evidence that was given to the Reasoner (echoed for traceability).
    reasoning_chain:
        The Reasoner's best output — a full chain-of-thought reasoning trace.
        Feed this directly into WriterInput.reasoning.
    result:
        Full NodeResult including uncertainty score and all k raw samples.
        uncertainty ∈ [0, 1]; higher = less self-consistent.

    Note
    ----
    Reasoner uncertainty is expected to be higher than Retriever / Writer even
    on clean inputs (~0.667 at k=3) because chain-of-thought phrasing varies
    across samples even when the conclusion is identical.  The calibrated
    threshold in config.py accounts for this.
    """
    question: str
    evidence: str
    reasoning_chain: str
    result: NodeResult

    def to_dict(self) -> dict:
        return {
            "node": "reasoner",
            "question": self.question,
            "evidence": self.evidence,
            "reasoning_chain": self.reasoning_chain,
            "uncertainty": self.result.uncertainty,
            "samples": self.result.samples,
        }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _reasoner_prompt(question: str, evidence: str) -> str:
    """Build the Reasoner system + user prompt."""
    return (
        "You are a Reasoning agent. Given the evidence below, reason step by step "
        "to derive the answer to the question. Show your full chain of thought.\n\n"
        f"Question: {question}\n"
        f"Evidence: {evidence}\n\n"
        "Reasoning:"
    )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_reasoner_pipeline(
    question: str,
    evidence: str,
    k: int = DEFAULT_K,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ReasonerOutput:
    """Run the Reasoner agent with self-consistency sampling.

    Parameters
    ----------
    question:
        The natural-language question to answer.
    evidence:
        The relevant sentences to reason over — typically the output from
        run_retriever_pipeline().selected_evidence.
    k:
        Number of independent samples for self-consistency.
    temperature:
        Sampling temperature.

    Returns
    -------
    ReasonerOutput
        Includes the full chain-of-thought and the NodeResult (with
        per-node uncertainty and all k raw samples).
    """
    prompt = _reasoner_prompt(question, evidence)
    result: NodeResult = sample_node(
        node_name="reasoner",
        prompt=prompt,
        k=k,
        temperature=temperature,
    )
    return ReasonerOutput(
        question=question,
        evidence=evidence,
        reasoning_chain=result.output,
        result=result,
    )


# ---------------------------------------------------------------------------
# Output persistence helper
# ---------------------------------------------------------------------------

def _save_output(output: ReasonerOutput, results_dir: str = RESULTS_DIR) -> str:
    """Append the reasoner output as a JSONL line.  Returns the file path."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "reasoner_output.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(output.to_dict(), ensure_ascii=False) + "\n")
    return path


def _load_latest_retriever_output(jsonl_path: str) -> dict:
    """Read the last line from a retriever_output.jsonl file."""
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
        prog="python -m src.pipelines.reasoner_pipeline",
        description=(
            "Standalone Reasoner agent — produces a chain-of-thought from evidence."
        ),
    )
    p.add_argument("--question", required=True, help="The question to reason about.")

    evidence_group = p.add_mutually_exclusive_group(required=True)
    evidence_group.add_argument(
        "--evidence",
        default=None,
        help="Evidence string to reason over.",
    )
    evidence_group.add_argument(
        "--from-retriever-output",
        default=None,
        metavar="JSONL",
        help=(
            "Path to a retriever_output.jsonl file. "
            "The last record's 'selected_evidence' is used as evidence."
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
        help="Do not write output to results/reasoner_output.jsonl.",
    )
    return p


def main(argv: Optional[list] = None) -> ReasonerOutput:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Resolve evidence source
    if args.from_retriever_output:
        record = _load_latest_retriever_output(args.from_retriever_output)
        evidence = record["selected_evidence"]
        print(f"[reasoner_pipeline] Loaded evidence from: {args.from_retriever_output}")
    else:
        evidence = args.evidence

    print(f"\n[reasoner_pipeline] Question : {args.question}")
    print(f"[reasoner_pipeline] Evidence : {evidence[:120].strip()}{'...' if len(evidence) > 120 else ''}")
    print(f"[reasoner_pipeline] k={args.k}  temperature={args.temperature}")
    print("[reasoner_pipeline] Running Reasoner agent...\n")

    output = run_reasoner_pipeline(
        question=args.question,
        evidence=evidence,
        k=args.k,
        temperature=args.temperature,
    )

    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  [REASONER RESULT]")
    print(f"  Uncertainty     : {output.result.uncertainty:.4f}  (0=certain, 1=random)")
    print(f"  Reasoning chain : {output.reasoning_chain[:400].strip()}")
    if len(output.result.samples) > 1:
        print(f"\n  All {len(output.result.samples)} samples:")
        for i, s in enumerate(output.result.samples):
            print(f"    [{i}] {s[:140].strip()}")
    print(sep)

    if not args.no_save:
        path = _save_output(output)
        print(f"\n[reasoner_pipeline] Output saved → {path}")

    return output


if __name__ == "__main__":
    main()
