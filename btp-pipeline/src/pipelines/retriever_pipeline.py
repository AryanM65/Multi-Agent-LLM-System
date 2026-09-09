"""
retriever_pipeline.py — Standalone Retriever agent pipeline.

The Retriever receives a question and raw candidate paragraphs (context)
and returns only the sentences directly relevant to answering the question.

Self-consistency sampling (k samples) is used to compute an uncertainty
score for this node.  The modal (most-agreed-upon) selection is returned.

Usage — as a module imported from other code:
    from src.pipelines.retriever_pipeline import run_retriever_pipeline
    out = run_retriever_pipeline(question="...", context="...")
    print(out.selected_evidence)   # the filtered sentences
    print(out.result.uncertainty)  # [0, 1]; higher = less self-consistent

Usage — as a standalone CLI:
    cd btp-pipeline/
    python -m src.pipelines.retriever_pipeline \\
        --question "What nationality is the director of Crocodile Dundee?" \\
        --context  "[Crocodile Dundee] Australian film directed by Peter Faiman."

    # or load context from a plain text file:
    python -m src.pipelines.retriever_pipeline \\
        --question "..." \\
        --context-file /path/to/context.txt

Output is printed to stdout AND appended to results/retriever_output.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Make the package importable when run as `python -m src.pipelines.retriever_pipeline`
# from inside btp-pipeline/.
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
class RetrieverInput:
    """Input to the Retriever pipeline."""
    question: str
    context: str          # Raw candidate paragraphs (multi-paragraph string)


@dataclass
class RetrieverOutput:
    """Output from the Retriever pipeline.

    Attributes
    ----------
    question:
        The original question (echoed for downstream chaining).
    selected_evidence:
        The Retriever's best output — sentences it judged most relevant.
        Feed this directly into ReasonerInput.evidence.
    result:
        Full NodeResult including uncertainty score and all k raw samples.
        uncertainty ∈ [0, 1]; higher = less self-consistent.
    """
    question: str
    selected_evidence: str
    result: NodeResult

    def to_dict(self) -> dict:
        return {
            "node": "retriever",
            "question": self.question,
            "selected_evidence": self.selected_evidence,
            "uncertainty": self.result.uncertainty,
            "samples": self.result.samples,
        }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _retriever_prompt(question: str, context: str) -> str:
    """Build the Retriever system + user prompt."""
    return (
        "You are a Retriever agent. Given the question and candidate paragraphs, "
        "select and return only the sentences that are directly relevant to answering "
        "the question. Do not add any commentary or explanation — only return the "
        "relevant sentences.\n\n"
        f"Question: {question}\n"
        f"Paragraphs:\n{context}\n\n"
        "Relevant sentences:"
    )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_retriever_pipeline(
    question: str,
    context: str,
    k: int = DEFAULT_K,
    temperature: float = DEFAULT_TEMPERATURE,
) -> RetrieverOutput:
    """Run the Retriever agent with self-consistency sampling.

    Parameters
    ----------
    question:
        The natural-language question to answer.
    context:
        Candidate paragraphs (multi-hop style, e.g. from HotpotQA distractor
        set or any free-text passage).
    k:
        Number of independent samples for self-consistency.
        More samples → more reliable uncertainty estimate (but slower).
    temperature:
        Sampling temperature.  Use DEFAULT_TEMPERATURE (0.7) for normal
        operation; NOISE_TEMPERATURE (1.2) to simulate a noise fault.

    Returns
    -------
    RetrieverOutput
        Includes the selected evidence and the full NodeResult (with
        per-node uncertainty score and all k raw samples).
    """
    prompt = _retriever_prompt(question, context)
    result: NodeResult = sample_node(
        node_name="retriever",
        prompt=prompt,
        k=k,
        temperature=temperature,
    )
    return RetrieverOutput(
        question=question,
        selected_evidence=result.output,
        result=result,
    )


# ---------------------------------------------------------------------------
# Output persistence helper
# ---------------------------------------------------------------------------

def _save_output(output: RetrieverOutput, results_dir: str = RESULTS_DIR) -> str:
    """Append the retriever output as a JSONL line.  Returns the file path."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "retriever_output.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(output.to_dict(), ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.pipelines.retriever_pipeline",
        description="Standalone Retriever agent — extracts relevant evidence from context.",
    )
    p.add_argument("--question", required=True, help="The question to answer.")
    p.add_argument(
        "--context",
        default=None,
        help="Candidate paragraphs as a string.",
    )
    p.add_argument(
        "--context-file",
        default=None,
        metavar="FILE",
        help="Path to a plain-text file whose contents are used as context.",
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
        help="Do not write output to results/retriever_output.jsonl.",
    )
    return p


def main(argv: Optional[list] = None) -> RetrieverOutput:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Resolve context source
    if args.context_file:
        with open(args.context_file, "r", encoding="utf-8") as f:
            context = f.read()
    elif args.context:
        context = args.context
    else:
        parser.error("Provide --context or --context-file.")

    print(f"\n[retriever_pipeline] Question : {args.question}")
    print(f"[retriever_pipeline] Context  : {context[:120].strip()}{'...' if len(context) > 120 else ''}")
    print(f"[retriever_pipeline] k={args.k}  temperature={args.temperature}")
    print("[retriever_pipeline] Running Retriever agent...\n")

    output = run_retriever_pipeline(
        question=args.question,
        context=context,
        k=args.k,
        temperature=args.temperature,
    )

    # Print result summary
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  [RETRIEVER RESULT]")
    print(f"  Uncertainty      : {output.result.uncertainty:.4f}  (0=certain, 1=random)")
    print(f"  Selected evidence: {output.selected_evidence[:300].strip()}")
    if len(output.result.samples) > 1:
        print(f"\n  All {len(output.result.samples)} samples:")
        for i, s in enumerate(output.result.samples):
            print(f"    [{i}] {s[:120].strip()}")
    print(sep)

    if not args.no_save:
        path = _save_output(output)
        print(f"\n[retriever_pipeline] Output saved → {path}")

    return output


if __name__ == "__main__":
    main()
