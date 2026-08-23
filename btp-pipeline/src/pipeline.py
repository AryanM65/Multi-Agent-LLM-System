"""
pipeline.py — Three-node pipeline: Retriever → Reasoner → Writer.

All calls are sequential.  Returns a PipelineTrace with per-node
NodeResults — uncertainties are never merged into a scalar.
"""

from __future__ import annotations

from src.config import DEFAULT_K, DEFAULT_TEMPERATURE
from src.nodes import NodeResult, PipelineTrace, sample_node

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def retriever_prompt(question: str, context: str) -> str:
    return (
        "You are a Retriever agent. Given the question and candidate paragraphs, "
        "select and return only the sentences that are directly relevant to answering "
        "the question. Do not add any commentary or explanation — only return the "
        "relevant sentences.\n\n"
        f"Question: {question}\n"
        f"Paragraphs:\n{context}\n\n"
        "Relevant sentences:"
    )


def reasoner_prompt(question: str, evidence: str) -> str:
    return (
        "You are a Reasoning agent. Given the evidence below, reason step by step "
        "to derive the answer to the question. Show your full chain of thought.\n\n"
        f"Question: {question}\n"
        f"Evidence: {evidence}\n\n"
        "Reasoning:"
    )


def writer_prompt(question: str, reasoning: str) -> str:
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

def run_pipeline(
    question: str,
    context: str,
    k: int = DEFAULT_K,
    temperature: float = DEFAULT_TEMPERATURE,
) -> PipelineTrace:
    """Run all three pipeline nodes sequentially and collect a PipelineTrace.

    The temperature argument applies uniformly to all three nodes unless
    a caller overrides per-node (e.g. noise injection raises temperature for
    the whole pipeline).  For contamination/ceiling faults, 0.7 is used
    throughout.

    Returns a PipelineTrace; call trace.uncertainties() to get the per-node
    dict — never pass it as a scalar.
    """
    trace = PipelineTrace(question=question)

    # Node 1 — Retriever
    r1 = sample_node(
        "retriever",
        retriever_prompt(question, context),
        k=k,
        temperature=temperature,
    )
    trace.add(r1)

    # Node 2 — Reasoner  (consumes Retriever's best output)
    r2 = sample_node(
        "reasoner",
        reasoner_prompt(question, r1.output),
        k=k,
        temperature=temperature,
    )
    trace.add(r2)

    # Node 3 — Writer  (consumes Reasoner's best output)
    r3 = sample_node(
        "writer",
        writer_prompt(question, r2.output),
        k=k,
        temperature=temperature,
    )
    trace.add(r3)

    return trace
