"""debug_topology_regression.py — confirms the generic topology-engine
build_prompt() reproduces the old hardcoded Retriever->Reasoner->Writer
chain's prompts EXACTLY, per the master plan's Section 4.1 regression-test
requirement ("run the new generic engine on default_chain_topology(...) and
confirm it reproduces identical behavior to the current fixed pipeline on
the same test questions before building any non-chain topology on top of
it").

This originally failed: build_prompt had silently dropped the trailing
continuation cue ("Relevant sentences:" / "Reasoning:" / "Final answer:")
and swapped field order vs. the old retriever_prompt/reasoner_prompt/
writer_prompt functions. Fixed in src/pipeline.py; this script is the
regression check to prevent it recurring.

Usage:
    python scripts/debug_topology_regression.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.pipeline import (
    retriever_prompt, reasoner_prompt, writer_prompt, build_prompt,
    RETRIEVER_INSTRUCTION, REASONER_INSTRUCTION, WRITER_INSTRUCTION,
)
from src.topology import default_chain_topology

TEST_CASES = [
    {
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "context": "Scott Derrickson is American. Ed Wood is American.",
        "evidence": "Scott Derrickson is American. Ed Wood is American.",
        "reasoning": "Both are American, therefore same nationality.",
    },
    {
        "question": "What government position was held by the woman who portrayed Corliss Archer?",
        "context": "Shirley Temple portrayed Corliss Archer. She served as Chief of Protocol.",
        "evidence": "Shirley Temple served as Chief of Protocol of the United States.",
        "reasoning": "The evidence states she served as Chief of Protocol.",
    },
]


def main():
    topo = default_chain_topology(RETRIEVER_INSTRUCTION, REASONER_INSTRUCTION, WRITER_INSTRUCTION)
    all_ok = True

    for case in TEST_CASES:
        q, c, ev, rs = case["question"], case["context"], case["evidence"], case["reasoning"]

        checks = [
            ("retriever", retriever_prompt(q, c), build_prompt(topo, "retriever", {}, q, c, None)),
            ("reasoner", reasoner_prompt(q, ev), build_prompt(topo, "reasoner", {"retriever": ev}, q, "", None)),
            ("writer", writer_prompt(q, rs), build_prompt(topo, "writer", {"reasoner": rs}, q, "", None)),
        ]
        for node, old, new in checks:
            ok = old == new
            all_ok = all_ok and ok
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {node} prompt identical (question={q[:40]!r}...)")
            if not ok:
                print(f"  OLD: {old!r}")
                print(f"  NEW: {new!r}")

    print(f"\n{'ALL PASS — generic engine reproduces old chain exactly' if all_ok else 'REGRESSION DETECTED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
