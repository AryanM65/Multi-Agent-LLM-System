"""debug_noise_scoping.py — isolated test confirming node-scoped noise faults
only elevate temperature for the targeted node (Fix 2.4 / plan Section 4.2.3).

Monkeypatches sample_node to record the temperature it's called with for each
node in a real run_pipeline() execution (mock backend, so it's fast and free),
across all three possible noise targets. Asserts only the targeted node saw
NOISE_TEMPERATURE and the other two saw DEFAULT_TEMPERATURE.

Usage:
    python scripts/debug_noise_scoping.py
"""

import os
import sys

os.environ["BTP_MOCK"] = "1"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.config as cfg
cfg.MOCK_MODE = True
import src.nodes as nodes_mod
nodes_mod.MOCK_MODE = True

import src.pipeline as pipeline_mod
from src.topology import default_chain_topology

QUESTION = "Were Scott Derrickson and Ed Wood of the same nationality?"
CONTEXT = "Scott Derrickson is American. Ed Wood is American."


def make_topology():
    return default_chain_topology(
        retriever_instruction="You are a Retriever agent.",
        reasoner_instruction="You are a Reasoner agent.",
        writer_instruction="You are a Writer agent.",
    )


def run_with_capture(target_node: str):
    calls = {}
    real_sample_node = nodes_mod.sample_node

    def spy_sample_node(node_name, prompt, k, temperature, **kwargs):
        calls[node_name] = temperature
        return real_sample_node(node_name, prompt, k=k, temperature=temperature, **kwargs)

    pipeline_mod.sample_node = spy_sample_node
    try:
        fault_config = {"type": "noise", "target_node": target_node}
        pipeline_mod.run_pipeline(
            make_topology(), QUESTION, CONTEXT,
            k=3, temperature=cfg.DEFAULT_TEMPERATURE, fault_config=fault_config,
        )
    finally:
        pipeline_mod.sample_node = real_sample_node
    return calls


def main():
    all_ok = True
    for target in ("retriever", "reasoner", "writer"):
        calls = run_with_capture(target)
        print(f"\ntarget_node={target!r}")
        for node, temp in calls.items():
            tag = "NOISE" if abs(temp - cfg.NOISE_TEMPERATURE) < 1e-6 else "default"
            print(f"  {node:10s} temperature={temp}  ({tag})")

        expected_noisy = {target}
        actual_noisy = {n for n, t in calls.items() if abs(t - cfg.NOISE_TEMPERATURE) < 1e-6}
        if actual_noisy != expected_noisy:
            print(f"  FAIL: expected only {expected_noisy} at NOISE_TEMPERATURE, got {actual_noisy}")
            all_ok = False
        else:
            print(f"  PASS: only {target} elevated, others at DEFAULT_TEMPERATURE")

    print(f"\n{'ALL PASS' if all_ok else 'SOME FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
