"""
config.py — Single source of truth for all tuneable constants.

UNCERTAINTY_THRESHOLD calibration history:
  Original placeholder: 0.3
  Phase 2 calibration (3 questions, k=3, clean gold context):
    Retriever  uncertainty: 0.0   (short factual sentence extraction)
    Reasoner   uncertainty: 0.667 (CoT phrasing variance — expected baseline)
    Writer     uncertainty: 0.0   (short final answers)
  Calibrated value: 0.75  (just above Reasoner's 0.667 baseline)

NODE_THRESHOLDS (Study 2): per-node threshold vector replacing the global
  threshold.  Retriever and Writer baselines are ~0.0, so a much tighter
  per-node threshold detects more real faults without raising false-positive
  rates.  Updated empirically once ≥15 clean-baseline trials exist.
"""

import os

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Backend: mlx_lm (Apple Silicon / Metal).
MODEL = "mlx-community/Qwen3-8B-4bit"

# ---------------------------------------------------------------------------
# Mock / CPU mode  (Work-stream 1)
# ---------------------------------------------------------------------------
# Set BTP_MOCK=1 in the environment to run without MLX (deterministic stubs).
# All scripts honour this flag; CLI tools also accept --mock to set it at runtime.
MOCK_MODE: bool = os.getenv("BTP_MOCK", "0") == "1"

# ---------------------------------------------------------------------------
# Self-consistency sampling
# ---------------------------------------------------------------------------
DEFAULT_K = 3              # samples per node — bump to 5 for cloud-scale Study 2
DEFAULT_TEMPERATURE = 0.7  # standard inference temperature
NOISE_TEMPERATURE = 1.2    # elevated temperature used for noise fault injection
MAX_TOKENS = 128           # max tokens for Retriever / Writer
MAX_TOKENS_REASONER = 200  # extra headroom for Reasoner chain-of-thought

# ---------------------------------------------------------------------------
# Diagnosis — global + per-node thresholds
# ---------------------------------------------------------------------------
UNCERTAINTY_THRESHOLD = 0.75   # global fallback (Study 1 / single-threshold mode)

# Study 2 per-node threshold vector.  Callers that support per-node thresholds
# should use needs_diagnosis_per_node() in diagnose.py which reads this dict.
# Override any value here after empirical calibration on ≥15 clean runs.
NODE_THRESHOLDS: dict = {
    "retriever":   0.30,   # baseline ≈ 0.0 on clean gold context
    "reasoner":    0.75,   # baseline ≈ 0.667 due to CoT phrasing variance
    "writer":      0.30,   # baseline ≈ 0.0 on clean gold context
    # Generic fallback for non-standard node IDs (used in multi-retriever topologies)
    "_default":    0.75,
}

DEFAULT_RETRIES = 2

# ---------------------------------------------------------------------------
# Fault injection taxonomy (Phase 3)
# ---------------------------------------------------------------------------
# The full Study 2+ fault grid is (fault_type × target_node).
# For the current chain topology, target nodes are retriever, reasoner, writer.
# Use build_fault_conditions() to get the complete 3×3 grid + clean control.
FAULT_TYPES  = ["noise", "contamination", "ceiling"]
TARGET_NODES = ["retriever", "reasoner", "writer"]
# NOTE: When non-chain topologies are used, call build_fault_conditions(topo)
# to auto-derive target nodes from topo.nodes.keys().


def build_fault_conditions(topo=None):
    """Return the full fault grid plus a clean control.

    Args:
        topo: Optional Topology object.  When provided, target nodes are derived
              from topo.nodes.keys() (supports any topology shape, not just the
              3-node chain).  When None, defaults to TARGET_NODES.

    Returns:
        List[Optional[dict]] — None for clean control (always first), then one
        dict per (fault_type × target_node) combination.
    """
    nodes = list(topo.nodes.keys()) if topo is not None else TARGET_NODES
    conditions = [None]   # clean control, always first
    for ftype in FAULT_TYPES:
        for node in nodes:
            conditions.append({"type": ftype, "target_node": node})
    return conditions


# ---------------------------------------------------------------------------
# Available named topologies  (Work-stream 2)
# ---------------------------------------------------------------------------
# Keys here match the topology_id values in src/topologies.py.
# Used by run_study2.py --topology flag.
TOPOLOGY_REGISTRY = [
    "chain",                  # default 3-node chain (Retriever→Reasoner→Writer)
    "dual_retriever_fanin",   # two independent Retrievers → Reasoner → Writer
    "parallel_reasoner",      # Retriever → (Reasoner-A ─┐) → Writer
                              #                           ├──
                              #             (Reasoner-B ─┘)   (shared Retriever parent)
    "deep_chain",             # Retriever → Reasoner-1 → Reasoner-2 → Writer
]

# ---------------------------------------------------------------------------
# Paths  (relative to btp-pipeline/)
# ---------------------------------------------------------------------------
DATA_DIR   = "./data/hotpotqa_distractor"
LOG_DIR    = "./logs"
RESULTS_DIR = "./results"

DEFAULT_LOG_PATH             = f"{LOG_DIR}/trials.jsonl"
DEFAULT_SKIP_LOG_PATH        = f"{LOG_DIR}/skipped_trials.jsonl"
DEFAULT_CONFUSION_MATRIX_PATH = f"{RESULTS_DIR}/confusion_matrix.csv"

# Study 2 paths
DEFAULT_STUDY2_LOG_PATH      = f"{LOG_DIR}/study2_trials.jsonl"
DEFAULT_STUDY2_SKIP_LOG_PATH = f"{LOG_DIR}/study2_skipped.jsonl"
DEFAULT_STUDY2_CM_DIR        = f"{RESULTS_DIR}/study2_confusion/"
