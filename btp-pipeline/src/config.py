"""
config.py — Single source of truth for all tuneable constants.

UNCERTAINTY_THRESHOLD calibration history:
  Original placeholder: 0.3
  Phase 2 calibration (3 questions, k=3, clean gold context):
    Retriever  uncertainty: 0.0   (short factual sentence extraction)
    Reasoner   uncertainty: 0.667 (CoT phrasing variance — expected baseline)
    Writer     uncertainty: 0.0   (short final answers)
  Calibrated value: 0.75  (just above Reasoner's 0.667 baseline)

TODO (Study 2 upgrade): replace global UNCERTAINTY_THRESHOLD with per-node
  threshold dict: {"retriever": 0.30, "reasoner": 0.75, "writer": 0.30}
  Retriever and Writer baselines are ~0.0, so a lower per-node threshold would
  detect more real faults while keeping false positives low.  Implement after
  collecting enough clean-baseline samples to estimate per-node std accurately.
"""

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Backend: mlx_lm (Apple Silicon / Metal) — replaces Ollama since the local
# model is mlx-community/Qwen3-8B-4bit (MLX safetensors, not GGUF).
MODEL = "mlx-community/Qwen3-8B-4bit"

# ---------------------------------------------------------------------------
# Self-consistency sampling
# ---------------------------------------------------------------------------
DEFAULT_K = 3              # samples per node — bump to 5 for cloud-scale Study 1
DEFAULT_TEMPERATURE = 0.7  # standard inference temperature
NOISE_TEMPERATURE = 1.2    # elevated temperature used for noise fault injection
MAX_TOKENS = 128           # max tokens for Retriever / Writer
MAX_TOKENS_REASONER = 200  # extra headroom for Reasoner chain-of-thought

# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------
UNCERTAINTY_THRESHOLD = 0.75
DEFAULT_RETRIES = 2

# ---------------------------------------------------------------------------
# Fault injection taxonomy (Phase 3)
# ---------------------------------------------------------------------------
# The full Study 2+ fault grid is (fault_type × target_node).
# For the current chain topology, target nodes are retriever, reasoner, writer.
# Use build_fault_conditions() to get the complete 3×3 grid + clean control.
FAULT_TYPES = ["noise", "contamination", "ceiling"]
TARGET_NODES = ["retriever", "reasoner", "writer"]
# NOTE: When non-chain topologies are introduced (dataset-generation phase),
# replace TARGET_NODES with topo.nodes.keys() at the call site.


def build_fault_conditions():
    """Return the full fault grid plus a clean control.

    Returns a list of dicts (or None for the clean control) with shape:
      None                                          — clean / no-fault control
      {"type": fault_type, "target_node": node_id} — one specific fault condition

    The clean control is always first so per-question baselines can be
    established before running any fault trials.
    """
    conditions = [None]  # clean control, always first
    for ftype in FAULT_TYPES:
        for node in TARGET_NODES:
            conditions.append({"type": ftype, "target_node": node})
    return conditions


# ---------------------------------------------------------------------------
# Paths  (relative to the project root: btp-pipeline/)
# ---------------------------------------------------------------------------
DATA_DIR = "./data/hotpotqa_distractor"
LOG_DIR = "./logs"
RESULTS_DIR = "./results"
DEFAULT_LOG_PATH = f"{LOG_DIR}/trials.jsonl"
DEFAULT_SKIP_LOG_PATH = f"{LOG_DIR}/skipped_trials.jsonl"
DEFAULT_CONFUSION_MATRIX_PATH = f"{RESULTS_DIR}/confusion_matrix.csv"
