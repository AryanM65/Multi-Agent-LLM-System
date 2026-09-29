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
# Backend selection
# ---------------------------------------------------------------------------
# BTP_BACKEND controls which LLM backend is used:
#   'ollama' (default) — local/cloud Ollama server, works on Windows/Linux/macOS CPU & GPU
#   'mlx'             — Apple Silicon MLX (macOS only, fastest on M-series chips)
#   'vllm'            — vLLM, GPU-only (Kaggle T4 etc.). See VLLM_MODEL below.
# BTP_MOCK=1 overrides this entirely (no LLM at all).
BACKEND: str = os.getenv("BTP_BACKEND", "ollama")

# ---------------------------------------------------------------------------
# Ollama model selection
# ---------------------------------------------------------------------------
# The Ollama model to use when BACKEND='ollama'.
# Override with BTP_OLLAMA_MODEL=<model_name> in the environment.
# Default: qwen3:8b (matches the MLX model for fair comparison).
OLLAMA_MODEL: str = os.getenv("BTP_OLLAMA_MODEL", "gpt-oss:20b-cloud")

# ---------------------------------------------------------------------------
# vLLM model selection
# ---------------------------------------------------------------------------
# The model to serve via vLLM when BACKEND='vllm'. Chosen for dataset
# generation (see plan.md Section 0): standard instruction-tuned model, no
# hidden reasoning pass (unlike gpt-oss), AWQ-quantized to leave real KV-cache
# headroom on a 16GB T4 (fp16 weights alone would be ~15.2GB, starving batching).
VLLM_MODEL: str = os.getenv("BTP_VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
VLLM_QUANTIZATION: str = os.getenv("BTP_VLLM_QUANTIZATION", "awq")
VLLM_DTYPE: str = os.getenv("BTP_VLLM_DTYPE", "float16")  # T4 (Turing) lacks native bf16 tensor cores
VLLM_GPU_MEMORY_UTILIZATION: float = float(os.getenv("BTP_VLLM_GPU_MEM_UTIL", "0.85"))
VLLM_MAX_MODEL_LEN: int = int(os.getenv("BTP_VLLM_MAX_MODEL_LEN", "4096"))

# ---------------------------------------------------------------------------
# Self-consistency sampling
# ---------------------------------------------------------------------------
DEFAULT_K = 5              # samples per node (cloud-scale Study 2; was 3 pre-Study-2)
DEFAULT_TEMPERATURE = 0.7  # standard inference temperature
# Raised from 1.2 -> 1.8 (2026-09-29): at 1.2 the noise fault only moved mean
# semantic uncertainty +0.062 vs 253 clean controls (real per-node std 0.165,
# z~0.37 -- weak, barely above measurement noise). 1.2 is a mild nudge above
# the 0.7 baseline; 1.8 is near vLLM's practical ceiling before output
# degenerates into incoherence. See model/data/enrich_discrepancy.py docstring
# and docs/model/model.md for the full measurement.
NOISE_TEMPERATURE = 1.8    # elevated temperature used for noise fault injection

# gpt-oss:20b-cloud reasons by default and returns that reasoning in a separate
# `message["thinking"]` field (see src/nodes.py:_ollama_generate). num_predict
# must cover BOTH the hidden reasoning pass and the final answer, or content
# comes back empty. Measured via scripts/debug_gptoss_thinking.py: reasoning
# alone commonly ran 400-1050 chars (~150-260 tokens); a 128-token budget left
# content="" in 5/5 isolated test calls. 300 tokens reliably produced non-empty
# Retriever content in follow-up testing; these values add margin above that.
#
# CONFIRMED SUFFICIENT FOR QWEN2.5-7B-INSTRUCT-AWQ (real calibration run via
# scripts/calibrate_vllm_model.py on Kaggle GPU, 3 questions, k=5, 45 total
# samples, 2026-09-24): 0/45 empty samples across all roles -- Qwen2.5 has no
# hidden-reasoning-pass problem at all (unlike gpt-oss), so these gpt-oss-era
# budgets, sized generously for that different problem, simply carry a lot of
# unused headroom for Qwen2.5 rather than being wrong. Observed real usage:
# retriever max=94 words (~120 tokens), reasoner max=226 words (~300 tokens),
# writer max=31 words. Could be tightened for efficiency but there is no
# correctness reason to; left as-is. FINAL ANSWER: marker compliance was
# 15/15 (100%). Re-run scripts/calibrate_vllm_model.py with more questions
# before trusting these as final if generation starts hitting longer/harder
# questions than the 3-question calibration sample covered.
MAX_TOKENS = 350           # max tokens for Retriever / Writer
MAX_TOKENS_REASONER = 600  # extra headroom for Reasoner chain-of-thought

# ---------------------------------------------------------------------------
# Diagnosis -- global + per-node thresholds
# ---------------------------------------------------------------------------
# IMPORTANT: These thresholds apply to SEMANTIC uncertainty (0.0-1.0 continuous range).
# The old lexical uncertainty at k=3 is mathematically capped at 0.667 (all 3 samples
# disagree), so a threshold of 0.75 is UNREACHABLE and would cause 0% detection rate.
# All threshold calibration must be done against semantic uncertainty values.

UNCERTAINTY_THRESHOLD = 0.50  # semantic fallback: flag if semantic uncertainty > 0.50

# Per-node semantic uncertainty thresholds.
# Calibrated from rescore_study1 data (semantic means on fault trials, gpt-oss):
#   retriever:  clean baseline ~0.10-0.15 | fault mean ~0.17-0.33
#   reasoner:   clean baseline ~0.10-0.20 | fault mean ~0.92-1.00
#   writer:     clean baseline ~0.05-0.10 | fault mean ~0.14-0.35
# Thresholds set at 2x the expected clean baseline to minimize false positives.
#
# STILL GPT-OSS-ERA VALUES. Not needed for dataset generation itself (the
# retry-then-reprobe diagnostic protocol these gate is disabled via
# --no-diagnose for the bulk run), so left unchanged rather than guessed at
# from a tiny sample. A preliminary Qwen2.5 clean-baseline reading exists
# (3 questions, 2026-09-24, via scripts/calibrate_vllm_model.py): reasoner
# semantic mean=0.47 std=0.41 (swung 0.0 -> 1.0 -> 0.42 across the 3
# questions -- high per-question variance, not a stable number yet), writer
# semantic mean=0.14 std=0.20. Both far too small a sample (n=3) to treat as
# a real calibration -- re-run with more questions (~15-20) before ever
# trusting NODE_THRESHOLDS for Qwen2.5 if the retry-then-reprobe protocol is
# evaluated later as the naive-baseline comparator (see plan.md / research
# design). Retriever has no semantic value by design (uses lexical/Jaccard
# instead) -- "no semantic values recorded" in a calibration run is expected,
# not a gap.
NODE_THRESHOLDS: dict = {
    "retriever":   0.25,   # semantic: clean ~0.10, faulted ~0.17-0.33
    "reasoner":    0.50,   # semantic: clean ~0.15, faulted ~0.92-1.00
    "writer":      0.25,   # semantic: clean ~0.07, faulted ~0.14-0.35
    # Generic fallback for non-standard node IDs (used in multi-retriever topologies)
    "_default":    0.50,
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
