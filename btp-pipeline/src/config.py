"""
config.py — Single source of truth for all tuneable constants.

IMPORTANT: UNCERTAINTY_THRESHOLD is currently a placeholder (0.3).
It MUST be recalibrated from actual Phase 1-2 output distributions
before trusting the confusion matrix numbers.  Typical calibration
procedure: collect uncertainties from the no-fault baseline (Phase 1)
and set the threshold at mean + 1σ, or at the 75th-percentile value.
"""

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Backend: mlx_lm (Apple Silicon / Metal) — replaces Ollama since the local
# model is mlx-community/Qwen3-8B-4bit (MLX safetensors, not GGUF).
# The Qwen3-8B-4bit is slightly larger than the originally spec'd Qwen2.5-7B
# but is the available local quantised model; all pipeline logic is identical.
MODEL = "mlx-community/Qwen3-8B-4bit"

# ---------------------------------------------------------------------------
# Self-consistency sampling
# ---------------------------------------------------------------------------
DEFAULT_K = 3              # samples per node — bump to 5 for cloud-scale Study 1
DEFAULT_TEMPERATURE = 0.7  # standard inference temperature
NOISE_TEMPERATURE = 1.2    # elevated temperature used by inject_noise
MAX_TOKENS = 128           # max tokens per generation — 128 is enough for QA answers
                           # Reasoner gets a bit more (chain-of-thought needs space)
MAX_TOKENS_REASONER = 200  # slightly more headroom for step-by-step reasoning


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------
# CALIBRATED from Phase 2 baseline (3 questions, k=3, clean gold context):
#   Retriever  uncertainty: 0.0   (short factual sentence extraction — fully consistent)
#   Reasoner   uncertainty: 0.667 (chain-of-thought phrasing varies across samples
#                                   even when the conclusion is identical — this is
#                                   expected for free-text reasoning output at k=3)
#   Writer     uncertainty: 0.0   (short final answers — fully consistent)
#
# Implication: threshold must be > 0.667 to avoid false positives on clean Reasoner
# output.  Setting to 0.75 — just above the baseline — so only *worse-than-baseline*
# inconsistency triggers diagnosis.  Recalibrate again after collecting Phase 3
# fault-injected distributions to confirm fault types push beyond this ceiling.
#
# For the cloud-scale Study 1 run with a larger model, recalibrate from that
# model's own baseline before comparing results.
UNCERTAINTY_THRESHOLD = 0.75

# Number of same-input retries in diagnose().
# Validate that local ceiling faults don't "fake recover" with 2 retries;
# raise to 3 if they do.
DEFAULT_RETRIES = 2

# ---------------------------------------------------------------------------
# Paths  (relative to the project root: btp-pipeline/)
# ---------------------------------------------------------------------------
DATA_DIR = "./data/hotpotqa_distractor"
LOG_DIR = "./logs"
RESULTS_DIR = "./results"
DEFAULT_LOG_PATH = f"{LOG_DIR}/trials.jsonl"
DEFAULT_CONFUSION_MATRIX_PATH = f"{RESULTS_DIR}/confusion_matrix.csv"
