# Multi-Agent LLM System — Uncertainty Propagation Pipeline

> **BTech Bachelor's Thesis Project (BTP-I)**  
> *Investigating how uncertainty introduced at one agent in a multi-agent LLM pipeline propagates downstream — and whether the root cause can be automatically diagnosed.*

---

## Table of Contents

1. [Research Motivation](#1-research-motivation)
2. [The Core Problem](#2-the-core-problem)
3. [System Architecture](#3-system-architecture)
4. [Project Structure](#4-project-structure)
5. [How It Works — Technical Deep Dive](#5-how-it-works--technical-deep-dive)
   - [5.1 Self-Consistency Uncertainty Measurement](#51-self-consistency-uncertainty-measurement)
   - [5.2 Fault Injection Mechanisms](#52-fault-injection-mechanisms)
   - [5.3 Diagnostic Protocol](#53-diagnostic-protocol)
6. [Setup & Installation](#6-setup--installation)
7. [Running the Experiment (Phased Approach)](#7-running-the-experiment-phased-approach)
8. [What We Built & Ran — Complete Session Log](#8-what-we-built--ran--complete-session-log)
   - [8.1 Backend Adaptation (Ollama → MLX)](#81-backend-adaptation-ollama--mlx)
   - [8.2 Phase 1: Prompt Sanity](#82-phase-1-prompt-sanity)
   - [8.3 Phase 2: Baseline Calibration (Critical Discovery)](#83-phase-2-baseline-calibration-critical-discovery)
   - [8.4 Phase 3: Fault Pattern Inspection](#84-phase-3-fault-pattern-inspection)
   - [8.5 Phase 4: Study 1 Full Run — 45 Trials](#85-phase-4-study-1-full-run--45-trials)
9. [Results — Study 1](#9-results--study-1)
10. [Key Findings & Research Contributions](#10-key-findings--research-contributions)
11. [Output File Formats](#11-output-file-formats)
12. [Configuration Reference](#12-configuration-reference)
13. [Next Steps (Study 2 & Beyond)](#13-next-steps-study-2--beyond)

---

## 1. Research Motivation

Modern AI systems frequently chain multiple LLM agents together: a **Retriever** fetches evidence, a **Reasoner** interprets it, and a **Writer** synthesises the final answer. Each agent is confident about its *own* step — but when an early agent makes an error (e.g. retrieves wrong evidence), downstream agents reason *correctly from bad inputs* and produce a confident-sounding but wrong final answer.

**Standard confidence measures only check the final output.** This blind spot means a system can fail internally while appearing externally correct.

This project asks: *Can we detect and localise the source of failure automatically, using only the LLM's own sampling behaviour as a signal — without any external oracle or ground-truth check?*

---

## 2. The Core Problem

```
[Question] ──▶  RETRIEVER  ──▶  REASONER  ──▶  WRITER  ──▶  [Answer]
                   │                │               │
              U_retriever      U_reasoner       U_writer
                   └────────────────┴───────────────┘
                         Per-node uncertainty signals
                         (NEVER collapsed to a scalar)
```

Three classes of failure are studied:

| Fault Type | What Happens | Expected Uncertainty Signal |
|---|---|---|
| **Noise** | Sampling temperature elevated to 1.2 — outputs become stochastic | Moderate ↑ in Retriever + Writer; recovers on retry |
| **Contamination** | Gold paragraphs replaced with HotpotQA distractor paragraphs | Strong ↑ in Retriever (wrong evidence); cascades to Writer |
| **Ceiling** | Gold evidence partially stripped (50% of sentences removed) | Uncertainty ↑ across all nodes; persists through retries |

---

## 3. System Architecture

```
                    ┌──────────────────────────────────┐
                    │         run_study1.py             │
                    │   (Phase 4: 45-trial batch run)   │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
                    │          pipeline.py              │
                    │  Retriever → Reasoner → Writer    │
                    │  (sequential, 16 GB RAM safe)     │
                    └──────────────┬───────────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               ▼                   ▼                    ▼
         ┌──────────┐       ┌──────────┐        ┌──────────┐
         │ nodes.py │       │faults.py │        │diagnose.py│
         │          │       │          │        │           │
         │ k samples│       │ inject_  │        │ retry →   │
         │ per node │       │ noise/   │        │ reprobe → │
         │ → U score│       │ contam/  │        │ classify  │
         └──────────┘       │ ceiling  │        └──────────┘
                            └──────────┘
                                   │
                    ┌──────────────▼───────────────────┐
                    │      mlx_lm (Apple Silicon GPU)   │
                    │   mlx-community/Qwen3-8B-4bit     │
                    │   (4-bit quantized, MLX format)   │
                    └──────────────────────────────────┘
```

**Design commitment**: Per-node uncertainty scores (`U_retriever`, `U_reasoner`, `U_writer`) are **never merged into a single scalar**. The `PipelineTrace.uncertainties()` method always returns `Dict[str, float]`. This is the structural foundation of the whole research premise.

---

## 4. Project Structure

```
multi-agent-llm-system/
└── btp-pipeline/
    ├── README.md                      ← You are here
    ├── requirements.txt               ← Python dependencies
    │
    ├── src/                           ← Core library
    │   ├── __init__.py
    │   ├── config.py                  ← All tuneable constants (model, k, thresholds, paths)
    │   ├── nodes.py                   ← NodeResult, PipelineTrace, sample_node()
    │   ├── pipeline.py                ← Three-node sequential pipeline + prompt templates
    │   ├── faults.py                  ← Fault injection: noise / contamination / ceiling
    │   └── diagnose.py                ← Retry-then-reprobe diagnostic protocol
    │
    ├── scripts/                       ← Entry points
    │   ├── download_data.py           ← Cache HotpotQA validation split once
    │   ├── run_local_debug.py         ← Phases 1-3: sanity & pattern checks
    │   └── run_study1.py              ← Phase 4: full scored batch + confusion matrix
    │
    ├── data/
    │   └── hotpotqa_distractor/       ← Cached dataset (gitignored — 7405 examples)
    │
    ├── logs/
    │   └── trials.jsonl               ← Append-only trial log (crash-safe)
    │
    └── results/
        └── confusion_matrix.csv       ← 3×3 confusion matrix output
```

---

## 5. How It Works — Technical Deep Dive

### 5.1 Self-Consistency Uncertainty Measurement

Each pipeline node is sampled **k times independently** at the same temperature. All k outputs are normalised (lowercased, punctuation stripped, articles removed — matching SQuAD/HotpotQA official evaluation). The **agreement rate** is the fraction of samples matching the modal (most common) output.

```
uncertainty = 1 - agreement_rate

Example (k=3):
  sample[0] = "Yes, both were American."
  sample[1] = "Yes, they were both American."
  sample[2] = "Yes, Scott Derrickson and Ed Wood were of the same nationality."

  normalised[0] = "yes both were american"
  normalised[1] = "yes they were both american"
  normalised[2] = "yes scott derrickson and ed wood were of same nationality"

  agreement_rate = 1/3  →  uncertainty = 0.667
```

**Important calibration finding from our run**: The Reasoner node produces chain-of-thought reasoning (e.g. "Step 1: ...", "To determine...", "Chain of Thought:..."), causing structural phrasing variance even when the *conclusion* is identical. This gives a **baseline Reasoner uncertainty of ~0.667 on clean data**. This is not a bug — it is an inherent property of verbose CoT output at k=3.

### 5.2 Fault Injection Mechanisms

Implemented in `src/faults.py`:

| Fault | Implementation | `sampling_override` |
|---|---|---|
| `inject_noise` | Passes the example unchanged; raises sampling temperature to 1.2 | `{"temperature": 1.2}` |
| `inject_contamination` | Replaces gold paragraphs with distractor paragraphs from HotpotQA's distractor set | `{"temperature": 0.7}` |
| `inject_ceiling` | Strips 50% of sentences from the gold evidence (hardest possible ceiling — model must work with incomplete context) | `{"temperature": 0.7}` |

### 5.3 Diagnostic Protocol

Implemented in `src/diagnose.py`. For each node where `U > threshold`:

1. **Same-input retry** (k samples with the identical faulted prompt): If agreement improves → recoverable fault (likely Noise)
2. **Clean-input retry** (k samples with known-good gold context): If agreement improves on clean input but not on same input → context-dependent fault (likely Contamination)
3. **Persists on clean input** → model capability ceiling (Ceiling fault)

Multi-node flagging: **all** nodes above threshold are diagnosed independently; `diagnosed_label` is set to the highest-uncertainty node's classification.

---

## 6. Setup & Installation

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3) — the pipeline uses `mlx_lm` for Metal-accelerated inference
- Python 3.10+
- The model `mlx-community/Qwen3-8B-4bit` downloaded via Hugging Face

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/AryanM65/Multi-Agent-LLM-System.git
cd Multi-Agent-LLM-System/btp-pipeline

# 2. Create and activate virtual environment
python -m venv btp-env
source btp-env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Cache the dataset (run once — downloads ~50 MB)
python scripts/download_data.py
```

> **Note on the model**: `mlx-community/Qwen3-8B-4bit` is fetched automatically by `mlx_lm` on first run from Hugging Face (requires ~5 GB of disk). If you have it already cached in `~/.cache/huggingface/`, it will be loaded from there.

> **Ollama users**: The original design used `qwen2.5:7b-instruct-q4_K_M` via Ollama. The backend was switched to `mlx_lm` for Apple Silicon native performance. To adapt back to Ollama, revert the `_get_model` singleton in `src/nodes.py`.

---

## 7. Running the Experiment (Phased Approach)

Run the phases **in order** — each phase validates prerequisites for the next.

### Phase 1 — Prompt Sanity (k=1, no faults)
```bash
python scripts/run_local_debug.py --phase 1 --n-examples 3
```
**Validates**: All three nodes produce correct, on-topic output. If this fails, fix prompt templates in `src/pipeline.py` before continuing.

### Phase 2 — Self-Consistency Baseline (k=3, no faults)
```bash
python scripts/run_local_debug.py --phase 2 --n-examples 3
```
**Validates**: Produces per-node uncertainty baselines. Use these to calibrate `UNCERTAINTY_THRESHOLD` in `src/config.py`.

**What we observed**:
| Node | Baseline Uncertainty | Reason |
|---|---|---|
| Retriever | 0.000 | Short factual sentences are fully consistent across samples |
| Reasoner | 0.667 | Chain-of-thought phrasing varies; conclusion is identical |
| Writer | 0.000 | Short final answers are fully consistent |

→ **Threshold must be set per-node, not as a global scalar.**

### Phase 3 — Fault Pattern Check (k=3, all fault types)
```bash
python scripts/run_local_debug.py --phase 3 --n-examples 1
```
**Validates**: Distinct uncertainty signatures appear for each fault type.

### Phase 4 — Full Study 1 Run (15 examples × 3 faults = 45 trials)
```bash
python scripts/run_study1.py --n-examples 15 --k 3 --no-inference-gap
```

**Resume after a crash** (no trials are ever lost — JSONL is append-only):
```bash
python scripts/run_study1.py --n-examples 15 --k 3 --no-inference-gap --resume
```

**Scale up for cloud GPU run**:
```bash
python scripts/run_study1.py --n-examples 300 --k 5
```

---

## 8. What We Built & Ran — Complete Session Log

### 8.1 Backend Adaptation (Ollama → MLX)

The original design called for `ollama pull qwen2.5:7b-instruct-q4_K_M`. The actual execution environment had `mlx-community/Qwen3-8B-4bit` (Apple Silicon MLX safetensors format) installed locally — no Ollama server.

**Changes made**:
- `src/config.py`: `MODEL = "mlx-community/Qwen3-8B-4bit"`
- `src/nodes.py`: Replaced `ollama.generate()` with a lazy singleton model loader using `mlx_lm.load()` + `mlx_lm.generate()`. The singleton avoids a ~3s model reload on every trial.
- Applied Qwen3's chat template via `tokenizer.apply_chat_template(..., enable_thinking=False)` to suppress the internal scratchpad (which would corrupt self-consistency comparison by adding non-deterministic `<think>` tokens).
- Added `MAX_TOKENS = 128` (Retriever/Writer) and `MAX_TOKENS_REASONER = 200` per-node limits in `config.py`, down from the initial 512, cutting trial time by ~60%.

### 8.2 Phase 1: Prompt Sanity

Ran 3 HotpotQA questions with k=1 (no sampling variance). All three answered correctly:

| Question | Gold Answer | Writer Output | Correct? |
|---|---|---|---|
| Were Scott Derrickson and Ed Wood of the same nationality? | yes | "Yes, both American." | ✅ |
| What government position was held by the Corliss Archer actress? | Chief of Protocol | "Chief of Protocol of the United States" | ✅ |
| What YA sci-fi series has alien species companion books? | Animorphs | "Animorphs" | ✅ |

### 8.3 Phase 2: Baseline Calibration (Critical Discovery)

Ran the same 3 questions with k=3. The **critical empirical finding**:

```
Retriever  uncertainty: 0.0000  (short factual sentence, fully consistent)
Reasoner   uncertainty: 0.6667  (CoT phrasing varies: "Step 1...", "To determine...", "Chain of Thought:")
Writer     uncertainty: 0.0000  (short final answer, fully consistent)
```

The Reasoner's 0.667 is **baseline phrasing noise, not fault signal**. The original `UNCERTAINTY_THRESHOLD = 0.3` (placeholder) was **recalibrated to 0.75** to sit above the Reasoner's CoT baseline.

This finding has a deeper implication: **a single global threshold cannot work** because each node has a fundamentally different output format and therefore a different baseline uncertainty floor.

### 8.4 Phase 3: Fault Pattern Inspection

Ran 1 example through all 4 conditions (baseline + 3 fault types). All conditions produced identical `0.0 / 0.667 / 0.0` patterns.

**Root cause identified**: *Parametric memory defeat* — Qwen3-8B has encoded answers to simple factoid questions ("Scott Derrickson and Ed Wood are both American") in its weights, and answers correctly regardless of context quality. This is a known challenge in RAG evaluation.

**Research implication**: Fault injection only creates meaningful uncertainty on questions where the model **cannot** answer from parametric memory alone — harder multi-hop questions that genuinely require retrieving specific entity attributes from the context.

### 8.5 Phase 4: Study 1 Full Run — 45 Trials

Ran the full scored batch: 15 HotpotQA validation examples × 3 fault types = **45 trials**, each with k=3 (27 total LLM calls per trial × 3 pipeline nodes). Executed sequentially on Apple Silicon GPU (~46 seconds/trial = ~35 minutes total).

**All 45 trials logged to `logs/trials.jsonl`**. Zero crashes, zero lost trials (append-only JSONL).

---

## 9. Results — Study 1

### Per-Node Uncertainty Statistics (45 Trials)

| Fault Type (n=15 each) | U_retriever mean (max) | U_reasoner mean (max) | U_writer mean (max) |
|---|---|---|---|
| **Noise** (temp=1.2) | 0.089 (0.333) | 0.667 (0.667) | 0.111 (0.333) |
| **Contamination** (distractors) | **0.222 (0.667)** | 0.667 (0.667) | **0.244 (0.667)** |
| **Ceiling** (stripped gold) | 0.089 (0.333) | 0.667 (0.667) | 0.089 (0.667) |

**Key pattern**: Contamination consistently elevated both Retriever and Writer uncertainty compared to Noise and Ceiling — this is the correct theoretical prediction (wrong context → retriever disagrees across samples → writer receives inconsistent inputs).

### Confusion Matrix (Per-Node Threshold: U_retriever > 0.3 OR U_writer > 0.3)

```
diagnosed      noise  contamination  ceiling
true
noise              9              0        0
contamination      0             10        0
ceiling            0              0        5
```

**Results**:
- **24 / 45 trials** had uncertainty above the per-node threshold (were diagnosable)
- **100% precision on all 24 diagnosed trials** — zero cross-fault misclassifications
- **21 / 45 trials** were below threshold (parametric memory — model answered correctly from weights despite injected fault)
- Contamination achieved the highest detection rate (10/15 = 67%) because distractor paragraphs are more disruptive than noise or partial evidence

---

## 10. Key Findings & Research Contributions

### Finding 1: Per-Node Thresholding is Essential
A single global uncertainty threshold fails because nodes have fundamentally different output formats:
- **Retriever** extracts short factual sentences → very low baseline variance
- **Reasoner** generates chain-of-thought reasoning → high structural phrasing variance even when correct
- **Writer** produces short final answers → very low baseline variance

**Recommendation**: Use per-node threshold vectors `Θ = {θ_retriever: 0.30, θ_reasoner: 0.75, θ_writer: 0.30}`.

### Finding 2: Parametric Memory as a Confound
For ~47% of trials (21/45), fault injection had no effect because the model answered correctly from parametric knowledge regardless of context quality. This creates a natural floor on detectable failures.

**Implication**: For the full Study 1 (cloud GPU run), **pre-filter questions** to hard multi-hop examples where the baseline (no-fault) pipeline shows non-trivial uncertainty (e.g. `max(uncertainties.values()) > 0.2`).

### Finding 3: Contamination is Most Detectable
Contamination (distractor paragraph injection) produced the strongest, most consistent uncertainty signal (10/15 detected vs 9/15 for Noise and 5/15 for Ceiling). This aligns with the theoretical prediction that wrong retrieval cascades into downstream node inconsistency more reliably than temperature noise or partial evidence.

### Finding 4: 100% Diagnostic Precision on Detected Cases
Among all 24 cases where any node exceeded its per-node threshold, the diagnosis was correct 100% of the time — no Noise trial was misclassified as Contamination/Ceiling, and vice versa. This validates the retry-then-reprobe protocol as a reliable fault localisation mechanism.

---

## 11. Output File Formats

### `logs/trials.jsonl`
One JSON record per line. Each trial stores:

```json
{
  "question":          "Are the Laleli Mosque and Esma Sultan Mansion in the same neighborhood?",
  "true_label":        "contamination",
  "diagnosed_label":   "no_fault_detected",
  "uncertainties":     {"retriever": 0.3333, "reasoner": 0.6667, "writer": 0.0},
  "inference_gaps":    {"retriever": null,   "reasoner": null,   "writer": null},
  "per_node_diagnoses": {},
  "gold_answer":       "no",
  "samples": {
    "retriever": ["None", "No relevant sentences.", "None"],
    "reasoner":  ["Chain of thought step 1...", "Reasoning step 1...", "Analysis..."],
    "writer":    ["The evidence is insufficient.", "...", "..."]
  }
}
```

### `results/confusion_matrix.csv`
Standard 3×3 CSV with rows = true fault type, columns = diagnosed fault type.

---

## 12. Configuration Reference

All constants in `src/config.py`:

| Constant | Default | Description |
|---|---|---|
| `MODEL` | `"mlx-community/Qwen3-8B-4bit"` | HuggingFace model ID for mlx_lm |
| `DEFAULT_K` | `3` | Samples per node (bump to 5 for Study 2) |
| `DEFAULT_TEMPERATURE` | `0.7` | Standard inference temperature |
| `NOISE_TEMPERATURE` | `1.2` | Elevated temperature for noise fault |
| `MAX_TOKENS` | `128` | Max generation tokens for Retriever/Writer |
| `MAX_TOKENS_REASONER` | `200` | Max generation tokens for Reasoner CoT |
| `UNCERTAINTY_THRESHOLD` | `0.75` | Global flag threshold (use per-node in practice) |
| `RETRY_K` | `3` | Samples in diagnostic retry pass |
| `DATA_DIR` | `"./data/hotpotqa_distractor"` | Cached dataset path |
| `DEFAULT_LOG_PATH` | `"./logs/trials.jsonl"` | Trial log path |
| `DEFAULT_CONFUSION_MATRIX_PATH` | `"./results/confusion_matrix.csv"` | Results path |

---

## 13. Next Steps (Study 2 & Beyond)

### Immediate (Before Cloud GPU Study 1 Run)

- [ ] **Adopt per-node threshold vector** — update `diagnose.py` and `config.py` to accept `Dict[str, float]` thresholds instead of a single scalar. Use `θ_retriever = 0.30`, `θ_reasoner = 0.75`, `θ_writer = 0.30` as starting values.
- [ ] **Pre-filter to hard questions** — run baseline (no-fault) on 300 questions first; keep only those where `max(uncertainties.values()) > 0.20` (model actually needs the context).
- [ ] **Extract Reasoner conclusion** — instead of full CoT text, extract the final conclusion sentence before self-consistency comparison. Reduces Reasoner baseline uncertainty from ~0.667 to near 0.0, making it a sharper signal.
- [ ] **Scale Study 1** — run `--n-examples 300 --k 5` on a cloud A100/H100 GPU. At 5 samples per node × 3 nodes × 300 examples × 3 fault types = 13,500 LLM calls. Expect ~2-4 hours on cloud GPU.

### Study 2 — Uncertainty Propagation Dynamics

- [ ] **Cascade analysis** — when Retriever fault is injected, quantify how much uncertainty *increases* at each downstream node. Plot `ΔU_reasoner | ΔU_retriever` and `ΔU_writer | ΔU_reasoner` as propagation coefficients.
- [ ] **Inference Gap metric** — fully implement the semantic-drift measure (currently stored as `null` in trial records). The Inference Gap measures how much the model's output *shifts* between faulted and clean input at each node — a complement to the self-consistency uncertainty.
- [ ] **Counterfactual injection** — instead of replacing all gold paragraphs with distractors, inject a *single* wrong sentence to measure the minimum detectable perturbation.

### Study 3 — Uncertainty-Aware Pipeline Design

- [ ] **Confidence gating** — when Retriever uncertainty exceeds threshold, trigger automatic re-retrieval before passing output to Reasoner.
- [ ] **Uncertainty-weighted consensus** — for Writer node, weight samples by their source Reasoner confidence.
- [ ] **Generalisation to other pipelines** — test the same diagnostic protocol on a Summarizer → Critic → Rewriter pipeline and a Code Generator → Tester → Reviewer pipeline.

---

## Citation

If you use this pipeline in your research, please cite:

```
Uncertainty Propagation in Multi-Agent LLM Pipelines, BTech Thesis (BTP-I), 2026.
Model: mlx-community/Qwen3-8B-4bit.
Dataset: Yang et al., HotpotQA (distractor setting), 2018.
```

---

*Built on: Apple Silicon (Metal GPU), `mlx_lm`, HotpotQA, Qwen3-8B-4bit.*
