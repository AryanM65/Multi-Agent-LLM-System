# Multi-Agent LLM System — Uncertainty Propagation Pipeline

> **BTech Bachelor's Thesis Project (BTP-I)**
> *Investigating how uncertainty introduced at one agent in a multi-agent LLM pipeline propagates downstream — and whether the root cause can be automatically diagnosed, using topology-aware signals as input to a GNN.*

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
   - [5.4 Topology Engine](#54-topology-engine)
6. [Setup & Installation](#6-setup--installation)
7. [Running the Experiment](#7-running-the-experiment)
8. [Dataset Generation Phase — Complete](#8-dataset-generation-phase--complete)
   - [8.1 Backend Migration: MLX/Ollama → vLLM on Kaggle GPU](#81-backend-migration-mlxollama--vllm-on-kaggle-gpu)
   - [8.2 Topology Pool & Question Source](#82-topology-pool--question-source)
   - [8.3 Bugs Found and Fixed During Data-Quality Audit](#83-bugs-found-and-fixed-during-data-quality-audit)
   - [8.4 Final Dataset Stats](#84-final-dataset-stats)
   - [8.5 Kaggle Workflow Notes](#85-kaggle-workflow-notes)
9. [Earlier Results — Study 1 (Local, 45 Trials)](#9-earlier-results--study-1-local-45-trials)
10. [Key Findings & Research Contributions](#10-key-findings--research-contributions)
11. [Output File Formats](#11-output-file-formats)
12. [Configuration Reference](#12-configuration-reference)
13. [Next Steps (GNN Training & Beyond)](#13-next-steps-gnn-training--beyond)

---

## 1. Research Motivation

Modern AI systems frequently chain multiple LLM agents together: a **Retriever** fetches evidence, a **Reasoner** interprets it, and a **Writer** synthesises the final answer. Each agent is confident about its *own* step — but when an early agent makes an error (e.g. retrieves wrong evidence), downstream agents reason *correctly from bad inputs* and produce a confident-sounding but wrong final answer.

**Standard confidence measures only check the final output.** This blind spot means a system can fail internally while appearing externally correct.

This project asks: *Can we detect and localise the source of failure automatically, using only the LLM's own sampling behaviour as a signal — and can a model trained on the pipeline's topology plus per-node uncertainty learn to identify both which node failed and why, without any external oracle?*

The project has two phases:
- **Phase 1 (complete)**: build the multi-agent pipeline, fault-injection framework, and self-consistency uncertainty measurement; validate the diagnostic protocol on a small local run.
- **Phase 2 (complete)**: scale to a labeled dataset across many topologies (chain, fan-out, fan-in, diamond, mixed DAGs) for training a topology-aware (GNN) fault classifier.

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

Real topologies are not always a simple 3-node chain — a pipeline may fan a Retriever out to multiple Reasoners, fan multiple Retrievers into one Reasoner, or form deeper mixed DAGs. The dataset generated in Phase 2 spans **20 distinct topologies** (14 seen during training, 6 held out for out-of-distribution testing) so a downstream model must learn a general, structure-aware notion of "where did this fail," not just memorise one fixed graph shape.

Three classes of fault are studied:

| Fault Type | What Happens | Expected Uncertainty Signal |
|---|---|---|
| **Noise** | Sampling temperature elevated to 1.2 — outputs become stochastic | Moderate ↑ at the targeted node; recovers on retry |
| **Contamination** | Gold/parent-node content replaced with a distractor from a substitute bank | Strong ↑ at the targeted node; cascades downstream |
| **Ceiling** | Evidence/instructions stripped or hardened at the targeted node — a genuine capability limit, not noise | Uncertainty ↑, persists through retries |

Each fault can target **any node** in the topology (not just the Retriever), and ground truth records both *which node* was targeted (`fault_config["target_node"]`) and *which fault type* (`true_label`).

---

## 3. System Architecture

```
                ┌────────────────────────────────────────┐
                │   scripts/run_study_vllm.py (current)   │
                │   GPU-batched generation across the     │
                │   full topology pool, on Kaggle T4       │
                └───────────────────┬──────────────────────┘
                                    │
                ┌───────────────────▼──────────────────────┐
                │              pipeline.py                  │
                │   Arbitrary DAG of Retriever/Reasoner/    │
                │   Writer nodes, topologically ordered      │
                └───────────────────┬──────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                          ▼
   ┌──────────────┐         ┌──────────────┐           ┌──────────────┐
   │   nodes.py    │         │  faults.py    │           │ diagnose.py   │
   │  k samples/   │         │ inject_noise/ │           │ retry →       │
   │  node → U     │         │ contamination/│           │ reprobe →     │
   │  score        │         │ ceiling       │           │ classify      │
   └──────────────┘         └──────────────┘           └──────────────┘
                                    │
                ┌───────────────────▼──────────────────────┐
                │                vLLM (batched)              │
                │      Qwen/Qwen2.5-7B-Instruct-AWQ          │
                │   (4-bit AWQ, float16, Kaggle T4 GPU)      │
                └────────────────────────────────────────┘
```

**Design commitment**: Per-node uncertainty scores are **never merged into a single scalar**. `PipelineTrace.uncertainties()` always returns `Dict[str, float]`, one entry per node in the topology. This is the structural foundation of the whole research premise, and it is exactly what makes a graph-structured (GNN) downstream model appropriate: each node in the topology graph carries its own feature vector.

**Backend history**: the pipeline originally ran on Ollama (`qwen2.5:7b-instruct-q4_K_M`), was adapted to Apple Silicon via `mlx_lm` (`mlx-community/Qwen3-8B-4bit`) for local development, and was finally migrated to **vLLM serving `Qwen/Qwen2.5-7B-Instruct-AWQ`** for the full dataset-generation run. The switch to vLLM was necessary for true cross-trial batching (`llm.chat()` over lists of conversations) and to Qwen2.5-AWQ because Kaggle's free T4 GPUs (compute capability 7.5) cannot run MXFP4-quantized models — that format requires Hopper/Ada-class hardware. AWQ 4-bit quantization fits the 7B model comfortably in the T4's 16GB VRAM with headroom for KV-cache.

---

## 4. Project Structure

```
multi-agent-llm-system/
├── README.md                          ← You are here
├── dataset/                           ← Phase 2 output: the labeled training dataset
│   ├── trials.jsonl                   ← 493 labeled trial records
│   ├── skipped.jsonl                  ← 15 skipped/invalid trial records (with reason)
│   ├── topology_pool.json             ← 20 topologies (14 train + 6 OOD), serialized DAGs
│   ├── dataset_description.md         ← Full schema doc: fields, input features, ground truth, worked example
│   └── generate_topology_pool.py      ← Regenerates topology_pool.json
├── dataset.zip                        ← Zipped copy of dataset/ for easy download
│
└── btp-pipeline/
    ├── requirements.txt               ← Python dependencies
    ├── plan.md                        ← Dataset-generation planning doc (Kaggle/vLLM decision history)
    ├── correct_project_context.md     ← Running project context/status log
    ├── HANDOFF_2026-09-25.md          ← Detailed session handoff: infra gotchas, bug writeups, timings
    │
    ├── src/                           ← Core library
    │   ├── config.py                  ← All tuneable constants (model, k, thresholds, backend, paths)
    │   ├── nodes.py                   ← NodeResult, PipelineTrace, sample_node(), vLLM batched generation
    │   ├── pipeline.py                ← Arbitrary-topology pipeline executor + prompt templates
    │   ├── topology.py                ← Topology / NodeSpec dataclasses, topological sort
    │   ├── faults.py                  ← Fault injection: noise / contamination / ceiling (role-aware)
    │   └── diagnose.py                ← Retry-then-reprobe diagnostic protocol
    │
    ├── scripts/                       ← Entry points
    │   ├── download_data.py           ← Cache HotpotQA validation split once
    │   ├── run_local_debug.py         ← Phases 1-3: sanity & pattern checks
    │   ├── run_study1.py              ← Chain-topology scored batch run (local backend)
    │   ├── run_study2.py              ← Multi-topology scored batch run
    │   ├── run_study_vllm.py          ← GPU-batched multi-topology generation (used for the dataset)
    │   ├── calibrate_vllm_model.py    ← vLLM backend calibration script
    │   ├── verify_results.py          ← Post-generation dataset sanity/statistics checker
    │   └── sample_examples.py         ← Builds the 30-question HotpotQA source pool
    │
    ├── data/
    │   ├── hotpotqa_distractor/       ← Cached dataset (gitignored)
    │   └── study30_examples.json      ← 30-question source pool used for dataset generation
    │
    ├── logs/
    │   └── trials.jsonl               ← Append-only trial log (crash-safe)
    │
    └── results/
        └── confusion_matrix.csv       ← 3×3 confusion matrix output (local Study 1)
```

---

## 5. How It Works — Technical Deep Dive

### 5.1 Self-Consistency Uncertainty Measurement

Each pipeline node is sampled **k times independently** at the same temperature (default `k=5`). All k outputs are normalised (lowercased, punctuation stripped, articles removed — matching SQuAD/HotpotQA official evaluation). Three complementary uncertainty metrics are computed:

```
lexical_uncertainty  = 1 - (fraction of samples matching the modal exact-match output)
semantic_uncertainty = entropy over embedding-cluster assignment of the k samples
jaccard_uncertainty  = 1 - mean pairwise Jaccard overlap (for multi-item Retriever outputs)
```

**Calibration finding**: the Reasoner node produces chain-of-thought reasoning, causing structural phrasing variance even when the *conclusion* is identical — this gives a non-zero baseline lexical uncertainty on clean data that is not itself a fault signal. Thresholds and the semantic/Jaccard metrics were added specifically to separate genuine fault signal from this baseline CoT noise.

### 5.2 Fault Injection Mechanisms

Implemented in `src/faults.py`, all now **role-aware** (dispatch on the node's role — `retriever`/`reasoner`/`writer` — rather than string-matching the node's name), so a fault can be injected at any node in any topology:

| Fault | Implementation |
|---|---|
| `inject_noise` | Passes input unchanged; raises sampling temperature to 1.2 at the targeted node |
| `inject_contamination` | Replaces the targeted node's input (gold paragraphs, or an upstream parent's output) with a distractor drawn from a substitute bank, excluding the clean value itself |
| `inject_ceiling` | Strips or hardens the targeted node's instructions/evidence, simulating a genuine capability ceiling rather than noise |

### 5.3 Diagnostic Protocol

Implemented in `src/diagnose.py`. For each node where `U > threshold`:

1. **Same-input retry** (k samples, identical faulted prompt): agreement improves → recoverable fault (likely Noise)
2. **Clean-input retry** (k samples, known-good input): improves on clean but not on faulted → context-dependent fault (likely Contamination)
3. **Persists on clean input** → model capability ceiling (Ceiling fault)

This diagnostic pass is optional during bulk dataset generation (`--no-diagnose` on `run_study_vllm.py`) since the dataset's ground truth already comes from the known injection, not from the diagnosis; diagnosis is a separate downstream evaluation of *how well the retry-based heuristic recovers that same ground truth*.

### 5.4 Topology Engine

`src/topology.py` defines `Topology` and `NodeSpec` dataclasses. Nodes are role-ordered (`ROLE_RANK`: retriever=0, reasoner=1, writer=2) and executed via Kahn's algorithm topological sort, so arbitrary DAGs — chains, fan-outs, fan-ins, diamonds, and mixed multi-layer graphs — all execute correctly and deterministically. The 20-topology pool used for dataset generation (`dataset/topology_pool.json`) was built by `dataset/generate_topology_pool.py` and split 14 train / 6 out-of-distribution.

---

## 6. Setup & Installation

### Prerequisites
- Python 3.10+
- For local/dev runs: Apple Silicon (`mlx_lm`) or Ollama, depending on backend
- For dataset-scale generation: a CUDA GPU with vLLM support (the dataset was generated on a Kaggle T4 via `run_study_vllm.py`)

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

### Backend selection

`src/config.py` reads `BACKEND` (`ollama` / `vllm` / `mlx`) to select the inference backend:

- `mlx` — Apple Silicon local dev, `mlx-community/Qwen3-8B-4bit`, no server needed.
- `ollama` — `qwen2.5:7b-instruct-q4_K_M` via a local Ollama server.
- `vllm` — `Qwen/Qwen2.5-7B-Instruct-AWQ`, AWQ 4-bit quantized, `float16` (T4 lacks native bf16). This is the backend used for all dataset-scale generation, and requires an NVIDIA GPU.

---

## 7. Running the Experiment

### Local sanity checks (any backend)
```bash
python scripts/run_local_debug.py --phase 1 --n-examples 3   # prompt sanity, k=1
python scripts/run_local_debug.py --phase 2 --n-examples 3   # self-consistency baseline, k=3
python scripts/run_local_debug.py --phase 3 --n-examples 1   # fault pattern check, all fault types
```

### Small scored batch (chain topology, local backend)
```bash
python scripts/run_study1.py --n-examples 15 --k 3 --no-inference-gap
python scripts/run_study1.py --n-examples 15 --k 3 --no-inference-gap --resume   # crash recovery, JSONL is append-only
```

### Full multi-topology GPU-batched generation (used for `dataset/`)
```bash
python scripts/run_study_vllm.py \
  --examples-per-topology-full-grid 2 \
  --examples-per-topology-partial 4 \
  --no-diagnose \
  --resume
```
`--resume` operates at topology-level granularity — safe to re-launch after a crash without re-running completed topologies. See [Section 8](#8-dataset-generation-phase--complete) for the full generation workflow and results.

### Post-generation verification
```bash
python scripts/verify_results.py --trials dataset/trials.jsonl --skipped dataset/skipped.jsonl
```

---

## 8. Dataset Generation Phase — Complete

The labeled dataset for training a topology-aware fault classifier/localizer is complete and checked into `dataset/` on the `dataset` branch (**493 trial records**, **15 skipped**). Full field-by-field schema, input-feature spec, ground-truth spec, and a worked example are in **[`dataset/dataset_description.md`](dataset/dataset_description.md)**.

### 8.1 Backend Migration: MLX/Ollama → vLLM on Kaggle GPU

Generating hundreds of trials across 20 topologies at k=5 samples/node requires real GPU throughput and batching, which the local MLX/Ollama backends don't provide. The plan (documented in `btp-pipeline/plan.md`) considered serving the existing MXFP4-quantized model via vLLM on Kaggle's free T4 GPUs, but **T4 (compute capability 7.5) cannot run MXFP4** — that format needs Hopper/Ada-class hardware. The backend was switched to **`Qwen/Qwen2.5-7B-Instruct-AWQ`** (4-bit AWQ, `float16`), which fits T4's 16GB VRAM with room for KV-cache, and calibrated before the full run (`scripts/calibrate_vllm_model.py`).

Generation used `llm.chat(List[List[message]], List[SamplingParams])` for genuine cross-trial batching — all pending LLM calls across a topology round are dispatched together rather than sequentially, which is what makes hundreds of trials on a free-tier GPU tractable.

### 8.2 Topology Pool & Question Source

- **Topology pool**: 20 topologies (14 train + 6 OOD), generated by `dataset/generate_topology_pool.py` using role-ordered random-DAG construction, serialized to `dataset/topology_pool.json`.
- **Question pool**: 30 HotpotQA questions (`data/study30_examples.json` — 15 from the original local runs + 15 newly sampled), sized to avoid repetition given trial-count caps (`--examples-per-topology-full-grid`, `--examples-per-topology-partial`) that keep the dataset near ~400-500 records instead of exploding combinatorially over topology × node × fault-type × question.

### 8.3 Bugs Found and Fixed During Data-Quality Audit

The first full generation run (410 records) completed without crashing, but a direct statistical/content inspection of the output (not just structural checks) surfaced three real correctness bugs in the fault-injection code, all fixed before the final regeneration:

1. **Ceiling role-dispatch bug** (`src/faults.py`) — `inject_ceiling()` dispatched on `target_node == "retriever"` (a string match against the node's *name*) instead of the node's *role*. In non-chain topologies, node names don't always match role names, so ceiling faults were silently mis-targeted. Fixed by adding an explicit `role` parameter threaded through `inject_ceiling()` and `_inject_ceiling_downstream()`, dispatching on `role` everywhere.
2. **Hardened instruction never applied** (`src/pipeline.py`) — `build_prompt()` built the hardened (ceiling-faulted) instruction but then used the original `node.instruction` in the final prompt string regardless, so ceiling faults had no actual effect on the prompt sent to the model. Fixed by routing the selected instruction (hardened when applicable) into the final prompt string, verified with a 4-point unit test (normal case, faulted case, normal-absent-when-faulted, no cross-node leakage).
3. **Contamination self-substitution** (`src/faults.py`) — the distractor substitute bank could select the clean value itself as its own "contamination," producing a no-op fault. Fixed by excluding the clean parent output from the substitute bank before sampling (returns `None`, causing the trial to be skipped, if the bank becomes empty).

Fixing bug #1 and #2 together increased ceiling-fault representation from 20 → 103 records (5.15×) on regeneration, since ceiling faults now actually reach the model. A known, low-severity, explicitly-scoped-out limitation remains: multi-parent contamination only corrupts the first parent, not all parents feeding a node.

### 8.4 Final Dataset Stats

| Metric | Value |
|---|---|
| Total trial records | 493 |
| Skipped records (with reason logged) | 15 |
| Ceiling-fault trials (post-fix) | 103 |
| Train-topology trials | 329 |
| OOD-topology trials | 164 |
| k (samples per node) | 5 |
| Backend | vLLM, `Qwen/Qwen2.5-7B-Instruct-AWQ` |
| GPU | Kaggle T4 |

Full per-fault-type and per-topology breakdowns, the record schema, input-feature vector spec (`[lexical_uncertainty, semantic_uncertainty, jaccard_uncertainty, role_one_hot×3]` per node + `edge_index` from the topology), ground-truth spec, and a worked example mapping a raw record to a GNN feature vector are all in **[`dataset/dataset_description.md`](dataset/dataset_description.md)**.

### 8.5 Kaggle Workflow Notes

Practical lessons from running generation on Kaggle, kept here since they'll matter for any future scale-up:

- **Dataset mount path**: a Kaggle Dataset attached to a Kernel mounts at `/kaggle/input/datasets/<owner>/<dataset-slug>`, not the shorter `/kaggle/input/<dataset-slug>` one might expect.
- **`scripts/` isn't a package on Kaggle**: add `CODE_ROOT/scripts` to `sys.path` and import script modules as top-level, rather than `from scripts.x import y`.
- **Real-time log streaming**: use `subprocess.Popen(stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)` and stream line-by-line in the kernel entry script — `subprocess.run(capture_output=True)` buffers all child output until the process exits, making a long run indistinguishable from a hung one. Combine with `kaggle kernels logs -f` (and `PYTHONUTF8=1` on Windows, to avoid a console-encoding crash) to watch progress live.
- **Verify dataset pushes**: `kaggle datasets version` can report "Upload successful" before the new content is actually live; always verify with a fresh `kaggle datasets download --unzip` into a clean directory, or poll `kaggle datasets files` until it reflects the new upload.
- **Dataset-as-code-delivery pattern**: rather than committing generation code to a Kaggle Notebook directly, the pipeline code was packaged as a Kaggle Dataset and mounted into a separate Kernel — this makes iterating on code (push a new Dataset version) independent from managing Kernel runs.

Full infra/bug detail: `btp-pipeline/HANDOFF_2026-09-25.md`.

---

## 9. Earlier Results — Study 1 (Local, 45 Trials)

This section documents the original local validation run (chain topology, MLX backend, before the vLLM/Kaggle migration) that established the core methodology.

### Per-Node Uncertainty Statistics (45 Trials)

| Fault Type (n=15 each) | U_retriever mean (max) | U_reasoner mean (max) | U_writer mean (max) |
|---|---|---|---|
| **Noise** (temp=1.2) | 0.089 (0.333) | 0.667 (0.667) | 0.111 (0.333) |
| **Contamination** (distractors) | **0.222 (0.667)** | 0.667 (0.667) | **0.244 (0.667)** |
| **Ceiling** (stripped gold) | 0.089 (0.333) | 0.667 (0.667) | 0.089 (0.667) |

**Key pattern**: Contamination consistently elevated both Retriever and Writer uncertainty compared to Noise and Ceiling — the correct theoretical prediction (wrong context → retriever disagrees across samples → writer receives inconsistent inputs).

### Confusion Matrix (Per-Node Threshold: U_retriever > 0.3 OR U_writer > 0.3)

```
diagnosed      noise  contamination  ceiling
true
noise              9              0        0
contamination      0             10        0
ceiling            0              0        5
```

- **24 / 45 trials** had uncertainty above the per-node threshold (were diagnosable)
- **100% precision on all 24 diagnosed trials** — zero cross-fault misclassifications
- **21 / 45 trials** were below threshold (parametric memory — model answered correctly from weights despite the injected fault)

---

## 10. Key Findings & Research Contributions

### Finding 1: Per-Node Thresholding is Essential
A single global uncertainty threshold fails because nodes have fundamentally different output formats: Retriever/Writer produce short, low-variance text; Reasoner produces verbose CoT with high structural variance even when correct. **Recommendation**: per-node threshold vectors, e.g. `Θ = {θ_retriever: 0.30, θ_reasoner: 0.75, θ_writer: 0.30}`.

### Finding 2: Parametric Memory as a Confound
Fault injection has no effect whenever the model can answer from parametric knowledge regardless of context quality — a natural floor on detectable failures. This motivated the multi-topology, multi-fault-target dataset design in Phase 2: spreading faults across more nodes and structures surfaces genuine failure modes that a single-chain, retriever-only fault design would miss.

### Finding 3: Contamination is Most Detectable (Locally)
In the original 45-trial local run, contamination produced the strongest, most consistent uncertainty signal. The larger Phase 2 dataset (with role-aware fault targeting across 20 topologies) enables checking whether this holds once faults can target the Reasoner and Writer directly, not just the Retriever.

### Finding 4: Topology Structure Changes Fault Cost, Not Just Fault Signal
Generation cost is driven by the longest dependency-chain *depth* in a topology (how many sequential rounds are needed) and by how many Reasoner-role nodes fall in the same round (Reasoner uses a larger token budget), not by raw node count — a wide fan-out topology with many same-layer nodes doesn't cost proportionally more than a narrow chain.

### Finding 5: Correctness Bugs Can Hide Behind "It Ran Successfully"
The three bugs in [Section 8.3](#83-bugs-found-and-fixed-during-data-quality-audit) all produced a dataset that generated cleanly, with no crashes or structural errors — they only surfaced under direct statistical/content inspection of the actual generated values. Structural validation (schema checks, "did it finish") is necessary but not sufficient; verifying a generated dataset requires checking that labeled fault conditions actually produced the expected qualitative effect on model output.

---

## 11. Output File Formats

### `dataset/trials.jsonl` (Phase 2 — primary training dataset)
See **[`dataset/dataset_description.md`](dataset/dataset_description.md)** for the full field-by-field schema, including topology metadata, per-node uncertainty triples, fault config, and ground-truth labels.

### `logs/trials.jsonl` (local/dev runs)
One JSON record per line:

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

Key constants in `src/config.py`:

| Constant | Default | Description |
|---|---|---|
| `BACKEND` | env-var driven | `"ollama"` / `"vllm"` / `"mlx"` — selects inference backend |
| `VLLM_MODEL` | `"Qwen/Qwen2.5-7B-Instruct-AWQ"` | Model served for dataset-scale generation |
| `VLLM_QUANTIZATION` | `"awq"` | 4-bit AWQ quantization (T4-compatible) |
| `VLLM_DTYPE` | `"float16"` | T4 lacks native bf16 |
| `DEFAULT_K` | `5` | Samples per node |
| `DEFAULT_TEMPERATURE` | `0.7` | Standard inference temperature |
| `NOISE_TEMPERATURE` | `1.2` | Elevated temperature for noise fault |
| `MAX_TOKENS` | `350` | Max generation tokens for Retriever/Writer |
| `MAX_TOKENS_REASONER` | `600` | Max generation tokens for Reasoner CoT |
| `UNCERTAINTY_THRESHOLD` | `0.75` | Global flag threshold (use per-node in practice) |
| `RETRY_K` | `3` | Samples in diagnostic retry pass |
| `DATA_DIR` | `"./data/hotpotqa_distractor"` | Cached dataset path |
| `DEFAULT_LOG_PATH` | `"./logs/trials.jsonl"` | Trial log path |
| `DEFAULT_CONFUSION_MATRIX_PATH` | `"./results/confusion_matrix.csv"` | Results path |

---

## 13. Next Steps (GNN Training & Beyond)

### Immediate
- [ ] **Train a GNN fault classifier/localizer** on `dataset/trials.jsonl` using the per-node feature vectors and `edge_index` spec in `dataset_description.md`; hold out the 6 OOD topologies for generalization testing.
- [ ] **k-fold cross-validation** given the dataset's size (493 records) — a single train/test split risks high-variance evaluation at this scale.
- [ ] **Extend the multi-parent contamination fix** — currently only the first parent's output is corrupted when a node has multiple parents; extend to corrupt all parents for full topology coverage.

### Propagation Dynamics
- [ ] **Cascade analysis** — when a fault is injected at node X, quantify how much uncertainty increases at each downstream node; plot propagation coefficients across topology depth.
- [ ] **Inference Gap metric** — fully implement the semantic-drift measure (currently `null` in local-run records): how much a node's output shifts between faulted and clean input.

### Pipeline Design
- [ ] **Confidence gating** — when a node's uncertainty exceeds its threshold, trigger automatic re-execution of that node before passing output downstream.
- [ ] **Generalisation** — test the trained localizer against non-QA pipelines (e.g. Summarizer → Critic → Rewriter) to check whether the topology-aware signal transfers.

---

## Citation

If you use this pipeline or dataset in your research, please cite:

```
Signature-Aware Orchestration for Uncertainty Propagation in Multi-Agent LLM Systems,
BTech Thesis (BTP-I), 2026.
Generation model: Qwen/Qwen2.5-7B-Instruct-AWQ (vLLM, Kaggle T4).
Dataset: Yang et al., HotpotQA (distractor setting), 2018.
```

---

*Dataset generated on: vLLM, `Qwen/Qwen2.5-7B-Instruct-AWQ`, Kaggle T4 GPU. Local development on: Apple Silicon (Metal GPU), `mlx_lm`, HotpotQA.*
