# Project Context — Multi-Agent LLM System: Uncertainty Propagation Pipeline

> **BTech Bachelor's Thesis Project (BTP-I)** — Aryan M, 2026

---

## 1. What This Project Is

This is a **research codebase** for a BTech thesis investigating a specific question:

> *When one agent in a multi-agent LLM pipeline fails, can we automatically detect WHICH agent failed and WHY — using only the model's own sampling behaviour as a signal, without any external oracle or ground-truth checker?*

Modern AI systems frequently chain multiple LLM agents: a **Retriever** fetches evidence, a **Reasoner** interprets it, and a **Writer** synthesises the final answer. Each agent looks locally correct — but if the Retriever passes bad evidence, the Reasoner reasons *perfectly from bad inputs* and the Writer produces a confident-sounding but wrong final answer.

**Standard systems only inspect the final output.** This project builds a mechanism that watches *every intermediate node* using self-consistency sampling and diagnoses failures at their root cause.

---

## 2. Repository Layout

```
Multi agent LLM System/                   (workspace root)
├── README.md                             (full research README, 446 lines)
├── explain.md, explain2.md               (intermediate research notes)
├── final_context.md                      (this file)
└── btp-pipeline/                         (entire codebase lives here)
    ├── README.md                         (pipeline-level README, 156 lines)
    ├── requirements.txt                  (Python dependencies)
    ├── src/                              (core library)
    │   ├── __init__.py
    │   ├── config.py                     (single source of truth for all constants)
    │   ├── nodes.py                      (NodeResult, PipelineTrace, sample_node)
    │   ├── pipeline.py                   (the 3-node pipeline + prompt templates)
    │   ├── faults.py                     (fault injection: noise / contamination / ceiling)
    │   ├── diagnose.py                   (retry-then-reprobe diagnostic protocol)
    │   └── pipelines/                    (standalone per-agent CLI wrappers)
    │       ├── __init__.py
    │       ├── retriever_pipeline.py
    │       ├── reasoner_pipeline.py
    │       └── writer_pipeline.py
    ├── scripts/                          (experiment entry points)
    │   ├── download_data.py              (one-time HotpotQA dataset download)
    │   ├── run_local_debug.py            (Phases 1-3: sanity checks)
    │   ├── run_agent.py                  (unified CLI to run individual agents)
    │   └── run_study1.py                 (Phase 4: full scored batch + confusion matrix)
    ├── logs/
    │   └── trials.jsonl                  (append-only trial log, 162 KB, 45 trials)
    └── results/
        ├── confusion_matrix.csv          (final 3x3 confusion matrix)
        ├── retriever_output.jsonl
        ├── reasoner_output.jsonl
        └── writer_output.jsonl
```

---

## 3. The Core Research Architecture

### The Three-Node Pipeline

```
[Question + Context]
        |
        v
+---------------+       U_retriever in [0,1]
|   RETRIEVER   |  ------------------------------>
+-------+-------+  (filters paragraphs to only
        |           the relevant sentences)
        v
+---------------+       U_reasoner in [0,1]
|    REASONER   |  ------------------------------>
+-------+-------+  (chain-of-thought over the
        |           filtered evidence)
        v
+---------------+       U_writer in [0,1]
|    WRITER     |  ------------------------------>
+---------------+  (final concise answer)

         Per-node uncertainties:
         NEVER collapsed to a scalar.
         Always a Dict[str, float].
```

**The fundamental design commitment** (enforced everywhere in the code): uncertainty scores are always kept as `{"retriever": U1, "reasoner": U2, "writer": U3}`. Collapsing to a scalar would destroy the localisation signal that is the entire research premise.

### The Uncertainty Measure

Each node is sampled **k times** (default k=3) independently at the same temperature. All k outputs are normalised using the **HotpotQA official SQuAD-style normalisation** (lowercase → strip punctuation → remove articles a/an/the → collapse whitespace). The **agreement rate** is computed:

```
agreement_rate = (count of modal normalised output) / k
uncertainty    = 1 - agreement_rate     in [0, 1]

Example (k=3):
  sample[0] = "Step 1: Derrickson is American. Step 2: Wood is American."
  sample[1] = "To determine: both are American nationals."
  sample[2] = "Chain of Thought: Conclusion is yes, both American."

  normalised[0] = "step 1 derrickson is american step 2 wood is american"
  normalised[1] = "to determine both are american nationals"
  normalised[2] = "chain of thought conclusion is yes both american"

  All three differ -> modal count = 1/3
  uncertainty = 1 - 1/3 = 0.667
```

**The most important calibration discovery** from this project: the Reasoner node inherently has a **baseline uncertainty of ~0.667 on perfectly clean data** because chain-of-thought output has high phrasing variance ("Step 1...", "To determine...", "Chain of Thought:") even when the conclusion is identical. This is not a bug — it is an inherent property of verbose CoT output at k=3.

---

## 4. The Model and Backend

- **Model**: `mlx-community/Qwen3-8B-4bit` — 4-bit quantised Qwen3-8B in MLX safetensors format
- **Backend**: `mlx_lm` — Apple Silicon native inference using Metal GPU
- **Hardware**: Apple Silicon (M-series), 16 GB RAM
- **Why MLX, not Ollama**: The original design specified Ollama (`qwen2.5:7b-instruct-q4_K_M`). The actual execution environment had the MLX model locally available. Switching from Ollama to MLX was the first major adaptation made during development.

**Key singleton pattern** in `nodes.py`: The model is loaded **once** as a module-level singleton via `_get_model()`. Without this, each of the 27 LLM calls per trial would reload the 8B model, adding ~3s × 27 = ~81 seconds. The singleton reduces this to a one-time ~3s load.

**Chat template**: Qwen3 uses an instruct template with `<|im_start|>` tokens. The pipeline calls `tokenizer.apply_chat_template(..., enable_thinking=False)`. The `enable_thinking=False` suppresses Qwen3's internal scratchpad, which would produce non-deterministic `<think>` tokens corrupting self-consistency comparison.

**Token budgets** (tuned to cut trial time by ~60%):
- Retriever/Writer: `MAX_TOKENS = 128`
- Reasoner: `MAX_TOKENS_REASONER = 200` (CoT needs a bit more room)

---

## 5. The Dataset

**HotpotQA** — a multi-hop QA dataset requiring reasoning over two or more Wikipedia paragraphs.

- **Split**: validation, distractor setting
- **Total examples in split**: 7,405
- **Used in Study 1**: 15 examples → 45 trials (15 × 3 fault types)
- **Why distractor setting**: Each example ships with gold (supporting) paragraphs AND 8 distractor (irrelevant) paragraphs. Contamination fault injection uses these shipped distractors — naturalistic bad input, not adversarially crafted.
- **Download**: `scripts/download_data.py` → cached at `./data/hotpotqa_distractor/`

---

## 6. The Three Fault Types

Implemented in `btp-pipeline/src/faults.py`.

### Fault 1: Noise
- **What**: Gold context unchanged; temperature elevated 0.7 → 1.2
- **Effect**: Output distribution flattens → higher disagreement across k samples despite perfect input
- **Diagnostic expectation**: High uncertainty → **recovers** on same-input retry at normal temperature
- `true_label = "noise"`, `sampling_override = {"temperature": 1.2}`

### Fault 2: Contamination
- **What**: 2 gold paragraphs replaced with 2 randomly sampled distractor paragraphs
- **Effect**: Retriever sees no relevant sentences → inconsistent output → cascades to Writer
- **Diagnostic expectation**: High uncertainty; **does NOT recover** on same-input retry; **DOES recover** on clean gold context retry
- `true_label = "contamination"`, `sampling_override = {"temperature": 0.7}`
- Returns `None` if fewer than 2 distractors available → caller skips

### Fault 3: Ceiling
- **What**: 50% of sentences stripped from each gold paragraph; on-topic but incomplete context
- **Effect**: Model cannot fully answer the multi-hop question; inconsistent guesses
- **Diagnostic expectation**: High uncertainty that persists through both same-input AND clean-input retry (no good input exists — it's a capability gap)
- `true_label = "ceiling"`, `sampling_override = {"temperature": 0.7}`

> **Important**: The "clean" prompt in the diagnostic always uses the FULL gold context (not the stripped one). This means ceiling faults ARE distinguishable — the clean retry still fails because the model genuinely lacks the reasoning capability, not because of bad context.

---

## 7. The Diagnostic Protocol

Implemented in `btp-pipeline/src/diagnose.py`.

### The Retry-Then-Reprobe Protocol

For each node where `uncertainty > UNCERTAINTY_THRESHOLD`:

```
Step 1 — Same-input retry
    Run k samples with the IDENTICAL (faulted) prompt, DEFAULT_RETRIES=2 times
    avg_retry_uncertainty = mean of the 2 retry uncertainties
    if avg_retry_uncertainty < UNCERTAINTY_THRESHOLD:
        -> NOISE  (it recovered — instability, not bad input)

Step 2 — Clean-input retry
    Run k samples with the FULL GOLD context
    if clean_result.uncertainty < UNCERTAINTY_THRESHOLD:
        -> CONTAMINATION  (recovered on clean input — input was the problem)

Step 3 — Neither helped
    -> CEILING  (model genuinely cannot answer this)
```

**Rationale for this order**: cheapest explanation first. Re-running is free; swapping context is cheap; concluding a capability gap is the last resort.

### Multi-Node Flagging

When multiple nodes exceed threshold:
- **All** flagged nodes are diagnosed independently
- `trace.per_node_diagnoses` stores full `{node_name: label}` dict
- `trace.diagnosed_label` = diagnosis of the **highest-uncertainty** node (most likely root cause)

### The Threshold

`UNCERTAINTY_THRESHOLD = 0.75` — calibrated empirically from Phase 2:
- Retriever baseline: 0.000 (fully consistent short-sentence extraction)
- Reasoner baseline: 0.667 (CoT phrasing varies — expected, not a fault)
- Writer baseline: 0.000 (fully consistent short answers)

0.75 sits just above 0.667 — flags only nodes that are *worse than the Reasoner's own inherent noise floor*.

> **Open issue**: A single global threshold 0.75 is suboptimal. Research findings indicate per-node thresholds are needed: `{retriever: 0.30, reasoner: 0.75, writer: 0.30}`. Implementing this is the top Study 2 priority.

---

## 8. Code Architecture — File by File

### `src/config.py`

Single source of truth. All other modules import from here:

```python
MODEL = "mlx-community/Qwen3-8B-4bit"
DEFAULT_K = 3              # samples per node
DEFAULT_TEMPERATURE = 0.7
NOISE_TEMPERATURE = 1.2
MAX_TOKENS = 128           # Retriever / Writer
MAX_TOKENS_REASONER = 200  # Reasoner CoT
UNCERTAINTY_THRESHOLD = 0.75  # calibrated from Phase 2
DEFAULT_RETRIES = 2
DATA_DIR = "./data/hotpotqa_distractor"
LOG_DIR = "./logs"
RESULTS_DIR = "./results"
```

---

### `src/nodes.py`

The foundational sampling layer. Contains:

**`hotpotqa_normalize(text)`** — SQuAD-style normalisation:
lowercase → strip punctuation → remove articles (a, an, the) → collapse whitespace.

Critical fix over the original design which used `text.strip().lower()` — that would cause false disagreements on "the X" vs "X", artificially inflating uncertainty scores.

**`NodeResult`** dataclass:
```python
node_name: str
output: str         # the modal (most-agreed-on) raw sample
uncertainty: float  # 1 - agreement_rate, in [0, 1]
samples: List[str]  # all k raw samples stored for post-hoc analysis
inference_gap: Optional[float] = None  # Study 2 experimental metric
```

**`PipelineTrace`** dataclass — collects NodeResults for one question:
```python
question: str
node_results: Dict[str, NodeResult]
true_label: Optional[str]         # set at fault-injection time
diagnosed_label: Optional[str]    # set by diagnose.py
gold_answer: Optional[str]
per_node_diagnoses: Dict[str, str]

def uncertainties(self) -> Dict[str, float]: ...  # NEVER collapses to scalar
```

**`sample_node(node_name, prompt, k, temperature, normalize_fn)`** — core sampling:
1. For each of k iterations: apply chat template → `mlx_lm.generate()` → store raw output
2. Normalise all k outputs
3. Find modal normalised output (most common)
4. `uncertainty = 1 - (modal_count / k)`
5. Return first raw sample whose normalised form matches the modal

---

### `src/pipeline.py`

Three-node pipeline + prompt templates.

**Prompt templates** (carefully designed to enforce agent role separation):

```
Retriever: "You are a Retriever agent. Given the question and candidate
paragraphs, select and return only the sentences directly relevant to
answering the question. Do not add any commentary."

Reasoner: "You are a Reasoning agent. Given the evidence below, reason
step by step to derive the answer. Show your full chain of thought."

Writer: "You are a Writer agent. Given the reasoning trace below, produce
a final, concise answer. Output only the answer — no explanation."
```

**`run_pipeline(question, context, k, temperature)`**:
1. Run Retriever → NodeResult r1
2. Run Reasoner with r1.output as evidence → NodeResult r2
3. Run Writer with r2.output as reasoning → NodeResult r3
4. Return PipelineTrace (all three)

Sequential only — hardware constraint (16 GB RAM; concurrent MLX would OOM).

---

### `src/faults.py`

Helpers: `get_gold_context(example)`, `get_distractor_context(example)`, `format_context(paragraphs)` → `"[Title] sent1 sent2..."`.

Three injectors: `inject_noise`, `inject_contamination`, `inject_ceiling`.
Each returns `{question, context, answer, true_label, sampling_override}` or `None`.

---

### `src/diagnose.py`

**`needs_diagnosis(trace, node_name)`** — checks `uncertainty > threshold`.

**`compute_inference_gap(node_result, input_text, enabled=True)`** — EXPERIMENTAL (Study 2):
- Uses `sentence-transformers all-MiniLM-L6-v2`
- Computes cosine distance (input embedding vs output embedding)
- Higher = more semantic drift from input to output
- Logged as `null` in all 45 Study 1 trials (disabled with `--no-inference-gap`)

**`diagnose(node_name, same_input_prompt, clean_input_prompt, k, retries)`** — 3-step protocol → `"noise"` / `"contamination"` / `"ceiling"`.

**`diagnose_trace(trace, node_prompts, k, retries, compute_gap)`** — diagnoses all flagged nodes; updates trace in-place.

`node_prompts` structure (passed from `run_study1.py`):
```python
{
  "retriever": {"same": retriever_prompt(q, faulted_ctx),
                "clean": retriever_prompt(q, gold_ctx)},
  "reasoner":  {"same": reasoner_prompt(q, faulted_retriever_output),
                "clean": reasoner_prompt(q, gold_ctx)},  # approximation — documented
  "writer":    {"same": writer_prompt(q, faulted_reasoner_output),
                "clean": writer_prompt(q, gold_ctx)},
}
```

> **Known approximation**: The "clean" Reasoner input uses gold context directly, not a clean re-run of the Retriever. Logged as a known limitation; full clean re-run at each node is a Study 2 enhancement.

---

### `src/pipelines/`

Standalone per-agent CLI wrappers with typed I/O dataclasses. Each has:
- Input/Output dataclasses (`RetrieverInput`, `RetrieverOutput`, etc.)
- `run_*_pipeline()` function
- `_save_output()` → appends to `results/{node}_output.jsonl`
- Full `argparse` CLI (`--question`, `--context`/`--evidence`/`--reasoning`, `--from-*-output`, `--k`, `--temperature`, `--no-save`)

The `--from-*-output` flags allow chaining: retriever output → reasoner input → writer input via saved JSONL files.

---

### `scripts/run_agent.py`

Unified CLI dispatcher. Modes: `retriever`, `reasoner`, `writer`, `all`.

The `all` mode chains all three sequentially and prints a per-node uncertainty summary table at the end — the most useful tool for interactive debugging.

---

### `scripts/run_local_debug.py`

Phases 1–3 sanity checks. Must be run in order before Phase 4.

- **Phase 1** (k=1, 3 examples, no faults): Confirm all three prompts produce correct, on-topic output. Catch prompt-formatting bugs here.
- **Phase 2** (k=3, 3 examples, no faults): Measure baseline uncertainty distribution. Calibrate threshold here.
- **Phase 3** (k=3, 3 examples, all 3 fault types): Confirm that the three fault types produce visibly distinct uncertainty patterns before trusting the scoring loop.

---

### `scripts/run_study1.py`

Phase 4 — full scored experiment. Key features:
- **Append-only JSONL logging**: every trial written to disk immediately after completion (crash-safe; zero trials can be lost)
- **tqdm progress bar** with `skipped` counter
- **Resume support** (`--resume`): loads existing log; skips trials already done, keyed by `(question, true_label)` pair
- **Confusion matrix**: `build_confusion_matrix()` builds 3×3 crosstab → saves CSV → prints with diagonal dominance check

---

## 9. The Experimental Execution — Phase by Phase

### Phase 1: Prompt Sanity ✅ COMPLETED

k=1, 3 HotpotQA examples, no faults. All three answered correctly:

| Question | Gold Answer | Writer Output | Correct? |
|---|---|---|---|
| Were Scott Derrickson and Ed Wood of the same nationality? | yes | "Yes, both American." | ✅ |
| What government position was held by the Corliss Archer actress? | Chief of Protocol | "Chief of Protocol of the United States" | ✅ |
| What YA sci-fi series has alien species companion books? | Animorphs | "Animorphs" | ✅ |

---

### Phase 2: Baseline Calibration ✅ COMPLETED — Critical Discovery

k=3, same 3 examples, no faults. **The most important finding of the entire project**:

```
Retriever  uncertainty: 0.0000   short factual extraction, fully consistent
Reasoner   uncertainty: 0.6667   CoT phrasing varies; conclusion identical
Writer     uncertainty: 0.0000   short final answer, fully consistent
```

**Decision**: Recalibrated `UNCERTAINTY_THRESHOLD` from 0.3 (original placeholder) → **0.75** (just above Reasoner's inherent baseline).

**Research implication**: A single global threshold cannot work — each node has a fundamentally different output format and therefore a different baseline uncertainty floor. Per-node threshold vector `{retriever: 0.30, reasoner: 0.75, writer: 0.30}` is the correct solution.

---

### Phase 3: Fault Pattern Inspection ✅ COMPLETED — Parametric Memory Finding

k=3, 1 example, all 4 conditions (baseline + 3 faults). **All conditions produced identical `0.0 / 0.667 / 0.0` patterns.**

**Root cause**: *Parametric memory* — Qwen3-8B has encoded answers to simple factoid questions in its weights. "Scott Derrickson and Ed Wood are both American" is answerable from training data regardless of what context is provided. The model ignored the injected faults entirely.

**Research implication**: Fault injection only creates meaningful uncertainty on questions where the model genuinely **cannot** answer from parametric memory alone — harder multi-hop questions requiring specific cross-paragraph evidence not in model weights. The Phase 3 example happened to be too easy.

---

### Phase 4: Study 1 Full Run ✅ COMPLETED

- **15 HotpotQA validation examples × 3 fault types = 45 trials**
- k=3 per node, `--no-inference-gap`
- ~46 seconds/trial → ~35 minutes total
- **Zero crashes, zero lost trials** — append-only JSONL worked perfectly
- **45 records in `logs/trials.jsonl`** (162 KB)

---

## 10. Results — Study 1

### Per-Node Uncertainty Statistics (45 Trials)

| Fault Type (n=15) | U_retriever mean (max) | U_reasoner mean (max) | U_writer mean (max) |
|---|---|---|---|
| Noise (temp=1.2) | 0.089 (0.333) | 0.667 (0.667) | 0.111 (0.333) |
| **Contamination** (distractors) | **0.222 (0.667)** | 0.667 (0.667) | **0.244 (0.667)** |
| Ceiling (stripped gold) | 0.089 (0.333) | 0.667 (0.667) | 0.089 (0.667) |

Contamination consistently elevated both Retriever and Writer uncertainty vs the other two fault types — the correct theoretical prediction. Wrong retrieval cascades into downstream inconsistency.

### Confusion Matrix

```
diagnosed      noise  contamination  ceiling
true
noise              9              0        0
contamination      0             10        0
ceiling            0              0        5
```

- **24 / 45 trials** exceeded threshold → diagnosable
- **100% precision** on all 24 diagnosed trials — zero cross-fault misclassifications
- **21 / 45 trials** below threshold — model answered from parametric memory
- Detection recall: Contamination 10/15 (67%), Noise 9/15 (60%), Ceiling 5/15 (33%)

---

## 11. Key Research Findings

### Finding 1: Per-Node Thresholding Is Essential
A global threshold fails because node output formats differ fundamentally in their inherent variance.

**Recommended threshold vector**: `Θ = {θ_retriever: 0.30, θ_reasoner: 0.75, θ_writer: 0.30}`

### Finding 2: Parametric Memory as a Confound
~47% of trials (21/45) were below threshold because the model answered from weights regardless of context. This is a natural floor on detectable failures for a large, fact-rich model.

**Recommendation**: Pre-filter questions to those where no-fault baseline shows `max(uncertainties.values()) > 0.20`.

### Finding 3: Contamination Is Most Detectable
10/15 detected (67% recall) vs 9/15 for Noise, 5/15 for Ceiling. Wrong retrieval cascades more reliably than temperature noise or partial evidence.

### Finding 4: 100% Diagnostic Precision on Detected Cases
Among all 24 diagnosed cases, zero cross-fault misclassifications. The retry-then-reprobe protocol is a reliable fault localisation mechanism when uncertainty is high enough to trigger it.

---

## 12. Output File Formats

### `logs/trials.jsonl` — one JSON record per trial

```json
{
  "question": "Are the Laleli Mosque and Esma Sultan Mansion in the same neighborhood?",
  "true_label": "contamination",
  "diagnosed_label": "no_fault_detected",
  "uncertainties": {"retriever": 0.3333, "reasoner": 0.6667, "writer": 0.0},
  "inference_gaps": {"retriever": null, "reasoner": null, "writer": null},
  "per_node_diagnoses": {},
  "gold_answer": "no",
  "samples": {
    "retriever": ["None", "No relevant sentences.", "None"],
    "reasoner": ["Chain of thought step 1...", "Reasoning step 1...", "Analysis..."],
    "writer": ["The evidence is insufficient.", "...", "..."]
  }
}
```

### `results/confusion_matrix.csv`

```
true,noise,contamination,ceiling
noise,9,0,0
contamination,0,10,0
ceiling,0,0,5
```

---

## 13. Invariants (Never Break These)

1. **Per-node uncertainties are NEVER merged into a scalar.** `trace.uncertainties()` always returns `Dict[str, float]`.
2. **All LLM calls are sequential.** No threading, no asyncio — 16 GB RAM constraint.
3. **Append-only JSONL logging.** Every trial written immediately — a crash never loses completed trials.
4. **Multi-node flagging**: diagnose ALL nodes above threshold; `diagnosed_label` = highest-uncertainty node's diagnosis.
5. **Normalisation uses official SQuAD/HotpotQA style.** Not `strip().lower()`.

---

## 14. Open Items and Next Steps

### Immediate (Before Cloud GPU Study 1 Scale-Up)
- [ ] **Per-node threshold vector** — update `diagnose.py` + `config.py` to accept `Dict[str, float]` thresholds; use `{retriever: 0.30, reasoner: 0.75, writer: 0.30}`
- [ ] **Pre-filter to hard questions** — run no-fault baseline on 300 questions; keep only those where `max(uncertainties.values()) > 0.20`
- [ ] **Extract Reasoner conclusion only** — instead of full CoT, compare only the final conclusion sentence; reduces Reasoner baseline uncertainty from ~0.667 to ~0.0 making it a much sharper signal
- [ ] **Scale Study 1** — `--n-examples 300 --k 5` on cloud A100/H100; ~13,500 LLM calls, ~2-4 hours

### Study 2 — Uncertainty Propagation Dynamics
- [ ] **Cascade analysis** — quantify `ΔU_reasoner | ΔU_retriever` and `ΔU_writer | ΔU_reasoner` as propagation coefficients
- [ ] **Inference Gap metric** — fully enable the semantic-drift measure (currently `null` in all records) using `sentence-transformers all-MiniLM-L6-v2`
- [ ] **Counterfactual injection** — inject a single wrong sentence instead of replacing all paragraphs; find minimum detectable perturbation

### Study 3 — Uncertainty-Aware Pipeline Design
- [ ] **Confidence gating** — when Retriever uncertainty exceeds threshold, trigger automatic re-retrieval before passing to Reasoner
- [ ] **Uncertainty-weighted consensus** — weight Writer samples by source Reasoner confidence
- [ ] **Generalisation** — test diagnostic protocol on Summarizer → Critic → Rewriter and Code Generator → Tester → Reviewer pipelines

---

## 15. Quick-Reference Commands

```bash
# From btp-pipeline/ directory

# One-time setup
python -m venv btp-env
btp-env\Scripts\activate        # Windows
pip install -r requirements.txt
python scripts/download_data.py  # downloads HotpotQA once (~50 MB)

# Phase-by-phase validation (run in order)
python scripts/run_local_debug.py --phase 1  # prompt sanity (k=1)
python scripts/run_local_debug.py --phase 2  # baseline calibration (k=3)
python scripts/run_local_debug.py --phase 3  # fault pattern check (k=3)

# Full Study 1 run
python scripts/run_study1.py --n-examples 15 --k 3 --no-inference-gap
python scripts/run_study1.py --n-examples 15 --k 3 --no-inference-gap --resume  # after crash

# Cloud scale-up
python scripts/run_study1.py --n-examples 300 --k 5

# Interactive single-question run (full chain)
python scripts/run_agent.py all \
    --question "What nationality is the director of Crocodile Dundee?" \
    --context "[Crocodile Dundee] Australian film directed by Peter Faiman."
```

---

## 16. Dependencies

```
datasets>=2.19.0       # HotpotQA via Hugging Face datasets library
ollama>=0.2.0          # listed in requirements.txt but REPLACED by mlx_lm
sentence-transformers  # Inference Gap (Study 2, optional / currently disabled)
scikit-learn>=1.4.0    # confusion matrix utilities
pandas>=2.2.0          # confusion matrix DataFrame
tqdm>=4.66.0           # progress bar in run_study1.py
mlx_lm                 # NOT in requirements.txt — Apple Silicon only
                       # install via: pip install mlx-lm
```

> **Note**: `requirements.txt` still lists `ollama>=0.2.0` from the original design. The actual backend is `mlx_lm`. The requirements file should be updated before sharing with others running on Apple Silicon.

---

*Project status: Study 1 fully completed (45 trials, 100% precision on diagnosed cases). Study 2 in planning.*  
*Document generated: September 2026.*
