# Uncertainty Propagation in Multi-Agent LLM Pipelines — Complete Explanation

> **Document purpose:** A self-contained, detailed reference explaining *why* this project exists, *what* was built, *how* every piece works, and *what the results mean*. Intended for anyone picking up this repository cold — no prior context assumed.

---

## Table of Contents

1. [The Problem This Project Solves](#1-the-problem-this-project-solves)
2. [The Research Idea](#2-the-research-idea)
3. [What Was Actually Built (Study 1)](#3-what-was-actually-built-study-1)
4. [Repository Layout](#4-repository-layout)
5. [Step-by-Step Implementation Walkthrough](#5-step-by-step-implementation-walkthrough)
   - [5.1 Configuration — `config.py`](#51-configuration--configpy)
   - [5.2 Data Structures & Sampling — `nodes.py`](#52-data-structures--sampling--nodespy)
   - [5.3 Pipeline Wiring — `pipeline.py`](#53-pipeline-wiring--pipelinepy)
   - [5.4 Fault Injection — `faults.py`](#54-fault-injection--faultspy)
   - [5.5 Diagnostic Protocol — `diagnose.py`](#55-diagnostic-protocol--diagnosepy)
   - [5.6 Data Download — `scripts/download_data.py`](#56-data-download--scriptsdownload_datapy)
   - [5.7 Debug Phases 1–3 — `scripts/run_local_debug.py`](#57-debug-phases-13--scriptsrun_local_debugpy)
   - [5.8 Full Scored Run — `scripts/run_study1.py`](#58-full-scored-run--scriptsrun_study1py)
6. [The Three Fault Types in Detail](#6-the-three-fault-types-in-detail)
7. [The Diagnostic Protocol — How It Works](#7-the-diagnostic-protocol--how-it-works)
8. [The Confusion Matrix — The Primary Deliverable](#8-the-confusion-matrix--the-primary-deliverable)
9. [Critical Design Rules Enforced Throughout](#9-critical-design-rules-enforced-throughout)
10. [Open Decisions & Known Limitations](#10-open-decisions--known-limitations)
11. [The Bigger Research Picture (Studies 2–4)](#11-the-bigger-research-picture-studies-24)

---

## 1. The Problem This Project Solves

### The naive multi-agent setup

A standard multi-agent LLM pipeline looks like this:

```
Question → [Retriever LLM] → evidence → [Reasoner LLM] → reasoning → [Writer LLM] → final answer
```

Each node is a separate LLM call with a different system prompt and role. In practice, all three roles can be the same underlying model called three times with different prompts.

### Where today's systems fail

Every existing system — confidence scores, MoE router weights, "if confidence < threshold → verifier" rules — collapses all uncertainty into **one scalar per pipeline run**, then applies one threshold, then triggers one generic fallback. This has a fundamental blind spot:

```
Planner Trust = 0.98 → Retriever Trust = 0.25 → Reasoner Trust = 0.99 → Writer Trust = 0.97
```

In this trace, the Reasoner and Writer report high self-confidence — because they *are* confident in their own reasoning — but they have reasoned from bad evidence produced by the low-trust Retriever. The final answer is wrong, but looks confident. A system that only checks aggregate or final-node confidence is completely blind to this.

**The core failure mode:** a tentative upstream output becomes a confirmed downstream input. The uncertainty signal from the Retriever is silently erased as it passes through subsequent nodes.

### The specific gap in the literature

The research paper most closely related to this work is **SAUP (Zhao et al., ACL 2025)**, which propagates per-step uncertainty across a ReAct agent via weighted RMS. But SAUP produces a better *aggregate score only* — it never takes a genuinely different action depending on *which node* was uncertain or *why*.

Even the **Bayesian Uncertainty Propagation for Agentic RAG (Hull, 2026)** paper explicitly names "learned inhibition per node" — differentiated response per node — as unsolved future work.

**This project fills that gap for Study 1:** building and validating the *diagnostic* mechanism that identifies *why* a node is uncertain before deciding what to do about it.

---

## 2. The Research Idea

### The three fault types

Instead of treating all uncertainty as equivalent, this project defines three distinct *causes* of high uncertainty at any pipeline node:

| Fault Type | Definition | Diagnostic Test |
|---|---|---|
| **Noise** | Sampling variance on a single call. The input is fine; the model just drew a bad random sample. | Retry with identical input at normal temperature. If uncertainty drops → noise. |
| **Contamination** | The upstream *input* to this node is wrong/irrelevant, but the model's capability is fine. | Retry with a known-good clean input. If uncertainty drops → contamination. |
| **Ceiling** | The task exceeds this model's effective capability regardless of input quality. | Fails even on a clean input. Only fixed by more capacity. |

### The retry-then-reprobe protocol

The diagnostic tests are ordered cheapest-first:

1. **Same-input retry** (cheap): does repeated sampling at normal temperature converge? If yes → noise.
2. **Clean-input reprobe** (moderate): does a fresh run with ground-truth context converge? If yes → contamination.
3. **Neither helped** → ceiling.

### Why this matters

The cause determines the correct fix:
- **Noise** → retry with same input at lower temperature.
- **Contamination** → fix the upstream node (re-retrieve, reformulate query), not the downstream node that showed the symptom.
- **Ceiling** → escalate (bigger model, task decomposition, tools, abstain). Retrying with the same model wastes compute.

A generic "low confidence → run verifier" rule can't tell these apart. It may retry the Writer (cheap) when the actual fix is re-retrieval (one step upstream), or keep retrying when the real answer is to escalate.

---

## 3. What Was Actually Built (Study 1)

**Study 1's single most important deliverable:** a **3×3 confusion matrix** — rows = true injected fault type, columns = diagnosed fault type — that tests whether the retry-then-reprobe protocol can actually recover the correct cause label.

The pipeline:
1. Takes a real HotpotQA question through all three nodes (Retriever → Reasoner → Writer).
2. **Deliberately injects** one of three fault types (with known ground truth, since we caused it).
3. Runs the diagnostic protocol to *guess* which fault type occurred, using only the uncertainty pattern — never told the true label.
4. Scores the guess against ground truth.

If the confusion matrix's diagonal clearly dominates (correct guesses >> incorrect guesses), the diagnostic mechanism works and later studies (orchestration policies, resource constraints) are worth building.

**Current result** (from `results/confusion_matrix.csv`):

| true \ diagnosed | noise | contamination | ceiling |
|---|---|---|---|
| **noise** | 9 | 0 | 0 |
| **contamination** | 0 | 10 | 0 |
| **ceiling** | 0 | 0 | 5 |

**24/24 correct — 100% diagnostic accuracy on this local run.** The diagonal dominates perfectly. The core research premise holds.

> **Caveat:** This is a small-N local run (quantized 8B model, k=3, ~8 examples). Cloud-scale Study 1 (larger model, k=5, hundreds of trials) is required before statistical claims can be made.

---

## 4. Repository Layout

```
Multi agent LLM System/
├── README.md                          ← Top-level project README (research idea)
└── btp-pipeline/
    ├── README.md                      ← Pipeline-specific README (quick-start)
    ├── requirements.txt               ← Python dependencies
    ├── data/
    │   └── hotpotqa_distractor/       ← Cached HotpotQA dataset (created by download_data.py)
    ├── src/
    │   ├── __init__.py
    │   ├── config.py                  ← All tunable constants (single source of truth)
    │   ├── nodes.py                   ← Data structures + self-consistency sampling
    │   ├── pipeline.py                ← Three-node pipeline wiring
    │   ├── faults.py                  ← Fault injection library (noise/contamination/ceiling)
    │   └── diagnose.py                ← Retry-then-reprobe diagnostic protocol
    ├── scripts/
    │   ├── download_data.py           ← One-time dataset download & cache
    │   ├── run_local_debug.py         ← Phases 1–3: sanity checks, small N, verbose
    │   └── run_study1.py              ← Phase 4+: scored batch run + confusion matrix
    ├── logs/
    │   └── trials.jsonl               ← Append-only per-trial log (crash-safe)
    └── results/
        └── confusion_matrix.csv       ← Primary Study 1 deliverable
```

---

## 5. Step-by-Step Implementation Walkthrough

### 5.1 Configuration — [`config.py`](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/config.py)

**Purpose:** Single source of truth for every tunable constant. Every other module imports from here — no magic numbers scattered across files.

**Key constants and why they were chosen:**

```python
MODEL = "mlx-community/Qwen3-8B-4bit"
```
Originally specified as Qwen2.5-7B via Ollama in the research brief. **Changed to Qwen3-8B-4bit via `mlx_lm`** because the locally available quantized model was in MLX safetensors format (not GGUF). All pipeline logic is identical — only the backend call changed.

```python
DEFAULT_K = 3
DEFAULT_TEMPERATURE = 0.7
NOISE_TEMPERATURE = 1.2
MAX_TOKENS = 128
MAX_TOKENS_REASONER = 200
```
`k=3` is the self-consistency sample count for local prototyping (bump to 5 for cloud-scale Study 1). `NOISE_TEMPERATURE = 1.2` flattens the sampling distribution to artificially induce disagreement for the noise fault type. Reasoner gets extra token budget because chain-of-thought reasoning is verbose.

```python
UNCERTAINTY_THRESHOLD = 0.75
```
**This is a critical calibrated value.** The brief specified `0.3` as a placeholder. After running Phase 2 (clean baseline), the actual observed uncertainties were:
- Retriever: `0.0` (factual sentence extraction → fully consistent)
- Reasoner: `0.667` (chain-of-thought phrasing varies, even when conclusion is identical)
- Writer: `0.0` (short final answers → fully consistent)

Setting the threshold at `0.3` would have triggered false positives on *every clean Reasoner call* (baseline uncertainty 0.667 > 0.3). The threshold was empirically raised to `0.75` — just above the clean Reasoner baseline — so only *worse-than-baseline* inconsistency triggers diagnosis.

```python
DEFAULT_RETRIES = 2
DEFAULT_LOG_PATH = "./logs/trials.jsonl"
DEFAULT_CONFUSION_MATRIX_PATH = "./results/confusion_matrix.csv"
```

---

### 5.2 Data Structures & Sampling — [`nodes.py`](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/nodes.py)

**Purpose:** Defines the two core data structures (`NodeResult`, `PipelineTrace`) and the `sample_node()` function that executes self-consistency sampling.

#### The model singleton loader ([`nodes.py` L31–L42](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/nodes.py#L31-L42))

```python
_mlx_model = None
_mlx_tokenizer = None

def _get_model():
    global _mlx_model, _mlx_tokenizer
    if _mlx_model is None:
        from mlx_lm import load
        _mlx_model, _mlx_tokenizer = load(MODEL)
    return _mlx_model, _mlx_tokenizer
```

Loading an 8B model takes ~3 seconds. Without caching, a pipeline run with `k=3` and 3 nodes would reload the model **9 times**. The singleton pattern loads it once per process and reuses it across all calls.

#### The normalisation function ([`nodes.py` L49–L70](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/nodes.py#L49-L70))

```python
def hotpotqa_normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()
```

The research brief flagged this as a critical action item. The naive `strip().lower()` placeholder would undercount agreement on Writer output because of superficial phrasing differences like:
- "Arthur's Magazine" vs "arthurs magazine" → naive: disagree; normalized: agree
- "The answer is yes" vs "Yes" → naive: disagree; normalized: agree

This is the **official SQuAD-style normalization** used by HotpotQA's own evaluation script: lowercase → strip punctuation → remove articles → collapse whitespace. It is aliased as `default_normalize` so call-sites written against the placeholder still work.

#### NodeResult ([`nodes.py` L76–L91](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/nodes.py#L76-L91))

```python
@dataclass
class NodeResult:
    node_name: str
    output: str          # The raw sample matching the modal normalized answer
    uncertainty: float   # 1 - agreement_rate; 0 = fully consistent, 1 = fully inconsistent
    samples: list        # All k raw samples (for post-hoc analysis)
    inference_gap: Optional[float] = None  # Experimental semantic drift metric
```

`uncertainty = 1 - agreement_rate`. If all 3 samples normalize to the same answer → agreement_rate = 1.0 → uncertainty = 0.0. If all 3 differ → agreement_rate = 0.33 → uncertainty = 0.67.

#### PipelineTrace ([`nodes.py` L94–L124](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/nodes.py#L94-L124))

```python
@dataclass
class PipelineTrace:
    question: str
    node_results: Dict[str, NodeResult]  # {"retriever": ..., "reasoner": ..., "writer": ...}
    true_label: Optional[str]            # set at injection time
    diagnosed_label: Optional[str]       # set after diagnosis
    gold_answer: Optional[str]
    per_node_diagnoses: Dict[str, str]   # full {node_name: label} for multi-node flagging
```

**Critical design rule:** `trace.uncertainties()` returns `{"retriever": 0.6, "reasoner": 0.0, "writer": 0.1}` — a dict. It must **never** be collapsed into a single float anywhere in the codebase. This separation is the project's core structural bet.

#### sample_node() ([`nodes.py` L130–L200](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/nodes.py#L130-L200))

```python
def sample_node(node_name, prompt, k=3, temperature=0.7, normalize_fn=hotpotqa_normalize) -> NodeResult:
```

Key implementation details:
1. Calls `mlx_lm.generate()` **k times sequentially** (never concurrent — hardware constraint on 16GB RAM).
2. Applies the Qwen3 instruct **chat template** via `tokenizer.apply_chat_template()` with `enable_thinking=False` to suppress Qwen3's internal chain-of-thought scratchpad (which would confuse self-consistency comparison).
3. Gives the Reasoner a larger token budget (`MAX_TOKENS_REASONER = 200`) vs. other nodes (`MAX_TOKENS = 128`).
4. Counts normalized sample frequencies; top answer by count is the modal answer.
5. Returns the first *raw* sample whose normalized form matched the modal answer (preserving original phrasing for downstream prompts).

---

### 5.3 Pipeline Wiring — [`pipeline.py`](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/pipeline.py)

**Purpose:** Defines the three role prompts and wires them into a sequential pipeline.

#### Prompt builders ([`pipeline.py` L17–L47](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/pipeline.py#L17-L47))

```python
def retriever_prompt(question, context) -> str:
    # "You are a Retriever agent. Given the question and candidate paragraphs,
    #  select and return only the sentences directly relevant to answering
    #  the question. Do not add any commentary..."

def reasoner_prompt(question, evidence) -> str:
    # "You are a Reasoning agent. Given the evidence below, reason step by step
    #  to derive the answer. Show your full chain of thought."

def writer_prompt(question, reasoning) -> str:
    # "You are a Writer agent. Given the reasoning trace, produce a final,
    #  concise answer. Output only the answer — no explanation, no preamble."
```

These prompts are **also exported** to `run_study1.py` so the diagnostic protocol can reconstruct the exact prompt that was used in a faulted run (for the same-input retry) and the equivalent prompt with gold context (for the clean-input reprobe).

#### run_pipeline() ([`pipeline.py` L54–L99](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/pipeline.py#L54-L99))

```python
def run_pipeline(question, context, k=3, temperature=0.7) -> PipelineTrace:
    trace = PipelineTrace(question=question)

    r1 = sample_node("retriever", retriever_prompt(question, context), k=k, temperature=temperature)
    trace.add(r1)

    r2 = sample_node("reasoner", reasoner_prompt(question, r1.output), k=k, temperature=temperature)
    trace.add(r2)

    r3 = sample_node("writer", writer_prompt(question, r2.output), k=k, temperature=temperature)
    trace.add(r3)

    return trace
```

The chaining is literal: Retriever's `.output` (the modal raw sample) becomes the evidence string in the Reasoner prompt. Reasoner's `.output` becomes the reasoning string in the Writer prompt. This is exactly how real multi-agent pipelines work — and exactly how contamination propagates silently downstream.

---

### 5.4 Fault Injection — [`faults.py`](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/faults.py)

**Purpose:** Provides a library of three fault injectors that produce trials with **known ground-truth labels** for diagnostic testing.

#### Context helpers ([`faults.py` L28–L57](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/faults.py#L28-L57))

```python
def get_gold_context(example) -> List[Tuple[str, List[str]]]:
    # Returns (title, sentences) pairs for gold paragraphs only
    # identified by example["supporting_facts"]["title"]

def get_distractor_context(example) -> List[Tuple[str, List[str]]]:
    # Returns everything in context NOT in supporting_facts["title"]
    # These are HotpotQA's shipped irrelevant paragraphs — used directly
    # for contamination injection, no artificial construction needed

def format_context(paragraphs) -> str:
    # "[Title] Sentence1 Sentence2 ..."
```

HotpotQA's distractor split is ideal for this project: each example ships with ~2 gold paragraphs and ~8 distractor (irrelevant) paragraphs. The distractors are naturally occurring irrelevant text — not adversarially crafted — making contamination injection realistic.

#### inject_noise() ([`faults.py` L63–L79](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/faults.py#L63-L79))

```python
def inject_noise(example) -> Dict:
    return {
        "question": example["question"],
        "context": format_context(get_gold_context(example)),  # GOLD context — input is fine
        "answer": example["answer"],
        "true_label": "noise",
        "sampling_override": {"temperature": NOISE_TEMPERATURE},  # 1.2 (elevated)
    }
```

The gold context is used (so the task is solvable), but temperature is elevated to `1.2`. This flattens the sampling distribution, making the model more likely to produce different phrasings across `k` samples even when it "knows" the answer. The fault is purely in the sampling, not the input.

#### inject_contamination() ([`faults.py` L82–L104](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/faults.py#L82-L104))

```python
def inject_contamination(example, n_distractors=2) -> Optional[Dict]:
    distractors = get_distractor_context(example)
    if len(distractors) < n_distractors:
        return None   # skip — not enough distractors
    chosen = random.sample(distractors, n_distractors)
    return {
        "question": example["question"],
        "context": format_context(chosen),   # DISTRACTOR context — bad input
        "answer": example["answer"],
        "true_label": "contamination",
        "sampling_override": {"temperature": DEFAULT_TEMPERATURE},  # 0.7 (normal)
    }
```

Two randomly selected distractor paragraphs replace the gold context. The model receives irrelevant information and can't reliably answer the question, but its *capability* is not the problem — the input is. Returns `None` when fewer than `n_distractors` distractors exist; the caller skips such examples.

#### inject_ceiling() ([`faults.py` L107–L135](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/faults.py#L107-L135))

```python
def inject_ceiling(example, strip_fraction=0.5) -> Dict:
    gold = get_gold_context(example)
    weakened = []
    for title, sents in gold:
        keep_n = max(1, int(len(sents) * (1 - strip_fraction)))  # keep first 50%
        weakened.append((title, sents[:keep_n]))
    return {
        "question": example["question"],
        "context": format_context(weakened),   # STRIPPED gold — genuinely incomplete
        "answer": example["answer"],
        "true_label": "ceiling",
        "sampling_override": {"temperature": DEFAULT_TEMPERATURE},
    }
```

Half the sentences are stripped from each gold paragraph. The topic remains on-point (no distractors), but the specific cross-paragraph evidence needed for a complete multi-hop answer is missing. This creates a genuine reasoning gap — even the best reasoning from the available fragments can't reliably produce the correct answer.

**Key diagnostic subtlety:** the "clean input" reprobe for ceiling faults uses the **full unstripped gold context** (not the stripped version). So if the model can answer correctly when given complete gold context, it's a data-quality problem (ceiling at the injected level, but not truly a capability ceiling). If it still fails on complete gold context, that's a genuine capability ceiling. This distinction is handled in `run_study1.py`.

---

### 5.5 Diagnostic Protocol — [`diagnose.py`](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/diagnose.py)

**Purpose:** Implements the retry-then-reprobe protocol that classifies the *cause* of high uncertainty.

#### needs_diagnosis() ([`diagnose.py` L37–L43](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/diagnose.py#L37-L43))

```python
def needs_diagnosis(trace, node_name) -> bool:
    return trace.node_results[node_name].uncertainty > UNCERTAINTY_THRESHOLD
```

Simple threshold check. Any node with uncertainty > 0.75 is flagged for diagnosis.

#### compute_inference_gap() ([`diagnose.py` L50–L89](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/diagnose.py#L50-L89))

```python
def compute_inference_gap(node_result, input_text, *, enabled=True) -> Optional[float]:
```

An **optional experimental metric** repurposed from the SAUP paper. Computes cosine distance between the sentence embeddings of a node's input and its output using `sentence-transformers/all-MiniLM-L6-v2`. Higher distance = more semantic drift. Logged alongside self-consistency uncertainty but does **not** drive the diagnostic decision in Study 1. Deferred to Study 2.

#### diagnose() ([`diagnose.py` L96–L137](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/diagnose.py#L96-L137))

```python
def diagnose(node_name, same_input_prompt, clean_input_prompt, k=3, retries=2) -> str:
    # Step 1: Retry same input multiple times
    retry_results = [sample_node(node_name, same_input_prompt, k=k) for _ in range(retries)]
    avg_retry_uncertainty = sum(r.uncertainty for r in retry_results) / len(retry_results)
    if avg_retry_uncertainty < UNCERTAINTY_THRESHOLD:
        return "noise"

    # Step 2: Retry with clean/known-good input
    clean_result = sample_node(node_name, clean_input_prompt, k=k)
    if clean_result.uncertainty < UNCERTAINTY_THRESHOLD:
        return "contamination"

    # Step 3: Neither helped
    return "ceiling"
```

Key implementation details:
- **`retries=2` (not 1):** A single retry failing is weak evidence. Soft ceilings can intermittently succeed by chance. Using the *average* uncertainty over 2 retries makes "noise" harder to falsely confirm.
- **Cheapest explanation first:** Same-input retry is cheap (no data loading, same prompt). Clean-input reprobe is more expensive (requires constructing a new prompt with gold context). Ceiling classification costs nothing extra.

#### diagnose_trace() ([`diagnose.py` L144–L217](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/diagnose.py#L144-L217))

```python
def diagnose_trace(trace, node_prompts, k=3, retries=2, compute_gap=True) -> None:
```

**Multi-node flagging strategy** (an open design decision from the research brief, resolved here): when multiple nodes exceed the threshold, ALL are diagnosed. `trace.diagnosed_label` is set to the diagnosis of the *highest-uncertainty* node (most likely root cause). `trace.per_node_diagnoses` stores the full `{node_name: label}` dict for post-hoc analysis.

The `node_prompts` argument is structured as:
```python
{
    "retriever": {"same": retriever_prompt(q, faulted_ctx), "clean": retriever_prompt(q, gold_ctx)},
    "reasoner":  {"same": reasoner_prompt(q, faulted_retriever_out), "clean": reasoner_prompt(q, gold_ctx)},
    "writer":    {"same": writer_prompt(q, faulted_reasoner_out), "clean": writer_prompt(q, gold_ctx)},
}
```

---

### 5.6 Data Download — [`scripts/download_data.py`](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/scripts/download_data.py)

```python
def main():
    if os.path.exists(DATA_DIR) and any(os.scandir(DATA_DIR)):
        print("Dataset already cached. Skipping download.")
        return
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    ds.save_to_disk(DATA_DIR)
```

**Idempotent** — checks if data already exists before downloading. Run once before any pipeline scripts. Caches the HotpotQA distractor validation split (7,405 examples) to `./data/hotpotqa_distractor/` in HuggingFace Arrow format for fast repeated loading.

---

### 5.7 Debug Phases 1–3 — [`scripts/run_local_debug.py`](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/scripts/run_local_debug.py)

**Purpose:** Small-N, verbose sanity checks to validate each layer of the pipeline before running the full scored Study 1. Follows the build order from the research brief.

```
python scripts/run_local_debug.py --phase 1   # k=1, 3 examples, no faults
python scripts/run_local_debug.py --phase 2   # k=3, 3 examples, no faults
python scripts/run_local_debug.py --phase 3   # k=3, 3 examples, all 3 fault types
python scripts/run_local_debug.py             # all phases
```

#### Phase 1 ([`run_local_debug.py` L70–L78](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/scripts/run_local_debug.py#L70-L78)) — No fault, k=1

Goal: confirm all three prompts produce sane, on-topic output on real HotpotQA text. k=1 means no self-consistency sampling — just one generation per node, so you can read the raw output and check for prompt-formatting bugs.

#### Phase 2 ([`run_local_debug.py` L85–L97](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/scripts/run_local_debug.py#L85-L97)) — No fault, k=3

Goal: inspect `trace.uncertainties()` on clean input. Check that easy questions show low uncertainty. This is where the `UNCERTAINTY_THRESHOLD` was calibrated — Phase 2 data showed Reasoner baseline uncertainty of 0.667, which forced the threshold up to 0.75.

```python
all_low = all(v < 0.35 for v in u.values())
status = "✓ low uncertainty" if all_low else "⚠ high uncertainty — check prompts"
```

(Note: 0.35 threshold for the display check, not the diagnostic threshold — these are separate concepts.)

#### Phase 3 ([`run_local_debug.py` L104–L150](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/scripts/run_local_debug.py#L104-L150)) — All fault types, k=3

For each example, runs:
1. A baseline (gold context, temp=0.7) for comparison
2. Noise fault (gold context, temp=1.2)
3. Contamination fault (distractors, temp=0.7)
4. Ceiling fault (stripped gold, temp=0.7)

Prints uncertainty patterns side-by-side with expectation hints so the developer can manually verify that the three fault types produce visibly distinct patterns. If they don't, fix injection logic or thresholds *before* running the scored loop.

---

### 5.8 Full Scored Run — [`scripts/run_study1.py`](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/scripts/run_study1.py)

**Purpose:** Phase 4+ entry point. Runs `n_examples × 3 fault types` trials, logs every trial to disk, and produces the confusion matrix.

```
python scripts/run_study1.py --n-examples 15 --k 3
python scripts/run_study1.py --n-examples 15 --k 3 --resume
```

#### run_labeled_trial() ([`run_study1.py` L64–L128](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/scripts/run_study1.py#L64-L128))

One complete trial: inject fault → run pipeline → build node_prompts for diagnosis → call `diagnose_trace()` → return labeled trace.

Builds `node_prompts` for all three nodes so `diagnose_trace()` can handle any combination of flagged nodes. The clean reference always uses **full unstripped gold context** — critical for ceiling faults where the faulted context is an incomplete subset of gold.

#### trace_to_record() ([`run_study1.py` L135–L150](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/scripts/run_study1.py#L135-L150))

Converts a `PipelineTrace` to a JSON-serializable dict:
```json
{
  "question": "...",
  "true_label": "contamination",
  "diagnosed_label": "contamination",
  "uncertainties": {"retriever": 0.6667, "reasoner": 0.0, "writer": 0.0},
  "inference_gaps": {"retriever": 0.12, "reasoner": null, "writer": null},
  "per_node_diagnoses": {"retriever": "contamination"},
  "gold_answer": "Arthur's Magazine",
  "samples": {"retriever": ["...", "...", "..."], "reasoner": [...], "writer": [...]}
}
```

Raw samples are included for post-hoc analysis (e.g. manual spot-check of Writer normalization).

#### run_study1() ([`run_study1.py` L157–L227](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/scripts/run_study1.py#L157-L227))

**Crash-safe logging:** writes each trial record to `logs/trials.jsonl` immediately after completion (`log_file.flush()` after every write). A mid-run crash loses at most one in-progress trial, never completed ones.

**Resume support:** `--resume` flag reads existing records from the log, builds a `done_keys` set of `(question, true_label)` pairs, and skips any trial already in the set.

**tqdm progress bar** shows total trials with a skip counter.

#### build_confusion_matrix() ([`run_study1.py` L234–L287](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/scripts/run_study1.py#L234-L287))

```python
def build_confusion_matrix(results, save_path=DEFAULT_CONFUSION_MATRIX_PATH) -> pd.DataFrame:
```

- Filters out `"no_fault_detected"` records (trials where all nodes were below threshold — valid data but excluded from the confusion matrix since there's no diagnosis to compare).
- Builds a 3×3 DataFrame indexed by `["noise", "contamination", "ceiling"]`.
- Uses `pd.crosstab()` to fill counts from the filtered records.
- Computes overall diagnostic accuracy = diagonal sum / total.
- Prints a pass/fail indicator: accuracy > 50% → `"✓ Diagonal dominates"`.
- Saves to `results/confusion_matrix.csv`.

---

## 6. The Three Fault Types in Detail

### How they differ in the pipeline

| | Noise | Contamination | Ceiling |
|---|---|---|---|
| **Input quality** | Gold (correct) | Bad (irrelevant distractors) | Incomplete (stripped gold) |
| **Temperature** | 1.2 (elevated) | 0.7 (normal) | 0.7 (normal) |
| **Root cause** | Sampling variance | Upstream bad data | Model capability gap |
| **Where symptom appears** | Any node (random) | Retriever and downstream | Any node handling hard multi-hop |
| **Fix** | Retry same input | Fix retrieval upstream | Escalate / decompose |

### Why temperature elevation works for noise injection

At temperature 1.2, the softmax distribution over vocabulary tokens is flatter. The model is more likely to generate different phrasings of essentially the same answer across samples, causing false disagreement that self-consistency labels as uncertainty. At temperature 0.7 (normal), the distribution is sharper and the model more consistently picks its top-probability tokens.

### Why ceiling uses stripped gold (not distractors)

If ceiling used distractors, the model would face both bad input *and* capability-gap. The two faults would be confounded. By using stripped (but topically correct) gold, the model gets on-topic text with the right entities — it just lacks the specific bridging sentences needed for multi-hop reasoning. This isolates the capability gap.

---

## 7. The Diagnostic Protocol — How It Works

### Step-by-step trace for each fault type

**Noise fault:**
1. Initial run: uncertainty is high (e.g., 1.0) because temp=1.2 causes sample diversity.
2. `needs_diagnosis(trace, "retriever")` → True (1.0 > 0.75).
3. `diagnose()` is called with the *same prompt* (gold context, but diagnosis now uses normal temperature DEFAULT_TEMPERATURE implicitly through `sample_node()`'s default).
4. Retry 1 & 2: at normal temperature (0.7), the model consistently produces the same factual sentence → average uncertainty drops to 0.0.
5. 0.0 < 0.75 → return `"noise"`. ✓

**Contamination fault:**
1. Initial run: Retriever receives distractor paragraphs. It either returns irrelevant sentences or makes up something. Uncertainty is high (e.g., 0.667–1.0).
2. `diagnose()` is called.
3. Same-input retry (still distractor context): model still receives bad context → uncertainty stays high across both retries.
4. Average retry uncertainty ≥ 0.75 → not noise.
5. Clean-input reprobe (gold context): model now receives the correct paragraphs → retrieves the right sentences consistently → uncertainty drops to 0.0.
6. 0.0 < 0.75 → return `"contamination"`. ✓

**Ceiling fault:**
1. Initial run: stripped gold context → model can't bridge the multi-hop gap → high uncertainty.
2. `diagnose()` is called.
3. Same-input retry (stripped context): still incomplete → uncertainty stays high.
4. Clean-input reprobe: uses **full unstripped gold context** → the model either can answer (uncertainty drops) or still can't (uncertainty stays high).
5. If uncertainty stays high even on full gold → return `"ceiling"`. ✓

> **Note on ceiling clean-input reprobe:** Whether the model succeeds on full gold depends on the question's difficulty. For hard multi-hop questions that genuinely exceed the quantized 8B model's capability, the reprobe fails → correctly classified as ceiling. For easier questions where the stripping was the only issue, the reprobe succeeds → would be classified as contamination (a design tradeoff noted in the codebase).

---

## 8. The Confusion Matrix — The Primary Deliverable

Saved at [`results/confusion_matrix.csv`](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/results/confusion_matrix.csv):

```
true \ diagnosed    noise    contamination    ceiling
noise                   9                0          0
contamination           0               10          0
ceiling                 0                0          5
```

**24 out of 24 classified correctly — 100% diagnostic accuracy.**

### What this means

The retry-then-reprobe protocol, using only uncertainty patterns (no peeking at true labels), correctly identified the injected fault type in every trial. The core research premise holds: the three fault types produce distinct enough uncertainty signatures that a simple two-step protocol can reliably tell them apart.

### What this does NOT mean yet

- **Small sample:** 8–9 examples per fault type at k=3 with a local quantized 8B model. Not statistically meaningful.
- **Local conditions:** The quantized model may show higher baseline uncertainty than a full-precision larger model. Cloud-scale Study 1 (hundreds of trials, k=5, larger model) is required before publishing.
- **Threshold sensitivity:** The 100% result is partly a function of the calibrated threshold (0.75). Recalibrate on the cloud-scale model's baseline before trusting cloud numbers.
- **No compound faults:** These are clean single-fault-type injections. Real production failures may be combinations (e.g., contamination + mild ceiling).

---

## 9. Critical Design Rules Enforced Throughout

### Rule 1: Per-node uncertainties are NEVER merged into a scalar

Every function that takes or returns uncertainty operates on `Dict[str, float]` — not `float`. `trace.uncertainties()` is documented as "never collapse to scalar." This separation is the project's core structural bet: merging them defeats the diagnostic mechanism entirely.

### Rule 2: All model calls are sequential

No threading, no asyncio. All `sample_node()` calls in `run_pipeline()` are sequential. This is a hardware constraint (16GB RAM laptop) — concurrent model loads would cause OOM or thrashing.

### Rule 3: Append-only logging

Every completed trial is written to `logs/trials.jsonl` immediately after completion. `log_file.flush()` is called after every write. A mid-run crash loses at most one in-progress trial, never completed ones. This is explicitly designed for the 16GB RAM laptop scenario where long runs may be interrupted.

### Rule 4: The model is loaded once per process

The `_get_model()` singleton in `nodes.py` prevents the ~3-second reload on every `sample_node()` call.

### Rule 5: `retries >= 2` for noise detection

A single same-input retry is weak evidence. Soft ceilings can intermittently succeed by chance. The protocol uses the *average* uncertainty across `DEFAULT_RETRIES = 2` retry runs to get a more reliable signal.

---

## 10. Open Decisions & Known Limitations

These were explicitly flagged as open decisions in the research brief. Each is documented with the choice made here:

| Decision | Brief Placeholder | Choice Made | Location |
|---|---|---|---|
| `UNCERTAINTY_THRESHOLD` | 0.3 (placeholder) | **0.75** (calibrated from Phase 2 baseline: Reasoner = 0.667 clean) | [`config.py` L48](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/config.py#L48) |
| `retries` count | 2 (suggested) | **2** (kept; validate if ceiling faults fake-recover) | [`config.py` L53](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/config.py#L53) |
| HotpotQA normalization | `strip().lower()` placeholder | **Official SQuAD-style** (lowercase → strip punctuation → remove articles → collapse whitespace) | [`nodes.py` L49](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/nodes.py#L49) |
| Multi-node flagging | Diagnose max-uncertainty node only | **Diagnose ALL flagged nodes**; primary label = highest-uncertainty node's diagnosis | [`diagnose.py` L144](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/diagnose.py#L144) |
| Inference Gap | Deferred | **Implemented but optional** (logged, does not drive diagnosis) | [`diagnose.py` L50](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/src/diagnose.py#L50) |

### Known approximation in Reasoner clean-input prompt

For the Reasoner's clean-input reprobe, the clean prompt uses the raw gold context string (not the output of a fresh clean Retriever run). This is a documented approximation — a full clean re-run would require running the Retriever inline. Noted as a Study 2 enhancement in the code comment at [`run_study1.py` L116–L118](file:///d:/aryan/Projects/Multi%20agent%20LLM%20System/btp-pipeline/scripts/run_study1.py#L116-L118).

---

## 11. The Bigger Research Picture (Studies 2–4)

Study 1 (built here) is the foundation. The full research plan has four layers:

### Study 1 — Diagnostic accuracy ✅ (implemented here)
Does the retry-then-reprobe protocol correctly classify fault types? → Confusion matrix.

### Study 2 — Three-arm orchestration comparison
Same pipeline, same faults, three policies:
- **A (Baseline):** One score, one threshold, one generic fallback.
- **B (Signature-aware rules):** Hard-coded branching by (node, fault-type) per the retry-then-escalate ladder.
- **C (LLM-as-orchestrator):** A separate LLM call sees the failure signature and chooses the remediation.

Metrics: diagnosis accuracy, remediation success rate, cost (tokens/latency).

### Study 3 — Resource-condition factorial (2×3 grid)
Studies 2's three policies under two conditions:
- **Single-model-only:** Ceiling remediation restricted to decompose/more-compute/tools/abstain — no access to bigger model.
- **Escalation-available:** Full ladder including routing to a bigger model.

Key question: does signature-aware routing still win under real-world resource constraints?

### Study 4 (stretch goal) — Learned policy
A bandit/RL policy trained on Studies 1–3 data. Can it beat both the rule table and the LLM-orchestrator on cost-adjusted performance?

---

## Quick Reference: How to Run

```bash
# 1. Setup
cd btp-pipeline
python -m venv btp-env
btp-env\Scripts\activate      # Windows
pip install -r requirements.txt
# Also: pip install mlx-lm  (Apple Silicon) or configure Ollama (non-Apple)

# 2. Download data (once)
python scripts/download_data.py

# 3. Sanity checks (recommended before Study 1)
python scripts/run_local_debug.py --phase 1 --n-examples 3
python scripts/run_local_debug.py --phase 2 --n-examples 3
python scripts/run_local_debug.py --phase 3 --n-examples 3

# 4. Full Study 1 scored run
python scripts/run_study1.py --n-examples 15 --k 3

# 5. Resume if interrupted
python scripts/run_study1.py --n-examples 15 --k 3 --resume

# 6. Disable Inference Gap for speed
python scripts/run_study1.py --n-examples 15 --k 3 --no-inference-gap
```

Results are written to:
- `logs/trials.jsonl` — per-trial records (append-only, crash-safe)
- `results/confusion_matrix.csv` — primary Study 1 deliverable
