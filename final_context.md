# BTP-I Project: Complete & Updated Context
**Project**: Signature-Aware Orchestration for Uncertainty Propagation in Multi-Agent LLM Systems
**Author**: Aryan Mehra
**Branch**: `updated-methods` (pushed to GitHub — AryanM65/Multi-Agent-LLM-System)
**Last updated**: 2026-09-24

---

## Table of Contents
1. [Research Question & Goal](#1-research-question--goal)
2. [Repository Layout (Current)](#2-repository-layout-current)
3. [System Architecture](#3-system-architecture)
4. [Model & Backend](#4-model--backend)
5. [Dataset: HotpotQA](#5-dataset-hotpotqa)
6. [Topology Engine (Phase 1 — NEW)](#6-topology-engine-phase-1--new)
7. [Uncertainty Calculation (Phase 2 — REDESIGNED)](#7-uncertainty-calculation-phase-2--redesigned)
8. [Fault Injection (Phase 3 — REDESIGNED)](#8-fault-injection-phase-3--redesigned)
9. [Diagnostic Protocol](#9-diagnostic-protocol)
10. [File-by-File Code Reference](#10-file-by-file-code-reference)
11. [Data Schemas: JSONL Logs](#11-data-schemas-jsonl-logs)
12. [Study 1 Results (Baseline)](#12-study-1-results-baseline)
13. [Key Research Findings](#13-key-research-findings)
14. [Invariants That Must Never Be Broken](#14-invariants-that-must-never-be-broken)
15. [Current State & What's Next](#15-current-state--whats-next)
16. [Command Cheatsheet](#16-command-cheatsheet)
17. [Dependency Map](#17-dependency-map)

---

## 1. Research Question & Goal

**Core question**: Can an automated diagnostic system, running inside a multi-agent LLM pipeline, reliably distinguish between three categories of failure — sampling noise, input contamination, and model capability ceiling — purely from the *pattern* of uncertainty signals propagating across pipeline nodes?

**Why it matters**: Multi-agent LLM systems fail silently. A single corrupted input at node 1 degrades nodes 2 and 3 downstream, but the root cause is invisible unless the system can trace uncertainty back to its origin. This project builds the tracing mechanism.

**Research hypothesis**: The three fault types produce structurally distinct uncertainty signatures across the Retriever→Reasoner→Writer chain:
- **Noise** → high uncertainty at exactly the targeted node, resolves on same-input retry
- **Contamination** → high uncertainty at the targeted node, persists on same-input retry, resolves on clean-input retry
- **Ceiling** → high uncertainty at the targeted node, persists through all retries (no good input exists)

**Scope (current)**: Study 1 — 3-node linear chain, single-hop HotpotQA-distractor questions, lexical uncertainty.
**Scope (next)**: Study 2 — cascade analysis, per-node threshold calibration, semantic uncertainty; Study 3 — non-linear topologies, GNN-based propagation.

---

## 2. Repository Layout (Current)

```
Multi agent LLM System/
├── btp-pipeline/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py          ← constants, fault taxonomy, paths
│   │   ├── topology.py        ← NEW: NodeSpec, Topology, DAG engine
│   │   ├── uncertainty.py     ← NEW: semantic/Jaccard/extraction metrics
│   │   ├── nodes.py           ← NodeResult (extended), PipelineTrace, sample_node
│   │   ├── pipeline.py        ← REWRITTEN: generic run_pipeline(topo,...), fault hooks
│   │   ├── faults.py          ← REWRITTEN: make_rng, node-targeted injectors
│   │   ├── diagnose.py        ← retry-then-reprobe + verify_against_baseline
│   │   └── pipelines/
│   │       ├── __init__.py
│   │       ├── retriever_pipeline.py   ← standalone single-node CLI
│   │       ├── reasoner_pipeline.py    ← standalone single-node CLI
│   │       └── writer_pipeline.py      ← standalone single-node CLI
│   ├── scripts/
│   │   ├── download_data.py           ← cache HotpotQA locally (run once)
│   │   ├── run_local_debug.py         ← UPDATED: phases 1/2/3/3x sanity checks
│   │   ├── run_study1.py              ← REWRITTEN: full scored batch run
│   │   ├── rescore_study1.py          ← NEW: retroactive re-scoring of old logs
│   │   └── run_agent.py               ← UPDATED: unified CLI (uses topology engine)
│   ├── data/
│   │   └── hotpotqa_distractor/       ← HuggingFace dataset cached on disk
│   ├── logs/
│   │   ├── trials.jsonl               ← append-only trial log (Study 1)
│   │   └── skipped_trials.jsonl       ← NEW: skipped trials with reason field
│   └── results/
│       ├── confusion_matrix.csv       ← 3×3 from Study 1
│       ├── study1_rescore.jsonl       ← NEW: per-trial retroactive re-scores
│       └── study1_rescore_table.csv   ← NEW: lexical vs semantic comparison table
├── final_context.md                   ← this file
├── explain.md
└── explain2.md
```

---

## 3. System Architecture

### 3.1 High-Level Flow

```
Question + Context
      │
      ▼
┌─────────────┐     self-consistency     ┌────────────────────────┐
│  RETRIEVER  │ ─── k=3 samples ──────▶  │  NodeResult            │
│  (role)     │     normalize → modal    │  .uncertainty (lexical) │
│             │     parse items → Jaccard│  .uncertainty_jaccard  │
└──────┬──────┘                          └────────────────────────┘
       │ output (selected evidence)
       ▼
┌─────────────┐     self-consistency     ┌────────────────────────┐
│  REASONER   │ ─── k=3 samples ──────▶  │  NodeResult            │
│  (role)     │     extract FINAL ANSWER │  .uncertainty (lexical) │
│             │     → semantic cluster   │  .uncertainty_semantic  │
└──────┬──────┘                          │  .conclusions          │
       │ output (reasoning chain)        └────────────────────────┘
       ▼
┌─────────────┐     self-consistency     ┌────────────────────────┐
│   WRITER    │ ─── k=3 samples ──────▶  │  NodeResult            │
│  (role)     │     semantic cluster     │  .uncertainty (lexical) │
│             │     over raw outputs     │  .uncertainty_semantic  │
└──────┬──────┘                          └────────────────────────┘
       │
       ▼
  PipelineTrace
  {node_results: {retriever, reasoner, writer},
   topology_id: "chain_3node_default",
   true_label, diagnosed_label, per_node_diagnoses}
       │
       ▼ (if any node uncertainty > UNCERTAINTY_THRESHOLD)
  diagnose_trace()  →  retry-then-reprobe protocol
       │
       ▼
  "noise" | "contamination" | "ceiling"
```

### 3.2 Topology as Data

The pipeline is no longer hardcoded. A `Topology` is a DAG of `NodeSpec` objects. The current 3-node chain is one specific instance:

```python
Topology(
  nodes={
    "retriever": NodeSpec("retriever", "retriever", RETRIEVER_INSTRUCTION),
    "reasoner":  NodeSpec("reasoner",  "reasoner",  REASONER_INSTRUCTION),
    "writer":    NodeSpec("writer",    "writer",    WRITER_INSTRUCTION),
  },
  edges=[("retriever","reasoner"), ("reasoner","writer")],
  topology_id="chain_3node_default",
)
```

`run_pipeline(topo, question, context, k, temperature, fault_config)` uses `topological_order()` (Kahn's algorithm) to determine execution order, then iterates and calls `sample_node()` per node.

### 3.3 Uncertainty Formula (per node)

**Lexical (all nodes, always computed)**:
```
uncertainty_lexical = 1 - (count_of_modal_normalized_answer / k)
```

**Semantic (Reasoner + Writer)**:
```
1. Embed k texts with all-MiniLM-L6-v2
2. Greedy cluster: assign to first cluster with cosine_sim >= 0.85
3. semantic_uncertainty = H(cluster_size_distribution) / log(k)
   where H = Shannon entropy
```

**Jaccard (Retriever, multi-item outputs only)**:
```
1. Parse each of k outputs into a set of item strings
2. Compute all pairwise Jaccard similarities
3. jaccard_uncertainty = 1 - mean_pairwise_jaccard
```

---

## 4. Model & Backend

| Property | Value |
|---|---|
| Model | `mlx-community/Qwen3-8B-4bit` |
| Backend | `mlx_lm` (Apple Silicon / Metal) |
| Quantisation | 4-bit MLX safetensors |
| Chat template | Qwen3 instruct (`apply_chat_template`, `enable_thinking=False`) |
| Token budgets | Retriever/Writer: 128 tokens; Reasoner: 200 tokens |
| Load strategy | Module-level singleton (`_get_model()`), loaded once per process |
| Default k | 3 samples per node |
| Default temperature | 0.7 (normal), 1.2 (noise fault) |
| Threading | Sequential only — no asyncio, no threading |

The `enable_thinking=False` flag suppresses Qwen3's internal `<think>` chain-of-thought scratchpad. Without it, the scratchpad would appear in each sample and confuse self-consistency comparison (two samples might have identical conclusions but completely different scratchpad text, causing false disagreement).

The Embedder (`all-MiniLM-L6-v2` via `sentence_transformers`) is also a singleton, lazy-loaded in `uncertainty.py::get_embedder()` and shared by all callers (including `diagnose.py::compute_inference_gap`).

---

## 5. Dataset: HotpotQA

- **Split**: `distractor` / `validation` — 7,405 examples
- **Loaded via**: HuggingFace `datasets.load_from_disk()` after `scripts/download_data.py`
- **Format per example**:
  ```python
  {
    "_id": str,
    "question": str,
    "answer": str,
    "context": {"title": [str,...], "sentences": [[str,...],...] },
    "supporting_facts": {"title": [str,...], "sent_id": [int,...] }
  }
  ```
- **Gold context** = paragraphs whose title is in `supporting_facts["title"]`
- **Distractor context** = all other paragraphs (4–6 per example)
- **Normalisation** (`hotpotqa_normalize`): lowercase → strip punctuation → remove articles (a/an/the) → collapse whitespace — matches official HotpotQA SQuAD-style evaluation

---

## 6. Topology Engine (Phase 1 — NEW)

### 6.1 `src/topology.py`

**`NodeSpec`** — specifies one node:
```python
@dataclass
class NodeSpec:
    node_id: str    # unique within topology e.g. "retriever", "R1"
    role: str       # "retriever" | "reasoner" | "writer"
    instruction: str  # system/task instruction text for this node's prompts
```

**`Topology`** — the DAG:
```python
@dataclass
class Topology:
    nodes: Dict[str, NodeSpec]
    edges: List[Tuple[str, str]]   # (from_node_id, to_node_id)
    topology_id: str = ""
```

**Validation chain** (`validate_topology`):
1. All nodes have valid roles
2. All edge endpoints reference real node IDs
3. Role ordering non-decreasing along every edge (`ROLE_RANK = {retriever:0, reasoner:1, writer:2}`)
4. Graph is acyclic (Kahn's topological sort — raises on cycle)
5. No isolated nodes in multi-node topologies

**Graph utilities**:
- `topological_order(topo)` → `List[str]` (execution order)
- `parents_of(topo, node_id)` → `List[str]` (which nodes feed this one)
- `children_of(topo, node_id)` → `List[str]`
- `source_nodes(topo)` → nodes with no incoming edges (read raw context)
- `sink_nodes(topo)` → nodes with no outgoing edges (final answer producers)

**Factory**: `default_chain_topology(ret_instr, rea_instr, wri_instr) → Topology`

### 6.2 `src/pipeline.py` (rewritten)

**`build_prompt(topo, node_id, outputs, question, context, fault_config)`**:
- Source nodes (no parents): builds `"{instruction}\n\nParagraphs:\n{context}\n\nQuestion: {question}"`
- Non-source nodes: assembles parent outputs as `[Input from {pid}]: {output}` block
- Phase 3 hook: if `fault_config["_corrupted_input"]` is set and this node is the target, substitutes the corrupted input into the prompt block

**`apply_fault_to_prompt(prompt, temperature, fault_config) → (prompt, temperature)`**:
- `noise` → returns `(prompt, NOISE_TEMPERATURE)` — only this node's temperature changes
- `contamination`/`ceiling` → no-op (corruption already applied in `build_prompt`)

**`run_pipeline(topo, question, context, k, temperature, fault_config=None) → PipelineTrace`**:
```python
validate_topology(topo)
order = topological_order(topo)
for node_id in order:
    prompt = build_prompt(topo, node_id, outputs, question, context, fault_config)
    if fault_config and fault_config["target_node"] == node_id:
        prompt, node_temperature = apply_fault_to_prompt(prompt, temperature, fault_config)
    result = sample_node(node_id, prompt, k, node_temperature,
                         normalize_fn=get_normalizer(role), role=role)
    outputs[node_id] = result.output
    trace.add(result)
return trace
```

**Backward-compatible prompt builders** (preserved):
- `retriever_prompt(question, context)` → str
- `reasoner_prompt(question, evidence)` → str
- `writer_prompt(question, reasoning)` → str

**`default_topology()` → `Topology`** — one-stop call for the canonical chain.

---

## 7. Uncertainty Calculation (Phase 2 — REDESIGNED)

### 7.1 Problem statement (why this was redesigned)

| Node | Old metric | Problem |
|---|---|---|
| Reasoner | Exact-match on full CoT output | 0.667 baseline uncertainty on clean data — CoT phrasing varies across samples even when the conclusion is identical |
| Retriever | Exact-match on full output string | "agree on 2/3 items" = "agree on 0/3 items" (all-or-nothing) |
| Writer | Exact-match | Adequate for most cases; degrades on paraphrase (yes/no, comparison questions) |

### 7.2 Design principle

**The old lexical metric is never deleted.** Every `NodeResult` stores `uncertainty` (lexical) as before. New metrics are stored in additional optional fields, enabling:
1. Before/after comparison against existing Study 1 logs
2. Richer per-node feature vectors for future GNN work

### 7.3 Extended `NodeResult` (in `src/nodes.py`)

```python
@dataclass
class NodeResult:
    node_name: str
    output: str
    uncertainty: float              # LEXICAL — always populated, unchanged

    # Phase 2 additions (all Optional, default None):
    uncertainty_semantic: Optional[float]        # Reasoner, Writer
    uncertainty_jaccard: Optional[float]         # Retriever (multi-item)
    conclusions: Optional[List[str]]             # Reasoner: extracted FINAL ANSWERs
    item_frequencies: Optional[Dict[str, float]] # Retriever: per-item inclusion rate
    samples: List[str]
    inference_gap: Optional[float]               # Study 2 / experimental
```

### 7.4 REASONER_INSTRUCTION (updated)

```python
REASONER_INSTRUCTION = (
    "You are a Reasoning agent. Given the evidence below, reason step by step "
    "to derive the answer to the question. Show your full chain of thought.\n\n"
    "End your response with a final line in exactly this format:\n"
    "FINAL ANSWER: <your one-sentence conclusion>"
)
```

`extract_conclusion(raw_output)` in `uncertainty.py` parses the `FINAL ANSWER:` marker.
If the model doesn't follow format, falls back to the last non-empty line.
`check_conclusion_marker(raw_output)` → bool, for monitoring prompt compliance rate.

### 7.5 `sample_node` role-branching (in `src/nodes.py`)

```python
def sample_node(node_name, prompt, k, temperature, normalize_fn, role=""):
    samples = [generate_once(...) for _ in range(k)]

    # Lexical (always)
    normed = [normalize_fn(s) for s in samples]
    modal, top_count = Counter(normed).most_common(1)[0]
    uncertainty_lexical = 1 - top_count/k
    output = first sample whose normalized form == modal

    if role == "reasoner":
        conclusions = [extract_conclusion(s) for s in samples]
        uncertainty_semantic = semantic_uncertainty(conclusions)

    elif role == "writer":
        uncertainty_semantic = semantic_uncertainty(samples)

    elif role == "retriever":
        item_sets = [parse_retrieved_items(s) for s in samples]
        if any(len(s) > 1 for s in item_sets):
            uncertainty_jaccard = jaccard_uncertainty(item_sets)
            item_frequencies = per_item_inclusion_frequency(item_sets)

    return NodeResult(...)
```

### 7.6 Retroactive re-scoring

`scripts/rescore_study1.py` reads the existing `logs/trials.jsonl` (Study 1), applies new metrics to saved raw samples, and produces:
- `results/study1_rescore.jsonl` — per-trial re-scored records
- `results/study1_rescore_table.csv` — aggregated comparison table (lexical vs semantic per fault_type × node)

For old Reasoner samples (no `FINAL ANSWER:` marker), uses `retroactive_extract_conclusion()` — heuristic keyword search (therefore/thus/hence/the answer is) rather than the new marker-based extraction.

---

## 8. Fault Injection (Phase 3 — REDESIGNED)

### 8.1 Problems with the original design

| # | Problem | Fix |
|---|---|---|
| 1 | All faults implicitly targeted only the Retriever | Full 3×3 fault grid: (type × target_node) |
| 2 | Noise scope ambiguous (pipeline-wide vs node-scoped?) | Noise is now explicitly node-scoped via `apply_fault_to_prompt` |
| 3 | Contamination: fixed 2-paragraph swap, random distractors | Proportional swap (50% of gold) + plausibility-ranked distractors |
| 4 | Ceiling: not verified that answer was actually removed | `inject_ceiling` returns `None` if answer survives stripping |
| 5 | No per-question no-fault control baseline | Control trial always runs first, baseline stored per question |
| 6 | Random corruption not reproducibly seeded | `make_rng(question_id, fault_type, target_node)` seeds every injector |

### 8.2 Fault taxonomy

```python
FAULT_TYPES  = ["noise", "contamination", "ceiling"]
TARGET_NODES = ["retriever", "reasoner", "writer"]

build_fault_conditions() → [
    None,                                         # clean control (always first)
    {"type": "noise",         "target_node": "retriever"},
    {"type": "noise",         "target_node": "reasoner"},
    {"type": "noise",         "target_node": "writer"},
    {"type": "contamination", "target_node": "retriever"},
    {"type": "contamination", "target_node": "reasoner"},
    {"type": "contamination", "target_node": "writer"},
    {"type": "ceiling",       "target_node": "retriever"},
    {"type": "ceiling",       "target_node": "reasoner"},
    {"type": "ceiling",       "target_node": "writer"},
]
# 10 conditions per question = 1 control + 9 fault trials
```

### 8.3 Reproducibility: `make_rng`

```python
def make_rng(question_id, fault_type, target_node) -> random.Random:
    seed_str = f"{question_id}_{fault_type}_{target_node}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)
```

Every injector takes `rng: random.Random` as a required argument. No calls to the global `random` module anywhere in `faults.py`.

### 8.4 Noise — node-scoped temperature elevation

`inject_noise(example, target_node, rng) → Dict`
Returns a `fault_config` dict. `apply_fault_to_prompt` in `pipeline.py` raises *only that node's* temperature to `NOISE_TEMPERATURE = 1.2`. All other nodes run at 0.7.

Diagnostic expectation: uncertainty ↑ at target → recovers on same-input retry at normal temperature → classified as NOISE.

### 8.5 Contamination — two variants

**Retriever target** (`inject_contamination_retriever`):
- Swaps `max(1, round(len(gold) * 0.5))` gold paragraphs with distractors
- Optional plausibility ranking: if an embedder is provided, the distractor most similar to the gold paragraph it replaces is preferred (from the top-k candidates)
- Returns `None` if insufficient distractors (caller logs to `skipped_trials.jsonl`)

**Reasoner / Writer target** (`inject_contamination_downstream`):
- The upstream parent node ran *cleanly*; its output is then replaced with a plausible-but-wrong substitute from a pre-built `substitute_bank`
- The bank is built from clean outputs of *other* questions (collected once before the main loop in `run_study1.py`)
- Stored as `fault_config["_corrupted_input"]`; consumed by `build_prompt` when assembling the targeted node's prompt

### 8.6 Ceiling — answer-survival verification required

**Retriever target** (`_inject_ceiling_retriever`):
- Uses `rng.sample()` to choose which sentences to keep (50% strip by default)
- Checks `answer.lower() in stripped_context.lower()`
- If the answer string survived: returns `None` → caller writes to `skipped_trials.jsonl` with `reason: "ceiling_answer_survived"`

**Reasoner / Writer target** (`_inject_ceiling_downstream`):
- Hardens the node's instruction (forcing information loss):
  - Reasoner: "AT MOST 2 reasoning steps"
  - Writer: "AT MOST 5 WORDS"
- Verification that this degrades performance is empirical (sanity check 3.8d), not checkable statically

### 8.7 Per-question control + baseline verification

```python
# run_study1.py trial loop (per question):
1. Run clean control trial → PipelineTrace → store baseline_uncertainties
2. For each fault_config in build_fault_conditions():
   - make_rng(question_id, type, target_node)
   - inject(...) → fault_config (or None → log skip with reason)
   - run_pipeline(topo, ..., fault_config=fault_config)
   - verify_against_baseline(observed_u, baseline_u, fault_config)
   - log_trial(record)  # includes verification z-score
```

**`verify_against_baseline`** (in `diagnose.py`):
```python
def verify_against_baseline(observed, baseline, fault_config,
                             assumed_std=0.10, z_thresh=1.0) -> Dict:
    z = (observed - baseline) / assumed_std
    return {"target_deviated": z > z_thresh, "z_score": z,
            "observed": observed, "baseline": baseline}
```

> TODO after ~15-20 control trials: replace `assumed_std=0.10` with an empirically estimated per-node standard deviation.

---

## 9. Diagnostic Protocol

### 9.1 Retry-then-Reprobe (unchanged logic, updated calls)

```
If uncertainty > UNCERTAINTY_THRESHOLD (0.75):

  Step 1: Retry same input × DEFAULT_RETRIES (2) at normal temperature
          → avg_retry_uncertainty < 0.75 → NOISE

  Step 2: Retry with known-good (clean gold) input
          → clean_result.uncertainty < 0.75 → CONTAMINATION

  Step 3: Neither helped → CEILING
```

The `diagnose()` function now passes `role` through to `sample_node()` so Phase 2 metrics are also computed during retry runs.

### 9.2 Multi-node flagging

When multiple nodes exceed the threshold, ALL are diagnosed. `PipelineTrace.diagnosed_label` = the diagnosis of the highest-uncertainty flagged node (most likely root cause). `PipelineTrace.per_node_diagnoses` stores the full `{node_name: label}` map.

### 9.3 Threshold calibration

```
UNCERTAINTY_THRESHOLD = 0.75

Calibrated from Phase 2 baseline (3 questions, k=3, clean gold context):
  Retriever  lexical uncertainty: 0.000
  Reasoner   lexical uncertainty: 0.667  ← inherent CoT phrasing variance
  Writer     lexical uncertainty: 0.000

Threshold set at 0.75 (just above the Reasoner's 0.667 baseline).

TODO Study 2: replace global threshold with per-node vector:
  {"retriever": 0.30, "reasoner": 0.75, "writer": 0.30}
```

### 9.4 Inference Gap (Study 2 / experimental)

`compute_inference_gap(node_result, input_text)` in `diagnose.py`:
- Embeds `input_text` and `node_result.output` using the shared embedder
- Returns `1 - cosine_similarity` (semantic drift between what came in and what came out)
- Stored in `NodeResult.inference_gap`
- **Does not drive diagnostic decisions** — logged for exploratory Study 2 analysis
- Disable with `--no-inference-gap` flag in `run_study1.py`

---

## 10. File-by-File Code Reference

### `src/config.py`
Single source of truth for all constants.

```python
MODEL = "mlx-community/Qwen3-8B-4bit"
DEFAULT_K = 3
DEFAULT_TEMPERATURE = 0.7
NOISE_TEMPERATURE = 1.2
MAX_TOKENS = 128
MAX_TOKENS_REASONER = 200
UNCERTAINTY_THRESHOLD = 0.75
DEFAULT_RETRIES = 2

FAULT_TYPES = ["noise", "contamination", "ceiling"]
TARGET_NODES = ["retriever", "reasoner", "writer"]
build_fault_conditions() → List[Optional[dict]]  # 10 conditions

DATA_DIR = "./data/hotpotqa_distractor"
LOG_DIR = "./logs"
RESULTS_DIR = "./results"
DEFAULT_LOG_PATH = "./logs/trials.jsonl"
DEFAULT_SKIP_LOG_PATH = "./logs/skipped_trials.jsonl"
DEFAULT_CONFUSION_MATRIX_PATH = "./results/confusion_matrix.csv"
```

---

### `src/topology.py` (**NEW**)
DAG data structures and validation.

Key exports:
- `ROLE_RANK: Dict[str, int]` — `{retriever:0, reasoner:1, writer:2}`
- `NodeSpec(node_id, role, instruction)` — dataclass, validates role in `__post_init__`
- `Topology(nodes, edges, topology_id)` — dataclass
- `validate_topology(topo)` — 5-check structural validation, raises `ValueError`
- `topological_order(topo) → List[str]` — Kahn's algorithm, raises on cycle
- `parents_of(topo, node_id) → List[str]`
- `children_of / source_nodes / sink_nodes` — graph traversal utilities
- `default_chain_topology(ret_instr, rea_instr, wri_instr) → Topology`

---

### `src/uncertainty.py` (**NEW**)
Role-specific uncertainty metrics.

Key exports:
- `get_embedder()` — lazy singleton `SentenceTransformer("all-MiniLM-L6-v2")`
- `extract_conclusion(raw_output) → str` — parses `FINAL ANSWER:` marker, last-line fallback
- `check_conclusion_marker(raw_output) → bool` — compliance monitor
- `semantic_uncertainty(texts, sim_threshold=0.85) → float` — normalized entropy over embedding clusters
- `jaccard_uncertainty(item_sets) → float` — 1 − mean pairwise Jaccard similarity
- `per_item_inclusion_frequency(item_sets) → Dict[str, float]` — per-item appearance rate
- `parse_retrieved_items(raw_output) → List[str]` — splits on newlines then commas
- `retroactive_extract_conclusion(raw_output) → str` — heuristic for old Study 1 logs (no marker)

---

### `src/nodes.py` (updated)
Self-consistency sampling engine.

Key exports:
- `hotpotqa_normalize(text) → str` — official SQuAD-style normalization
- `default_normalize` — alias for backward compatibility
- `get_normalizer(role) → Callable` — returns `hotpotqa_normalize` for all roles (hook for future differentiation)
- `NodeResult` — extended dataclass (see §7.3)
- `PipelineTrace` — extended with `topology_id: str = ""`; adds `semantic_uncertainties()` method
- `sample_node(node_name, prompt, k, temperature, normalize_fn, role="") → NodeResult` — role-branched

---

### `src/pipeline.py` (rewritten)
Generic topology execution engine.

Key exports:
- `RETRIEVER_INSTRUCTION`, `REASONER_INSTRUCTION`, `WRITER_INSTRUCTION` — module-level constants
- `retriever_prompt / reasoner_prompt / writer_prompt` — backward-compatible builders
- `build_prompt(topo, node_id, outputs, question, context, fault_config) → str`
- `apply_fault_to_prompt(prompt, temperature, fault_config) → (str, float)`
- `run_pipeline(topo, question, context, k, temperature, fault_config=None) → PipelineTrace`
- `default_topology() → Topology` — wraps `default_chain_topology` with module-level instructions

---

### `src/faults.py` (rewritten)
Node-targeted, reproducibly-seeded fault injectors.

Key exports (helpers preserved unchanged):
- `get_gold_context(example) → List[Tuple[str, List[str]]]`
- `get_distractor_context(example) → List[Tuple[str, List[str]]]`
- `format_context(paragraphs) → str`

New:
- `make_rng(question_id, fault_type, target_node) → random.Random` — MD5-seeded
- `split_sentences(paragraph_text) → List[str]`
- `inject_noise(example, target_node, rng) → Dict` — always returns (no failure case)
- `inject_contamination_retriever(example, rng, swap_fraction=0.5, embedder=None, top_k=3) → Optional[Dict]`
- `inject_contamination_downstream(example, target_node, clean_parent_output, substitute_bank, rng, embedder=None, top_k=5) → Optional[Dict]`
- `inject_ceiling(example, target_node, rng, strip_fraction=0.5) → Optional[Dict]` — dispatches to `_inject_ceiling_retriever` or `_inject_ceiling_downstream`
- `_inject_ceiling_retriever` — strips sentences, verifies answer removed, returns None if answer survived
- `_inject_ceiling_downstream` — hardened instruction variant for Reasoner/Writer targets

---

### `src/diagnose.py` (updated)
Retry-then-reprobe diagnostic protocol.

Key exports:
- `needs_diagnosis(trace, node_name) → bool` — lexical uncertainty > UNCERTAINTY_THRESHOLD
- `verify_against_baseline(observed, baseline, fault_config, assumed_std=0.10, z_thresh=1.0) → Dict` — **NEW Phase 3**
- `compute_inference_gap(node_result, input_text, enabled=True) → Optional[float]` — uses shared `get_embedder()` from uncertainty.py
- `diagnose(node_name, same_input_prompt, clean_input_prompt, k, retries, role="") → str`
- `diagnose_trace(trace, node_prompts, k, retries, compute_gap, node_roles=None) → None`

---

### `scripts/run_local_debug.py` (updated)
Sanity check script, 4 modes:

| Flag | What it does |
|---|---|
| `--phase 1` | k=1, no fault, inspect prompt output quality |
| `--phase 2` | k=3, no fault, print lexical + semantic uncertainties, verify below threshold |
| `--phase 3` | k=3, 3 fault types at Retriever, compare against baseline |
| `--phase 3x` | **NEW** — 2 examples × full 10-condition grid; prints table; reproducibility check |

---

### `scripts/run_study1.py` (rewritten)
Full scored batch runner.

Trial loop (per question):
1. **Clean control** trial → PipelineTrace → `baseline_uncertainties` dict
2. **Build substitute bank** (once before loop, 10 pilot examples)
3. **Each fault condition** from `build_fault_conditions()`:
   - `make_rng(question_id, type, target_node)`
   - Call appropriate injector → `None` → `_skip_record()` → `skipped_trials.jsonl`
   - `run_pipeline(topo, ..., fault_config=...)`
   - `diagnose_trace(trace, node_prompts, node_roles=node_roles)`
   - `verify_against_baseline(...)` → stored in record
   - Append to `trials.jsonl`

JSONL record now includes: `topology_id`, `semantic_uncertainties`, `is_control`, `fault_config`, `verification`, `conclusions`.

`--resume` reconstructs `done_keys` as 4-tuples `(question, true_label, type, target_node)`.

---

### `scripts/rescore_study1.py` (**NEW**)
Retroactive re-scoring for Phase 2 DoD.

```bash
python scripts/rescore_study1.py [--log-path logs/trials.jsonl]
```
Outputs:
- `results/study1_rescore.jsonl` — per-trial records with new metrics
- `results/study1_rescore_table.csv` — mean lexical vs semantic per fault_type × node

---

### `scripts/run_agent.py` (updated)
`run_all_mode` now uses `run_pipeline(default_topology(), ...)` instead of calling individual pipeline modules. Per-module modes (`retriever`, `reasoner`, `writer`) still call individual standalone pipelines for interactive single-node testing.

---

## 11. Data Schemas: JSONL Logs

### `logs/trials.jsonl` (one record per trial)
```json
{
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "true_label": "contamination",
  "diagnosed_label": "contamination",
  "is_control": false,
  "fault_config": {"type": "contamination", "target_node": "retriever"},
  "topology_id": "chain_3node_default",
  "uncertainties": {"retriever": 0.6667, "reasoner": 0.6667, "writer": 0.0},
  "semantic_uncertainties": {"retriever": null, "reasoner": 0.12, "writer": 0.05},
  "inference_gaps": {"retriever": null, "reasoner": null, "writer": null},
  "per_node_diagnoses": {"retriever": "contamination"},
  "gold_answer": "yes",
  "verification": {"target_deviated": true, "z_score": 6.67, "observed": 0.6667, "baseline": 0.0},
  "samples": {
    "retriever": ["sample1...", "sample2...", "sample3..."],
    "reasoner":  ["sample1...", "sample2...", "sample3..."],
    "writer":    ["sample1...", "sample2...", "sample3..."]
  },
  "conclusions": {
    "reasoner": ["Yes, they are both American.", "Yes, both were American.", "..."]
  }
}
```

### `logs/skipped_trials.jsonl` (**NEW**)
```json
{
  "_is_skip": true,
  "question_id": "5a8b57f25542995d1e6f1371",
  "question": "...",
  "fault_config": {"type": "ceiling", "target_node": "retriever"},
  "reason": "ceiling_answer_survived"
}
```

Possible `reason` values:
- `"ceiling_answer_survived"` — gold answer string still present in stripped context
- `"contamination_insufficient_distractors"` — not enough distractor paragraphs
- `"contamination_empty_substitute_bank"` — bank not populated yet
- `"contamination_no_parent_for_{node}"` — topology issue

---

## 12. Study 1 Results (Baseline)

Study 1 used the **original** pipeline (pre-Phase 2/3 redesign):
- 15 questions × 3 fault types = 45 trials
- All faults targeted only the Retriever
- Lexical uncertainty only
- Global UNCERTAINTY_THRESHOLD = 0.75

**Confusion matrix (45/45 trials diagnosed)**:

|  | diagnosed: noise | diagnosed: contamination | diagnosed: ceiling |
|---|---|---|---|
| **true: noise** | 15 | 0 | 0 |
| **true: contamination** | 0 | 14 | 1 |
| **true: ceiling** | 0 | 0 | 15 |

- **Accuracy**: 44/45 = 97.8%
- **Contamination precision**: 100%; recall: 93.3% (1 ceiling misclassified as contamination)
- **Clean detection rate**: 100% (all 15 noise examples recovered on retry)

> These results are on old-pipeline data. The redesigned system (Phases 1-3) produces different (richer) logs and must be re-evaluated with the new 10-condition-per-question protocol.

---

## 13. Key Research Findings

### Finding 1: Reasoner's inherent 0.667 baseline uncertainty
On clean gold-context inputs, the Reasoner reliably produces uncertainty ≈ 0.667 under k=3 self-consistency with lexical matching. This is not a fault — it reflects CoT phrasing variance. Two of three samples typically state the same conclusion using different words, and after normalization they still differ in structure. The Phase 2 fix (FINAL ANSWER extraction + semantic clustering) should reduce this to near 0.

### Finding 2: Parametric memory confound
For easy/common HotpotQA questions, the model answers correctly even when given completely wrong (distractor) context. This inflates contamination recall (model isn't confused by bad input if it already knows the answer). Mitigation: pre-filter to "hard" multi-hop questions (baseline uncertainty > 0.2 on clean context).

### Finding 3: Per-node thresholding required
A global threshold of 0.75 produces zero false positives on Retriever and Writer (clean baseline ≈ 0.0) but barely catches the Reasoner (clean baseline ≈ 0.667 → threshold is barely above it). Per-node thresholds `{retriever: 0.30, reasoner: 0.75, writer: 0.30}` would give much tighter fault detection at Retriever and Writer without sacrificing Reasoner specificity.

### Finding 4: Contamination is the most detectable fault
Contamination produces the largest uncertainty delta at the Retriever and cascades clearly to the Writer. Noise is the hardest to consistently detect at k=3 (the delta above baseline is smaller). Ceiling is reliably detected but the 1 misclassification suggests that some ceiling scenarios partially recover with clean-input retry (soft ceiling).

---

## 14. Invariants That Must Never Be Broken

1. **Per-node uncertainties are NEVER collapsed to a scalar.** `PipelineTrace.uncertainties()` returns a `Dict[str, float]`. No caller may call `sum()` or `mean()` over this dict for any diagnostic purpose.

2. **All model calls are sequential.** No threading, no asyncio, no concurrent futures. Apple Silicon MLX constraint.

3. **The lexical uncertainty metric is never deleted.** `NodeResult.uncertainty` persists regardless of what new metrics are added. All new metrics are additional optional fields.

4. **`validate_topology()` is called on every `run_pipeline()` invocation** (already guaranteed — it's the first line of `run_pipeline`).

5. **Every injector uses `make_rng`, not the global `random` module.** Re-running any trial with the same `(question_id, fault_type, target_node)` must produce byte-identical corrupted input.

6. **Skipped trials must be logged.** `inject_ceiling()` returning `None` is not a silent skip — it must result in a record written to `skipped_trials.jsonl` with an explicit `reason` field.

7. **Control trial always runs first per question.** The per-question baseline uncertainties must exist before any fault trials for that question are run (required for `verify_against_baseline`).

---

## 15. Current State & What's Next

### Current state (as of branch `updated-methods`, 2026-09-24)

| Phase | Status |
|---|---|
| Phase 1: Topology-agnostic pipeline | ✅ Complete — all 11 files pass syntax + import checks |
| Phase 2: Uncertainty redesign | ✅ Complete — semantic/Jaccard metrics, FINAL ANSWER prompt, retroactive rescore script |
| Phase 3: Fault injection redesign | ✅ Complete — make_rng, node-scoped noise, proportional contamination, verified ceiling, skip log, baseline verification |
| Phase 3 sanity checks (`--phase 3x`) | ⏳ **Pending** — needs to be run on the actual Apple Silicon machine (requires MLX + real model) |
| Phase 2 validation (Reasoner semantic ≈ 0 on clean) | ⏳ **Pending** — same as above |
| Retroactive rescore of Study 1 logs | ⏳ **Pending** — run `python scripts/rescore_study1.py` |

### Next steps (Study 2)

1. **Run Phase 3 extended sanity checks**:
   ```bash
   python scripts/run_local_debug.py --phase 3x --n-examples 2
   ```
   Must verify: (a) noise@node changes only that node, (b) contamination prompts show real corruption, (c) skip log populated, (d) re-run produces identical corrupted text.

2. **Run Phase 2 validation**:
   ```bash
   python scripts/run_local_debug.py --phase 2 --n-examples 3
   ```
   Expect: Reasoner `uncertainty_semantic` ≈ 0.0 on clean data (fixing the 0.667 artifact).

3. **Retroactive rescore of Study 1**:
   ```bash
   python scripts/rescore_study1.py
   ```

4. **Run a 5-question pilot with the new study design**:
   ```bash
   python scripts/run_study1.py --n-examples 5 --k 3 --no-inference-gap
   ```
   Manually inspect the resulting JSONL — confirm all 10 conditions logged, skip log populated, verification z-scores reasonable.

5. **Implement per-node threshold vector** in `config.py` and `diagnose.py`:
   ```python
   NODE_THRESHOLDS = {"retriever": 0.30, "reasoner": 0.75, "writer": 0.30}
   ```

6. **Pre-filter HotpotQA to hard examples** (baseline Retriever uncertainty > 0.2) to mitigate the parametric memory confound.

7. **Study 2 bulk run**: scale to 50+ questions × 10 conditions × k=5.

8. **Study 3**: introduce non-chain topologies (fan-in/fan-out) via the new `Topology` data structure.

---

## 16. Command Cheatsheet

```bash
# --- Setup (run once) ---
cd btp-pipeline
python scripts/download_data.py

# --- Sanity checks (run before any bulk study) ---
python scripts/run_local_debug.py --phase 1 --n-examples 3
python scripts/run_local_debug.py --phase 2 --n-examples 3
python scripts/run_local_debug.py --phase 3 --n-examples 3
python scripts/run_local_debug.py --phase 3x --n-examples 2  # NEW: §3.8 checks

# --- Retroactive re-scoring of Study 1 logs ---
python scripts/rescore_study1.py

# --- Study run ---
python scripts/run_study1.py --n-examples 5 --k 3 --no-inference-gap
python scripts/run_study1.py --n-examples 15 --k 3 --no-inference-gap --resume

# --- Single agent interactive testing (standalone node CLIs) ---
python scripts/run_agent.py retriever \
    --question "Were Scott Derrickson and Ed Wood of the same nationality?" \
    --context "[Scott Derrickson] American director..."

python scripts/run_agent.py all \
    --question "Were Scott Derrickson and Ed Wood of the same nationality?" \
    --context "[Scott Derrickson] American director..." \
    --k 3

# --- Git ---
git status                            # check what's staged
git checkout updated-methods          # switch to the new branch
git log --oneline -5                  # recent commits
```

---

## 17. Dependency Map

```
External dependencies:
  mlx_lm                  ← model inference (Apple Silicon)
  datasets                ← HotpotQA loading
  sentence_transformers   ← semantic uncertainty + inference gap
  numpy                   ← embedding arithmetic
  pandas                  ← confusion matrix + rescore table
  tqdm                    ← progress bar
  hashlib, random         ← stdlib (make_rng reproducibility)

Internal import graph:
  config.py
      ↑
  topology.py (no src imports)
  uncertainty.py (numpy, sentence_transformers only)
      ↑
  nodes.py ← config, uncertainty
      ↑
  pipeline.py ← config, nodes, topology
  faults.py ← config (only)
  diagnose.py ← config, nodes, uncertainty
      ↑
  scripts/run_study1.py ← config, diagnose, faults, nodes, pipeline
  scripts/run_local_debug.py ← config, faults, pipeline
  scripts/rescore_study1.py ← config, uncertainty
  scripts/run_agent.py ← config, pipeline, pipelines/*
```

**No circular imports.** `topology.py` and `uncertainty.py` are leaf nodes in the dependency graph (they import nothing from other `src/` modules). `pipeline.py` imports `topology` + `nodes`; `diagnose.py` imports `nodes` + `uncertainty` — no cycles.
