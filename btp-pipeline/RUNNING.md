# RUNNING.md — Execution Guide

Complete step-by-step commands for running the BTP pipeline on both Windows (mock mode) and Apple Silicon (real model).

---

## Prerequisites

### On Apple Silicon (macOS, real model)
```bash
# Create a virtual environment (uv recommended, or conda)
pip install uv
uv venv .venv && source .venv/bin/activate

# Install all dependencies
uv pip install mlx-lm datasets sentence-transformers numpy pandas tqdm

# Download the HotpotQA dataset (one-time, ~200MB)
cd btp-pipeline/
python scripts/download_data.py
```

### On Windows (mock mode — no MLX needed)
```powershell
python -m venv .venv
.venv\Scripts\activate

# MLX not available on Windows — install everything else
pip install datasets sentence-transformers numpy pandas tqdm

# No dataset download needed in mock mode
```

---

## Environment variables

| Variable | Value | Effect |
|---|---|---|
| `BTP_MOCK` | `1` | Use deterministic mock backend (no MLX, no dataset needed) |
| `BTP_MOCK` | `0` (default) | Use real MLX model |

```powershell
# Windows PowerShell
$env:BTP_MOCK = "1"

# macOS / bash
export BTP_MOCK=1
```

Or use the `--mock` CLI flag with any script that supports it.

---

## Step 1 — Sanity checks (run before any study)

```bash
cd btp-pipeline/

# Phase 1: k=1, no fault, verify prompt output quality
python scripts/run_local_debug.py --phase 1 --n-examples 3

# Phase 2: k=3, no fault, verify semantic uncertainty ≈ 0 on clean Reasoner
# Expected: uncertainty_semantic ≈ 0.0 (FINAL ANSWER extraction working)
python scripts/run_local_debug.py --phase 2 --n-examples 3

# Phase 3: fault injection basic patterns
python scripts/run_local_debug.py --phase 3 --n-examples 3

# Phase 3x: full §3.8 extended sanity checks (2 examples × full grid)
# Checks: noise node-scope, contamination prompt corruption, ceiling skip log,
#         reproducibility of corrupted text across re-runs
python scripts/run_local_debug.py --phase 3x --n-examples 2
```

**Expected results on clean data (Phase 2)**:
```
retriever:  uncertainty_lexical ≈ 0.00,  uncertainty_jaccard ≈ 0.00
reasoner:   uncertainty_lexical ≈ 0.67,  uncertainty_semantic ≈ 0.00  ← key check
writer:     uncertainty_lexical ≈ 0.00,  uncertainty_semantic ≈ 0.00
```
If `uncertainty_semantic` for the Reasoner is NOT near 0 on clean input, check that the Reasoner instruction includes `FINAL ANSWER:` and `extract_conclusion()` is finding it.

---

## Step 2 — Study 1 (chain topology, 10-condition grid)

```bash
# 5-question pilot (confirm logs structure before full run)
python scripts/run_study1.py --n-examples 5 --k 3 --no-inference-gap

# Check logs/trials.jsonl — confirm:
#   - is_control: true entries present (one per question, first)
#   - skipped_trials.jsonl populated with reason fields
#   - verification.target_deviated present for fault trials

# Full Study 1 run
python scripts/run_study1.py --n-examples 15 --k 3 --no-inference-gap

# Resume from a crash
python scripts/run_study1.py --n-examples 15 --k 3 --no-inference-gap --resume
```

---

## Step 3 — Retroactive rescore (Phase 2 before/after comparison)

```bash
# Applies new semantic/Jaccard metrics to saved samples in logs/trials.jsonl
python scripts/rescore_study1.py

# Output: results/study1_rescore_table.csv
# Terminal: colour-coded Lex (old) vs Sem (new) table per fault_type × node
```

**What to look for**:
- Reasoner `semantic_mean` should be much lower than `lexical_mean` on clean baseline (fixing the 0.667 artifact)
- Reasoner `semantic_mean` should RISE under contamination / ceiling faults (semantic divergence)

---

## Step 4 — Study 2 (multi-topology, 10-condition grid)

```bash
# ── Windows (mock mode, no MLX, no dataset) ──────────────────────────────
$env:BTP_MOCK = "1"

# Single topology, mock
python scripts/run_study2.py --n-examples 3 --k 3 --topology chain --mock

# All 4 topologies, mock
python scripts/run_study2.py --n-examples 3 --k 3 --topology all --mock

# ── Apple Silicon (real model) ────────────────────────────────────────────
# All topologies, 5 questions pilot
python scripts/run_study2.py --n-examples 5 --k 3 --topology all --no-inference-gap

# Full run, all topologies
python scripts/run_study2.py --n-examples 15 --k 5 --topology all

# Single topology
python scripts/run_study2.py --n-examples 15 --k 5 --topology dual_retriever_fanin
python scripts/run_study2.py --n-examples 15 --k 5 --topology parallel_reasoner
python scripts/run_study2.py --n-examples 15 --k 5 --topology deep_chain

# Resume from crash
python scripts/run_study2.py --n-examples 15 --k 5 --topology all --resume
```

**Outputs**:
```
logs/study2_trials.jsonl          ← append-only trial log (all topologies)
logs/study2_skipped.jsonl         ← skipped trials with reason field
results/study2_confusion/
  chain.csv
  dual_retriever_fanin.csv
  parallel_reasoner.csv
  deep_chain.csv
  summary.csv                     ← accuracy per topology + aggregate
```

---

## Step 5 — Interactive single-node testing

```bash
# Run only the Retriever
python scripts/run_agent.py retriever \
    --question "Were Scott Derrickson and Ed Wood of the same nationality?" \
    --context "[Scott Derrickson] American director. [Ed Wood] American filmmaker."

# Run the full pipeline via the topology engine
python scripts/run_agent.py all \
    --question "Were Scott Derrickson and Ed Wood of the same nationality?" \
    --context "[Scott Derrickson] American director. [Ed Wood] American filmmaker." \
    --k 3

# Mock mode (no MLX)
python scripts/run_agent.py all \
    --question "Test question?" \
    --context "Test context." \
    --k 3 --mock
```

---

## Passing criteria at each step

| Step | Check |
|---|---|
| Phase 1 sanity | All 3 nodes produce on-topic, non-empty text |
| Phase 2 sanity | `uncertainty_semantic` ≈ 0 at Reasoner on clean input |
| Phase 3x sanity | (a) noise changes only target node uncertainty; (b) contamination shows real prompt change; (c) skip log populated; (d) re-run → identical corrupted text |
| Study 1 pilot | All 10 conditions logged per question; z-score > 1 for most fault trials |
| Rescore | Reasoner semantic_mean < lexical_mean on clean; rises on contamination |
| Study 2 mock | All 4 topologies run without errors in mock mode |
| Study 2 real | Confusion matrix diagonal dominates (accuracy > 50%) per topology |

---

## Fault grid reference (10 conditions per question)

| # | Type | Target node | Expected primary uncertainty ↑ |
|---|---|---|---|
| 0 | clean (control) | — | None (baseline) |
| 1 | noise | retriever | retriever only |
| 2 | noise | reasoner | reasoner only |
| 3 | noise | writer | writer only |
| 4 | contamination | retriever | retriever ↑↑, writer cascades |
| 5 | contamination | reasoner | reasoner ↑↑, writer cascades |
| 6 | contamination | writer | writer ↑↑ |
| 7 | ceiling | retriever | retriever ↑, persists on retry |
| 8 | ceiling | reasoner | reasoner ↑, persists on retry |
| 9 | ceiling | writer | writer ↑, persists on retry |

---

## Topology shapes

| Name | Shape | Key hypothesis |
|---|---|---|
| `chain` | Ret → Rea → Wri | Baseline reference |
| `dual_retriever_fanin` | Ret-A + Ret-B → Rea → Wri | Evidence disagreement propagates into Reasoner |
| `parallel_reasoner` | Ret → Rea-A + Rea-B → Wri | Asymmetric fault at one Reasoner detectable at Writer |
| `deep_chain` | Ret → Rea-1 → Rea-2 → Wri | Second Reasoner attenuates first Reasoner's fault |
