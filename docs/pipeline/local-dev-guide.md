# Local Development Guide — Running the Pipeline Without a Cloud GPU

> For iterating on pipeline code (`src/`, `scripts/`) locally, either against a real local model (MLX on Apple Silicon, or Ollama) or in deterministic mock mode (no model needed at all — useful on any machine, including Windows). For the actual dataset-scale generation run, see [`docs/infra/kaggle-and-lightning-setup.md`](../infra/kaggle-and-lightning-setup.md) instead — this doc is for local sanity checks and small-scale experiments only.

---

## Setup

### Apple Silicon (macOS, real model via MLX)
```bash
pip install uv
uv venv .venv && source .venv/bin/activate
uv pip install mlx-lm datasets sentence-transformers numpy pandas tqdm
cd btp-pipeline/
python scripts/download_data.py   # one-time, ~200MB HotpotQA cache
```

### Windows (mock mode — no MLX available on Windows)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install datasets sentence-transformers numpy pandas tqdm
# No dataset download needed in mock mode
```

### Mock mode environment variable
| Variable | Value | Effect |
|---|---|---|
| `BTP_MOCK` | `1` | Deterministic mock backend — no model, no dataset download needed |
| `BTP_MOCK` | `0` (default) | Real model (MLX or Ollama, per `src/config.py`'s `BACKEND`) |

```powershell
$env:BTP_MOCK = "1"        # Windows PowerShell
export BTP_MOCK=1          # macOS / bash
```
Or pass `--mock` to any script that supports it.

---

## Step 1 — Sanity checks (always run before any scored study)

```bash
cd btp-pipeline/
python scripts/run_local_debug.py --phase 1 --n-examples 3   # k=1, no fault: prompt sanity
python scripts/run_local_debug.py --phase 2 --n-examples 3   # k=3, no fault: self-consistency baseline
python scripts/run_local_debug.py --phase 3 --n-examples 3   # k=3, all 3 fault types: pattern check
python scripts/run_local_debug.py --phase 3x --n-examples 2  # full grid + reproducibility check
```

**Expected on clean data (Phase 2)**:
```
retriever:  uncertainty_lexical ≈ 0.00,  uncertainty_jaccard ≈ 0.00
reasoner:   uncertainty_lexical ≈ 0.67,  uncertainty_semantic ≈ 0.00  ← key check
writer:     uncertainty_lexical ≈ 0.00,  uncertainty_semantic ≈ 0.00
```
The Reasoner's ~0.67 *lexical* baseline is expected (chain-of-thought phrasing varies even when the conclusion is identical) — not a bug. If `uncertainty_semantic` is NOT near 0 for the Reasoner on clean input, check that the Reasoner instruction includes `FINAL ANSWER:` and that `extract_conclusion()` is finding it.

**Phase 3 expected patterns**:

| Fault type | Expected uncertainty pattern |
|---|---|
| Noise | Moderate ↑ vs. baseline; recovers on same-input retry |
| Contamination | Retriever ↑↑; Writer cascades |
| Ceiling | ↑ across nodes; persists through both retry types |

If patterns are indistinguishable, fix injection logic or recalibrate `UNCERTAINTY_THRESHOLD` in `src/config.py` before proceeding — don't move on to a scored run with an uncalibrated threshold.

---

## Step 2 — Study 1 (chain topology, single-target fault grid)

```bash
python scripts/run_study1.py --n-examples 5 --k 3 --no-inference-gap    # pilot: confirm log structure
python scripts/run_study1.py --n-examples 15 --k 3 --no-inference-gap   # full local run
python scripts/run_study1.py --n-examples 15 --k 3 --no-inference-gap --resume   # crash recovery
```
Check `logs/trials.jsonl` for: `is_control: true` entries (one per question, first), `skipped_trials.jsonl` populated with `reason` fields, `verification.target_deviated` present on fault trials.

Produces `logs/trials.jsonl` and `results/confusion_matrix.csv`.

---

## Step 3 — Retroactive rescore (compare lexical vs. semantic uncertainty on saved logs)

```bash
python scripts/rescore_study1.py
```
Applies newer semantic/Jaccard metrics to samples already saved in `logs/trials.jsonl`, without re-running the model. Outputs `results/study1_rescore_table.csv`. Expect: Reasoner `semantic_mean` well below `lexical_mean` on clean baseline (fixing the lexical-only 0.667 artifact), and rising under contamination/ceiling faults.

---

## Step 4 — Study 2 (multi-topology fault grid)

```bash
# Mock mode (any machine, no model)
$env:BTP_MOCK = "1"
python scripts/run_study2.py --n-examples 3 --k 3 --topology chain --mock
python scripts/run_study2.py --n-examples 3 --k 3 --topology all --mock

# Real model
python scripts/run_study2.py --n-examples 15 --k 5 --topology all
python scripts/run_study2.py --n-examples 15 --k 5 --topology dual_retriever_fanin
python scripts/run_study2.py --n-examples 15 --k 5 --topology all --resume
```
Outputs: `logs/study2_trials.jsonl`, `logs/study2_skipped.jsonl`, `results/study2_confusion/{topology}.csv` + `summary.csv`.

---

## Step 5 — Interactive single-node testing

```bash
python scripts/run_agent.py retriever --question "..." --context "..."
python scripts/run_agent.py all --question "..." --context "..." --k 3
python scripts/run_agent.py all --question "Test?" --context "Test." --k 3 --mock
```
See [`docs/pipeline/standalone-agents.md`](standalone-agents.md) for the full standalone-pipeline API (running any single agent independently, chaining via saved JSONL, etc.).

---

## Passing criteria at each step

| Step | Check |
|---|---|
| Phase 1 sanity | All 3 nodes produce on-topic, non-empty text |
| Phase 2 sanity | `uncertainty_semantic` ≈ 0 at Reasoner on clean input |
| Phase 3x sanity | Noise changes only target node; contamination shows real prompt change; skip log populated; re-run → identical corrupted text |
| Study 1 pilot | All conditions logged per question; z-score > 1 for most fault trials |
| Rescore | Reasoner `semantic_mean` < `lexical_mean` on clean; rises on contamination |
| Study 2 mock | All topologies run without errors |
| Study 2 real | Confusion matrix diagonal dominates (accuracy > 50%) per topology |

---

## Threshold calibration procedure

`UNCERTAINTY_THRESHOLD` in `src/config.py` should never be trusted as a placeholder — recalibrate whenever the model changes:
1. Collect all node uncertainties from a Phase 2 (no-fault baseline) run.
2. Compute `mean + 1σ` or the 75th percentile of that baseline distribution.
3. Set `UNCERTAINTY_THRESHOLD` to that value.
4. Re-run Phase 3 with the calibrated threshold and verify the three fault types still produce visibly distinct patterns.
