# BTP Pipeline — Uncertainty Propagation in Multi-Agent LLM Systems

**Study 1** of a BTech thesis investigating how uncertainty introduced at one agent
in a multi-agent LLM pipeline propagates downstream, and whether the *cause*
can be diagnosed automatically.

---

## Project Structure

```
btp-pipeline/
├── data/hotpotqa_distractor/  ← cached dataset (gitignored)
├── src/
│   ├── config.py              ← tuneable constants (k, threshold, paths)
│   ├── nodes.py               ← NodeResult, PipelineTrace, sample_node()
│   ├── pipeline.py            ← Retriever → Reasoner → Writer
│   ├── faults.py              ← inject_noise / contamination / ceiling
│   └── diagnose.py            ← retry-then-reprobe protocol
├── scripts/
│   ├── download_data.py       ← cache HotpotQA once
│   ├── run_local_debug.py     ← phases 1-3: prompt sanity & fault pattern check
│   └── run_study1.py          ← phase 4: scored batch run + confusion matrix
├── logs/trials.jsonl          ← append-only trial log
└── results/confusion_matrix.csv
```

---

## Setup

> **Full documentation** (research context, architecture, results, next steps) is in the [root README](../README.md).

```bash
# 1. Create and activate virtual environment
python -m venv btp-env
source btp-env/bin/activate   # Windows: btp-env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Cache the dataset (run once — ~50 MB download)
python scripts/download_data.py
```

> **Model**: This pipeline uses `mlx_lm` with `mlx-community/Qwen3-8B-4bit` on Apple Silicon GPU.
> The model is fetched automatically from HuggingFace on first run (~5 GB).
> To use a different backend, update `_get_model()` in `src/nodes.py` and `MODEL` in `src/config.py`.

---

## Execution Order

Follow the phases **in order** — do not skip to run_study1.py without validating earlier phases.

### Phase 1 — Prompt sanity (k=1, 3 examples, no faults)
```bash
python scripts/run_local_debug.py --phase 1
```
**Expected:** All three nodes produce sane, on-topic output. No errors.

### Phase 2 — Self-consistency baseline (k=3, 3 examples, no faults)
```bash
python scripts/run_local_debug.py --phase 2
```
**Expected:** `uncertainties` dict printed per example. Easy questions should
show all nodes < 0.35 uncertainty (adjusted for quantized model, per §8 of the brief).
If uncertainty is high here, check prompt formatting and normalization, not the fault logic.

### Phase 3 — Fault injection sanity (k=3, 3 examples, all fault types)
```bash
python scripts/run_local_debug.py --phase 3
```
**Expected patterns:**

| Fault type    | Expected uncertainty pattern                                      |
|---------------|-------------------------------------------------------------------|
| Noise         | Moderate ↑ vs baseline; should recover on same-input retry        |
| Contamination | Retriever uncertainty ↑↑; reasoner may follow                     |
| Ceiling       | Uncertainty ↑ across nodes; persists through both retry types     |

If patterns are **indistinguishable**, fix injection logic or recalibrate
`UNCERTAINTY_THRESHOLD` in `src/config.py` before proceeding.

### Phase 4 — Full scored run (15 examples, 3 fault types each)
```bash
python scripts/run_study1.py --n-examples 15 --k 3
```
Produces `logs/trials.jsonl` and `results/confusion_matrix.csv`.

**Resume after a crash:**
```bash
python scripts/run_study1.py --n-examples 15 --k 3 --resume
```

**Scale up (cloud GPU run — bump k and n_examples):**
```bash
python scripts/run_study1.py --n-examples 300 --k 5
```

---

## Threshold Calibration

`UNCERTAINTY_THRESHOLD = 0.3` in `src/config.py` is a placeholder.

**Calibration procedure (after Phase 1-2 baseline):**
1. Collect all node uncertainties from Phase 2 (no-fault baseline).
2. Compute `mean + 1σ` or the 75th percentile of the baseline distribution.
3. Set `UNCERTAINTY_THRESHOLD` to that value.
4. Re-run Phase 3 with the calibrated threshold and verify distinct patterns.

---

## JSONL Record Format

Each line in `logs/trials.jsonl`:

```json
{
  "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
  "true_label": "contamination",
  "diagnosed_label": "contamination",
  "uncertainties": {"retriever": 0.6, "reasoner": 0.0, "writer": 0.1},
  "inference_gaps": {"retriever": 0.12, "reasoner": null, "writer": null},
  "per_node_diagnoses": {"retriever": "contamination"},
  "gold_answer": "Arthur's Magazine",
  "samples": {
    "retriever": ["sample1...", "sample2...", "sample3..."],
    "reasoner":  ["..."],
    "writer":    ["..."]
  }
}
```

---

## Design Commitments

1. **Per-node uncertainties are never merged into a scalar.** `trace.uncertainties()` always returns a `Dict[str, float]` — this is the core structural bet of the project.
2. **All Ollama calls are sequential.** No threading, no asyncio — hardware constraint (16 GB RAM).
3. **Append-only JSONL logging.** Every trial is written to disk immediately — a crash never loses completed trials.
4. **Multi-node flagging:** when multiple nodes exceed threshold, all are diagnosed; `diagnosed_label` is the highest-uncertainty node's diagnosis.

---

## Open Decisions (deferred, not silently assumed)

| # | Item | Status |
|---|------|--------|
| 1 | `UNCERTAINTY_THRESHOLD` recalibration | ⚠ Deferred until Phase 1-2 data collected |
| 2 | `retries=2` ceiling-fault false-recovery validation | ⚠ Validate after local run |
| 3 | Writer normalization spot-check | ⚠ Manual; use `samples` field in JSONL |
| 4 | Multi-node flagging strategy | ✓ Resolved: diagnose all, surface max-uncertainty |
| 5 | Inference Gap (semantic drift) | ✓ Implemented as optional experimental metric; Study 2 for decision-driving |
