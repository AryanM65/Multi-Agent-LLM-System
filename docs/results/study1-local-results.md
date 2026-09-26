# Study 1 — Original Local Validation Results

> The first validation run establishing the core methodology, before the migration to vLLM/Kaggle for dataset-scale generation. Run on Apple Silicon (MLX, `mlx-community/Qwen3-8B-4bit`), chain topology only, all faults targeting the Retriever. Kept for historical/methodology reference — the actual training dataset is the much larger, bug-fixed, multi-topology `dataset/trials.jsonl` (see [`docs/dataset/dataset_description.md`](../dataset/dataset_description.md)), not this run.

---

## Confusion matrix (45 trials: 15 questions × 3 fault types)

```
                diagnosed: noise   contamination   ceiling
true: noise            15               0             0
true: contamination     0              14             1
true: ceiling            0               0            15
```

- **Accuracy**: 44/45 = 97.8%
- Contamination precision: 100%; recall: 93.3% (one ceiling trial misclassified as contamination — a "soft ceiling" that partially recovered on clean-input reprobe)
- Clean-detection rate: 100% (all 15 noise examples recovered on same-input retry)

## Per-node uncertainty statistics

| Fault type (n=15 each) | U_retriever mean (max) | U_reasoner mean (max) | U_writer mean (max) |
|---|---|---|---|
| Noise (temp=1.2) | 0.089 (0.333) | 0.667 (0.667) | 0.111 (0.333) |
| Contamination (distractors) | **0.222 (0.667)** | 0.667 (0.667) | **0.244 (0.667)** |
| Ceiling (stripped gold) | 0.089 (0.333) | 0.667 (0.667) | 0.089 (0.667) |

**Key pattern**: contamination consistently elevated both Retriever and Writer uncertainty relative to noise and ceiling — matching the theoretical prediction (wrong context → Retriever disagrees across samples → Writer receives inconsistent input).

Of the 45 trials, only 24 exceeded the per-node diagnostic threshold and were actually diagnosable — the other 21 were cases where the model answered correctly from parametric memory despite the injected fault (a known confound; see Finding 2 below).

## Lexical vs. semantic uncertainty — before/after comparison

A later pass re-scored the same trials with the (then-new) semantic-uncertainty metric, comparing against a slightly larger 56-trial log:

| Metric | OLD (lexical only) | NEW (+ semantic, per-node thresholds) |
|---|---|---|
| Detection rate | 0.0% | 17.9% |
| Accuracy (of detected) | 0.0% | 40.0% |

| Node | OLD lexical mean | NEW lexical mean | NEW semantic mean |
|---|---|---|---|
| Reasoner | 0.6667 | 0.0655 | 0.095 |
| Retriever | 0.1333 | 0.0417 | — (not computed for Retriever) |
| Writer | 0.1481 | 0.0952 | 0.1335 |

The large lexical-mean drop for the Reasoner between OLD and NEW reflects a prompt-format fix (`FINAL ANSWER:` marker + `extract_conclusion()`), not the semantic metric itself — extracting just the conclusion sentence, rather than exact-matching the full chain-of-thought, removes most of the phrasing-variance noise at the source.

## Key findings from this phase

1. **Per-node thresholding is essential.** A single global threshold fails because nodes have fundamentally different output formats — Retriever/Writer are short and low-variance, Reasoner is verbose CoT with high structural variance even when correct.
2. **Parametric memory is a confound.** For easy/common questions, the model answers correctly from its own training-time knowledge regardless of injected context corruption — a natural floor on detectable failures that motivated spreading faults across more nodes/topologies in the later multi-topology dataset, rather than only ever targeting the Retriever.
3. **Contamination was the most detectable fault** in this small local run — worth re-checking against the full multi-topology dataset now that faults can target the Reasoner and Writer directly.
4. **Structural bugs can hide behind "it ran successfully."** All of this phase's real correctness issues (documented in `docs/model/model.md` and the infra doc) were caught by directly inspecting generated content for statistical sanity, not by structural/schema validation alone — the same discipline that later caught the multi-parent-contamination and `retriever_c` coverage bugs in the full dataset generation run.
