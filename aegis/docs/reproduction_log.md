# Reproduction log (Phase 3)

## 3a. Qwen2.5-7B-Instruct zero-shot (paper Table 1 row)

Run: Kaggle kernel `aryanmahajan7/aegis-zeroshot-full` (T4, vLLM, `Qwen/Qwen2.5-7B-Instruct-AWQ`), 2026-11-02.
Raw outputs, run info and the full kernel log are in `aegis/results/kaggle_logs/full/`. Scored with
`python -m aegis.src.eval.score_zeroshot aegis/results/kaggle_logs/full --log` (ledger entry phase `3a`).
Wall time: 12,630 s for 600 test prompts, 3,862 s for 184 Who&When prompts (about 21 s and 21 s per prompt).

Numbers are micro / macro F1 in percent.

| Benchmark, level | Ours | Paper | Difference (micro, macro) |
|---|---|---|---|
| Aegis-Bench, pair | 8.01 / 2.48 | 5.02 / 2.52 | +3.0, -0.0 |
| Aegis-Bench, agent | 49.24 / 10.69 | 27.55 / 14.49 | +21.7, -3.8 |
| Aegis-Bench, error | 20.15 / 12.72 | 14.96 / 11.36 | +5.2, +1.4 |
| Who&When, pair | 2.98 / 1.85 | 2.31 / 1.14 | +0.7, +0.7 |
| Who&When, agent | 35.16 / 23.14 | 40.92 / 23.50 | -5.8, -0.4 |
| Who&When, error | 6.23 / 3.35 | 3.64 / 1.77 | +2.6, +1.6 |
| Avg (12 numbers) | 14.67 | 12.43 | +2.2 |

Run health: Aegis-Bench 8 of 600 outputs unparseable (counted as empty), 9 with finish reason other than `stop`
(hit the 512-token cap), 25 prompts truncated; Who&When 0 unparseable, 25 truncated.

### Verdict: **close on most rows, not matched on Aegis-Bench agent-level micro-F1**

- Matched or close (within about 2 points): Aegis-Bench pair macro, error macro; Who&When pair, agent macro, and all Who&When
  micro/macro rows except agent micro (-5.8) and error micro (+2.6).
- Not matched: Aegis-Bench agent micro-F1 is 49.24 versus 27.55 published (+21.7 points). Our macro agent-F1 is lower
  (10.69 vs 14.49), so the model gets the common agents right more often but our hallucinated agent names
  (for example `Solver6`) create extra zero-F1 classes in the macro average.
- We do not know why the agent-micro gap is so large. Candidates, none tested:
  1. The paper's test set may be the repo's `evaluation/data/aegis.jsonl` (600 records, no ids in common with the HF test file;
     see dataset_audit.md) rather than the HF file we used.
  2. AWQ 4-bit weights instead of full precision.
  3. Different prompt or system message than the repo's `local_model.py` (we copied that one; the paper may have used the API-model variant).
  4. Parse policy: the authors skip unparseable outputs, we count them as empty.
  5. The paper's Qwen row may have used sampling (the repo's local inference sets `do_sample=True`); ours is greedy.

### Deviations from the paper protocol
- AWQ 4-bit quantized model, greedy decoding (temperature 0), `max_new_tokens` 512.
- 16,384-token context; prompts over the budget (14,874 conversation tokens) are cut to the first 30% and last 70% with a marker.
  This hit 25 of 600 test and 25 of 184 Who&When prompts. The authors do not truncate.
- HF `Aegis-Bench/test.jsonl` as the test set (decision D1); Who&When from the repo's processed `whowhen.jsonl` (184 records, the paper says 127).
- Unparseable output counts as an empty prediction (the authors skip it).
- Test data was used for this reproduction only; nothing was tuned on it.

### Optional follow-up
Re-run on the repo's `evaluation/data/aegis.jsonl` to test explanation 1. This costs about 3.5 GPU-hours on Kaggle and was not run.

## 3b. DCL-style (frozen encoder) re-implementation

No DCL code or checkpoint was released (decision D8). `aegis/src/baselines/dcl_style.py` re-implements the published design from its
description: bag of turns, projection to 128-d, agent-name and error-mode prototype banks (initialised from text embeddings),
top-K=3 multiple-instance pooling per agent, asymmetric BCE + supervised contrastive (tau 0.07) + hierarchical-consistency losses
(weights 1.0 / 0.4 / 0.6). It is **named DCL-style and is not DCL**: the encoder is frozen (cached MiniLM; the paper fine-tunes it), lr 1e-3
with early stopping on val loss instead of lr 1e-4 for 2 x 500 steps, agent prototypes keyed by agent name, batch 64. About 0.2M trainable
parameters. One decision threshold is tuned on validation; then test and Who&When are scored once per seed (3 seeds, mean shown).
Ledger: phase `3b`.

Micro / macro F1, percent. "paper-split" trains on the released train set; "grouped" trains on the query-disjoint train set (46% fewer records).

| Benchmark, level | Published DCL | DCL-style, paper-split | DCL-style, grouped |
|---|---|---|---|
| Aegis-Bench pair | 8.33 / 5.30 | 15.06 / 8.56 | 11.59 / 6.01 |
| Aegis-Bench agent | 22.93 / 20.23 | 69.88 / 57.22 | 66.47 / 51.82 |
| Aegis-Bench error | 24.73 / 27.70 | 34.34 / 33.23 | 28.96 / 27.14 |
| Who&When pair | 1.60 / 0.77 | 3.09 / 1.04 | 4.08 / 1.12 |
| Who&When agent | 8.40 / 6.07 | 37.95 / 31.44 | 37.90 / 30.00 |
| Who&When error | 14.67 / 10.57 | 10.61 / 6.98 | 12.50 / 8.46 |

### Verdict: **not matched** (and not intended as a reproduction)

- Our frozen-encoder model scores higher than the published DCL on 11 of 12 micro-F1 numbers; the exception is Who&When error micro-F1 on the
  paper-split (10.6 vs 14.7), which the grouped model also narrowly misses (12.5).
- The gap is largest at agent level (69.9 vs 22.9). The trivial floor on the same metric is about 70 (baselines_log.md), so the published
  DCL agent-level numbers sit far below a rule that predicts every present agent. We cannot explain this from the paper alone; plausible causes are that
  the published model predicts agent names in a different vocabulary than the labels, or restricts its outputs, or the published agent-level metric
  differs from the released `evaluate.py` that we ported.
- Because the published DCL cannot be matched or even bracketed by a simpler model on the same metrics, comparisons against the published DCL
  number should be reported with the trivial floor and the flat-MLP baseline next to them, and the headline claims should rest on pair-level
  and macro numbers on the grouped split.
- On validation (grouped), the flat MLP (pair 23.1) beats this DCL-style model (pair 13.5): the prototype/top-K architecture does not
  help over mean pooling here. This is evidence only about this frozen-encoder variant.
- Test and Who&When were each scored once per seed with a configuration chosen on validation (no tuning on them).

## 3c. Aegis-SFT, Aegis-GRPO, proprietary models
Not rerun. Quote the published numbers.
