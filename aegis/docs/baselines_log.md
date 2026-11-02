# Phase 4 baselines (validation only)

All numbers: micro / macro F1 in percent, computed with `aegis/src/metrics.py` (authors' definition) on the **validation** split.
The test set and Who&When are NOT used here. Two validation protocols (decision D2):
- **paper**: released train (7146) / val (1787). Val queries overlap train (1753 of 1787 share a query), so it is optimistic.
- **grouped**: test queries removed, train/val re-split by query (3865 / 940). No query overlap.

Ledger: `aegis/results/metrics_history.jsonl`, phase `4`.

## Trivial floors (important context for every number in this project)

On validation, "predict every present agent with every one of the 14 modes" gives:

| Split | Agent micro / macro | Error micro / macro | Pair micro |
|---|---|---|---|
| paper | 70.5 / 64.2 | 28.2 / 28.1 | 7.7 |
| grouped | 73.4 / 64.7 | 29.9 / 29.8 | 8.2 |

Reason: 54-56% of the agents in a trajectory are faulty (about 2.3 of 4.4), so agent-level micro-F1 has a floor near 70, and with 14
balanced modes the error-level micro-F1 floor is near 28.

**Consequence:** the paper's DCL row (agent micro 22.93, error micro 24.73 on the test set) and the zero-shot Qwen rows sit *below* this
trivial floor on the same metric definition (the floor was measured on validation, not test; the test label statistics are
similar: 53.3% of agents faulty). Beating DCL's published average is therefore not by itself evidence of a good model. Report
the trivial floor and the frequency baseline next to every claim, and lean on pair-level and macro numbers.

## Frequency baseline (`baselines/frequency.py`)

Per framework, always predict the k most common (agent, mode) pairs from train (only agents that occur in the trajectory); k tuned on val.

| Split | k | Pair | Agent | Error |
|---|---|---|---|---|
| paper | 13 | 13.47 / 5.09 | 81.00 / 63.72 | 28.36 / 27.77 |
| grouped | 15 | 15.23 / 8.32 | 84.11 / 77.20 | 31.27 / 30.73 |

(k = 13 and 15 are close to the top of the searched range 1-15; the floor is flat there, so this does not change the conclusion.)

## Flat MLP (`baselines/flat_mlp.py`), no graph

Per (trajectory, agent): mean-pooled cached MiniLM turn embeddings (384) + framework one-hot + agent-name one-hot + 4 count scalars (432 dims).
MLP 2x256, dropout 0.3, AdamW 1e-3, BCE on 14 mode logits + 1 faulty-agent logit, early stopping on val loss, thresholds
(t_a, t_m) tuned on val, at least one pair forced per trajectory. 3 seeds, mean +- std.

| Split | Pair micro | Pair macro | Agent micro | Agent macro | Error micro | Error macro |
|---|---|---|---|---|---|---|
| paper | 32.19 +- 3.36 | 24.04 +- 1.41 | 82.33 +- 3.02 | 73.21 +- 2.32 | 41.26 +- 1.40 | 41.11 +- 1.15 |
| grouped | 23.05 +- 0.19 | 18.99 +- 1.13 | 83.95 +- 0.20 | 77.53 +- 0.58 | 36.01 +- 0.21 | 35.67 +- 0.49 |

## Reading the results

1. **Query leakage matters.** Pair micro-F1 drops from 32.2 to 23.1 and error micro from 41.3 to 36.0 when the same queries are removed
   from training. Results on the released split overstate generalization. Use the grouped split for claims.
2. **Content carries signal beyond frequency.** Flat MLP beats the frequency baseline clearly at pair level (23.1 vs 15.2 grouped)
   and error level (36.0 vs 31.3), while agent level is near the floor for every method (84 vs 84). The agent-level metric barely separates methods.
3. The flat MLP is the bar the graph model must clear in Phase 5. The floor and frequency rows go in the final table.
4. Caveats: thresholds on the grouped split sit at the lower edge of the grid (0.1); a finer grid could add a little. Not tuned further
   to avoid overfitting val.

## Flat MLP with top-k decoding (`baselines/flat_topk.py`, phase `4b`), validation only

Same trained MLP and probabilities as above; only the decoder changes, using label structure from `error_analysis.md`
(never-faulty roles masked, k faulty agents per framework taken from train, argmax mode per agent plus extra modes above t_extra tuned on val).
3 seeds, mean +- std, micro / macro F1 in percent.

| Split | Decoder | Pair | Agent | Error |
|---|---|---|---|---|
| paper | threshold | 32.19 / 24.04 | 82.33 / 73.21 | 41.26 / 41.11 |
| paper | top-k | 31.04 / 20.84 | 82.32 / 66.08 | 38.50 / 37.70 |
| grouped | threshold | 23.05 / 18.99 | 83.95 / 77.53 | 36.01 / 35.67 |
| grouped | top-k | 24.06 / 16.68 | 84.46 / 75.01 | 32.66 / 30.55 |

Verdict: **top-k does not help overall.** On the grouped split it gains +1.0 pair micro and +0.5 agent micro, but loses 3.4 error micro and
every macro number (pair macro -2.3, agent macro -2.5, error macro -5.1). On the paper split it is worse or equal on all six micro and macro numbers.
The hypothesis that a fixed per-framework agent count is a better decoder is not supported. The tuned t_extra sits at the lower grid edge (0.3) in every run,
so the best settings add many extra modes per agent, which moves toward the "predict everything" floor. The threshold decoder stays the default for Phase 5.
The two-feature probe (cosine similarity to the query and to peers) from `error_analysis.md` has not been run.

## Relational probe (`baselines/flat_probe.py`, phase `4c`), validation only

Flat MLP plus 4 extra features per (trajectory, agent): cosine of the agent's mean turn vector to the query, max turn-to-query cosine,
cosine to the mean of the other agents' turns, min turn-to-peers cosine. Query vectors are cached MiniLM embeddings (1,457 new queries embedded locally,
77 s). Same model, decoder and tuning as the flat MLP. 3 seeds, mean +- std, micro / macro F1 in percent.

| Split | Features | Pair | Agent | Error |
|---|---|---|---|---|
| paper | base | 32.19 +- 3.36 / 24.04 | 82.33 +- 3.02 / 73.21 | 41.26 +- 1.40 / 41.11 |
| paper | + relational | 34.17 +- 0.13 / 24.21 | 80.15 +- 1.11 / 70.64 | 41.50 +- 0.16 / 41.13 |
| grouped | base | 23.05 +- 0.19 / 18.99 | 83.95 +- 0.20 / 77.53 | 36.01 +- 0.21 / 35.67 |
| grouped | + relational | 23.61 +- 0.27 / 19.35 | 84.16 +- 0.40 / 77.63 | 36.41 +- 0.11 / 35.68 |

Verdict: **small, consistent gain on the grouped split; inconclusive on the paper split.**
- Grouped: pair micro rises in all three seeds (22.91 to 23.49, 23.31 to 23.98, 22.92 to 23.35), mean +0.56; error micro +0.40; agent micro +0.21 (within noise); macro numbers move by 0.1 to 0.4.
- Paper: pair micro +2.0, but the base seeds vary by 3.4 (seed 2 is the weak one), and agent micro drops 2.2, so I would not read this as a gain.
- The gain on the grouped split is about 2 seed standard deviations on pair micro and far smaller than the gap to what a trained graph model would need to show. This is weak evidence that peer/query comparison carries a little signal. It does not by itself justify the graph; it sets the bar: a graph model should clearly beat 23.6 pair micro on the grouped split to be worth its complexity.
- Test was not touched.
