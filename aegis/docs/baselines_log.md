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
