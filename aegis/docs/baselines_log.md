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

## Turn-graph GNN (`gnn/turn_gnn.py`, `graph.py`, phase `5`), validation only

Graph per trajectory: one node per turn (frozen MiniLM vector) plus a query node; edges for consecutive turns, consecutive turns of the same agent, and
query-to-turn / turn-to-query (decision D3 plus the query node). 128-d, relation-specific mean aggregation, residual + LayerNorm. Agent vector = mean of its turn nodes,
concatenated with the same framework / agent-name / count features as the flat MLP, then an MLP head. Same loss, decoder, early stopping and threshold tuning as the flat MLP.
"0 layers" is the control: same projection and pooling but no message passing. 3 seeds, mean +- std, micro / macro F1 in percent.
Raw logs: `aegis/results/runs/gnn_L2.log`, `gnn_L0.log`. Ledger: phase `5`. Test not touched. The GNN was run without the guide's approval of the dataset switch having been confirmed in this log; treat these as exploratory.

| Split | Model | Params | Pair | Agent | Error |
|---|---|---|---|---|---|
| grouped | flat MLP | - | 23.05 +- 0.19 / 18.99 | 83.95 +- 0.20 / 77.53 | 36.01 +- 0.21 / 35.67 |
| grouped | flat MLP + relational probe | - | 23.61 +- 0.27 / 19.35 | 84.16 +- 0.40 / 77.63 | 36.41 +- 0.11 / 35.68 |
| grouped | pooled-projection, 0 layers | 164k | 25.63 +- 0.49 / 21.26 | 83.74 +- 0.22 / 75.74 | 37.46 +- 0.34 / 37.30 |
| grouped | GNN, 2 layers | 329k | 26.47 +- 0.52 / 21.00 | 83.94 +- 0.51 / 76.54 | 38.04 +- 0.32 / 37.59 |
| paper | flat MLP | - | 32.19 +- 3.36 / 24.04 | 82.33 +- 3.02 / 73.21 | 41.26 +- 1.40 / 41.11 |
| paper | pooled-projection, 0 layers | 164k | 31.36 +- 2.92 / 22.59 | 84.05 +- 2.88 / 71.70 | 41.02 +- 0.95 / 40.96 |
| paper | GNN, 2 layers | 329k | 36.88 +- 0.29 / 24.75 | 80.97 +- 0.46 / 68.77 | 44.04 +- 0.18 / 43.58 |

### Reading the results
1. **Most of the grouped-split gain over the flat MLP is not the graph.** The 0-layer control already gains +2.6 pair micro and +1.5 error micro over the flat MLP.
   It differs from the flat MLP by projecting each turn before pooling (instead of pooling raw vectors) and having a different optimiser path, so the gain is credited to
   per-turn nonlinear projection, not to message passing.
2. **Message passing adds a small amount on grouped.** GNN vs 0 layers: pair micro +0.84 (higher in 2 of 3 seeds: 26.20 vs 26.28, 26.01 vs 25.11, 27.21 vs 25.52),
   error micro +0.58 (higher in all 3 seeds), agent micro +0.20, pair macro -0.26, error macro +0.29. With 3 seeds and std about 0.5 this is suggestive, not established.
3. **On the paper split the graph gap is larger** (pair +5.5, error +3.0 vs the 0-layer control) but that split has query leakage, so a model with more capacity to
   memorise query-specific patterns gains more there. Do not use it as evidence for the graph; claims rest on grouped.
4. **Agent-level micro is flat across all models** (about 84 on grouped), consistent with the trivial floor (73-84). Agent macro is slightly lower for the GNN than for the flat MLP.
5. Caveats: the paper-split 0-layer seed 0 hit the 40-epoch cap, so that control may be under-trained; no hyper-parameter search was done for any model;
   one graph design (D3 + query node) was tried; the encoder is frozen MiniLM.
6. Next step before any claim: seeds 3 -> 5 on grouped for GNN vs 0 layers, an edge-type ablation (drop same-agent, drop query), then test once.

## GNN ablation, grouped split, 5 seeds (`gnn/turn_gnn.py --drop`, phase `5`), validation only

Seeds 0-4, mean +- std, micro / macro F1 in percent. "Drop X" removes that edge type from message passing in both layers
(the unused weights still count in the parameter total). Logs: `aegis/results/runs/g5_*.log`. Test not touched.

| Model | Pair | Agent | Error |
|---|---|---|---|
| flat MLP (3 seeds, from above) | 23.05 +- 0.19 / 18.99 | 83.95 / 77.53 | 36.01 / 35.67 |
| 0 layers (no message passing) | 25.66 +- 0.40 / 21.35 | 84.10 +- 0.50 / 76.36 | 37.67 +- 0.38 / 37.56 |
| GNN, all edges | 26.86 +- 0.61 / 21.40 | 84.05 +- 0.69 / 75.50 | 38.18 +- 0.56 / 37.73 |
| GNN, drop same-agent edges | 26.56 +- 0.68 / 20.90 | 83.62 +- 0.97 / 76.16 | 38.45 +- 0.40 / 38.12 |
| GNN, drop sequence edges | 26.62 +- 0.53 / 20.95 | 83.09 +- 0.64 / 75.69 | 38.43 +- 0.41 / 38.01 |
| GNN, drop query node | 25.55 +- 0.40 / 19.81 | 82.98 +- 1.83 / 75.81 | 37.08 +- 0.81 / 36.47 |

### Reading the results
1. **The query node carries the graph's gain.** Removing it drops pair micro by 1.3 (26.86 to 25.55), which is below the 0-layer control (25.66). Removing the same-agent edges (-0.30) or the
   sequence edges (-0.24) changes pair micro by less than one standard deviation, and error micro goes up slightly (38.18 to 38.45 / 38.43). So, in this design, the useful
   message passing is "each turn sees the query", which is the same signal the earlier cosine-to-query probe suggested. Turn-order and same-agent structure show no measurable benefit here.
2. **Graph vs control:** GNN is above the 0-layer control on pair micro in 5 of 5 seeds (+1.2 on average) but on error micro in only 3 of 5 (+0.5 on average, inside noise).
   Agent-level numbers are the same everywhere (about 84 micro), and macro F1 does not separate the models.
3. **Run-to-run noise is larger than the seed std suggests.** Seeds 0-2 of the full GNN gave pair micro 26.20 / 26.01 / 27.21 in the earlier 3-seed run and 27.62 / 26.94 / 27.13 in this
   run with identical seeds and code (the 0-layer control reproduced exactly). The likely cause is non-deterministic summation order in the edge aggregation (`index_add_`) amplified by
   early stopping, but this was not isolated. Treat differences below about 1 point as unresolved.
4. **Honest summary for the write-up:** a frozen-encoder model that projects each turn and lets turns see the query beats the flat mean-pooled MLP by about 3.8 pair micro and 2.2 error micro
   on the grouped split. About two thirds of that comes from the per-turn projection, the rest from query-aware message passing. There is no evidence here that turn-to-turn graph structure
   (sequence, same-agent edges) helps. Agent-level micro-F1 remains at the trivial floor for every model.
