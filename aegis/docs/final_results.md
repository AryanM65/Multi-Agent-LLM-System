# Final held-out results (Phase 6 re-run, 2026-11-12)

Protocol: train on a split's train set, early-stop and choose thresholds on its validation set, then score the 600-record test set and
Who&When once. 5 seeds, mean +- std. Micro-F1 / macro-F1 in percent. Logs: `aegis/results/runs/final_test_rerun_grouped.log`,
`final_test_rerun_paper.log`. Ledger rows: phase `6r` (they supersede the phase-6 Who&When rows, which were wrong; old rows are kept).

## Bug fixed
Standardisation used `sd = std + 1e-6`. The `who_when` framework one-hot never fires in train, so on Who&When that column became 1,000,000,
the sigmoids saturated, and every model emitted the same single pair (identical scores, zero spread). Fix: columns with std < 1e-3 get
`sd = 1` (`flat_mlp.py`, `turn_gnn.py`). Test and validation inputs are unchanged by the fix; test numbers reproduce the earlier run.

## Test set, grouped protocol (leakage removed)
| Model | Pair | Agent | Error |
|---|---|---|---|
| floor | 7.35 / 6.35 | 68.58 / 61.98 | 27.14 / 26.99 |
| frequency (k=13) | 12.14 / 5.35 | 80.45 / 72.05 | 28.47 / 28.34 |
| flat MLP | 19.83 / 12.99 | 78.74 / 65.27 | 33.01 / 32.62 |
| 0-layer control | 20.91 / 14.47 | 79.43 / 67.23 | 34.74 / 34.52 |
| 2-layer GNN | 21.87 / 14.46 | 78.62 / 67.28 | 35.30 / 34.85 |

## Test set, paper protocol (released split, leaky)
| Model | Pair | Agent | Error |
|---|---|---|---|
| floor | 7.35 / 6.35 | 68.58 / 61.98 | 27.14 / 26.99 |
| frequency (k=13) | 13.85 / 6.04 | 80.37 / 69.84 | 29.02 / 28.48 |
| flat MLP | 33.08 / 21.62 | 80.86 / 67.67 | 40.43 / 39.60 |
| 0-layer control | 31.37 / 21.99 | 82.52 / 68.63 | 39.82 / 39.57 |
| 2-layer GNN | 36.20 / 24.15 | 82.12 / 70.99 | 43.18 / 42.34 |

Seed std on the paper split is 2.4-2.9 points on pair micro, so the GNN lead there (+3.1 over the flat MLP, +4.8 over the control)
is about one std and is not established. On the grouped split the GNN is +1.0 pair / +0.6 error over the control, also within noise.

## Who&When (pair micro / agent micro / error micro)
| Model | grouped | paper |
|---|---|---|
| floor | 3.42 / 38.32 / 13.33 | same |
| flat MLP | 4.25 / 39.60 / 10.83 | 1.89 / 34.43 / 6.73 |
| 0-layer control | 2.94 / 33.55 / 9.89 | 1.71 / 30.11 / 7.78 |
| 2-layer GNN | 3.60 / 38.04 / 10.22 | 1.98 / 35.47 / 8.34 |

All models are at or below the trivial floor on Who&When (free-form agent names, different domain). No claim is made there.

## Reading
- Agent-level micro-F1 sits near the floor for every model; pair and error level carry the signal.
- Question leakage inflates pair micro by about 12-14 points (paper vs grouped).
- Earlier claim "the GNN gain over the control vanishes on test" is softened: +1.0 pair on grouped test, still within noise.

# Fine-tuned encoder (phase 6ft, 2026-11-12)

MiniLM-L6 fine-tuned end-to-end on the GROUPED train split (Kaggle T4, `aegis/scripts/kaggle_finetune_minilm.py`; flat-MLP head, BCE,
encoder lr 2e-5, head lr 1e-3, early stop on grouped-val loss: best epoch 3, val BCE 0.1595 -> 0.1387). Then every turn/query was
re-embedded and the same downstream models were re-run. Test and Who&When were never used for training or selection. Logs:
`aegis/results/kaggle_logs/ft_full/`, `aegis/results/runs/ft_eval.log`, `final_test_finetuned_grouped.log`.
Note: the encoder's early stopping used the grouped validation set (slightly optimistic validation numbers; test is clean).

## Grouped validation, frozen vs fine-tuned (3 seeds, micro pair / agent / error)
| Model | Frozen | Fine-tuned |
|---|---|---|
| flat MLP | 22.93 / 83.71 / 35.93 | 36.93 / 80.22 / 44.29 |
| 0-layer control | 25.69 / 83.90 / 37.70 | 33.40 / 85.33 / 43.02 |
| 2-layer GNN | 26.93 / 83.82 / 38.16 | 35.60 / 86.08 / 44.82 |

## Test set, grouped protocol, 5 seeds (micro pair / agent / error)
| Model | Frozen (phase 6r) | Fine-tuned (phase 6ft) |
|---|---|---|
| flat MLP | 19.83 / 78.74 / 33.01 | 29.05 / 79.08 / 38.80 |
| 0-layer control | 20.91 / 79.43 / 34.74 | 29.68 / 78.05 / 39.39 |
| 2-layer GNN | 21.87 / 78.62 / 35.30 | 28.46 / 82.38 / 40.25 |

Fine-tuning adds about +7 to +9 pair micro and +4 to +6 error micro on test (seed std 1.5-2.9), well outside the seed spread.
The graph does not add to the fine-tuned flat model on test (pair 28.5 vs 29.1); the 2-layer GNN is best on agent (82.4) and error (40.3).
Who&When stays at or below the floor (pair micro 2.1-3.7), so no claim is made there.


# Clean-encoder results (2026-11-12): encoder early stopping on a held-out slice of TRAIN, validation untouched

Setup: MiniLM fine-tuned end to end (`aegis/scripts/kaggle_finetune_minilm.py --es-holdout 0.15`): 3,234 grouped-train trajectories for training,
631 (115 held-out questions) for the encoder's early stopping, the 940 grouped-validation trajectories never seen by the encoder.
Two encoder seeds (es_s0, es_s1; best epoch 4 in both). Downstream models (flat MLP, 0-layer control, 2-layer GNN) are trained on the vectors,
thresholds tuned on grouped validation, then the 600-record test set and Who&When are scored once, 5 seeds each.
Logs: `aegis/results/kaggle_logs/ft_es_s{0,1}/`, `aegis/results/runs/ft_eval_es_s{0,1}.log`, `final_test_es_s{0,1}.log`, `whowhen_study_es_s{0,1}.log`.
The encoder here saw 16% less training data than the earlier encoder (3,234 vs 3,865), so numbers are not directly comparable with the section above.

## Grouped validation (3 seeds per encoder, mean of the two encoders; micro pair / agent / error)
| Model | Clean encoders | Earlier encoder (early-stopped on val, optimistic) |
|---|---|---|
| flat MLP | 31.66 / 86.19 / 42.49 | 36.93 / 80.22 / 44.29 |
| 0-layer control | 31.35 / 86.78 / 42.06 | 33.40 / 85.33 / 43.02 |
| 2-layer GNN | 33.05 / 87.83 / 43.85 | 35.60 / 86.08 / 44.82 |

## Test set, grouped protocol, base features (5 seeds; micro pair / agent / error; mean of the two clean encoders)
| Model | es_s0 | es_s1 | Mean |
|---|---|---|---|
| floor | 7.35 / 68.58 / 27.14 | same | 7.35 / 68.58 / 27.14 |
| flat MLP | 27.05 / 81.03 / 38.68 | 25.80 / 81.34 / 38.45 | 26.43 / 81.19 / 38.57 |
| 0-layer control | 27.91 / 79.79 / 39.11 | 25.21 / 80.98 / 38.21 | 26.56 / 80.39 / 38.66 |
| 2-layer GNN | 27.14 / 82.67 / 39.23 | 27.46 / 81.11 / 39.09 | 27.30 / 81.89 / 39.16 |
Avg12 (paper definition, includes Who&When): flat 30.73 / 30.92, control 29.85 / 30.36, GNN 30.18 / 30.23; floor 25.59.
The encoder seed moves test pair by about 1 point, as much as the differences between models. The GNN edge over the flat MLP (+0.9 pair, +0.6 error)
is suggestive, not established.

## Who&When study (zero-shot: no Who&When label is used for training or tuning; flat MLP; 5 seeds; mean of the two clean encoders)
Floor on Who&When: pair 3.42, agent 38.32, error 13.33 (micro).
| Variant | Who&When pair / agent / error (micro) | AEGIS test pair / agent / error (micro) |
|---|---|---|
| base (framework + agent-name one-hot) | 4.16 / 43.00 / 11.15 | 26.43 / 81.19 / 38.57 |
| name-free (those columns zeroed) | 5.29 / 45.48 / 12.74 | 25.21 / 77.29 / 37.84 |
| name-embedding (name-free + MiniLM embedding of the agent-name text) | 5.04 / 40.85 / 15.04 | **29.41 / 79.78 / 39.09** |
Per encoder (pair / agent / error micro on Who&When): es_s0 base 3.78 / 43.71 / 9.92, namefree 5.74 / 47.58 / 12.52, nameemb 4.39 / 40.44 / 13.61;
es_s1 base 4.53 / 42.28 / 12.37, namefree 4.83 / 43.37 / 12.96, nameemb 5.69 / 41.26 / 16.46.
Top-1 decoding (one pair per record) was worse than thresholds for every variant and is not used.

Published Who&When rows (paper Table 1, micro pair / agent / error): DCL 1.60 / 8.40 / 14.67; Aegis-SFT 4.03 / 51.14 / 9.87; o3 7.41 / 53.10 / 14.88.
Reading: the name-embedding variant is the best overall, with a gain that replicates on both clean encoders: AEGIS test pair +3.0 over base
(+2.2 and +3.8 per encoder; within noise of a single seed but consistent), Who&When error 15.04 (above the floor 13.33, DCL 14.67, o3 14.88, Aegis-SFT 9.87)
and pair 5.04 (above the floor 3.42 and Aegis-SFT 4.03, below o3 7.41). Agent level on Who&When stays near the floor (40.85 vs 38.32) and far below Aegis-SFT/o3 (51-53);
the name-free variant helps agent level most (45.48) but costs about 4 pair points in-domain.
Caveats: (1) the three variants were defined after looking at Who&When results from the earlier encoder, so there is mild selection on the evaluation set;
(2) Who&When has 184 records, 20 of whose labels name an agent absent from the turns (maximum agent recall about 89%);
(3) seed noise is large (test pair std up to 2.9); (4) Who&When labels were mapped by Gemini-2.5-Flash from human notes;
(5) the clean encoders were trained on 3,234 trajectories, not 3,865.


# End-to-end graph fine-tune (2026-11-12): does the graph help when the encoder is trained through it?

Setup (`aegis/scripts/kaggle_finetune_graph.py`, Kaggle T4, one seed, seed 0): MiniLM and the turn-graph GNN trained together (encoder lr 2e-5, rest 1e-3, bs 8, 6 epochs,
about 130 s/epoch), early stopping on a held-out slice of TRAIN (3,234 train / 631 early-stop / 940 validation untouched). `--layers 2` = graph with message passing;
`--layers 0` = control (same model, no message passing). Predictions exported and scored on CPU with `aegis/src/eval/score_graph_ft.py`
(thresholds tuned on grouped validation only; test and Who&When scored once). Logs: `aegis/results/kaggle_logs/ftg_l2/`, `ftg_l0/`, `aegis/results/runs/score_ftg_l{2,0}.log`.
Best early-stop loss: layers 2 0.1431 (epoch 4), layers 0 0.1440 (epoch 6): essentially equal.

| Model (micro pair / agent / error) | Grouped validation | Grouped test | Who&When (base features; floor 3.42 / 38.32 / 13.33) |
|---|---|---|---|
| End-to-end, 0 layers (control) | 30.74 / 86.56 / 42.15 | 26.00 / 80.88 / 38.38 | 2.40 / 41.18 / 9.76 |
| End-to-end, 2-layer graph | 29.59 / 87.74 / 39.88 | 23.69 / 80.15 / 35.22 | 2.90 / 42.55 / 7.54 |
| Two-stage (fine-tuned encoder, then models on its vectors), flat MLP, mean of 2 encoders | 31.66 / 86.19 / 42.49 | 26.43 / 81.19 / 38.57 | 4.16 / 43.00 / 11.15 |
| Two-stage, 0-layer control | 31.35 / 86.78 / 42.06 | 26.56 / 80.39 / 38.66 | n/a (see final_test logs) |
| Two-stage, 2-layer GNN | 33.05 / 87.83 / 43.85 | 27.30 / 81.89 / 39.16 | n/a (see final_test logs) |
Avg12 (paper definition): end-to-end control 30.72, end-to-end graph 29.06; floor 25.59.
Per-framework test pair micro (end-to-end graph vs control): agentverse 25.2 vs 29.1, dylan 18.1 vs 18.3, llm_debate 28.8 vs 28.9, macnet 25.9 vs 28.6,
magentic_one 19.2 vs 18.4 (n=28), smoagents 4.7 vs 7.8 (n=48): the graph is not better in any framework with a meaningful sample.

Reading (honest): training the encoder through the graph does NOT help. The 2-layer end-to-end model is 2.3 pair points below its own 0-layer control on test
(23.7 vs 26.0) and 1.2 below it on validation, and both end-to-end models are at or below the simpler two-stage pipeline (about 26.4-27.3 test pair).
This is one seed and one encoder run, and seed noise on test pair is about 1-3 points, so the gap between the graph and its control is suggestive, not established;
what the evidence does support is that there is no sign of a graph gain here. Together with the two-stage result (GNN +0.9 pair over flat, within noise),
the defensible claim is: a graph over the turns adds at most a small, unproven gain; most of the improvement comes from fine-tuning the text encoder.
Who&When with the base (name-dependent) features stays at or below the floor for both end-to-end models.


## End-to-end graph fine-tune, name-free variant (framework and agent-name one-hot columns zeroed; same setup, one seed)
Logs: `aegis/results/kaggle_logs/ftg_l2nf/`, `ftg_l0nf/`, `aegis/results/runs/score_ftg_l2nf.log`, `score_ftg_l0nf.log`.
| Model (micro pair / agent / error) | Grouped validation | Grouped test | Who&When (floor 3.42 / 38.32 / 13.33) |
|---|---|---|---|
| E2E name-free, 0 layers (control) | 23.72 / 83.77 / 35.88 | 20.19 / 77.77 / 33.38 | 4.41 / 47.62 / 9.97 |
| E2E name-free, 2-layer graph | 27.86 / 87.00 / 40.81 | 22.90 / 81.95 / 35.72 | 5.04 / 39.18 / 9.90 |
Avg12: control 27.97, graph 29.47 (floor 25.59). Best early-stop loss: graph 0.1477 (epoch 5), control 0.1544 (epoch 4).
Reading: here the graph beats its control by +2.7 pair on test and +4.1 on validation, the opposite direction from the name-dependent pair above (graph 2.3 below control).
Each pair is a single seed, so the sign of the graph effect is not stable across the two comparisons; this is consistent with noise (seed std on test pair is about 1-3 points)
and does not support a claim that the graph helps or hurts. Name-free end-to-end models are clearly worse in-domain than name-dependent ones (control 20.2 vs 26.0) and
reach Who&When pair 4.4-5.0 (above the floor) but error 9.9-10.0 (below the floor 13.33); the two-stage name-embedding flat MLP above remains the best Who&When setting (pair 5.04 / error 15.04 on average).
