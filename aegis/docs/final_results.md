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
