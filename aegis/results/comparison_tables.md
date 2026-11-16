# Comparison tables (generated; do not edit by hand)

Micro-F1 in percent. 'reported' rows are copied from the AEGIS paper's Table 1 (arXiv 2509.14295v6); 'ours' rows are run by us (frozen MiniLM + flat MLP, 5 seeds) with record-level bootstrap 95% intervals in brackets. The paper does not name its test file; the authors' repo file `evaluation/data/aegis.jsonl` matches its description (100 per benchmark) and is primary (decision D11).

## AEGIS-Bench (pair / agent / error micro-F1)

| Row | Source | Pair | Agent | Error |
|---|---|---|---|---|
| Random | reported | 0.33 | 4.54 | 11.23 |
| DCL | reported | 8.33 | 22.93 | 24.73 |
| Qwen2.5-7B-Instruct (zero-shot) | reported | 5.02 | 27.55 | 14.96 |
| Qwen2.5-14B-Instruct (zero-shot) | reported | 5.47 | 35.78 | 20.24 |
| o3 (zero-shot) | reported | 7.86 | 40.31 | 22.37 |
| Aegis-SFT | reported | 16.62 | 76.53 | 27.53 |
| Aegis-GRPO | reported | 6.84 | 49.74 | 21.19 |
| *Trivial floor (repo_test)* | ours | 6.47 [6.2, 6.7] | 62.00 [59.7, 64.3] | 26.19 [25.4, 27.0] |
| *Trivial floor (hf_test)* | ours | 7.35 [7.1, 7.6] | 68.58 [67.1, 70.0] | 27.14 [26.4, 27.9] |
| frequency baseline, released train/val, repo_test | ours | 11.92 [11.0, 12.9] | 69.83 [67.0, 72.4] | 27.43 [26.3, 28.5] |
| frequency baseline, released train/val, hf_test | ours | 13.85 [13.0, 14.7] | 80.37 [79.0, 81.7] | 29.02 [27.9, 30.1] |
| shortcut-only (no text), released train/val, repo_test | ours | 11.90 [10.3, 13.6] | 59.07 [56.2, 61.9] | 24.09 [22.0, 26.1] |
| shortcut-only (no text), released train/val, hf_test | ours | 14.30 [12.8, 15.8] | 69.14 [67.3, 71.0] | 26.38 [24.3, 28.4] |
| content-only (no name/framework), released train/val, repo_test | ours | 20.28 [18.7, 21.7] | 71.14 [68.3, 73.9] | 35.23 [33.5, 36.9] |
| content-only (no name/framework), released train/val, hf_test | ours | 23.22 [21.7, 24.6] | 76.90 [75.4, 78.3] | 36.43 [34.7, 38.1] |
| full model, released train/val, repo_test | ours | 26.15 [24.0, 28.4] | 68.45 [65.5, 71.2] | 37.29 [35.2, 39.3] |
| full model, released train/val, hf_test | ours | 33.02 [30.8, 35.1] | 80.72 [79.1, 82.2] | 40.36 [38.0, 42.5] |
| frequency baseline, released train/val, test-duplicates removed, repo_test | ours | 10.77 [10.0, 11.5] | 69.55 [66.7, 72.2] | 27.24 [26.3, 28.2] |
| frequency baseline, released train/val, test-duplicates removed, hf_test | ours | 12.81 [12.0, 13.5] | 80.45 [79.2, 81.7] | 28.87 [27.9, 29.9] |
| shortcut-only (no text), released train/val, test-duplicates removed, repo_test | ours | 10.75 [9.2, 12.4] | 58.35 [55.5, 61.2] | 23.54 [21.5, 25.7] |
| shortcut-only (no text), released train/val, test-duplicates removed, hf_test | ours | 11.74 [10.3, 13.3] | 68.37 [66.6, 70.1] | 24.93 [22.9, 26.9] |
| content-only (no name/framework), released train/val, test-duplicates removed, repo_test | ours | 16.79 [15.5, 18.0] | 69.62 [66.9, 72.3] | 30.89 [29.3, 32.4] |
| content-only (no name/framework), released train/val, test-duplicates removed, hf_test | ours | 18.77 [17.6, 19.9] | 75.17 [73.7, 76.6] | 32.60 [31.1, 34.0] |
| full model, released train/val, test-duplicates removed, repo_test | ours | 20.02 [18.5, 21.5] | 68.45 [65.6, 71.0] | 31.43 [29.7, 33.0] |
| full model, released train/val, test-duplicates removed, hf_test | ours | 23.84 [22.3, 25.4] | 79.13 [77.6, 80.5] | 34.13 [32.4, 35.8] |

## Who&When (zero-shot: no Who&When label used for training or tuning)

| Row | Source | Pair | Agent | Error |
|---|---|---|---|---|
| Random | reported | 0.11 | 1.06 | 8.74 |
| DCL | reported | 1.60 | 8.40 | 14.67 |
| Qwen2.5-7B-Instruct (zero-shot) | reported | 2.31 | 40.92 | 3.64 |
| Qwen2.5-14B-Instruct (zero-shot) | reported | 0.00 | 49.88 | 1.56 |
| o3 (zero-shot) | reported | 7.41 | 53.10 | 14.88 |
| Aegis-SFT | reported | 4.03 | 51.14 | 9.87 |
| Aegis-GRPO | reported | 2.45 | 54.43 | 4.15 |
| *Trivial floor (who_when)* | ours | 3.42 [3.1, 3.8] | 38.32 [35.0, 41.8] | 13.33 [13.3, 13.3] |
| shortcut-only (no text), released train/val, who_when | ours | 4.75 [2.6, 7.0] | 34.08 [28.8, 39.7] | 11.33 [8.3, 14.6] |
| content-only (no name/framework), released train/val, who_when | ours | 3.48 [2.2, 4.9] | 41.75 [37.0, 46.6] | 10.79 [8.6, 13.2] |
| full model, released train/val, who_when | ours | 1.85 [0.8, 3.2] | 33.97 [29.1, 39.2] | 8.05 [5.8, 10.7] |
| shortcut-only (no text), released train/val, test-duplicates removed, who_when | ours | 4.95 [2.7, 7.5] | 38.08 [33.2, 43.0] | 10.87 [7.5, 14.7] |
| content-only (no name/framework), released train/val, test-duplicates removed, who_when | ours | 2.71 [1.6, 3.9] | 40.45 [35.7, 45.4] | 10.22 [8.1, 12.4] |
| full model, released train/val, test-duplicates removed, who_when | ours | 1.96 [1.0, 3.1] | 33.88 [29.0, 39.2] | 7.63 [5.6, 10.0] |

## Effect of verbatim-duplicate leakage on the full model (pair micro-F1)

| Test file | Full model, released train/val | Same, duplicates removed | Drop (pair) |
|---|---|---|---|
| repo_test | 26.15 [24.0, 28.4] | 20.02 [18.5, 21.5] | 6.13 |
| hf_test | 33.02 [30.8, 35.1] | 23.84 [22.3, 25.4] | 9.19 |

Notes: (1) reported agent-level numbers sit below our trivial floor; we report them as published and do not interpret them. (2) The shortcut-only model never reads the conversation. (3) Who&When rows here are the simple model, not the ensemble with block-dropout and label shift in the draft; those need the S4 runs.
