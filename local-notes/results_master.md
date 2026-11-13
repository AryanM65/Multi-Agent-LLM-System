# Master results record: every approach tried and every result (AEGIS-Bench and Who&When)

LOCAL ONLY (`local-notes/`, git-excluded). Never commit or push. Written 2026-11-12 from `aegis/docs/final_results.md`, `local-notes/context.md` section 7, the run logs in `aegis/results/runs/`, and the ledger `aegis/results/metrics_history.jsonl`. If a number here disagrees with a log, the log wins.
All numbers are percent, micro-F1 unless "macro" is stated, order "pair / agent / error". "Floor" = predict every present agent with all 14 modes. avg12 = mean of the 12 F1 numbers (paper Table 1 "Avg.").
Companion files: `approaches_tried.md` (narrative), `ideas.md` (next round), `loop.md` (the loop), `file.md`, `context.md`, `paper_review.md`.

## 0. Reference rows

| Row | Value |
|---|---|
| AEGIS test floor | pair 7.35 / 6.35 (micro/macro), agent 68.58 / 61.98, error 27.14 / 26.99 |
| Who&When floor | 3.42 / 2.85, 38.32 / 38.41, 13.33 / 12.34 |
| avg12 floor | 25.59 |
| Paper: Random | avg 4.08 (AB 0.33/0.21, 4.54/3.56, 11.23/11.15) |
| Paper: DCL (3 seeds) | AB 8.33/5.30, 22.93/20.23, 24.73/27.70; W&W 1.60/0.77, 8.40/6.07, 14.67/10.57; avg 12.61 |
| Paper: Qwen2.5-7B zero-shot | AB 5.02/2.52, 27.55/14.49, 14.96/11.36; W&W 2.31/1.14, 40.92/23.50, 3.64/1.77; avg 12.43 |
| Paper: Qwen2.5-14B | avg 13.99 |
| Paper: Aegis-SFT | avg 26.51; W&W 4.03 / 51.14 / 9.87; AB pair 16.6, agent 76.5, error 27.5 |
| Paper: Aegis-GRPO | avg 18.41 |
| Paper: o3 | avg 20.24; W&W 7.41 / 53.10 / 14.88 |
Note: the paper's agent-level numbers (DCL 22.9, Qwen 27.6) sit below the trivial floor (68.6): unexplained.

## 1. Setup facts that shape every result
- AEGIS-Bench: 9,533 trajectories, 6 frameworks, 14 modes; released split train 7,146 / val 1,787 / test 600 (HF `test.jsonl` used as primary).
- Question leakage: 588 of 600 test queries also appear in train (430 of 438 unique), so the released split leaks.
- Grouped split (ours, query-disjoint): train 3,865 / val 940; test unchanged (600).
- Clean encoder protocol: encoder early-stopped on a held-out slice of train (3,234 train / 631 early-stop), validation untouched.
- Who&When: 184 records, exactly 1 faulty agent each, free-form agent names; 20 labels name an agent absent from the turns (max agent recall about 89%); labels mapped to the 14 modes by Gemini-2.5-Flash; mode mix skewed (FM-2.6 26%, FM-3.3 21%).
- Leak-proof loader: never uses `num_agents`, `num_injected_agents`, `ground_truth.*`, `injection_strategy`, per-turn `is_injected`.
- Seed noise: test pair std 1.5-2.9 for the fine-tuned models, 2.4-2.9 on the paper split; same-seed GNN runs differed up to about 1.4 pair before the one-thread fix.

## 2. Zero-shot Qwen2.5-7B-Instruct-AWQ reproduction (phase 3a, Kaggle `aegis-zeroshot-full`)
vLLM on T4, authors' prompt, greedy, 512 new tokens, 16,384-token context (25 of 600 test and 25 of 184 Who&W prompts truncated); 8 of 600 test outputs unparseable.
| Row | Ours | Paper |
|---|---|---|
| AB pair micro/macro | 8.01 / 2.48 | 5.02 / 2.52 |
| AB agent | 49.24 / 10.69 | 27.55 / 14.49 |
| AB error | 20.15 / 12.72 | 14.96 / 11.36 |
| W&W pair | 2.98 / 1.85 | 2.31 / 1.14 |
| W&W agent | 35.16 / 23.14 | 40.92 / 23.50 |
| W&W error | 6.23 / 3.35 | 3.64 / 1.77 |
| avg12 | 14.67 | 12.43 |
Unexplained: AB agent micro 49.2 vs 27.6. Untested explanations: other test file, 4-bit weights, prompt variant, parse rule, sampling vs greedy.

## 3. DCL-style rebuild (phase 3b; frozen encoder, 3 seeds, test)
Not the real DCL: frozen encoder, about 0.2M params, lr 1e-3 with early stopping, batch 64, no clean-twin positives, agent prototypes keyed by name text, bilinear pair term, mean of top-3 turn scores, single threshold.
- Paper-split trained: AB pair 15.06 / 8.56, agent 69.88 / 57.22, error 34.34 / 33.23; W&W pair 3.09 / 1.04, agent 37.95 / 31.44, error 10.61 / 6.98.
- Grouped trained: AB pair 11.59 / 6.01, agent 66.47 / 51.82, error 28.96 / 27.14; W&W pair 4.08 / 1.12, agent 37.90 / 30.00, error 12.50 / 8.46.
- Above published DCL on 10 of 12 micro numbers (exception: W&W error micro). On grouped validation worse than the flat MLP (pair 13.5 vs 23.1).

## 4. Frozen MiniLM models (phases 4-6r)
Validation, 3 seeds (5 for GNN ablations):
- Frequency: paper split k=13 pair 13.47 / 5.09, agent 81.00, error 28.36; grouped k=15 pair 15.23 / 8.32, agent 84.11 / 77.20, error 31.27 / 30.73.
- Flat MLP: paper split pair 32.19 +- 3.36, agent 82.33, error 41.26; grouped pair 23.05 +- 0.19 / 18.99, agent 83.95 / 77.53, error 36.01 / 35.67.
- Top-k decoder: grouped pair +1.0, agent +0.5, error -3.4, all macros lower: not adopted.
- Relational probe (+4 cosine features): grouped pair 23.61 (higher in 3 of 3 seeds), error 36.41; paper split inconclusive.
- GNN ablations (5 seeds, grouped): 0-layer control 25.66 +- 0.40; full GNN 26.86 +- 0.61; drop same-agent edges 26.56; drop sequence edges 26.62; drop query node 25.55 (about control). Error micro 37.67 / 38.18 / 38.45 / 38.43 / 37.08. Params: control 164k, GNN 329k.

Test (5 seeds, scored once; phase 6r re-run after the Who&When bug fix):
| Model | Grouped test | Paper-split test (leaky) |
|---|---|---|
| frequency (k=13) | 12.14 / 80.45 / 28.47 | 13.85 / 80.37 / 29.02 |
| flat MLP | 19.83 / 78.74 / 33.01 | 33.08 / 80.86 / 40.43 |
| 0-layer control | 20.91 / 79.43 / 34.74 | 31.37 / 82.52 / 39.82 |
| 2-layer GNN | 21.87 / 78.62 / 35.30 | 36.20 / 82.12 / 43.18 |
Macro (pair / agent / error): grouped flat 12.99 / 65.27 / 32.62, control 14.47 / 67.23 / 34.52, GNN 14.46 / 67.28 / 34.85. Question leakage inflates pair micro by about 12-14 points (paper split vs grouped).
Who&When (frozen, base features): grouped flat 4.25 / 39.60 / 10.83, control 2.94 / 33.55 / 9.89, GNN 3.60 / 38.04 / 10.22; paper split flat 1.89 / 34.43 / 6.73, control 1.71 / 30.11 / 7.78, GNN 1.98 / 35.47 / 8.34. At or below floor.
Who&When bug: standardisation sd = std + 1e-6 made the never-firing `who_when` framework column 1,000,000, saturating the sigmoids so every model output the same pair (2.72 / 33.70 / 10.87). Fixed: columns with std < 1e-3 get sd = 1.

## 5. Fine-tuned MiniLM, early-stopped on validation (phase 6ft; encoder trained on grouped train 3,865)
Kaggle `aegis-ft-full`: encoder lr 2e-5, head lr 1e-3, bs 8, fp16, max length 256, at most 30 turns per trajectory; val loss 0.1595 / 0.1442 / 0.1387 / 0.1390 / 0.1392, best epoch 3, about 148 s/epoch. Caveat: validation numbers slightly optimistic (the encoder used validation for early stopping); test is clean.
- Validation (3 seeds), frozen -> fine-tuned: flat 22.93 / 83.71 / 35.93 -> 36.93 / 80.22 / 44.29; control 25.69 / 83.90 / 37.70 -> 33.40 / 85.33 / 43.02; GNN 26.93 / 83.82 / 38.16 -> 35.60 / 86.08 / 44.82.
- Test (grouped, 5 seeds): flat 29.05 +- 2.24 / 79.08 / 38.80 (macro 17.76 / 59.70 / 38.04); control 29.68 +- 2.92 / 78.05 / 39.39; GNN 28.46 +- 1.52 / 82.38 / 40.25 (macro 19.47 / 67.39 / 41.04). Gain from fine-tuning about +7 to +9 pair, +4 to +6 error. Graph adds nothing to the fine-tuned flat model on pair.
- Who&When: flat 3.71 / 40.18 / 10.20; control 2.14 / 37.98 / 9.81; GNN 2.55 / 36.98 / 9.35 (at or below floor).

## 6. Larger encoders (validation only; test not scored)
- all-mpnet-base-v2: failed twice (v1 CUDA OOM at bs 4; v2 `MPNetModel does not support gradient checkpointing`).
- bge-base-en-v1.5 (110M, 768-d, 4 epochs, patience 2, bs 2, grad-ckpt; about 2.5 h, 760 s/epoch): val BCE 0.13945 (best epoch 2) vs MiniLM 0.13874. Validation, 3 seeds: flat 34.89 / 83.57 / 43.77; control 35.40 / 83.24 / 43.51; GNN 33.02 / 87.58 / 43.42 (MiniLM FT: 36.93, 33.40, 35.60 pair). No clear gain, 5x slower.

## 7. Clean encoders (early stop on a held-out slice of train; 2 encoder seeds es_s0, es_s1, best epoch 4 in both)
Validation (3 seeds per encoder, mean of the two): flat 31.66 / 86.19 / 42.49; control 31.35 / 86.78 / 42.06; GNN 33.05 / 87.83 / 43.85.
Test (grouped, 5 seeds, base features):
| Model | es_s0 | es_s1 | Mean |
|---|---|---|---|
| flat MLP | 27.05 / 81.03 / 38.68 | 25.80 / 81.34 / 38.45 | 26.43 / 81.19 / 38.57 |
| 0-layer control | 27.91 / 79.79 / 39.11 | 25.21 / 80.98 / 38.21 | 26.56 / 80.39 / 38.66 |
| 2-layer GNN | 27.14 / 82.67 / 39.23 | 27.46 / 81.11 / 39.09 | 27.30 / 81.89 / 39.16 |
avg12: flat 30.73 / 30.92, control 29.85 / 30.36, GNN 30.18 / 30.23 (floor 25.59). Encoder seed moves test pair about 1 point; GNN edge (+0.9 pair, +0.6 error) within noise.

## 8. Who&When study (zero-shot, flat MLP, 5 seeds, mean of the two clean encoders)
| Variant | Who&When | AEGIS test |
|---|---|---|
| base (framework + name one-hot) | 4.16 / 43.00 / 11.15 | 26.43 / 81.19 / 38.57 |
| name-free (those columns zeroed) | 5.29 / 45.48 / 12.74 | 25.21 / 77.29 / 37.84 |
| name-embedding (name-free + MiniLM embedding of the name text) | 5.04 / 40.85 / 15.04 | 29.41 / 79.78 / 39.09 (best) |
Per encoder (W&W): es_s0 base 3.78 / 43.71 / 9.92, name-free 5.74 / 47.58 / 12.52, name-embedding 4.39 / 40.44 / 13.61; es_s1 base 4.53 / 42.28 / 12.37, name-free 4.83 / 43.37 / 12.96, name-embedding 5.69 / 41.26 / 16.46. Name-embedding gain in-domain: +2.2 and +3.8 per encoder.
Top-1 decoding (one pair per record) worse than thresholds in every variant: dropped.
Versus published W&W: pair 5.04 above floor 3.42 and Aegis-SFT 4.03, below o3 7.41; error 15.04 above floor 13.33, DCL 14.67, o3 14.88, Aegis-SFT 9.87; agent 40.85 near the floor and far below Aegis-SFT 51.14 and o3 53.10.
Caveats: variants defined after seeing Who&When results from the earlier encoder (mild selection); 20 unreachable labels; noisy labels; seed noise.

## 9. End-to-end graph fine-tune (encoder trained through the graph; Kaggle T4, one seed, 6 epochs, about 130 s/epoch)
| Model | Val | Test | Who&When | avg12 |
|---|---|---|---|---|
| name-dependent, 0-layer control | 30.74 / 86.56 / 42.15 | 26.00 / 80.88 / 38.38 | 2.40 / 41.18 / 9.76 | 30.72 |
| name-dependent, 2-layer graph | 29.59 / 87.74 / 39.88 | 23.69 / 80.15 / 35.22 | 2.90 / 42.55 / 7.54 | 29.06 |
| name-free, 0-layer control | 23.72 / 83.77 / 35.88 | 20.19 / 77.77 / 33.38 | 4.41 / 47.62 / 9.97 | 27.97 |
| name-free, 2-layer graph | 27.86 / 87.00 / 40.81 | 22.90 / 81.95 / 35.72 | 5.04 / 39.18 / 9.90 | 29.47 |
Best early-stop loss: l2 0.1431 (epoch 4), l0 0.1440 (epoch 6); name-free l2 0.1477 (epoch 5), l0 0.1544 (epoch 4). Name-dependent per-framework test pair (graph vs control): agentverse 25.2 vs 29.1, dylan 18.1 vs 18.3, llm_debate 28.8 vs 28.9, macnet 25.9 vs 28.6, magentic_one 19.2 vs 18.4 (n=28), smoagents 4.7 vs 7.8 (n=48).
Reading: the graph effect flips sign between the two pairs (-2.3 then +2.7 test pair), so no stable graph effect; name-free end-to-end is clearly worse in-domain than name-dependent.

## 10. Did the GNN help? (pair micro, grouped)
| Setting | Flat | Control | GNN | GNN vs flat |
|---|---|---|---|---|
| Frozen, val (5 seeds) | 23.05 | 25.66 | 26.86 | +3.8 |
| Frozen, test | 19.83 | 20.91 | 21.87 | +2.0 |
| Fine-tuned (val-early-stopped), test | 29.05 | 29.68 | 28.46 | -0.6 |
| Clean encoder, val | 31.66 | 31.35 | 33.05 | +1.4 |
| Clean encoder, test | 26.43 | 26.56 | 27.30 | +0.9 |
| End-to-end vs its control (name-dependent / name-free) | | 26.00 / 20.19 | 23.69 / 22.90 | -2.3 / +2.7 |
Frozen-encoder gain came entirely from the query node. Defensible claim: a turn graph adds at most a small, unproven gain.

## 11. What led to improvement
1. Fine-tuning the encoder: +9 pair (earlier encoder), +6.6 (clean), error +4 to +6. (Paper's own DCL practice.)
2. Agent-name text embedding: AEGIS test pair +3.0; W&W error 15.04, pair 5.04.
3. Name-free features: W&W pair 4.16 -> 5.29, agent 43.0 -> 45.5; costs 1-4 pair in-domain.
Little or nothing: graph, relational features (+0.6), top-k and top-1 decoding, bge-base.

## 12. Comparison with the paper (AEGIS-Bench test unless noted)
- Earlier encoder, flat MLP: pair 29.1 / agent 79.1 / error 38.8 vs DCL 8.3 / 22.9 / 24.7 and Aegis-SFT 16.6 / 76.5 / 27.5; avg12 about 29.4 vs 12.6 and 26.5, but the floor averages 25.6.
- Clean-encoder best (name-embedding): 29.41 / 79.78 / 39.09 in-domain.
- Who&W agent level is still near the floor and not beaten (Aegis-SFT 51.14, o3 53.10).

## 13. Error analysis (30 trajectories; `aegis/docs/error_analysis.md`)
Injected faults often leave tell-tale phrases (e.g. "incorrect formula"); mode labels often do not match the visible symptom; some faults are invisible in text; symptoms propagate to other agents. Faulty-agent count nearly fixed per framework (smoagents 1 of 4; magentic_one 1.00; llm_debate 2 of 3; agentverse 3 of 4; macnet mean 2.22; dylan 2.61). Modes balanced (top mode 10-12%).

## 14. Old project (own generated data; separate from AEGIS)
DAG pipelines of retriever / reasoner / writer on HotpotQA with injected faults (noise, contamination, ceiling), Qwen2.5-7B-AWQ via vLLM. Naive max-uncertainty 0.145 OOD top-1; flat RandomForest 0.249; GNN about 0.395-0.44 (selected on the OOD set, so optimistic); ablations (PCA, depth, hidden dim, focal loss) plateaued around 0.40-0.44.

## 15. Where the raw evidence is
`aegis/docs/final_results.md` (tables); `aegis/results/runs/` (`final_test_*`, `ft_eval_*`, `whowhen_study_*`, `score_ftg_*`, `chain_status.log`); `aegis/results/kaggle_logs/` (`ft_full`, `ft_es_s0`, `ft_es_s1`, `ft_bge`, `ft_mpnet_v1`, `ft_mpnet_v2`, `ftg_smoke`, `ftg_l2`, `ftg_l0`, `ftg_l2nf`, `ftg_l0nf`); ledger `aegis/results/metrics_history.jsonl` (phases 6r, 6ft, 7, 8).
