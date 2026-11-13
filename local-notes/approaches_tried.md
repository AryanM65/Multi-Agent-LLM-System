# Everything we tried beyond the paper, with results (consolidated 2026-11-12)

LOCAL ONLY (`local-notes/`, git-excluded). Never commit or push. Written from the chat answers to: "all the approaches we tried except the paper's and their results", "did the GNN lead to any improvement", "what exactly led to improvement". Sources of the numbers: `aegis/docs/final_results.md`, `aegis/docs/error_analysis.md`, `aegis/results/runs/*.log`, ledger `aegis/results/metrics_history.jsonl`.
All AEGIS-Bench numbers are micro-F1 (pair / agent / error) unless noted. Floor = predict every present agent with all 14 modes. Test floor 7.35 / 68.58 / 27.14; Who&When floor 3.42 / 38.32 / 13.33; avg12 floor 25.59.

## A. Evaluation changes
- Grouped split (query-disjoint train/val/test handling; 588 of 600 test queries were also in train under the paper split): leakage inflated scores; flat MLP pair on validation fell 32.2 to 23.1.
- Trivial floor added to every table: agent-level floor is about 70, so several published agent rows are below it.
- Leak-proof loader: never reads `num_agents`, `num_injected_agents`, `ground_truth.*`, `injection_strategy`, per-turn `is_injected`.

## B. Baselines (frozen MiniLM, grouped validation)
- Frequency baseline (most common pairs per framework, k=15): pair 15.23, agent 84.11, error 31.27.
- Flat MLP (mean of each agent's turn vectors): 23.05 / 83.95 / 36.01.
- DCL-style rebuild (frozen encoder, not the real DCL): test pair 15.06 (paper split) or 11.59 (grouped); above published DCL on 10 of 12 micro numbers but a different model.
- Zero-shot Qwen2.5-7B reproduction: avg12 14.67 vs paper 12.43; agent micro 49.2 vs paper 27.6 (unexplained).

## C. Ideas that did not help (or little)
- Top-k decoding: pair +1.0, agent +0.5, error -3.4, macros lower; dropped.
- Relational features (similarity to query and peers): pair +0.6 on the grouped split, higher in all 3 seeds; small.
- Top-1 decoding on Who&When: worse than thresholds in every variant.
- bge-base encoder: not clearly better on validation, 5x slower; test not scored.
- all-mpnet-base-v2: failed twice (CUDA OOM, then no gradient checkpointing support).

## D. Turn-graph GNN on frozen vectors
- Design: one node per turn plus a query node, 4 edge types (sequence, same-agent, query-to-turn, turn-to-query), 2 layers, with a 0-layer control.
- Grouped validation pair (5 seeds): flat 23.05, control 25.66, GNN 26.86.
- Ablations: removing the query node removed the whole gain; removing sequence or same-agent edges changed almost nothing.
- Grouped test pair: flat 19.83, control 20.91, GNN 21.87.

## E. Fine-tuning the text encoder (largest gain)
- Frozen -> fine-tuned (early-stopped on validation): test pair flat / control / GNN 19.83 / 20.91 / 21.87 -> 29.05 / 29.68 / 28.46 (+7 to +9 pair, +4 to +6 error).
- Clean version (early stopping on a held-out slice of train, 3,234 train): test 26.43 / 26.56 / 27.30; validation flat 31.66, control 31.35, GNN 33.05 (earlier val-early-stopped version was optimistic). Lower partly from not touching validation and partly from 16% less train data; the two cannot be separated.
- This is the paper's own DCL practice (trains its encoder end to end), so not a new contribution.

## F. End-to-end graph fine-tune (encoder trained through the graph; one seed each)
- Name-dependent: control val 30.74 / test 26.00; 2-layer graph val 29.59 / test 23.69 (graph worse by 2.3 test). Who&When 2.40 / 41.18 / 9.76 vs 2.90 / 42.55 / 7.54. avg12 30.72 vs 29.06.
- Name-free: control val 23.72 / test 20.19; 2-layer graph val 27.86 / test 22.90 (graph better by 2.7 test). Who&When 4.41 / 47.62 / 9.97 vs 5.04 / 39.18 / 9.90. avg12 27.97 vs 29.47.
- The sign flips, so no stable graph effect; read as noise.

## G. Who&When (zero-shot; no Who&When labels used), flat MLP, mean of 2 clean encoders
| Variant | Who&When | AEGIS test |
|---|---|---|
| base (names + framework) | 4.16 / 43.00 / 11.15 | 26.43 / 81.19 / 38.57 |
| name-free | 5.29 / 45.48 / 12.74 | 25.21 / 77.29 / 37.84 |
| name-embedding | 5.04 / 40.85 / 15.04 | 29.41 / 79.78 / 39.09 (best) |
Published Who&When rows: DCL 1.60 / 8.40 / 14.67; Aegis-SFT 4.03 / 51.14 / 9.87; o3 7.41 / 53.10 / 14.88. Caveats: variants chosen after seeing Who&When; 184 records with 20 unreachable labels (max agent recall about 89%); labels mapped by Gemini-2.5-Flash; seed noise 1-3 points.

## H. Analysis and fixes
- Error analysis (30 trajectories): injected faults often leave tell-tale phrases; mode labels often do not match the visible symptom; some faults invisible in text.
- Who&When standardisation bug (training feature 0 became 1,000,000 on Who&When so every model gave the same answer): fixed.
- GNN reproducibility: same-seed runs differed through multi-threaded float sums; fixed with one thread, bit-identical (max abs diff 0.0).

## I. Old project (own generated data, separate)
Naive max-uncertainty 0.145 top-1; flat RandomForest 0.249; GNN about 0.395-0.44 (selected on the held-out set, optimistic); many ablations (PCA, depth, hidden dim, focal loss) plateaued around 0.40-0.44.

## J. Did the GNN help? (pair micro, grouped split)
| Setting | Flat | Control | GNN | GNN vs flat |
|---|---|---|---|---|
| Frozen, validation (5 seeds) | 23.05 | 25.66 | 26.86 | +3.8 |
| Frozen, test (5 seeds) | 19.83 | 20.91 | 21.87 | +2.0 |
| Fine-tuned (val-early-stopped), test | 29.05 | 29.68 | 28.46 | -0.6 |
| Clean encoder, validation | 31.66 | 31.35 | 33.05 | +1.4 |
| Clean encoder, test | 26.43 | 26.56 | 27.30 | +0.9 |
Reading: the frozen-encoder gain came entirely from the query node; it shrank once the encoder was fine-tuned; the latest gap (+0.9 pair, +0.6 error) is within seed noise; the end-to-end test is inconclusive. GNN gave slightly better agent macro / error in several settings (weak evidence). What would settle it: more seeds, and a graph built from true agent connections (not stored in the dataset; our edges are guesses).

## K. What led to improvement, in order of size
1. Fine-tuning the text encoder: +9 pair (earlier encoder), +6.6 (clean); error +4 to +6; holds for every model type. Likely partly because faults leave style-specific phrasing (part of the score may be injector-style recognition).
2. Agent-name text embedding: AEGIS test pair 26.43 -> 29.41 (+3.0; +2.2 and +3.8 per encoder); Who&When error 15.04 and pair 5.04 (above floors).
3. Removing name features: Who&When pair 4.16 -> 5.29, agent 43.0 -> 45.5; costs 1-4 pair points in-domain.
Little or nothing: graph (at most about +1, unproven), relational features (+0.6), top-k/top-1 decoding, bigger encoder (bge-base).
Evaluation-only effects: grouped split lowered every score (honest numbers, not a model gain); the Who&When standardisation fix moved it from identical garbage to real predictions.
Caveats: most numbers rest on 1-5 seeds, noise 1-3 points; Who&When variants chosen after seeing its results.

## L. Added 2026-11-12 (Who&When push; details in results.md and timeline.md)
- More seeds of the clean 256-token MiniLM encoder (es_s2..es_s6): the peer-contrast gain (+3.4 test pair mean over six encoders) replicates on every one; Who&W unchanged within noise.
- Encoder variants: 512 tokens per message (AEGIS test pair 34.4 base, 36.3 peer; Who&W error falls to 9-11), MiniLM-L12 (small gain), L12 with 512 tokens (best AEGIS 37.26 peer, Who&W at or below floor), bge-small (no difference), chunked messages (same AEGIS level, worse Who&W; dropped).
- CPU improvement variants: peer-contrast, name-dropout, both (peernd). Name-dropout alone and peernd are the only Who&W-positive changes (agent +2 to +6).
- Best single Who&W so far: len512_s1 peernd 7.09 / 44.75 / 14.12 (AEGIS test 33.99).
- Written and queued/running: ww_focus.py (ensemble over encoders, name-dropout, label-shift correction), ww_attn.py (attention pooling over turns, CPU), end-to-end attention-pooling encoder (`--attn` flag, kernel aegis-attn on account D).
- Planned: name-to-role mapping, scalar-dropout, test-time adaptation (separate transductive row), leave-one-framework-out validation, name-shuffle encoder; escalation ladder (sibling counterfactual pairs, de-propagation, small LM reader, content-flow edges) in timeline.md.
