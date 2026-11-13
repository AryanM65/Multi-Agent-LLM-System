# Our implemented approaches versus every approach in the paper

LOCAL ONLY (`local-notes/`, git-excluded). Never commit or push. Format rule from the user: bullet points, no tables.
FULL VALUES: every run and every seed (micro and macro, all three levels) is at the bottom of this file in the appendix between the FULL_VALUES markers; regenerate with `python local-notes/gen_full_values.py`.

Written 2026-11-12 night. Only approaches we actually implemented and scored are kept here. The gains were computed by script from the numbers below, so each is exact subtraction.


## Sources and how to read it
- Paper numbers: the paper's Table 1 as recorded in `new-plan.md` section 3.5 (micro-F1, percent). Paper approaches: Random, DCL (their small trained model), Qwen2.5-7B and 14B zero-shot, Aegis-SFT (14B fine-tuned), Aegis-GRPO (14B), o3 zero-shot.
- Our numbers: from `aegis/docs/final_results.md`, `results_master.md`, `results.md` and the run logs. All are micro-F1 on the 600-conversation AEGIS-Bench test set (our grouped protocol, harder than the paper's) or on Who&When (zero-shot, no Who&When labels used).
- "Gain" means our number minus the paper's, in percentage points; positive = we are better, negative = we are worse.
- Floor (predict every present agent with all 14 modes, no learning): AEGIS pair 7.35 / agent 68.58 / error 27.14; Who&When 3.42 / 38.32 / 13.33. Agent and error gains over weak paper rows are partly just the floor.
- The paper's average over 12 numbers: floor 25.59; Random 4.08, DCL 12.61, Qwen-7B 12.43, Qwen-14B 13.99, Aegis-GRPO 18.41, o3 20.24, Aegis-SFT 26.51. Ours: about 30.2 to 30.9 for the clean-encoder flat and graph models (gain about +4 over Aegis-SFT, +18 over DCL).

## Short answer
- AEGIS-Bench pair: every learned approach of ours beats every paper approach, including Aegis-SFT (the best paper row, 16.62). The frequency baseline (12.14) beats all paper rows except Aegis-SFT.
- AEGIS-Bench agent level: from the frozen flat MLP upward we beat every paper row, including Aegis-SFT (76.53), by about 2 to 5 points; this is close to the floor (68.58), so it is a modest result.
- AEGIS-Bench error level: we beat every paper row; the learned approaches are 5 to 13 points above Aegis-SFT (27.53).
- Who&When: we do NOT beat every paper approach on all three levels.
  - Pair: we beat all but o3 (7.41); the name-dropout run (6.04) is still 1.37 below o3.
  - Agent: we beat Random, DCL and Qwen-7B (close to it) but stay far below Aegis-SFT, Aegis-GRPO, o3 and Qwen-14B.
  - Error: name-embedding (15.04) and name-dropout (14.54) are on par with DCL (14.67) and o3 (14.88) and above the others.
- Caveats: much of our gain over the paper comes from fine-tuning the text encoder, which is the paper's own recipe; the test file may differ from the paper's (not verified); our numbers move 1 to 3 points with the random seed; the Who&When variants were chosen after seeing Who&When results; peer-contrast and name-dropout are single-encoder, 3-seed results so far.

## Part 1: AEGIS-Bench test (micro-F1)
Paper rows used: Random 0.33 / 4.54 / 11.23; DCL 8.33 / 22.93 / 24.73; Qwen-7B 5.02 / 27.55 / 14.96; Qwen-14B 5.47 / 35.78 / 20.24; Aegis-SFT 16.62 / 76.53 / 27.53; Aegis-GRPO 6.84 / 49.74 / 21.19; o3 7.86 / 40.31 / 22.37 (pair / agent / error).


### Frequency baseline (no learning; grouped test)
- What it is: predicts the most common faulty pairs of each framework; reads no text.
- Our result: pair 12.14, agent 80.45, error 28.47.
- Gain over each paper approach (our number minus theirs, in points; pair / agent / error):
  - vs Random: +11.81 / +75.91 / +17.24
  - vs DCL (the paper's own small model): +3.81 / +57.52 / +3.74
  - vs Qwen2.5-7B zero-shot: +7.12 / +52.90 / +13.51
  - vs Qwen2.5-14B zero-shot: +6.67 / +44.67 / +8.23
  - vs Aegis-SFT (14B fine-tuned): -4.48 / +3.92 / +0.94
  - vs Aegis-GRPO (14B): +5.30 / +30.71 / +7.28
  - vs o3 zero-shot: +4.28 / +40.14 / +6.10

### Flat MLP on frozen MiniLM vectors (grouped test, 5 seeds)
- What it is: frozen text encoder, each agent's messages averaged, small network on top.
- Our result: pair 19.83, agent 78.74, error 33.01.
- Gain over each paper approach (our number minus theirs, in points; pair / agent / error):
  - vs Random: +19.50 / +74.20 / +21.78
  - vs DCL (the paper's own small model): +11.50 / +55.81 / +8.28
  - vs Qwen2.5-7B zero-shot: +14.81 / +51.19 / +18.05
  - vs Qwen2.5-14B zero-shot: +14.36 / +42.96 / +12.77
  - vs Aegis-SFT (14B fine-tuned): +3.21 / +2.21 / +5.48
  - vs Aegis-GRPO (14B): +12.99 / +29.00 / +11.82
  - vs o3 zero-shot: +11.97 / +38.43 / +10.64

### Turn graph (GNN) on frozen vectors (grouped test, 5 seeds)
- What it is: same vectors, plus message passing over a graph of messages and the question.
- Our result: pair 21.87, agent 78.62, error 35.30.
- Gain over each paper approach (our number minus theirs, in points; pair / agent / error):
  - vs Random: +21.54 / +74.08 / +24.07
  - vs DCL (the paper's own small model): +13.54 / +55.69 / +10.57
  - vs Qwen2.5-7B zero-shot: +16.85 / +51.07 / +20.34
  - vs Qwen2.5-14B zero-shot: +16.40 / +42.84 / +15.06
  - vs Aegis-SFT (14B fine-tuned): +5.25 / +2.09 / +7.77
  - vs Aegis-GRPO (14B): +15.03 / +28.88 / +14.11
  - vs o3 zero-shot: +14.01 / +38.31 / +12.93

### Fine-tuned MiniLM, flat MLP (earlier encoder, grouped test, 5 seeds)
- What it is: the encoder itself is trained on the task, then the flat MLP runs on its vectors.
- Our result: pair 29.05, agent 79.08, error 38.80.
- Gain over each paper approach (our number minus theirs, in points; pair / agent / error):
  - vs Random: +28.72 / +74.54 / +27.57
  - vs DCL (the paper's own small model): +20.72 / +56.15 / +14.07
  - vs Qwen2.5-7B zero-shot: +24.03 / +51.53 / +23.84
  - vs Qwen2.5-14B zero-shot: +23.58 / +43.30 / +18.56
  - vs Aegis-SFT (14B fine-tuned): +12.43 / +2.55 / +11.27
  - vs Aegis-GRPO (14B): +22.21 / +29.34 / +17.61
  - vs o3 zero-shot: +21.19 / +38.77 / +16.43

### Clean fine-tuned encoders, turn graph (mean of 2 encoders, 5 seeds each)
- What it is: same, but the encoder was stopped on a held-out slice of training data (validation untouched, 16% less training data).
- Our result: pair 27.30, agent 81.89, error 39.16.
- Gain over each paper approach (our number minus theirs, in points; pair / agent / error):
  - vs Random: +26.97 / +77.35 / +27.93
  - vs DCL (the paper's own small model): +18.97 / +58.96 / +14.43
  - vs Qwen2.5-7B zero-shot: +22.28 / +54.34 / +24.20
  - vs Qwen2.5-14B zero-shot: +21.83 / +46.11 / +18.92
  - vs Aegis-SFT (14B fine-tuned): +10.68 / +5.36 / +11.63
  - vs Aegis-GRPO (14B): +20.46 / +32.15 / +17.97
  - vs o3 zero-shot: +19.44 / +41.58 / +16.79

### Name-embedding flat MLP (mean of 2 clean encoders, 5 seeds each)
- What it is: agent names are embedded as text instead of one-hot columns.
- Our result: pair 29.41, agent 79.78, error 39.09.
- Gain over each paper approach (our number minus theirs, in points; pair / agent / error):
  - vs Random: +29.08 / +75.24 / +27.86
  - vs DCL (the paper's own small model): +21.08 / +56.85 / +14.36
  - vs Qwen2.5-7B zero-shot: +24.39 / +52.23 / +24.13
  - vs Qwen2.5-14B zero-shot: +23.94 / +44.00 / +18.85
  - vs Aegis-SFT (14B fine-tuned): +12.79 / +3.25 / +11.56
  - vs Aegis-GRPO (14B): +22.57 / +30.04 / +17.90
  - vs o3 zero-shot: +21.55 / +39.47 / +16.72

### Peer-contrast features (one clean encoder es_s0, 3 seeds, new this push)
- What it is: adds how each agent differs from the other agents and from the question; replication on a second encoder is still running.
- Our result: pair 31.56, agent 78.99, error 40.13.
- Gain over each paper approach (our number minus theirs, in points; pair / agent / error):
  - vs Random: +31.23 / +74.45 / +28.90
  - vs DCL (the paper's own small model): +23.23 / +56.06 / +15.40
  - vs Qwen2.5-7B zero-shot: +26.54 / +51.44 / +25.17
  - vs Qwen2.5-14B zero-shot: +26.09 / +43.21 / +19.89
  - vs Aegis-SFT (14B fine-tuned): +14.94 / +2.46 / +12.60
  - vs Aegis-GRPO (14B): +24.72 / +29.25 / +18.94
  - vs o3 zero-shot: +23.70 / +38.68 / +17.76

## Part 2: Who&When (zero-shot, micro-F1)
Paper rows used: Random 0.11 / 1.06 / 8.74; DCL 1.60 / 8.40 / 14.67; Qwen-7B 2.31 / 40.92 / 3.64; Qwen-14B 0.00 / 49.88 / 1.56; Aegis-SFT 4.03 / 51.14 / 9.87; Aegis-GRPO 2.45 / 54.43 / 4.15; o3 7.41 / 53.10 / 14.88 (pair / agent / error).


### Base features (mean of 2 clean encoders)
- What it is: agent-name and framework one-hot columns as in-domain.
- Our result: pair 4.16, agent 43.00, error 11.15.
- Gain over each paper approach (our number minus theirs, in points; pair / agent / error):
  - vs Random: +4.05 / +41.94 / +2.41
  - vs DCL (the paper's own small model): +2.56 / +34.60 / -3.52
  - vs Qwen2.5-7B zero-shot: +1.85 / +2.08 / +7.51
  - vs Qwen2.5-14B zero-shot: +4.16 / -6.88 / +9.59
  - vs Aegis-SFT (14B fine-tuned): +0.13 / -8.14 / +1.28
  - vs Aegis-GRPO (14B): +1.71 / -11.43 / +7.00
  - vs o3 zero-shot: -3.25 / -10.10 / -3.73

### Name-free (mean of 2 clean encoders)
- What it is: those columns zeroed so nothing depends on names that do not exist in Who&When.
- Our result: pair 5.29, agent 45.48, error 12.74.
- Gain over each paper approach (our number minus theirs, in points; pair / agent / error):
  - vs Random: +5.18 / +44.42 / +4.00
  - vs DCL (the paper's own small model): +3.69 / +37.08 / -1.93
  - vs Qwen2.5-7B zero-shot: +2.98 / +4.56 / +9.10
  - vs Qwen2.5-14B zero-shot: +5.29 / -4.40 / +11.18
  - vs Aegis-SFT (14B fine-tuned): +1.26 / -5.66 / +2.87
  - vs Aegis-GRPO (14B): +2.84 / -8.95 / +8.59
  - vs o3 zero-shot: -2.12 / -7.62 / -2.14

### Name-embedding (mean of 2 clean encoders)
- What it is: names embedded as text.
- Our result: pair 5.04, agent 40.85, error 15.04.
- Gain over each paper approach (our number minus theirs, in points; pair / agent / error):
  - vs Random: +4.93 / +39.79 / +6.30
  - vs DCL (the paper's own small model): +3.44 / +32.45 / +0.37
  - vs Qwen2.5-7B zero-shot: +2.73 / -0.07 / +11.40
  - vs Qwen2.5-14B zero-shot: +5.04 / -9.03 / +13.48
  - vs Aegis-SFT (14B fine-tuned): +1.01 / -10.29 / +5.17
  - vs Aegis-GRPO (14B): +2.59 / -13.58 / +10.89
  - vs o3 zero-shot: -2.37 / -12.25 / +0.16

### Name-dropout training (one clean encoder es_s0, 3 seeds, new this push)
- What it is: training data doubled with a copy that hides the name embedding, so the model also works without names.
- Our result: pair 6.04, agent 47.04, error 14.54.
- Gain over each paper approach (our number minus theirs, in points; pair / agent / error):
  - vs Random: +5.93 / +45.98 / +5.80
  - vs DCL (the paper's own small model): +4.44 / +38.64 / -0.13
  - vs Qwen2.5-7B zero-shot: +3.73 / +6.12 / +10.90
  - vs Qwen2.5-14B zero-shot: +6.04 / -2.84 / +12.98
  - vs Aegis-SFT (14B fine-tuned): +2.01 / -4.10 / +4.67
  - vs Aegis-GRPO (14B): +3.59 / -7.39 / +10.39
  - vs o3 zero-shot: -1.37 / -6.06 / -0.34

## Not included here
- Approaches that did not help or are mixed (top-k decoding, relational features alone, per-mode thresholds, the alternative decoders, the larger encoder tests, the one-seed end-to-end graph runs) are in `results_master.md` and `results.md`.
- The DCL-style rebuild and the zero-shot Qwen reproduction are our implementations of paper approaches, so they are not repeated here; see `results_master.md` sections 2 and 3.
- Ideas not implemented yet (sibling features, content-flow edges, attention pooling, label-shift correction) are in `ideas.md`.

## Part 3: Who&When in detail (everything we know and tried)

### What Who&When is and how we use it
- Who&When is a second benchmark of failed multi-agent runs, annotated by humans with the responsible agent and the failing step. The AEGIS authors mapped its labels onto the same 14 error modes (with Gemini-2.5-Flash) and use it as the out-of-domain test.
- Our copy: 184 records (`evaluation/data/whowhen.jsonl`), exactly one faulty agent per record, free-form agent names (no fixed roles), a different domain from the AEGIS frameworks.
- We use it zero-shot: no Who&When label is ever used for training, early stopping or threshold tuning. Models train on AEGIS grouped-train, thresholds are tuned on AEGIS grouped-validation, and Who&When is scored once per configuration.
- Floor (predict every present agent with all 14 modes): pair 3.42, agent 38.32, error 13.33.
- Limits of the benchmark itself: 20 of 184 labels name an agent that never speaks in the turns, so the best possible agent recall is about 89%; labels are machine-mapped from human notes; the mode mix is skewed (FM-2.6 about 26%, FM-3.3 about 21%, FM-2.2 12.5%, FM-1.1 11%) while AEGIS is balanced, so a model trained on AEGIS faces a label shift.

### Paper approaches on Who&When (micro-F1, pair / agent / error)
- Random 0.11 / 1.06 / 8.74
- DCL 1.60 / 8.40 / 14.67
- Qwen2.5-7B zero-shot 2.31 / 40.92 / 3.64
- Qwen2.5-14B zero-shot 0.00 / 49.88 / 1.56
- Aegis-SFT 4.03 / 51.14 / 9.87
- Aegis-GRPO 2.45 / 54.43 / 4.15
- o3 zero-shot 7.41 / 53.10 / 14.88
- Observation: the best published pair score is only 7.41, and the paper's gains from fine-tuning on AEGIS carry over to Who&When only weakly (Qwen-14B agent 49.88 to Aegis-SFT 51.14).

### Our Who&When history, in the order it happened (pair / agent / error)
- First run (frozen encoder, all models): every model output the same single pair (2.72 / 33.70 / 10.87, zero spread). Cause: a bug. The `who_when` framework column never fires in training, so its standardisation divided by about 1e-6 and became 1,000,000 on Who&When, saturating every output. Fixed (columns with std below 1e-3 now use std 1). Old rows are kept in the ledger and marked wrong.
- After the fix, frozen encoder, grouped training:
  - flat MLP 4.25 / 39.60 / 10.83
  - 0-layer control 2.94 / 33.55 / 9.89
  - turn graph 3.60 / 38.04 / 10.22
  - On the paper's own training split: flat 1.89 / 34.43 / 6.73, control 1.71 / 30.11 / 7.78, graph 1.98 / 35.47 / 8.34.
  - Reading: at or below the floor; no claim.
- DCL-style rebuild: trained on the paper split 3.09 / 37.95 / 10.61; trained on the grouped split 4.08 / 37.90 / 12.50.
- Zero-shot Qwen2.5-7B reproduction: 2.98 / 35.16 / 6.23 (paper row 2.31 / 40.92 / 3.64).
- Fine-tuned encoder (first version): flat 3.71 / 40.18 / 10.20; control 2.14 / 37.98 / 9.81; graph 2.55 / 36.98 / 9.35. Still at or below the floor.
- Clean encoders (mean of 2), flat MLP, the three feature variants:
  - base (name and framework columns): 4.16 / 43.00 / 11.15
  - name-free (those columns zeroed): 5.29 / 45.48 / 12.74
  - name-embedding (names embedded as text): 5.04 / 40.85 / 15.04
  - Per encoder: es_s0 base 3.78 / 43.71 / 9.92, name-free 5.74 / 47.58 / 12.52, name-embedding 4.39 / 40.44 / 13.61; es_s1 base 4.53 / 42.28 / 12.37, name-free 4.83 / 43.37 / 12.96, name-embedding 5.69 / 41.26 / 16.46.
  - One-pair-per-record decoding (using the stated fact that Who&When has one faulty agent): worse than thresholds in every variant; dropped.
- End-to-end graph fine-tune (one seed): name-dependent control 2.40 / 41.18 / 9.76, graph 2.90 / 42.55 / 7.54; name-free control 4.41 / 47.62 / 9.97, graph 5.04 / 39.18 / 9.90. No win; error below the floor.
- New this push (one clean encoder es_s0, 3 seeds, improve_cpu):
  - base 5.14 / 41.00 / 14.01
  - peer-contrast features 4.75 / 37.60 / 14.18 (gain in-domain, small loss here)
  - name-dropout training 6.04 / 47.04 / 14.54 (best Who&When pair among our name-using models and the best agent score of any of our variants)

### Gain over the floor and over the paper (our best Who&When settings)
- Name-free: gain over floor +1.87 pair / +7.16 agent / -0.59 error. Over DCL +3.69 / +37.08 / -1.93. Over Aegis-SFT +1.26 / -5.66 / +2.87. Over o3 -2.12 / -7.62 / -2.14.
- Name-embedding: over floor +1.62 / +2.53 / +1.71. Over DCL +3.44 / +32.45 / +0.37. Over Aegis-SFT +1.01 / -10.29 / +5.17. Over o3 -2.37 / -12.25 / +0.16.
- Name-dropout (es_s0): over floor +2.62 / +8.72 / +1.21. Over DCL +4.44 / +38.64 / -0.13. Over Aegis-SFT +2.01 / -4.10 / +4.67. Over o3 -1.37 / -6.06 / -0.34.
- Average over the 12 numbers (paper "Avg.", includes Who&When): our clean-encoder models score about 30.2 to 30.9; the floor alone scores 25.59.

### What helped on Who&When and what did not
- Helped: removing dependence on names (name-free, name-dropout), because Who&When agent names are free-form and share nothing with AEGIS names; embedding the name text (best error score and best in-domain score together).
- Did not help: the turn graph, end-to-end fine-tuning through the graph, one-pair-per-record decoding, peer-contrast features (they help in-domain only).
- Still open: agent level stays far below Aegis-SFT, Aegis-GRPO and o3 (about 51 to 54); ideas not yet tried for Who&When are label-shift correction (estimate Who&When's mode mix without labels), name shuffling variants, and leave-one-framework-out validation to choose the recipe without looking at Who&When.

### Honest caveats specific to Who&When
- The three feature variants were defined after looking at Who&When results from the earlier encoder, so there is mild selection on the evaluation set.
- 184 records is small, so one record moves pair F1 by a visible amount; seed noise on pair is about 0.5 to 1 point.
- Label mapping noise, the 20 unreachable labels, and the paper's 32,768-token context for Who&When (we cut messages at 256 tokens) make comparison with the paper approximate.

<!-- FULL_VALUES_START -->
## Appendix: full values (every run and seed, micro AND macro F1, not only averages)
Generated from the ledger `aegis/results/metrics_history.jsonl` by `local-notes/gen_full_values.py` (re-run to refresh). Each bullet is one model on one split; below it one line per seed with pair / agent / error as micro/macro percent.

### Paper rows, all 12 numbers (Table 1; micro / macro F1, percent)
- Random: AEGIS pair 0.33/0.21, agent 4.54/3.56, error 11.23/11.15; Who&When pair 0.11/0.05, agent 1.06/0.83, error 8.74/7.14; average 4.08
- DCL: AEGIS pair 8.33/5.30, agent 22.93/20.23, error 24.73/27.70; Who&When pair 1.60/0.77, agent 8.40/6.07, error 14.67/10.57; average 12.61
- Qwen2.5-7B zero-shot: AEGIS pair 5.02/2.52, agent 27.55/14.49, error 14.96/11.36; Who&When pair 2.31/1.14, agent 40.92/23.50, error 3.64/1.77; average 12.43
- Qwen2.5-14B zero-shot: AEGIS pair 5.47/2.20, agent 35.78/12.71, error 20.24/5.91; Who&When pair 0.00/0.00, agent 49.88/33.19, error 1.56/1.35; average 13.99
- Aegis-SFT (14B fine-tuned): AEGIS pair 16.62/9.99, agent 76.53/47.97, error 27.53/27.66; Who&When pair 4.03/2.08, agent 51.14/36.94, error 9.87/7.77; average 26.51
- Aegis-GRPO (14B): AEGIS pair 6.84/2.55, agent 49.74/18.38, error 21.19/16.10; Who&When pair 2.45/1.49, agent 54.43/40.88, error 4.15/2.67; average 18.41
- o3 zero-shot: AEGIS pair 7.86/2.27, agent 40.31/23.27, error 22.37/16.76; Who&When pair 7.41/3.98, agent 53.10/42.55, error 14.88/8.63; average 20.24
- Trivial floor (ours): AEGIS test pair 7.35/6.35, agent 68.58/61.98, error 27.14/26.99; Who&When pair 3.42/2.85, agent 38.32/38.41, error 13.33/12.34; average 25.59

### Zero-shot Qwen2.5-7B reproduction (phase 3a)
- qwen2.5-7b-instruct-awq_zeroshot | scored on aegis_bench | split tag test+who_when | 1 run(s); each value is micro/macro F1
  - seed 0: pair 8.01/2.48, agent 49.24/10.69, error 20.15/12.72
- qwen2.5-7b-instruct-awq_zeroshot | scored on who_when | split tag test+who_when | 1 run(s); each value is micro/macro F1
  - seed 0: pair 2.98/1.85, agent 35.16/23.14, error 6.23/3.35

### Frozen MiniLM models, final held-out scoring (phase 6r; split 'grouped' = our protocol, 'paper' = released leaky split)
- flat_mlp | scored on test | split tag test_grouped | 5 run(s); each value is micro/macro F1
  - seed 0: pair 19.95/13.32, agent 79.24/65.54, error 33.30/33.28
  - seed 1: pair 19.58/12.98, agent 78.85/65.69, error 32.39/32.25
  - seed 2: pair 19.46/12.18, agent 78.19/65.33, error 32.44/31.87
  - seed 3: pair 20.01/12.80, agent 78.63/63.23, error 33.57/32.87
  - seed 4: pair 20.13/13.68, agent 78.79/66.54, error 33.33/32.82
- floor_all_agents_all_modes | scored on test | split tag test_grouped | 1 run(s); each value is micro/macro F1
  - seed 0: pair 7.35/6.35, agent 68.58/61.98, error 27.14/26.99
- frequency | scored on test | split tag test_grouped | 1 run(s); each value is micro/macro F1
  - seed 0: pair 12.14/5.35, agent 80.45/72.05, error 28.47/28.34
- turn_gnn_L0 | scored on test | split tag test_grouped | 5 run(s); each value is micro/macro F1
  - seed 0: pair 21.51/15.01, agent 79.86/66.17, error 34.96/34.39
  - seed 1: pair 19.38/12.89, agent 78.71/67.85, error 33.62/33.36
  - seed 2: pair 21.04/14.84, agent 79.20/68.38, error 35.05/34.88
  - seed 3: pair 20.83/14.89, agent 79.95/67.00, error 34.81/34.58
  - seed 4: pair 21.79/14.73, agent 79.43/66.74, error 35.25/35.36
- turn_gnn_L2 | scored on test | split tag test_grouped | 5 run(s); each value is micro/macro F1
  - seed 0: pair 21.67/13.83, agent 79.01/67.45, error 34.57/34.35
  - seed 1: pair 20.71/13.71, agent 78.35/68.40, error 34.60/34.76
  - seed 2: pair 22.20/14.03, agent 78.22/66.32, error 35.76/34.48
  - seed 3: pair 22.71/16.27, agent 78.43/67.55, error 36.15/35.78
  - seed 4: pair 22.05/14.48, agent 79.07/66.70, error 35.42/34.86
- flat_mlp | scored on test | split tag test_paper | 5 run(s); each value is micro/macro F1
  - seed 0: pair 33.39/20.59, agent 79.16/64.87, error 40.34/39.15
  - seed 1: pair 28.41/20.36, agent 84.11/68.65, error 38.65/38.50
  - seed 2: pair 35.13/23.05, agent 81.18/69.68, error 42.20/41.63
  - seed 3: pair 34.01/22.41, agent 80.46/67.21, error 40.19/38.94
  - seed 4: pair 34.45/21.67, agent 79.38/67.91, error 40.78/39.77
- floor_all_agents_all_modes | scored on test | split tag test_paper | 1 run(s); each value is micro/macro F1
  - seed 0: pair 7.35/6.35, agent 68.58/61.98, error 27.14/26.99
- frequency | scored on test | split tag test_paper | 1 run(s); each value is micro/macro F1
  - seed 0: pair 13.85/6.04, agent 80.37/69.84, error 29.02/28.48
- turn_gnn_L0 | scored on test | split tag test_paper | 5 run(s); each value is micro/macro F1
  - seed 0: pair 34.11/22.78, agent 79.20/66.46, error 39.83/39.31
  - seed 1: pair 29.31/21.02, agent 84.35/69.06, error 39.60/39.77
  - seed 2: pair 28.63/22.43, agent 84.48/69.74, error 38.63/38.60
  - seed 3: pair 35.72/23.04, agent 79.47/67.79, error 41.69/40.86
  - seed 4: pair 29.11/20.69, agent 85.11/70.09, error 39.35/39.34
- turn_gnn_L2 | scored on test | split tag test_paper | 5 run(s); each value is micro/macro F1
  - seed 0: pair 36.61/23.12, agent 80.47/68.46, error 43.25/42.05
  - seed 1: pair 36.80/23.34, agent 81.61/71.17, error 42.67/41.35
  - seed 2: pair 30.84/22.79, agent 85.36/74.47, error 40.92/40.85
  - seed 3: pair 39.00/25.97, agent 81.11/68.86, error 45.15/44.31
  - seed 4: pair 37.76/25.53, agent 82.05/72.00, error 43.91/43.14
- flat_mlp | scored on who_when | split tag who_when_grouped | 5 run(s); each value is micro/macro F1
  - seed 0: pair 3.82/0.87, agent 37.24/22.90, error 9.87/6.24
  - seed 1: pair 4.92/1.37, agent 37.86/21.63, error 10.85/6.00
  - seed 2: pair 4.93/1.29, agent 39.90/21.35, error 10.76/5.54
  - seed 3: pair 3.76/1.42, agent 40.26/28.51, error 13.31/8.13
  - seed 4: pair 3.83/0.42, agent 42.71/25.84, error 9.39/4.84
- floor_all_agents_all_modes | scored on who_when | split tag who_when_grouped | 1 run(s); each value is micro/macro F1
  - seed 0: pair 3.42/2.85, agent 38.32/38.41, error 13.33/12.34
- frequency | scored on who_when | split tag who_when_grouped | 1 run(s); each value is micro/macro F1
  - seed 0: pair 0.00/0.00, agent 0.00/0.00, error 0.00/0.00
- turn_gnn_L0 | scored on who_when | split tag who_when_grouped | 5 run(s); each value is micro/macro F1
  - seed 0: pair 2.93/1.59, agent 33.55/23.72, error 10.48/7.33
  - seed 1: pair 3.13/1.48, agent 32.43/21.86, error 10.10/7.06
  - seed 2: pair 2.58/1.19, agent 31.25/23.49, error 9.74/7.90
  - seed 3: pair 3.34/1.57, agent 38.10/27.44, error 9.30/6.17
  - seed 4: pair 2.72/0.96, agent 32.41/21.44, error 9.83/7.61
- turn_gnn_L2 | scored on who_when | split tag who_when_grouped | 5 run(s); each value is micro/macro F1
  - seed 0: pair 4.45/2.13, agent 38.24/30.44, error 11.37/6.63
  - seed 1: pair 3.35/1.74, agent 36.65/27.54, error 9.08/5.92
  - seed 2: pair 3.41/1.14, agent 42.28/31.77, error 10.71/6.92
  - seed 3: pair 4.08/1.90, agent 35.29/25.12, error 11.57/7.82
  - seed 4: pair 2.71/1.14, agent 37.71/26.71, error 8.37/5.55
- flat_mlp | scored on who_when | split tag who_when_paper | 5 run(s); each value is micro/macro F1
  - seed 0: pair 1.78/0.94, agent 38.00/21.95, error 5.20/4.72
  - seed 1: pair 2.71/1.09, agent 34.36/22.41, error 7.22/4.78
  - seed 2: pair 2.14/0.73, agent 33.18/21.71, error 8.31/4.99
  - seed 3: pair 1.41/0.72, agent 33.33/20.76, error 7.05/5.06
  - seed 4: pair 1.39/0.20, agent 33.25/22.80, error 5.87/3.47
- floor_all_agents_all_modes | scored on who_when | split tag who_when_paper | 1 run(s); each value is micro/macro F1
  - seed 0: pair 3.42/2.85, agent 38.32/38.41, error 13.33/12.34
- frequency | scored on who_when | split tag who_when_paper | 1 run(s); each value is micro/macro F1
  - seed 0: pair 0.00/0.00, agent 0.00/0.00, error 0.00/0.00
- turn_gnn_L0 | scored on who_when | split tag who_when_paper | 5 run(s); each value is micro/macro F1
  - seed 0: pair 1.45/0.24, agent 26.97/20.26, error 5.96/5.45
  - seed 1: pair 1.64/0.47, agent 26.14/23.26, error 8.09/5.83
  - seed 2: pair 2.58/1.63, agent 35.51/31.70, error 10.46/6.40
  - seed 3: pair 1.01/0.37, agent 29.24/21.30, error 4.64/4.03
  - seed 4: pair 1.88/0.49, agent 32.69/24.32, error 9.76/8.83
- turn_gnn_L2 | scored on who_when | split tag who_when_paper | 5 run(s); each value is micro/macro F1
  - seed 0: pair 2.65/0.67, agent 33.69/25.10, error 8.47/5.24
  - seed 1: pair 0.75/0.16, agent 31.40/28.65, error 5.75/3.57
  - seed 2: pair 1.78/1.03, agent 39.86/32.39, error 8.99/7.88
  - seed 3: pair 2.93/0.92, agent 34.98/26.87, error 11.69/8.74
  - seed 4: pair 1.80/0.78, agent 37.40/33.65, error 6.79/3.27

### Fine-tuned MiniLM (phase 6ft; split without a tag = first encoder stopped on validation; tags es_s0 / es_s1 = clean encoders)
- flat_mlp | scored on test | split tag test_grouped | 5 run(s); each value is micro/macro F1
  - seed 0: pair 30.10/16.79, agent 76.00/55.22, error 38.33/35.77
  - seed 1: pair 31.07/18.19, agent 77.49/53.36, error 39.59/38.01
  - seed 2: pair 31.30/18.83, agent 77.49/60.01, error 39.10/38.15
  - seed 3: pair 25.77/17.23, agent 81.91/62.71, error 38.42/39.60
  - seed 4: pair 27.01/17.75, agent 82.49/67.19, error 38.57/38.68
- floor_all_agents_all_modes | scored on test | split tag test_grouped | 1 run(s); each value is micro/macro F1
  - seed 0: pair 7.35/6.35, agent 68.58/61.98, error 27.14/26.99
- frequency | scored on test | split tag test_grouped | 1 run(s); each value is micro/macro F1
  - seed 0: pair 12.14/5.35, agent 80.45/72.05, error 28.47/28.34
- turn_gnn_L0 | scored on test | split tag test_grouped | 5 run(s); each value is micro/macro F1
  - seed 0: pair 25.95/17.50, agent 81.48/64.06, error 38.07/38.23
  - seed 1: pair 26.50/19.08, agent 81.36/67.44, error 39.12/40.15
  - seed 2: pair 32.85/19.34, agent 75.90/59.56, error 41.52/39.75
  - seed 3: pair 30.73/17.75, agent 75.11/54.49, error 37.91/34.90
  - seed 4: pair 32.39/19.71, agent 76.39/57.74, error 40.34/38.46
- turn_gnn_L2 | scored on test | split tag test_grouped | 5 run(s); each value is micro/macro F1
  - seed 0: pair 28.72/20.71, agent 83.49/69.92, error 40.80/41.91
  - seed 1: pair 27.99/19.63, agent 82.94/70.04, error 40.34/41.38
  - seed 2: pair 27.28/19.27, agent 83.29/67.32, error 39.62/40.86
  - seed 3: pair 27.03/18.40, agent 82.67/68.41, error 40.38/40.85
  - seed 4: pair 31.25/19.31, agent 79.50/61.27, error 40.13/40.22
- flat_mlp | scored on test | split tag test_grouped_es_s0 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 26.20/17.36, agent 81.67/65.16, error 38.92/39.84
  - seed 1: pair 30.97/17.91, agent 77.33/60.14, error 39.24/37.62
  - seed 2: pair 26.13/17.29, agent 82.14/64.19, error 38.36/39.27
  - seed 3: pair 25.92/17.37, agent 81.71/63.40, error 38.57/39.85
  - seed 4: pair 26.03/16.86, agent 82.30/64.09, error 38.30/38.80
- floor_all_agents_all_modes | scored on test | split tag test_grouped_es_s0 | 1 run(s); each value is micro/macro F1
  - seed 0: pair 7.35/6.35, agent 68.58/61.98, error 27.14/26.99
- frequency | scored on test | split tag test_grouped_es_s0 | 1 run(s); each value is micro/macro F1
  - seed 0: pair 12.14/5.35, agent 80.45/72.05, error 28.47/28.34
- turn_gnn_L0 | scored on test | split tag test_grouped_es_s0 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 26.10/17.32, agent 81.22/67.29, error 39.09/39.68
  - seed 1: pair 24.48/17.24, agent 81.10/68.15, error 38.04/39.69
  - seed 2: pair 31.23/18.37, agent 77.18/62.41, error 39.54/37.73
  - seed 3: pair 32.24/19.90, agent 78.25/62.28, error 40.49/38.78
  - seed 4: pair 25.48/18.12, agent 81.18/68.64, error 38.39/39.47
- turn_gnn_L2 | scored on test | split tag test_grouped_es_s0 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 27.88/16.50, agent 81.27/63.97, error 38.47/38.16
  - seed 1: pair 27.11/18.42, agent 83.70/70.97, error 39.47/40.82
  - seed 2: pair 26.80/19.16, agent 83.10/67.54, error 39.49/41.20
  - seed 3: pair 26.58/17.58, agent 82.81/67.15, error 39.32/40.24
  - seed 4: pair 27.31/18.03, agent 82.47/63.76, error 39.40/40.28
- flat_mlp | scored on test | split tag test_grouped_es_s1 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 25.72/16.65, agent 81.40/64.91, error 38.40/38.80
  - seed 1: pair 25.53/15.80, agent 80.96/66.52, error 38.16/37.86
  - seed 2: pair 25.81/16.35, agent 81.00/66.42, error 38.58/38.85
  - seed 3: pair 25.78/17.01, agent 81.53/59.03, error 38.34/39.32
  - seed 4: pair 26.14/16.88, agent 81.81/64.21, error 38.78/38.69
- floor_all_agents_all_modes | scored on test | split tag test_grouped_es_s1 | 1 run(s); each value is micro/macro F1
  - seed 0: pair 7.35/6.35, agent 68.58/61.98, error 27.14/26.99
- frequency | scored on test | split tag test_grouped_es_s1 | 1 run(s); each value is micro/macro F1
  - seed 0: pair 12.15/5.38, agent 80.45/72.05, error 28.20/27.70
- turn_gnn_L0 | scored on test | split tag test_grouped_es_s1 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 25.12/16.79, agent 80.98/66.39, error 37.47/37.47
  - seed 1: pair 24.71/17.05, agent 80.72/65.41, error 37.74/38.94
  - seed 2: pair 24.69/17.48, agent 80.84/64.97, error 38.50/40.22
  - seed 3: pair 25.05/17.12, agent 81.13/68.75, error 37.97/38.69
  - seed 4: pair 26.50/18.61, agent 81.24/68.24, error 39.37/40.02
- turn_gnn_L2 | scored on test | split tag test_grouped_es_s1 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 27.29/18.04, agent 82.29/66.94, error 39.53/39.47
  - seed 1: pair 26.99/17.62, agent 82.66/67.35, error 39.97/40.52
  - seed 2: pair 26.30/19.03, agent 82.16/62.92, error 39.02/41.02
  - seed 3: pair 30.18/18.00, agent 76.71/60.20, error 38.29/36.22
  - seed 4: pair 26.55/17.61, agent 81.73/62.31, error 38.64/38.56
- flat_mlp | scored on who_when | split tag who_when_grouped | 5 run(s); each value is micro/macro F1
  - seed 0: pair 3.33/1.13, agent 40.90/27.35, error 8.28/5.12
  - seed 1: pair 4.02/1.01, agent 38.54/23.98, error 9.13/5.54
  - seed 2: pair 2.64/1.29, agent 36.49/25.41, error 9.86/6.84
  - seed 3: pair 4.31/1.44, agent 42.77/33.17, error 11.45/8.38
  - seed 4: pair 4.24/1.16, agent 42.18/31.48, error 12.26/9.03
- floor_all_agents_all_modes | scored on who_when | split tag who_when_grouped | 1 run(s); each value is micro/macro F1
  - seed 0: pair 3.42/2.85, agent 38.32/38.41, error 13.33/12.34
- frequency | scored on who_when | split tag who_when_grouped | 1 run(s); each value is micro/macro F1
  - seed 0: pair 0.00/0.00, agent 0.00/0.00, error 0.00/0.00
- turn_gnn_L0 | scored on who_when | split tag who_when_grouped | 5 run(s); each value is micro/macro F1
  - seed 0: pair 2.00/0.95, agent 38.60/33.92, error 10.08/8.51
  - seed 1: pair 3.59/1.71, agent 40.26/34.21, error 13.54/10.01
  - seed 2: pair 2.33/0.90, agent 37.97/33.04, error 9.67/5.72
  - seed 3: pair 0.65/0.19, agent 36.22/30.16, error 7.24/5.69
  - seed 4: pair 2.13/0.85, agent 36.85/29.00, error 8.53/5.58
- turn_gnn_L2 | scored on who_when | split tag who_when_grouped | 5 run(s); each value is micro/macro F1
  - seed 0: pair 4.29/2.20, agent 37.59/33.94, error 11.87/8.64
  - seed 1: pair 1.26/0.55, agent 35.05/29.28, error 7.58/6.49
  - seed 2: pair 1.69/0.59, agent 35.96/32.11, error 7.67/6.93
  - seed 3: pair 3.44/1.83, agent 38.75/33.48, error 10.25/8.10
  - seed 4: pair 2.05/0.64, agent 37.57/27.79, error 9.40/7.25
- flat_mlp | scored on who_when | split tag who_when_grouped_es_s0 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 3.76/2.10, agent 42.83/34.66, error 10.69/8.17
  - seed 1: pair 3.27/1.05, agent 43.58/32.75, error 7.56/5.61
  - seed 2: pair 4.06/1.71, agent 42.52/36.18, error 11.37/9.28
  - seed 3: pair 3.83/2.23, agent 43.45/36.68, error 9.96/7.97
  - seed 4: pair 3.96/1.68, agent 46.15/35.84, error 10.00/8.12
- floor_all_agents_all_modes | scored on who_when | split tag who_when_grouped_es_s0 | 1 run(s); each value is micro/macro F1
  - seed 0: pair 3.42/2.85, agent 38.32/38.41, error 13.33/12.34
- frequency | scored on who_when | split tag who_when_grouped_es_s0 | 1 run(s); each value is micro/macro F1
  - seed 0: pair 0.00/0.00, agent 0.00/0.00, error 0.00/0.00
- turn_gnn_L0 | scored on who_when | split tag who_when_grouped_es_s0 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 1.95/0.76, agent 37.80/35.33, error 9.25/7.88
  - seed 1: pair 1.91/1.19, agent 37.13/34.70, error 9.00/7.16
  - seed 2: pair 1.43/1.01, agent 34.94/31.47, error 9.17/7.45
  - seed 3: pair 1.17/0.19, agent 37.79/33.43, error 6.12/4.29
  - seed 4: pair 1.27/0.57, agent 37.80/32.79, error 8.72/7.94
- turn_gnn_L2 | scored on who_when | split tag who_when_grouped_es_s0 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 2.18/1.26, agent 33.92/29.91, error 7.68/6.58
  - seed 1: pair 2.81/1.59, agent 37.28/34.52, error 9.64/7.78
  - seed 2: pair 2.77/1.73, agent 37.16/32.83, error 8.99/6.79
  - seed 3: pair 3.43/2.16, agent 35.95/32.92, error 9.87/7.31
  - seed 4: pair 2.00/1.16, agent 34.58/34.27, error 7.45/5.55
- flat_mlp | scored on who_when | split tag who_when_grouped_es_s1 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 4.96/2.88, agent 42.22/34.83, error 12.73/9.22
  - seed 1: pair 4.09/2.22, agent 41.39/34.65, error 11.91/9.20
  - seed 2: pair 4.95/2.60, agent 40.96/32.91, error 13.17/10.85
  - seed 3: pair 3.44/1.84, agent 42.42/35.52, error 11.81/9.25
  - seed 4: pair 5.22/2.78, agent 44.40/35.36, error 12.22/9.76
- floor_all_agents_all_modes | scored on who_when | split tag who_when_grouped_es_s1 | 1 run(s); each value is micro/macro F1
  - seed 0: pair 3.42/2.85, agent 38.32/38.41, error 13.33/12.34
- frequency | scored on who_when | split tag who_when_grouped_es_s1 | 1 run(s); each value is micro/macro F1
  - seed 0: pair 0.00/0.00, agent 0.00/0.00, error 0.00/0.00
- turn_gnn_L0 | scored on who_when | split tag who_when_grouped_es_s1 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 3.14/1.97, agent 38.19/35.28, error 11.62/9.38
  - seed 1: pair 2.84/1.87, agent 36.11/32.32, error 11.82/8.89
  - seed 2: pair 3.37/1.84, agent 39.02/34.59, error 10.68/9.20
  - seed 3: pair 2.16/1.29, agent 35.95/32.84, error 10.63/9.16
  - seed 4: pair 3.96/2.04, agent 40.07/32.75, error 11.31/9.16
- turn_gnn_L2 | scored on who_when | split tag who_when_grouped_es_s1 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 3.00/1.50, agent 38.92/33.18, error 9.44/7.53
  - seed 1: pair 1.91/1.36, agent 37.62/36.20, error 11.20/9.29
  - seed 2: pair 4.20/2.45, agent 36.33/34.15, error 11.65/9.33
  - seed 3: pair 3.29/1.65, agent 36.70/31.84, error 11.04/7.98
  - seed 4: pair 3.20/1.83, agent 35.94/30.96, error 8.74/7.04

### Who&When study variants (phase 7; base / namefree / nameemb, decoders thr and top1; tags = encoder)
- flat_mlp_base_thr | scored on test | split tag test_grouped_es_s0 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 26.20/17.36, agent 81.67/65.16, error 38.92/39.84
  - seed 1: pair 30.97/17.91, agent 77.33/60.14, error 39.24/37.62
  - seed 2: pair 26.13/17.29, agent 82.14/64.19, error 38.36/39.27
  - seed 3: pair 25.92/17.37, agent 81.71/63.40, error 38.57/39.85
  - seed 4: pair 26.03/16.86, agent 82.30/64.09, error 38.30/38.80
- flat_mlp_nameemb_thr | scored on test | split tag test_grouped_es_s0 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 31.78/19.26, agent 79.28/65.43, error 39.90/39.28
  - seed 1: pair 26.65/18.08, agent 82.91/67.61, error 38.76/39.62
  - seed 2: pair 24.95/17.19, agent 81.91/70.47, error 37.31/38.17
  - seed 3: pair 30.78/18.70, agent 77.57/59.80, error 39.02/38.64
  - seed 4: pair 32.16/19.38, agent 77.98/61.15, error 40.14/38.92
- flat_mlp_namefree_thr | scored on test | split tag test_grouped_es_s0 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 25.55/13.03, agent 77.91/57.17, error 38.58/38.62
  - seed 1: pair 24.28/12.47, agent 77.94/58.78, error 38.00/38.24
  - seed 2: pair 25.39/13.75, agent 78.05/57.85, error 38.37/38.94
  - seed 3: pair 24.18/12.21, agent 77.76/57.44, error 37.47/37.92
  - seed 4: pair 24.04/12.94, agent 78.25/57.83, error 37.38/38.02
- flat_mlp_base_thr | scored on test | split tag test_grouped_es_s1 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 25.72/16.65, agent 81.40/64.91, error 38.40/38.80
  - seed 1: pair 25.53/15.80, agent 80.96/66.52, error 38.16/37.86
  - seed 2: pair 25.81/16.35, agent 81.00/66.42, error 38.58/38.85
  - seed 3: pair 25.78/17.01, agent 81.53/59.03, error 38.34/39.32
  - seed 4: pair 26.14/16.88, agent 81.81/64.21, error 38.78/38.69
- flat_mlp_nameemb_thr | scored on test | split tag test_grouped_es_s1 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 27.00/18.47, agent 83.15/70.55, error 39.61/40.71
  - seed 1: pair 25.95/17.01, agent 82.96/69.79, error 38.19/38.67
  - seed 2: pair 30.34/18.28, agent 76.84/64.12, error 38.48/37.57
  - seed 3: pair 31.30/18.36, agent 77.21/58.80, error 38.48/37.82
  - seed 4: pair 33.23/19.28, agent 77.93/63.43, error 41.03/39.38
- flat_mlp_namefree_thr | scored on test | split tag test_grouped_es_s1 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 25.14/13.08, agent 78.05/57.92, error 37.80/38.06
  - seed 1: pair 24.72/12.91, agent 77.34/59.69, error 38.18/37.97
  - seed 2: pair 25.26/13.32, agent 77.10/53.10, error 38.28/38.77
  - seed 3: pair 28.45/13.88, agent 72.35/51.38, error 36.96/35.21
  - seed 4: pair 25.04/13.66, agent 78.11/56.57, error 37.32/37.68
- flat_mlp_base_thr | scored on test | split tag test_grouped_ft_full | 10 run(s); each value is micro/macro F1
  - seed 0: pair 30.10/16.79, agent 76.00/55.22, error 38.33/35.77
  - seed 0: pair 30.10/16.79, agent 76.00/55.22, error 38.33/35.77
  - seed 1: pair 31.07/18.19, agent 77.49/53.36, error 39.59/38.01
  - seed 1: pair 31.07/18.19, agent 77.49/53.36, error 39.59/38.01
  - seed 2: pair 31.30/18.83, agent 77.49/60.01, error 39.10/38.15
  - seed 2: pair 31.30/18.83, agent 77.49/60.01, error 39.10/38.15
  - seed 3: pair 25.77/17.23, agent 81.91/62.71, error 38.42/39.60
  - seed 3: pair 25.77/17.23, agent 81.91/62.71, error 38.42/39.60
  - seed 4: pair 27.01/17.75, agent 82.49/67.19, error 38.57/38.68
  - seed 4: pair 27.01/17.75, agent 82.49/67.19, error 38.57/38.68
- flat_mlp_nameemb_thr | scored on test | split tag test_grouped_ft_full | 5 run(s); each value is micro/macro F1
  - seed 0: pair 32.27/18.40, agent 78.89/58.94, error 38.80/36.91
  - seed 1: pair 32.45/19.48, agent 80.22/61.44, error 41.10/39.85
  - seed 2: pair 32.72/19.50, agent 79.51/60.97, error 41.18/39.31
  - seed 3: pair 31.75/19.42, agent 79.03/60.88, error 39.17/38.07
  - seed 4: pair 27.38/19.04, agent 82.45/68.31, error 38.74/39.54
- flat_mlp_namefree_thr | scored on test | split tag test_grouped_ft_full | 10 run(s); each value is micro/macro F1
  - seed 0: pair 25.75/13.80, agent 79.46/57.45, error 38.56/39.18
  - seed 0: pair 25.75/13.80, agent 79.46/57.45, error 38.56/39.18
  - seed 1: pair 25.70/14.91, agent 78.96/55.74, error 37.76/38.28
  - seed 1: pair 25.70/14.91, agent 78.96/55.74, error 37.76/38.28
  - seed 2: pair 26.14/14.05, agent 78.54/54.39, error 38.68/39.38
  - seed 2: pair 26.14/14.05, agent 78.54/54.39, error 38.68/39.38
  - seed 3: pair 26.93/15.08, agent 79.00/52.33, error 37.86/38.20
  - seed 3: pair 26.93/15.08, agent 79.00/52.33, error 37.86/38.20
  - seed 4: pair 25.12/13.08, agent 79.49/57.00, error 37.52/37.31
  - seed 4: pair 25.12/13.08, agent 79.49/57.00, error 37.52/37.31
- flat_mlp_base_thr | scored on who_when | split tag who_when_grouped_es_s0 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 3.76/2.10, agent 42.83/34.66, error 10.69/8.17
  - seed 1: pair 3.27/1.05, agent 43.58/32.75, error 7.56/5.61
  - seed 2: pair 4.06/1.71, agent 42.52/36.18, error 11.37/9.28
  - seed 3: pair 3.83/2.23, agent 43.45/36.68, error 9.96/7.97
  - seed 4: pair 3.96/1.68, agent 46.15/35.84, error 10.00/8.12
- flat_mlp_base_top1 | scored on who_when | split tag who_when_grouped_es_s0 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 1.63/1.21, agent 32.61/22.07, error 5.98/4.10
  - seed 1: pair 2.17/0.83, agent 36.96/23.51, error 5.98/3.47
  - seed 2: pair 1.09/0.35, agent 31.52/20.65, error 6.52/4.90
  - seed 3: pair 1.09/0.76, agent 31.52/19.03, error 5.43/4.20
  - seed 4: pair 2.72/0.89, agent 36.41/21.76, error 7.07/4.93
- flat_mlp_nameemb_thr | scored on who_when | split tag who_when_grouped_es_s0 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 3.94/2.36, agent 39.62/33.36, error 13.31/10.65
  - seed 1: pair 5.39/3.26, agent 41.45/35.52, error 14.07/9.34
  - seed 2: pair 6.10/3.39, agent 41.93/37.49, error 14.64/10.86
  - seed 3: pair 2.53/1.44, agent 39.48/31.95, error 12.87/8.98
  - seed 4: pair 4.01/2.56, agent 39.70/33.56, error 13.18/9.48
- flat_mlp_nameemb_top1 | scored on who_when | split tag who_when_grouped_es_s0 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 1.63/1.42, agent 28.26/19.62, error 8.15/6.35
  - seed 1: pair 2.17/1.54, agent 28.26/20.52, error 8.15/5.35
  - seed 2: pair 2.17/1.54, agent 26.63/17.85, error 8.15/5.86
  - seed 3: pair 1.63/1.10, agent 23.91/15.26, error 8.70/5.31
  - seed 4: pair 2.17/1.52, agent 28.80/18.91, error 8.70/7.07
- flat_mlp_namefree_thr | scored on who_when | split tag who_when_grouped_es_s0 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 5.55/3.11, agent 48.41/38.31, error 11.69/8.40
  - seed 1: pair 4.49/2.86, agent 45.05/37.50, error 11.98/9.31
  - seed 2: pair 6.13/3.45, agent 46.04/37.77, error 13.95/9.84
  - seed 3: pair 5.98/3.35, agent 47.93/39.08, error 12.42/9.41
  - seed 4: pair 6.57/3.11, agent 50.46/40.42, error 12.54/9.54
- flat_mlp_namefree_top1 | scored on who_when | split tag who_when_grouped_es_s0 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 2.72/1.77, agent 38.59/20.27, error 5.98/4.45
  - seed 1: pair 1.09/0.74, agent 35.33/23.05, error 7.07/6.61
  - seed 2: pair 2.72/1.76, agent 36.41/23.72, error 6.52/5.62
  - seed 3: pair 1.63/1.19, agent 38.59/22.29, error 5.98/4.63
  - seed 4: pair 1.09/0.54, agent 38.59/24.51, error 5.43/4.68
- flat_mlp_base_thr | scored on who_when | split tag who_when_grouped_es_s1 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 4.96/2.88, agent 42.22/34.83, error 12.73/9.22
  - seed 1: pair 4.09/2.22, agent 41.39/34.65, error 11.91/9.20
  - seed 2: pair 4.95/2.60, agent 40.96/32.91, error 13.17/10.85
  - seed 3: pair 3.44/1.84, agent 42.42/35.52, error 11.81/9.25
  - seed 4: pair 5.22/2.78, agent 44.40/35.36, error 12.22/9.76
- flat_mlp_base_top1 | scored on who_when | split tag who_when_grouped_es_s1 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 1.63/1.17, agent 32.07/21.56, error 8.70/5.04
  - seed 1: pair 0.00/0.00, agent 32.61/22.56, error 7.07/4.10
  - seed 2: pair 1.09/0.55, agent 33.70/24.40, error 10.33/9.21
  - seed 3: pair 0.54/0.44, agent 32.07/21.56, error 7.61/5.23
  - seed 4: pair 2.72/0.81, agent 38.04/25.49, error 7.61/5.74
- flat_mlp_nameemb_thr | scored on who_when | split tag who_when_grouped_es_s1 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 6.67/3.55, agent 40.00/35.33, error 16.89/12.30
  - seed 1: pair 6.08/3.32, agent 42.76/39.39, error 15.86/11.33
  - seed 2: pair 5.41/2.80, agent 41.04/34.25, error 16.97/12.92
  - seed 3: pair 5.31/2.70, agent 41.08/35.17, error 18.03/13.53
  - seed 4: pair 4.98/2.45, agent 41.44/35.06, error 14.54/10.64
- flat_mlp_nameemb_top1 | scored on who_when | split tag who_when_grouped_es_s1 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 3.26/2.29, agent 26.63/18.69, error 10.87/6.88
  - seed 1: pair 1.63/1.40, agent 25.00/17.27, error 8.70/5.96
  - seed 2: pair 2.17/1.11, agent 27.17/19.40, error 11.41/7.70
  - seed 3: pair 2.72/1.95, agent 23.91/14.90, error 11.96/5.78
  - seed 4: pair 1.09/0.87, agent 26.09/17.68, error 9.78/7.24
- flat_mlp_namefree_thr | scored on who_when | split tag who_when_grouped_es_s1 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 5.68/3.10, agent 43.23/36.23, error 13.62/8.89
  - seed 1: pair 3.85/2.21, agent 40.73/31.73, error 13.18/9.83
  - seed 2: pair 5.39/2.50, agent 42.30/33.29, error 13.62/9.04
  - seed 3: pair 4.05/2.30, agent 42.89/32.67, error 11.30/6.84
  - seed 4: pair 5.18/2.37, agent 47.68/39.34, error 13.07/9.07
- flat_mlp_namefree_top1 | scored on who_when | split tag who_when_grouped_es_s1 | 5 run(s); each value is micro/macro F1
  - seed 0: pair 2.72/1.82, agent 34.78/24.58, error 8.15/4.95
  - seed 1: pair 1.63/1.29, agent 32.61/24.69, error 8.15/4.27
  - seed 2: pair 3.80/1.75, agent 35.33/25.58, error 9.24/5.01
  - seed 3: pair 2.72/1.80, agent 35.33/23.71, error 9.24/5.48
  - seed 4: pair 2.17/1.03, agent 37.50/25.25, error 5.98/3.96
- flat_mlp_base_thr | scored on who_when | split tag who_when_grouped_ft_full | 10 run(s); each value is micro/macro F1
  - seed 0: pair 3.33/1.13, agent 40.90/27.35, error 8.28/5.12
  - seed 0: pair 3.33/1.13, agent 40.90/27.35, error 8.28/5.12
  - seed 1: pair 4.02/1.01, agent 38.54/23.98, error 9.13/5.54
  - seed 1: pair 4.02/1.01, agent 38.54/23.98, error 9.13/5.54
  - seed 2: pair 2.64/1.29, agent 36.49/25.41, error 9.86/6.84
  - seed 2: pair 2.64/1.29, agent 36.49/25.41, error 9.86/6.84
  - seed 3: pair 4.31/1.44, agent 42.77/33.17, error 11.45/8.38
  - seed 3: pair 4.31/1.44, agent 42.77/33.17, error 11.45/8.38
  - seed 4: pair 4.24/1.16, agent 42.18/31.48, error 12.26/9.03
  - seed 4: pair 4.24/1.16, agent 42.18/31.48, error 12.26/9.03
- flat_mlp_base_top1 | scored on who_when | split tag who_when_grouped_ft_full | 10 run(s); each value is micro/macro F1
  - seed 0: pair 3.80/1.05, agent 39.13/22.72, error 7.61/4.28
  - seed 0: pair 3.80/1.05, agent 39.13/22.72, error 7.61/4.28
  - seed 1: pair 2.17/0.53, agent 33.70/20.09, error 7.61/3.69
  - seed 1: pair 2.17/0.53, agent 33.70/20.09, error 7.61/3.69
  - seed 2: pair 2.17/1.24, agent 33.15/22.03, error 7.07/4.07
  - seed 2: pair 2.17/1.24, agent 33.15/22.03, error 7.07/4.07
  - seed 3: pair 2.17/0.45, agent 36.41/24.17, error 7.61/4.66
  - seed 3: pair 2.17/0.45, agent 36.41/24.17, error 7.61/4.66
  - seed 4: pair 2.17/0.13, agent 41.85/25.51, error 7.61/4.53
  - seed 4: pair 2.17/0.13, agent 41.85/25.51, error 7.61/4.53
- flat_mlp_nameemb_thr | scored on who_when | split tag who_when_grouped_ft_full | 5 run(s); each value is micro/macro F1
  - seed 0: pair 3.61/1.75, agent 40.08/29.83, error 11.49/7.27
  - seed 1: pair 4.46/2.16, agent 43.56/32.89, error 11.72/7.31
  - seed 2: pair 4.80/2.61, agent 41.57/32.93, error 14.68/8.67
  - seed 3: pair 5.59/2.57, agent 40.00/33.49, error 15.60/8.39
  - seed 4: pair 5.89/3.20, agent 43.45/38.14, error 15.24/10.67
- flat_mlp_nameemb_top1 | scored on who_when | split tag who_when_grouped_ft_full | 5 run(s); each value is micro/macro F1
  - seed 0: pair 0.54/0.46, agent 29.89/18.11, error 8.70/6.04
  - seed 1: pair 1.63/0.55, agent 29.89/16.85, error 7.61/3.96
  - seed 2: pair 1.63/0.84, agent 27.72/16.14, error 8.15/5.21
  - seed 3: pair 2.17/1.41, agent 27.17/17.64, error 11.96/5.67
  - seed 4: pair 3.26/1.47, agent 30.98/19.82, error 8.70/4.95
- flat_mlp_namefree_thr | scored on who_when | split tag who_when_grouped_ft_full | 10 run(s); each value is micro/macro F1
  - seed 0: pair 6.31/3.12, agent 47.97/35.25, error 13.32/8.78
  - seed 0: pair 6.31/3.12, agent 47.97/35.25, error 13.32/8.78
  - seed 1: pair 4.81/1.97, agent 46.64/33.64, error 12.37/8.89
  - seed 1: pair 4.81/1.97, agent 46.64/33.64, error 12.37/8.89
  - seed 2: pair 6.40/3.20, agent 48.66/36.69, error 15.05/9.86
  - seed 2: pair 6.40/3.20, agent 48.66/36.69, error 15.05/9.86
  - seed 3: pair 5.45/2.44, agent 48.78/37.77, error 12.46/7.76
  - seed 3: pair 5.45/2.44, agent 48.78/37.77, error 12.46/7.76
  - seed 4: pair 6.05/2.88, agent 51.11/38.36, error 13.21/9.59
  - seed 4: pair 6.05/2.88, agent 51.11/38.36, error 13.21/9.59
- flat_mlp_namefree_top1 | scored on who_when | split tag who_when_grouped_ft_full | 10 run(s); each value is micro/macro F1
  - seed 0: pair 3.26/0.81, agent 41.85/25.52, error 7.61/4.40
  - seed 0: pair 3.26/0.81, agent 41.85/25.52, error 7.61/4.40
  - seed 1: pair 3.80/1.06, agent 39.13/22.92, error 9.24/6.25
  - seed 1: pair 3.80/1.06, agent 39.13/22.92, error 9.24/6.25
  - seed 2: pair 3.80/1.66, agent 38.59/22.59, error 8.15/4.79
  - seed 2: pair 3.80/1.66, agent 38.59/22.59, error 8.15/4.79
  - seed 3: pair 3.26/0.63, agent 37.50/23.75, error 6.52/4.30
  - seed 3: pair 3.26/0.63, agent 37.50/23.75, error 6.52/4.30
  - seed 4: pair 3.80/1.12, agent 42.39/24.12, error 8.15/6.20
  - seed 4: pair 3.80/1.12, agent 42.39/24.12, error 8.15/6.20

### End-to-end graph fine-tune (phase 8; ftg_l2 = 2-layer graph, ftg_l0 = control, nf = name-free)
- graph_finetune_e2e | scored on test | split tag test_ftg_l0 | 1 run(s); each value is micro/macro F1
  - seed 0: pair 26.00/18.59, agent 80.88/65.88, error 38.38/39.44
- graph_finetune_e2e | scored on test | split tag test_ftg_l0nf | 1 run(s); each value is micro/macro F1
  - seed 0: pair 20.19/11.05, agent 77.77/55.30, error 33.38/31.91
- graph_finetune_e2e | scored on test | split tag test_ftg_l2 | 1 run(s); each value is micro/macro F1
  - seed 0: pair 23.69/16.62, agent 80.15/64.89, error 35.22/36.00
- graph_finetune_e2e | scored on test | split tag test_ftg_l2nf | 1 run(s); each value is micro/macro F1
  - seed 0: pair 22.90/13.67, agent 81.95/63.73, error 35.72/36.17
- graph_finetune_e2e | scored on who_when | split tag who_when_ftg_l0 | 1 run(s); each value is micro/macro F1
  - seed 0: pair 2.40/1.03, agent 41.18/35.74, error 9.76/9.35
- graph_finetune_e2e | scored on who_when | split tag who_when_ftg_l0nf | 1 run(s); each value is micro/macro F1
  - seed 0: pair 4.41/2.06, agent 47.62/35.00, error 9.97/7.04
- graph_finetune_e2e | scored on who_when | split tag who_when_ftg_l2 | 1 run(s); each value is micro/macro F1
  - seed 0: pair 2.90/0.79, agent 42.55/33.92, error 7.54/4.52
- graph_finetune_e2e | scored on who_when | split tag who_when_ftg_l2nf | 1 run(s); each value is micro/macro F1
  - seed 0: pair 5.04/2.84, agent 39.18/35.46, error 9.90/7.07

<!-- FULL_VALUES_END -->
