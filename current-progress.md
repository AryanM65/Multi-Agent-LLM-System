# Current progress (as of 2026-11-03)

Branch: `current-progress` (a snapshot of `aegis_v1`). Read `context.md` first for the background and vocabulary.
All numbers are percentages. "micro / macro" means micro-F1 / macro-F1. Every number below comes from a log or the ledger
(`aegis/results/metrics_history.jsonl`); where a number is partial or doubtful this is said explicitly.

## 1. One-paragraph summary

We moved the thesis to the published AEGIS-Bench dataset (arXiv 2509.14295), built and tested the data loading and the
authors' scoring code, reproduced the paper's zero-shot Qwen row approximately, wrote our own version of the paper's small
baseline (DCL-style), built simple baselines, found that the released split leaks questions and that the paper's scores sit
below a trivial "predict everything" floor, and then built a graph model over the conversation. The graph model beats the
simplest neural baseline by about 4 points of pair-level micro-F1 on validation, but almost all of that gain comes from
projecting each turn before averaging and from letting turns see the query; the turn-to-turn graph edges add nothing
measurable. On the held-out test set (partly scored) the graph's extra gain over the control mostly disappears. The final
test run was killed by the operating system for low memory before it finished, and its Who&When numbers look wrong, so
the final table is **not complete**.

## 2. What was done, phase by phase

### Phase 0: setup
- Branch `aegis_v1` created from `model_v1`. New code under `aegis/`; the old project was not modified.
- Cloned the authors' code to `aegis/third_party/AEGIS/`, downloaded the Hugging Face dataset to `aegis/data/` (both git-ignored).
- `aegis/src/utils.py`: seeding, config hash, git commit lookup, and `log_run`, which appends one line per run to the ledger.
- `requirements-aegis.txt`, a seed policy, and a decisions log.

### Phase 1: data audit (`aegis/docs/dataset_audit.md`)
- Counts, labels per framework, and a leakage review of every field.
- **Query leakage:** 588 of 600 test questions also occur in train, and 1,753 of 1,787 validation questions occur in train.
- The turn field `is_injected` (macnet only), `ground_truth.*`, `metadata.num_injected_agents`, `metadata.num_agents` and
  `injection_strategy` would give the answer away. They are never loaded (`aegis/src/loaders.py` reads a whitelist).
- The repo's `evaluation/data/aegis.jsonl` is a different 600-record file from the Hugging Face test file (no ids in common).
  Which one the paper used is unknown. We use the Hugging Face file (decision D1).
- Decisions D1-D10 are in `aegis/docs/decisions.md`.

### Phase 2: evaluation harness
- `aegis/src/schema.py`, `loaders.py`: typed records, mode-name normalisation, paper split and grouped split.
- `aegis/src/metrics.py`: a port of the authors' `evaluate.py`. It was compared line by line with the authors' file again on
  2026-11-02 and matches (same class union for macro-F1, same micro/macro formulas, same pair/agent/error projections). The only
  difference is that we count an unparseable model output as an empty prediction, while the authors skip it.
- 16 unit tests pass (`aegis/tests/`).
- Sanity check with a random baseline fitted by hand: pair 0.40/0.19, agent 5.86/4.03, error 11.74/11.53 against the paper's
  0.33/0.21, 4.54/3.56, 11.23/11.15. Close, not identical (the paper's sampler is unknown).

### Phase 3a: reproduce the zero-shot Qwen row (`aegis/docs/reproduction_log.md`)
- Qwen2.5-7B-Instruct (AWQ 4-bit) with vLLM on a Kaggle T4, the authors' prompt, greedy decoding, 512 new tokens, 16,384-token context
  (long prompts are cut to the first 30% and last 70%). 600 test and 184 Who&When prompts took about 4.5 hours in total.
- Raw outputs and the full log are saved in `aegis/results/kaggle_logs/full/`. Three earlier smoke-test attempts failed on library
  conflicts (torch/torchaudio/torchvision versions); their logs are kept in `smoke_v1..v3`.

| Benchmark, level | Ours | Paper |
|---|---|---|
| Aegis-Bench pair | 8.01 / 2.48 | 5.02 / 2.52 |
| Aegis-Bench agent | 49.24 / 10.69 | 27.55 / 14.49 |
| Aegis-Bench error | 20.15 / 12.72 | 14.96 / 11.36 |
| Who&When pair | 2.98 / 1.85 | 2.31 / 1.14 |
| Who&When agent | 35.16 / 23.14 | 40.92 / 23.50 |
| Who&When error | 6.23 / 3.35 | 3.64 / 1.77 |
| Avg12 | 14.67 | 12.43 |

- Verdict: close on most rows; **not matched** on Aegis-Bench agent micro-F1 (49.2 vs 27.6). The cause is unknown. Candidates, none tested:
  the paper may have used the other test file, 4-bit weights, a different prompt, a different parse rule, or sampling instead of greedy decoding.
  The metric code is ruled out (identical to the authors'), and the 8 unparseable outputs cannot explain a 21-point gap.
- Optional follow-up not run: repeat on `evaluation/data/aegis.jsonl` (about 3.5 GPU-hours).

### Phase 3b: DCL-style baseline
- The paper's DCL has no released code, so `aegis/src/baselines/dcl_style.py` re-implements the described design with a **frozen**
  MiniLM encoder (the paper fine-tunes it): bag of turns, top-3 attention pooling per agent, agent and mode prototype banks,
  asymmetric BCE + supervised contrastive + hierarchical consistency loss. It is named "DCL-style" and is not DCL.
- Test results (3 seeds, mean): trained on the paper split pair 15.06/8.56, agent 69.88/57.22, error 34.34/33.23; trained on the grouped
  split pair 11.59/6.01, agent 66.47/51.82, error 28.96/27.14. The published DCL is pair 8.33/5.30, agent 22.93/20.23, error 24.73/27.70.
- It beats the published DCL on 10 of 12 micro-F1 numbers, so it is **not** a reproduction. On validation it is worse than the flat MLP.

### Embeddings
- All 99,406 unique turn texts and 1,502 queries were embedded once with all-MiniLM-L6-v2 (max length 256) and cached on disk keyed by the
  SHA-1 of the text (`aegis/src/embed.py`). The big job ran on a Kaggle GPU in 275 seconds (a local run was killed for low memory);
  the log is in `aegis/results/kaggle_logs/embed/`. The cache itself is git-ignored.

### Phase 4: simple baselines (validation only, `aegis/docs/baselines_log.md`)
- **Trivial floor** on validation: agent micro 70.5 (paper split) / 73.4 (grouped), error micro 28.2 / 29.9. The published DCL (22.9, 24.7) and the
  zero-shot rows are below these floors. So beating DCL's published average is not by itself evidence of a good model; every table carries the floor.
- Frequency baseline, grouped validation: pair 15.23/8.32, agent 84.11/77.20, error 31.27/30.73.
- Flat MLP (3 seeds): paper split pair 32.19, agent 82.33, error 41.26; grouped split pair 23.05, agent 83.95, error 36.01 (micro).
  The drop from the paper split to the grouped split shows how much the question leakage inflates scores.

### Error analysis (`aegis/docs/error_analysis.md`)
Counted over all train data, plus a manual read of 18 trajectories (3 per framework; the plan asked for 5, so this is a smaller sample):
- Every framework except macnet gives each faulty agent exactly one mode.
- Some roles are never faulty: macnet Node-1 and Node-2, `User` in llm_debate and smoagents, `System` in smoagents, `user` in magentic_one.
- The number of faulty agents is nearly fixed per framework (for example smoagents and magentic_one always 1).
- Modes are close to balanced within each benchmark, so the mode must come from the text.
- Some injected faults are visible against the query or peers; others are invisible from the text, which limits any text model.

### Phase 4b: top-k decoding (no gain)
Using the fixed-count structure as a decoder (take the top k agents per framework, mask never-faulty roles) did not help: on grouped validation pair micro
+1.0 and agent micro +0.5 but error micro -3.4 and every macro number lower. Threshold decoding stays the default. (`flat_topk.py`)

### Phase 4c: relational probe (small gain)
Adding four features (cosine of an agent's text to the query, and to the other agents) to the flat MLP raised grouped validation pair micro from 23.05 to
23.61 (higher in all 3 seeds), error micro +0.4; inconclusive on the paper split. (`flat_probe.py`)

### Phase 5: graph model (exploratory, validation only)
- `aegis/src/graph.py`: one node per turn (frozen MiniLM vector) plus a query node. Edge types: consecutive turns, consecutive turns of the same agent,
  query to turn, turn to query. `aegis/src/gnn/turn_gnn.py`: two relation-specific message-passing layers (128-d), mean of an agent's turns,
  concatenated with the same framework/agent-name/count features as the flat MLP, MLP head, 14 mode outputs plus one faulty-agent output.
- Grouped validation, 5 seeds (mean +- std):

| Model | Pair | Agent | Error |
|---|---|---|---|
| flat MLP (3 seeds) | 23.05 +- 0.19 | 83.95 | 36.01 |
| 0-layer control | 25.66 +- 0.40 | 84.10 | 37.67 |
| GNN, all edges | 26.86 +- 0.61 | 84.05 | 38.18 |
| GNN without same-agent edges | 26.56 +- 0.68 | 83.62 | 38.45 |
| GNN without sequence edges | 26.62 +- 0.53 | 83.09 | 38.43 |
| GNN without the query node | 25.55 +- 0.40 | 82.98 | 37.08 |

- Reading: removing the query node removes the whole gain over the control; removing the other edge types changes little. So the useful part is "each
  turn can see the question", not the conversation structure. Agent-level micro-F1 is about 84 for every model (the floor).
- Same-seed runs of the full GNN were not exactly repeatable (pair micro 27.62 vs 26.20 for seed 0), so differences below about 1 point are unresolved.

### Phase 6: final held-out scoring (`aegis/src/eval/final_test.py`) — INCOMPLETE
- Protocol: train on a split's train set, early-stop and choose thresholds on its validation set, then score test and Who&When once. A dry run on slices of
  validation passed first.
- The run was **killed by the system for low memory** after finishing the grouped protocol for the floor, frequency, flat MLP (5 seeds), 0-layer control
  (5 seeds) and 4 of 5 GNN seeds. The paper protocol did not start. The partial log is `aegis/results/runs/final_test.log`; 32 phase-6 rows are in the ledger.
- Test results so far, grouped protocol, micro-F1:

| Model | Pair | Agent | Error |
|---|---|---|---|
| floor | 7.35 | 68.58 | 27.14 |
| frequency | 12.15 | 80.45 | 28.20 |
| flat MLP (5 seeds) | 19.70 | 79.22 | 32.76 |
| 0-layer control (5 seeds) | 21.07 | 79.71 | 34.68 |
| GNN (4 seeds, my mean) | 21.14 | about 78.8 | 34.94 |

- The projection-then-pool gain over the flat MLP holds on test (+1.4 pair, +1.9 error). The GNN's gain over the control does **not** (+0.2 pair, +0.3 error,
  against +1.2 and +0.5 on validation).
- **Who&When rows from this run are not trustworthy.** The flat MLP and the 0-layer control gave identical Who&When scores (pair 2.72, agent 33.70, error 10.87)
  with zero spread across seeds, which should be impossible for different models. The cause has not been found, and these rows must not be used.

## 3. Problems met and how they were handled
- Kaggle runtime library conflicts (torch/torchaudio/torchvision after installing vLLM): fixed by uninstalling only torchaudio.
- Local embedding job killed for low memory: embedded on a Kaggle GPU instead, and made the embedder write shards incrementally.
- A wrong count in a written verdict ("11 of 12") was corrected to "10 of 12".
- The system killed the final test run for low memory; it was not restarted automatically.

## 4. Open questions and caveats
1. The guide has not yet approved the switch to AEGIS-Bench, so the graph work is exploratory.
2. Why the paper's agent-level scores (DCL 22.9, zero-shot Qwen 27.6) are below the trivial floor, and why our zero-shot agent micro is 49.2, is unexplained.
3. The Who&When bug in the final scorer.
4. One graph design was tried, with a frozen encoder and no hyper-parameter search; the dataset has no stored communication edges.
5. The error analysis read 18 trajectories, not 30.
6. Agent-level micro-F1 cannot separate any of the models (all near the floor), so claims rest on pair level and on macro scores.

## 5. What remains
1. Find the Who&When bug using a few predictions, then re-run the missing pieces of the final scoring one protocol at a time once memory allows
   (duplicate phase-6 ledger rows need to be replaced or tagged).
2. Write `aegis/docs/final_results.md` with the final table next to the published rows.
3. Get the guide's decision on the dataset switch.
4. Optional: the `aegis.jsonl` zero-shot repeat, longer GNN tuning, a fine-tuned encoder.

## 6. How to reproduce (from the repo root, with the data and embedding cache in place)
```
python -m pytest aegis/tests -q                         # 16 tests
python -m aegis.src.baselines.frequency                  # frequency baseline
python -m aegis.src.baselines.flat_mlp --seeds 3         # flat MLP
python -m aegis.src.baselines.flat_probe --seeds 3       # relational probe
python -m aegis.src.gnn.turn_gnn --seeds 5 --splits grouped --layers 2   # graph model (--layers 0 = control, --drop same|seq|query)
python -m aegis.src.eval.final_test --dry-run            # code-path check without reading test
```
The Kaggle jobs are `aegis/scripts/kaggle_zeroshot_qwen.py` and `aegis/scripts/kaggle_embed_minilm.py`.

## 7. Where to read more
`aegis/docs/dataset_audit.md`, `decisions.md`, `reproduction_log.md`, `baselines_log.md`, `error_analysis.md`; the ledger
`aegis/results/metrics_history.jsonl`; training logs in `aegis/results/runs/`; saved GPU job outputs in `aegis/results/kaggle_logs/`.
