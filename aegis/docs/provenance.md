# Provenance: where every number in the paper comes from (written 2026-11-16)

Rule: a number may appear in the paper only if a row below names its script, command, log and result file. Ledger = `aegis/results/metrics_history.jsonl`
(one JSON line per run; fields `phase`, `model`, `seed`, `split`, `config_hash`, `git_commit`). Published numbers are copied, never recomputed.

| Paper content | Result file | Generating script and command | Log / ledger | Notes |
|---|---|---|---|---|
| Published rows (Random, DCL, zero-shot LLMs, Aegis-SFT, ...) | `aegis/results/published_rows.json` | `aegis/scripts/extract_published_rows.py <paper text> <json>` | AEGIS paper Table 1, arXiv 2509.14295v6 | Copied as reported; 25 rows; values re-read against the PDF on 2026-11-16 |
| Scorer equals the authors' scorer | pytest result | `aegis/tests/test_scorer_parity.py` (200 fuzz sets, end-to-end JSONL, unparseable-rule pin) | commit b75ab5a | The only difference is the unparseable-output rule (authors skip, we score empty) |
| Test-file comparison, exact-duplicate counts, 100-per-benchmark check | `aegis/results/test_file_comparison.json`, `aegis/docs/dataset_audit.md` (addendum), `aegis/docs/decisions.md` D11 | `python -m aegis.scripts.compare_test_files` | commit 6f13277 | Conversation identity = SHA-1 of `agent_name TAB content` over all turns |
| One model on both test files, released protocol | `aegis/results/preds/s2_flat_paper_seed*_*.json` | `python -m aegis.src.eval.test_file_check --seeds 5` | `aegis/results/runs/s2_test_file_check_2026-11-16.log`; ledger phase `S2` | Frozen MiniLM + flat MLP; embeddings recomputed locally (`embed_local_2026-11-16.log`) |
| Same, duplicates removed | `aegis/results/preds/s2_flat_paper_dedup_seed*_*.json` | `... test_file_check --seeds 5 --dedup` | `s2_test_file_check_dedup_2026-11-16.log`; ledger phase `S2` | |
| Floors, frequency, shortcut-only, content-only, full; bootstrap intervals | `aegis/results/block_ablation.json` | `python -m aegis.src.eval.block_ablation --seeds 5` | `aegis/results/runs/s3_block_ablation_2026-11-16.log`; ledger phase `S3` | 2,000 record-level bootstrap resamples, seeds pooled |
| Comparison tables (Markdown) | `aegis/results/comparison_tables.md` | `python -I aegis/scripts/build_comparison_tables.py` | generated from the two files above | Do not edit by hand |
| 14 encoders under the published protocol | `aegis/results/kaggle_logs/s4_s0 ... s4_s13/` (`ft_log.json`, `run_info.json`, kernel log); vectors in `.../ft_shards` (local, git-ignored, 163 MB each) | `aegis/scripts/make_s4_inputs.py`, `make_s4_kernel.py`, `s4_status_pull.py`, `s4_requeue_loop.py`; manifest `kaggle_logs/s4_manifest.md` | `aegis/results/runs/s4_requeue_loop_2026-11-16.log` | Train 5,844 / early-stop 1,068 / val 1,657 (untouched); 101,009 texts embedded; 5 Kaggle accounts |
| S4 ensemble (block-dropout masks, label shift), three test sets | `aegis/results/s4_ensemble.json`, `aegis/results/preds/s4_ensemble_seed0_preds.json` | `python -m aegis.src.eval.s4_ensemble --encoders 0,...,13 --seeds 3 --tag s4` | `aegis/results/runs/s4_ensemble_2026-11-16.log`; ledger phase `S4` | Thresholds from released val; no Who&When label used; label shift is transductive |
| Old query-disjoint results (draft Tables V-X, 32.96 / 8.24 / mask grid) | `local-notes/results_master.md`, `local-notes/timeline.md`, ledger phases `6r`, `6ft`, `7`, `8`, `10` | `aegis/src/eval/ww_block.py`, `ww_focus.py`, `final_test.py` (see `aegis/docs/final_results.md`) | `aegis/results/kaggle_logs/es_s*`, `aegis/results/runs/*.log` | Labelled "query-disjoint, as logged"; not re-run on this machine |
| Frozen-judge studies (Tables XI-XIII) | `aegis/results/rr/final_rrf1.json`, `rr_k12/` | `aegis/src/eval/rr_*.py` (runbook `local-notes/rr_runbook.md`) | `aegis/results/kaggle_logs/*judge*` | Pre-registered; not re-run |

## Known gaps (do not hide)
- The embeddings used for S2/S3/S4 downstream scoring were recomputed locally on 2026-11-16 (MPS); the old Kaggle-GPU cache is gone. A check that the new vectors reproduce the old numbers: the HF-test, released-protocol
  flat-MLP result is 33.02 pair / 80.72 agent / 40.36 error micro-F1 here vs 33.08 / 80.86 / 40.43 in the earlier ledger (phase `6r`).
- Smoke-test ledger rows (`s4_ens1_*`, config `encoders: [7]`) are exploratory and must not be quoted.
- `aegis-raw-data/whowhen.jsonl` on four Kaggle accounts is 184 bytes larger than the authors' repo copy (probably newlines); embeddings are keyed on content, so unaffected, but unverified.
- Ledger phase-6 rows (Who&When bug) are kept for history and superseded by phase `6r`.
