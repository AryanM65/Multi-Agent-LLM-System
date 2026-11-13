# The 5-minute work loop (2026-11-12): what it was, what it ran, how to rebuild it

LOCAL ONLY (`local-notes/`, git-excluded). Never commit or push. The Kaggle token is deliberately NOT written here: it lives in `kaggle.txt` at the repo root and is passed per command as the env var `KAGGLE_API_TOKEN`.

## 1. What the loop was
- A session-only cron job (final id `118a6b8f`, every 5 minutes, auto-expires after 7 days, deleted at the end). Each tick re-sent one long prompt ("AEGIS work loop tick") that told Claude to: check the state, do the next unfinished step, update `local-notes/file.md` and `local-notes/context.md` after every step, append a one-line status to both files every tick, and report briefly.
- An earlier GPU-monitor cron (every 5 minutes) watched the first Kaggle fine-tune kernels. Both crons are gone; nothing is scheduled now. If a new loop is wanted, recreate it with the loop skill or CronCreate and an explicit state description in the prompt (an earlier prompt without explicit state would have re-launched an already-running job).
- User wishes behind it: "keep a loop of every 5 minutes", "update the local notes side by side", "give an update after every 5 minutes", "after you do one thing please update the local notes also", "after you complete the loop show me the results and whether they completed".

## 2. Standing rules inside the loop prompt
- No commits or pushes. `local-notes/` never committed.
- Keep every log in the repo. The moment a Kaggle kernel is COMPLETE, pull its log and output into `aegis/results/kaggle_logs/<name>/` BEFORE anything else; never stop or delete a RUNNING kernel (CLAUDE.md: logs are unreadable while RUNNING and lost if deleted early).
- Token only via `KAGGLE_API_TOKEN` env var; never written to a file or notes.
- Environment for local runs: Python 3.12 (`/c/Python312/python`), `PYTHONPATH=<repo root>`, `PYTHONUTF8=1`, `USE_TF=0 TRANSFORMERS_NO_TF=1` (avoids a TensorFlow/protobuf ImportError), one heavy local job at a time (about 4 GB free RAM while the chain ran).
- Never repeat a finished step; read the tail of `local-notes/file.md` and the logs to see what is done.

## 3. The plan the loop executed (all steps DONE)
A. Background chain (scratchpad `chain.sh`): `final_test` then `whowhen_study` for the clean encoders `es_s0` and `es_s1`; progress in `aegis/results/runs/chain_status.log` (`CHAIN_DONE ...`, final `CHAIN_ALL_DONE`); logs `final_test_es_s{0,1}.log`, `whowhen_study_es_s{0,1}.log`.
B. Wrote `aegis/src/eval/score_graph_ft.py`, a CPU scorer for the `preds.json` exported by the end-to-end graph fine-tune (thresholds tuned on val only via `flat_mlp.tune`; micro/macro pair/agent/error; floors; avg12; per-framework test breakdown; ledger phase 8 with `--tag`).
C1. Wrote the clean-encoder grouped-test and Who&When tables into `aegis/docs/final_results.md`.
C2. Local CPU smoke test of `aegis/scripts/kaggle_finetune_graph.py` (`--limit 16 --epochs 1 --bs 4`).
C3. Kaggle smoke kernel `aegis-ftg-smoke` (`--limit 64 --epochs 1 --bs 8`); log in `aegis/results/kaggle_logs/ftg_smoke/`.
C4. Two full end-to-end kernels: `aegis-ftg-l2` (`--layers 2`) and `aegis-ftg-l0` (`--layers 0`), both `--epochs 6 --patience 2 --bs 8 --es-holdout 0.15 --seed 0`; logs in `ftg_l2/`, `ftg_l0/`.
C5. Scored both with `score_graph_ft.py` (tags `ftg_l2`, `ftg_l0`), comparison written into `final_results.md`.
C6. Name-free pair `aegis-ftg-l2nf` and `aegis-ftg-l0nf` (same plus `--name-free`); logs in `ftg_l2nf/`, `ftg_l0nf/`; scored with tags `ftg_l2nf`, `ftg_l0nf`; written into `final_results.md`.
D. Deleted the cron and gave the final summary (19:2x on 2026-11-12). Ticks ran about every 5 minutes from the first chain start through 19:14, when both name-free kernels completed.

## 4. Commands that were used (token via env var only)
- Status: `KAGGLE_API_TOKEN=<token> kaggle kernels status aryanmahajan7/aegis-ftg-l2nf`
- Save output (do this first when COMPLETE): `kaggle kernels output aryanmahajan7/aegis-ftg-l2nf -p aegis/results/kaggle_logs/ftg_l2nf` (also saves the kernel log).
- Launch: build a kernel dir from the scratchpad template `tpl/kernel-metadata.json` (pulled once with `kaggle kernels pull -m aryanmahajan7/aegis-ft-bge`): main file = one `sys.argv += [...]` line followed by the script; `dataset_sources` = `aegis-raw-data` + `aegis-grouped-ids`; GPU on; then `kaggle kernels push -p <dir>`.
- Score: `PYTHONUTF8=1 PYTHONPATH=. USE_TF=0 /c/Python312/python -m aegis.src.eval.score_graph_ft --preds aegis/results/kaggle_logs/<name>/preds.json --tag <tag> > aegis/results/runs/score_<tag>.log`
- Who&When study: `python -m aegis.src.eval.whowhen_study --ft-dir <ft dir> --tag <tag> --seeds 5`; final test: `python -m aegis.src.eval.final_test --ft-dir <ft dir> --tag <tag>`.
- Kaggle runs took about 130 s per epoch on a T4 (6 epochs, about 15-20 minutes each; two kernels ran in parallel).

## 5. Where the loop's outputs are
- Logs: `aegis/results/kaggle_logs/{ft_bge,ft_es_s0,ft_es_s1,ftg_smoke,ftg_l2,ftg_l0,ftg_l2nf,ftg_l0nf}/`; `aegis/results/runs/` (`ft_eval_*.log`, `final_test_es_s*.log`, `whowhen_study_*.log`, `score_ftg_*.log`, `chain_status.log`).
- Results: `aegis/docs/final_results.md` (clean-encoder, Who&When and end-to-end graph sections); ledger `aegis/results/metrics_history.jsonl` (phases 7 and 8).
- Notes: `local-notes/file.md` (story, entry per step and tick), `local-notes/context.md` (technical reference), `local-notes/ideas.md` (next-round brainstorm).
- Result headline: fine-tuning the encoder and name-embedding features gave the real gains; the graph (two-stage or end-to-end) gave no stable gain (end-to-end graph vs control: -2.3 pair name-dependent, +2.7 name-free, one seed each).

## 6. State now
No cron, no kernel running, nothing committed or pushed, many uncommitted files (see the last entry of `file.md`).

## 7. Current loop (replaces section 6; set 2026-11-12 23:02, cron job 8c7ff033, every 5 minutes, session-only, expires after 7 days)
- Purpose: entire focus on improving Who&When without hurting AEGIS-Bench; LoRA language-model side track skipped for now.
- Accounts: four Kaggle accounts, tokens read per command from kaggle.txt by line number into env var KAGGLE_API_TOKEN, never written anywhere: line 1 = A (aryanmahajan665), line 3 = B (aryanmahajan7), line 5 = C (amahaj55), line 7 = D (aryanomgh65). Each has private datasets aegis-grouped-ids and aegis-raw-data.
- Every tick, in order: (1) status of each unsaved kernel; (2) COMPLETE kernel: save log and output into aegis/results/kaggle_logs/<short>/ first, then queue CPU scoring; (3) record results in results.md, refresh the full-values appendix, add a dated line to timeline.md, append status lines to file.md and context.md (the user wants the notes updated inside the loop every tick); (4) when ww_focus lands write verdicts (ensemble vs single, label-shift vs none, nd vs peernd; Who&W first, AEGIS second); (5) Who&W CPU ideas one at a time (name-to-role mapping, scalar-dropout, test-time adaptation as separate transductive row, leave-one-framework-out validation); (6) idle GPU gets a Who&W job (name-shuffle encoder, leave-one-framework-out encoders, more seeds of the winning pooling); (7) about every 3 hours a new paper brainstorm round (ideas.md section 7 style) and a check of the escalation ladder in timeline.md; (8) brief bullet report.
- Constraints: one heavy local CPU job at a time (chain scripts chain6.sh, chain7.sh in the scratchpad, status lines in aegis/results/runs/chain_status.log); no commits or pushes; local-notes never committed; no markdown tables.
- Stop: when the user says stop, or after 2026-11-15; then cancel the cron and write a final summary with honest caveats.
- Kernel names and short names so far: len512 family (len512, len512_s1, len512_s2, len512_s3), l12 (l12, l12len512, l12len512_s1, l12len512_s2), chunk, bgesmall, es_s2..es_s6, attn (account D, end-to-end attention pooling).
