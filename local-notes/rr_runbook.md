# rr_runbook.md: how to run the "retriever-only" and "model shortlist + LLM judge (+ AEGIS retrieval)" experiments (Track A)

Written 2026-11-14. Source of the idea and the rules: `local-notes/rerank_rag_context.md` (the hand-off file). Format rule for this project: bullets only, no tables. Nothing here has been committed or pushed beyond what the user already pushed (branch 13-november), and no full run has been started: only scripts exist, tested on tiny inputs.

---

## 1. What was written (all new files, nothing existing was changed except as noted)

- `aegis/src/eval/rr_common.py`: shared helpers. Retrieval-feature block (`retrieval_block`), label-shift helper (same arithmetic as ww_block.py), probability-dump reader/writer, shortlist builder, bootstrap confidence intervals. The environment variable `RR_DIR` redirects all outputs (used only for smoke tests).
- `aegis/src/eval/rr_models.py`: ww_block.py (four-mask block-dropout) plus an optional retrieval block (`--retr 1`) and a dump of the ensemble probabilities to `aegis/results/rr/dump_<tag>.npz`. Run with `--retr 0` for the baseline dump and `--retr 1` for the retriever-only variant.
- `aegis/src/eval/rr_gate0.py`: Gate 0 (recall@k, hit@k, agent-recall@k, oracle re-rank F1, reachable share) from a dump. Writes `aegis/results/rr/gate0_<tag>.json`.
- `aegis/src/eval/rr_candidates.py`: top-k shortlist per trajectory plus mode-first labelled retrieval examples (`examples_rag`) and a random-examples control (`examples_rand`). Writes `aegis/results/rr/cand_<tag>_<split>.jsonl`. Has `--subsample N` (seeded random subset of val/test; Who&When always keeps all 184).
- `aegis/scripts/kaggle_judge.py`: frozen-LLM judge (Kaggle T4). One shared conversation prefix per record (key/value cache reused), yes/no log-odds read from one next-token distribution, option-order swap for bias control, three variants (plain, rag, rand) from one prefix, incremental output, time budget, several candidate files per model load.
- `aegis/scripts/rr_make_judge_kernel.py`: builds the Kaggle kernel directory and the small candidates dataset (no network, no token).
- `aegis/src/eval/rr_fuse.py`: fusion, tuning, pre-commit and final scoring. `--phase tune` uses AEGIS validation only and writes a pre-commit file; `--phase final` refuses to run without it and refuses to overwrite an existing result.
- Ledger hygiene: smoke tests wrote 24 fake or tiny-subset rows into `aegis/results/metrics_history.jsonl` (tags containing `rr_smoke` and `rr_fuse_tt_`). I moved them to `aegis/results/rr/smoke_ledger_rows_moved_out.jsonl` (backup of the full ledger is in the scratchpad as `metrics_history.backup.jsonl`). Smoke artifacts are in `aegis/results/rr/smoke/`. The scripts no longer log in smoke mode (`--limit` in rr_models, `RR_DIR` set in rr_fuse).

---

## 2. What was tested and what was NOT

- Tested on CPU with a tiny subset (one encoder, 400 trajectories, one head seed): rr_models (with retrieval block), rr_gate0, rr_candidates, rr_fuse (tune and final, on SYNTHETIC judge scores in a scratch directory, only to check the code path), the retrieval block (leakage masking checked on a planted duplicate), the judge script (Qwen2.5-0.5B-Instruct on CPU, two records, variants plain/rag/rand, both formats), and the kernel builder (generated script compiles).
- The synthetic fusion test shows the code recovers a planted signal and that the random-examples control stays at the shortlist level. Those Who&When numbers are meaningless: the synthetic judge scores were generated from labels purely to exercise the code, in a scratch folder.
- NOT tested: the GPU path, 4-bit loading (bitsandbytes), real speed on a T4, a 7B model, and the Kaggle transformers version (`DynamicCache.crop` and `logits_to_keep` were tested with transformers 4.53.3 on CPU; the script falls back if `logits_to_keep` is missing but needs `crop`). Measure first with `--limit 20` before launching a long run.
- No full rr_models run has been done, so there are no Gate 0 numbers yet.

---

## 3. Answers to the "verify before building" list (section 9 of the hand-off)

- Saved probabilities: they were NOT saved by ww_block.py (it only kept them in memory). rr_models.py now saves them (per head seed, mask and split, after the encoder ensemble, before label-shift). Label-shift is applied in rr_common.split_probs for Who&When only.
- Injection plan text: the AEGIS training records have `output.faulty_agents[].injection_strategy` (a short tag such as `prompt_injection`), `ground_truth.injected_agents`, `ground_truth.correct_answer` and `is_injection_successful`. No free-text injection plan was seen in the first record. So RAG examples carry labels, not rationales. Original successful trajectories: not checked (`correct_answer` is only the final answer).
- The 14 mode definitions: `aegis/third_party/AEGIS/evaluation/prompt.txt` (the paper's evaluation prompt). They are copied verbatim into `kaggle_judge.py`. The conversation layout also follows the paper (`Step {step} - {agent}:`).
- Positional scalar in features.py: yes, scalar 4 is the first-appearance rank of the agent (0 to 1).
- Judge model that fits a T4: Qwen2.5-7B-Instruct in 4-bit is the default and is already in the local Hugging Face cache (so a download exists on this machine, but Kaggle needs internet on, which the kernels use). Prefix caching is done by reusing the key/value cache and cropping back to the prefix length (no vLLM needed).

---

## 4. A finding that matters for ALL Who&When numbers (agent-name mismatch)

- 20 of the 184 Who&When true pairs (10.9%) name an agent that never appears under that exact name in the turns. Examples: the label says `Orchestrator` but the turns use `Orchestrator (thought)`, `Orchestrator (-> WebSurfer)`, `Orchestrator (termination condition)`; the label says `Websurfer` but the turns use `WebSurfer` (case).
- Consequence: those records cannot score on agent or pair F1 under exact matching, whatever the model predicts. This affects every Who&When row so far, and the paper's own scorer matches raw names as ours does. Gate 0 reports this as `reachable_share` (about 89%).
- IMPLEMENTED 2026-11-14 as a SEPARATE extra row only (user approved: "lets do it like this only"): rr_common.norm_agent/norm_pairs (one fixed rule on predictions AND labels), rr_models prints/logs `who_when_norm` next to `who_when`, rr_gate0 prints `model_f1_name_normalised` and `reachable_share_name_normalised`, rr_fuse --phase final prints a NAME-NORMALISED block with bootstrap intervals. The raw-name rows stay the headline (comparable with the paper); the normalised rows must always be labelled post-hoc. Earlier results (wbes14 etc.) were not recomputed. Candidate fix to try later and report as a separate, disclosed row: normalise agent names for Who&When scoring (strip a trailing parenthetical like ` (thought)` and compare case-insensitively), applied to both predictions and labels. It uses no model change and no label for fitting, but I found the pattern by looking at the labels, so it must be described as a data-handling normalisation chosen after inspection. Decide with the user before using it in the headline.

---

## 4b. Honest caveats about the design

- The judge's AEGIS-validation subset is a seeded random sample, so the AEGIS frameworks are mixed; the leave-one-framework-out check inside `rr_fuse --phase tune` re-tunes weights and thresholds with one validation framework held out, but the encoders are NOT retrained, so it is weaker than a true leave-one-framework-out.
- Judge scores are z-scored within each split. This uses the unlabeled judge scores of that split (transductive in the same weak sense as label-shift). It must be stated.
- Weights come from a 5-value grid per judge variant, chosen by 2-fold cross-fitting on validation halves split by query; the headline row is chosen by a fixed rule (best cross-fit validation objective among shortlist_model, plain, rag, plain+rag).
- A shortlist can never add a true pair that is outside the top-k. Gate 0 shows how much that costs.

---

## 5. Run order (CPU first, GPU only if the gates pass)

Environment for every command (Windows Git Bash): `cd` to the repo root, `export PYTHONPATH=. PYTHONUTF8=1 USE_TF=0 TRANSFORMERS_NO_TF=1 USE_TORCH=1`, Python is `/c/Python312/python`. One heavy local CPU job at a time. Log every run under `aegis/results/runs/`.

1. **Baseline dump (CPU, about 1.2 hours; ww_block on 14 encoders took about 1.2 hours):**
   - `python -m aegis.src.eval.rr_models --encoders es_s0,es_s1,es_s2,es_s3,es_s4,es_s5,es_s6,es_s7,es_s8,es_s9,es_s10,es_s11,es_s12,es_s13 --seeds 3 --retr 0 --tag rr_base > aegis/results/runs/rr_models_rr_base.log 2>&1`
   - Sanity check: it should reproduce `wbes14` (AEGIS 32.96; Who&When +shift 8.24 / 47.72 / 17.12; same code path and seeds). If it does not, stop and find out why before going on.
2. **Gate 0 (CPU, minutes):**
   - `python -m aegis.src.eval.rr_gate0 --dump aegis/results/rr/dump_rr_base.npz --tag rr_base`
   - Rule (a heuristic bar of ours): recall@5 at least 1.5 times recall@1 on AEGIS validation. Write the printed numbers, including reachable share, the Who&When recall@k and the oracle ceilings, to `results.md`. The Who&When columns are diagnostics only; k must be chosen from AEGIS validation.
3. **Retriever-only variant (CPU, about 1.5 hours plus about 10 minutes of retrieval features per encoder set, not measured):**
   - `python -m aegis.src.eval.rr_models --encoders <same 14> --seeds 3 --retr 1 --tag rr_retr > aegis/results/runs/rr_models_rr_retr.log 2>&1`
   - Compare against rr_base with the same encoders and seeds: Who&When first (names+peer mask with and without shift), then AEGIS test. Run Gate 0 on the new dump as well. Expect a small AEGIS effect and probably nothing on Who&When (kNN scored at the floor on Who&When). If retrieval features do not help on AEGIS validation either, the RAG judge's premise is weak.
4. **Candidates (CPU, minutes):** use the better of the two dumps (by AEGIS validation, not Who&When):
   - `python -m aegis.src.eval.rr_candidates --dump aegis/results/rr/dump_<rr_base or rr_retr>.npz --tag rrc --k 5 --r 2 --retr-enc es_s0 --subsample 300 --splits val,test,who_when`
   - Look at the printed number of weak-match drops. `--min-sim` defaults to 0 (never drop); to use a cutoff, pick it from the AEGIS validation similarity distribution, not from Who&When.
   - Test candidates are only needed for the final phase; leave them out of the first judge run to save GPU time (`--splits val,who_when`).
5. **Speed check on a GPU account (Kaggle T4):** build a kernel with `rr_make_judge_kernel.py ... --limit 20`, run it, read the `[val] n/n records (x min)` line, and extrapolate. Rough guess before measuring (not measured): about 15 to 40 seconds per record for three variants with five candidates and two order passes on a 7B 4-bit model, so 300 validation records plus 184 Who&When records may take several hours. Split over accounts with `--skip/--limit` (outputs get a `_s<skip>` suffix and rr_fuse concatenates them). Use `--minutes` (global budget) so a kernel always ends cleanly.
   - Account usernames and token lines in `kaggle.txt`: line 1 = aryanmahajan665, 3 = aryanmahajan7, 5 = amahaj55, 7 = aryanomgh65. Each needs the candidate dataset (`<user>/aegis-rr-cands`) and already has `<user>/aegis-raw-data`. Token only as env var `KAGGLE_API_TOKEN` per command, never written to any file or note.
   - When a kernel is COMPLETE, pull its output into `aegis/results/kaggle_logs/<short>/` first, then copy `judge_*.jsonl` and `run_info_*.json` into `aegis/results/rr/`.
6. **Tune and pre-commit (CPU, minutes; AEGIS validation only):**
   - `python -m aegis.src.eval.rr_fuse --phase tune --dump-tag <dump> --cand-tag rrc --variants plain,rag,rand --tag rrf1`
   - Read Gate 1 and Gate 2 from its output. Stop here if Gate 1 fails (the plain judge does not beat the shortlist-only model on held-out frameworks).
7. **Final (CPU, once):** `python -m aegis.src.eval.rr_fuse --phase final --dump-tag <dump> --cand-tag rrc --variants plain,rag,rand --tag rrf1`. This needs test and Who&When judge scores. It prints every row with bootstrap intervals and paired differences against the model alone, and it logs rows to the ledger. Report all rows, including those that do not help.

---

## 6. Rules to keep (from the hand-off, section 10)

- No commits or pushes unless the user says so. Do not use Who&When data or labels for training, tuning, retrieval or examples. Do not tune thresholds, weights, k or the min-sim cutoff on Who&When. Score Who&When once with the pre-committed configuration.
- Work on copies; append new run names to the ledger; keep raw logs.
- State clearly which numbers are transductive (label-shift, z-scoring) and which choices were post hoc (the name normalisation, if used).
- Citations in the hand-off were written from memory or search summaries; verify before the thesis.
