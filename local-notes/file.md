# What we have done so far — the full story in simple language

Written 2026-11-12. This file is local only. It is never committed or pushed (it sits in `local-notes/`, which is excluded
through `.git/info/exclude`). The companion file `context.md` (same folder) is the technical reference; this file is the story.

All scores are percentages. "F1" is a score from 0 to 100 where higher is better. "micro" means all predictions are pooled
together, so common classes dominate. "macro" means every class counts equally.

---

## Part 1. What is this project about?

This is the BTech thesis project (BTP-I, Group 121, NSUT) of Aryan Mahajan (with Yogesh Kumar and Shubham Yadav named in the plan).

A **multi-agent LLM system** is a team of language-model "agents" that talk to each other to solve a task, for example a maths
problem, a coding problem or a web search. Sometimes the team fails because one or more agents made a mistake.

Our task is called **failure attribution**:

> Look at the whole conversation of a failed run. Say **which agents were faulty** and **what kind of mistake each one made**.

There are 14 kinds of mistake, named FM-1.1 to FM-3.3. Examples: FM-1.1 means an agent deviated from the task specification;
FM-2.6 means an agent reasoned inconsistently; FM-3.2 means verification steps were removed. A prediction is a set of pairs
`(agent name, error mode)`.

---

## Part 2. How the project started, and why it changed

### 2.1 The first version (the "old project")
At first we built our own data. We built small pipelines of agents (a "retriever", "reasoner" and "writer" in different shapes),
ran them on HotpotQA questions with Qwen2.5-7B, and **deliberately injected one fault** into one agent at a time (three fault types:
noise, contamination, ceiling). Because we injected the faults ourselves, we knew the right answer for certain. We measured how much
each agent's repeated answers disagreed ("uncertainty") and trained a graph neural network (GNN) to say which agent was the faulty one.

What we built there:
- 20 pipeline shapes (14 for training, 6 held out for out-of-distribution testing).
- About 1,762 trials with 5 samples per agent (the "k=5" dataset) and later a 10-sample ("k=10", about 1,785 trials) version.
- Generation ran on free Kaggle and Lightning AI T4 GPUs with vLLM and a 4-bit Qwen2.5-7B-Instruct-AWQ model.
- We found and fixed three real bugs in the fault injection by actually reading generated text, and regenerated the whole dataset.
- Best old result: about 0.395 top-1 accuracy on the held-out shapes, against 0.145 for "pick the most uncertain agent" and 0.249
  for a non-graph random forest. That number was chosen using the held-out set, so it is optimistic.

This old work is still in `btp-pipeline/`, `model/`, `dataset/` and `docs/`. We did not touch it afterwards.

### 2.2 The change
The requirement changed: we must use a **dataset from a published paper** and compare against **published papers only**. So on
1 November 2026 we switched to **AEGIS-Bench** (Kong et al., ICLR 2026, arXiv 2509.14295). `new-plan.md` (written that day) is the plan.

Important: your guide has **not yet approved** this switch. Everything from the graph model onward is therefore called "exploratory".

---

## Part 3. What AEGIS-Bench is

- 9,533 conversations ("trajectories") from six multi-agent frameworks: agentverse, dylan, llm_debate, macnet, magentic_one, smoagents.
- Built by taking successful runs and injecting errors into them, keeping runs that then fail. So labels are known by construction.
- Each record has a question (query), a list of turns (who said what), and the label: a set of faulty `(agent, mode)` pairs.
- Released split: 7,146 train, 1,787 validation, 600 test. Plus a second benchmark, **Who&When** (a different, human-labelled set;
  184 records in the processed file we have, and the paper (v6) says Who&When has 184 annotated errors, so that matches; an earlier "127" note was not supported).
- **No agent-to-agent connections are stored.** If we want a graph, we have to invent one.

---

## Part 4. Step by step: what we did on AEGIS-Bench

### Phase 0 — Setup (1 Nov)
Created branch `aegis_v1` from the old branch, a new folder `aegis/` (the old project was left alone), the authors' code cloned
to `aegis/third_party/AEGIS/`, the Hugging Face data downloaded to `aegis/data/`. Wrote helper code for seeds, config hashes and a
**ledger** (`aegis/results/metrics_history.jsonl`) where every run adds one line.

### Phase 1 — Data audit (1 Nov)
We checked everything before trusting it. The findings that matter most:
1. **Question leakage.** 588 of 600 test questions also appear in the training data (and 1,753 of 1,787 validation questions).
   Each successful run has several faulty copies, and the released split did not keep copies of the same task together. A model
   can therefore memorise questions and look better than it is.
2. **Fields that give the answer away.** `num_injected_agents`, `num_agents`, `ground_truth.*`, `injection_strategy` and a per-turn
   flag `is_injected` (present only for the macnet framework). We never load them. The loader reads only a short whitelist of fields.
3. **Two different test files.** The repo's `evaluation/data/aegis.jsonl` and the Hugging Face test file have 600 records each and
   no ids in common. We cannot tell which one the paper used, so we use the Hugging Face one.
4. **No DCL code.** The paper's small model (DCL) has no released code.

### Phase 2 — Scoring code (1–2 Nov)
We ported the authors' scoring script exactly (pair level, agent level, error level, micro and macro F1) and checked it line by line
twice. Only difference: we count an unreadable model answer as an empty prediction; the authors skip it. 16 unit tests pass.
A random baseline we fitted by hand came out close to the paper's random row (not identical, because the paper's sampler is unknown).

### Phase 3a — Reproducing the zero-shot Qwen row (2 Nov)
We ran Qwen2.5-7B-Instruct (4-bit) with the authors' prompt on a free Kaggle T4 GPU: 600 test prompts plus 184 Who&When prompts,
about 4.5 hours. Result: close to the paper on most rows, but **not** on agent-level micro-F1 (ours 49.2, paper 27.6). We do not know
why. Candidate reasons (none tested): a different test file, a different prompt, sampling versus greedy decoding, the 4-bit weights.
Three earlier attempts failed on library version conflicts; we kept their logs.

### Phase 3b — Our version of the paper's small model, "DCL-style" (2 Nov)
Since no code exists, we **wrote it from scratch** from the paper's description. It is named DCL-style because it is not the real DCL:
its text encoder is frozen (the paper fine-tunes it), it has about 0.2M trainable parameters (the paper: 23–35M), and its contrastive
loss is our own simplification. It scored higher than the published DCL on 10 of 12 micro-F1 numbers, but that does not mean it is
better; it is a different model. On our own validation data it is worse than a plain neural network.

### Embeddings (2 Nov)
All 99,406 distinct turn texts and 1,502 queries were turned into 384-number vectors with the small encoder all-MiniLM-L6-v2 and cached.
A local run was killed for low memory, so we did the big job on a Kaggle GPU (275 seconds).

### Phase 4 — Simple baselines (2–3 Nov)
- **The trivial floor**: predict every agent that appears, with all 14 error modes. Because about 54–56% of agents are faulty, this
  already scores about 70 on agent-level micro-F1 and about 28 on error-level micro-F1. **Several published rows are below this
  floor**, so beating them proves little. Every table of ours includes the floor.
- **Frequency baseline**: always predict the most common pairs for that framework, ignoring text.
- **Flat MLP**: a small neural net on each agent's averaged turn vectors (no graph).
- Result on validation: the flat MLP beats the frequency baseline at pair and error level. Removing the question leakage lowers the
  score a lot (pair micro-F1 32.2 on the released split versus 23.1 on our grouped split), which shows how much leakage inflates.

### Error analysis (3 Nov)
Counting over all training labels and reading 18 conversations (we planned 30): every framework except macnet gives one mode to each
faulty agent; some roles are never faulty (scaffold roles such as `User`); the number of faulty agents is almost fixed per framework;
modes are balanced; some faults are visible against the question or peers and some are invisible in the text (which limits any text model).

### Phase 4b — "Top-k" decoding idea (3 Nov)
Idea: since the number of faulty agents is nearly fixed, take the top-k agents. Result: **no overall gain** (small gain in pair, big
loss in error). We kept the threshold rule.

### Phase 4c — Relational probe (3 Nov)
Added four features: how similar each agent's text is to the question and to the other agents. Small consistent gain on the grouped
split (+0.6 pair micro-F1), inconclusive on the released split.

### Phase 5 — Turn graph model (GNN) (3 Nov, exploratory)
Each turn is a node (its frozen vector), plus one extra node for the question. Four kinds of edges: consecutive turns, consecutive turns
of the same agent, question to turn, turn to question. Two message-passing layers, then average the turns of each agent, add simple
features, and predict 14 error modes plus "is this agent faulty".
- We also built a **0-layer control** (same model with message passing switched off) to see whether the graph or just the extra layers helped.
- Result on grouped validation (5 seeds): flat MLP 23.1, 0-layer control 25.7, full GNN 26.9 (pair micro-F1).
- Removing the question node removed the whole gain over the control. Removing the other edge types changed almost nothing.
  **So the useful part was "each turn can see the question", not the conversation structure.**

### Phase 6 — Final held-out scoring (first attempt, 3 Nov) — incomplete
Plan: train on a split's train set, choose settings on validation, score the 600-record test set and Who&When **once**.
The run was killed by the operating system for low memory (only the grouped protocol was partly done). And the Who&When numbers
for the flat MLP and the 0-layer control were identical with zero spread across seeds, which should be impossible.

---

## Part 5. What we did in this working session (12 Nov 2026)

### 5.1 Finding and fixing the Who&When bug
We standardise inputs with `(x − mean) / (std + 1e-6)`. The "framework = who_when" column never switches on in the training data, so
its std is 0 and the divisor became 0.000001. On Who&When that column is 1, so it turned into **1,000,000**. The network's outputs
saturated and every model returned the same single guess. We confirmed this by measuring: the maximum standardised value was 1,000,000.
Fix (in `flat_mlp.py` and `turn_gnn.py`): if a column's std is below 0.001, divide by 1 instead. Test and validation numbers are not
affected, because those columns are zero there.

### 5.2 Re-scoring both protocols
We added a `--protocol` option so one protocol can run at a time (the first run died from memory), and tagged the new ledger rows as
phase `6r`. Old rows are kept. Results (test, 5 seeds):

Grouped protocol (leakage removed), micro-F1 pair / agent / error:
- floor 7.4 / 68.6 / 27.1
- frequency 12.1 / 80.5 / 28.5
- flat MLP 19.8 / 78.7 / 33.0
- 0-layer control 20.9 / 79.4 / 34.7
- 2-layer GNN 21.9 / 78.6 / 35.3

Released-split protocol (leaky): flat MLP 33.1, control 31.4, GNN 36.2 pair micro-F1. Seed spread is about 2.4–2.9, so the GNN lead
there is about one standard deviation and is not established. Who&When stayed at or below the trivial floor for every model.
This was committed and pushed to the branch `aegis-whenwhen-fix-final-results` (before you asked us to stop pushing).

### 5.3 The biggest improvement: fine-tuning the text encoder
All earlier models used a **frozen** encoder (the vectors never change). We wrote a Kaggle job that trains the encoder **together
with** a small prediction head, using only the grouped training data and early stopping on grouped validation loss. Then it re-encodes
every turn and question with the improved encoder.
- Two smoke tests first: the first failed because of a mistake in our wrapper (an import placed in the wrong spot — no GPU time lost),
  the second passed.
- Full run: 5 epochs ran, best at epoch 3 (validation loss 0.1595 → 0.1387), about 148 seconds per epoch, 100,863 texts exported.
- Then the same three downstream models were re-run on the new vectors.

Grouped **validation**, 3 seeds, pair micro-F1: flat MLP 22.9 → 36.9; control 25.7 → 33.4; GNN 26.9 → 35.6.
Grouped **test**, 5 seeds (scored once): flat MLP 19.8 → **29.1**, control 20.9 → **29.7**, GNN 21.9 → **28.5**; error micro-F1
33.0 → 38.8, 34.7 → 39.4, 35.3 → 40.3; agent micro-F1 for the GNN rose to 82.4.
Two honest points: (1) the encoder's early stopping used the same validation set, so validation numbers are slightly optimistic, while
test is clean; (2) once the encoder is fine-tuned, the flat MLP is as good as the GNN, so the graph is not shown to help. The previous
"graph gain" came largely from a weak frozen encoder. This was committed and pushed to branch `aegis-finetuned-encoder` (again before
you asked us to stop pushing). Some later commits exist locally only.

### 5.4 Trying a larger encoder (in progress)
- `all-mpnet-base-v2` (110M parameters, 768-number vectors) first ran out of GPU memory at 4 conversations per batch (v1).
- We added gradient checkpointing and batch size 2 (v2). It failed immediately because the MPNet model in Hugging Face does not support
  gradient checkpointing. No GPU time was used. Logs of both are saved.
- We switched to `BAAI/bge-base-en-v1.5` (same size, BERT-based, supports checkpointing). Kernel `aryanmahajan7/aegis-ft-bge` was launched
  and is **running now**. Its log cannot be read until it finishes. We also made the downstream code accept any vector width (not just 384).

### 5.5 Kaggle GPU access
You gave a Kaggle API token. We pass it only through an environment variable for each command. It is not written to any file, and it
is not in these notes. A tiny Kaggle dataset `aegis-grouped-ids` (the grouped split's record ids) was created; the raw data was already
on Kaggle in `aegis-raw-data`.

### 5.6 The 5-minute loop
At your request a recurring job (every 5 minutes, session-only, expires after 7 days) re-runs the "continue improving" instruction.
Each round I report progress, say what I need, and say whether a Kaggle GPU is needed.

---

## Part 6. How our work differs from the paper's usual practice
1. Grouped split in addition to the released split (the released one leaks questions).
2. The trivial floor appears in every table.
3. We use the Hugging Face test file because the paper's file is unclear.
4. Unreadable outputs count as empty (the authors skip them).
5. Zero-shot used 4-bit Qwen with greedy decoding and a 16k-token window with truncation of the longest prompts (25 of 600).
6. We never use answer-leaking fields.
7. Our DCL-style model is our own reconstruction; the flat MLP, turn graph and fine-tuned runs are our own models.
8. Our decoding uses two thresholds tuned on validation, with at least one pair per conversation.
9. One T4 GPU and a CPU, against four RTX 4090s in the paper.
10. Who&When has 184 records here (paper v6: 184 annotated errors; no evaluated-set size is given).
Fine-tuning the encoder end to end is actually **closer** to the paper's DCL practice than our frozen baselines were.

---

## Part 7. Problems we hit and how we handled them
- Kaggle library version conflicts (torch/torchaudio): uninstall only torchaudio.
- Local embedding job killed for low memory: moved it to a Kaggle GPU and made it write files in pieces.
- A wrong count in a written verdict ("11 of 12") was corrected to "10 of 12".
- Final scoring killed for low memory: re-run one protocol at a time.
- Who&When identical scores: the standardisation bug above.
- Our own wrapper bug put a line before a `from __future__` import: fixed.
- GPU out of memory with the 110M-parameter encoder: gradient checkpointing and smaller batches.
- The Kaggle log download failed once with a Windows text-encoding error: set `PYTHONUTF8=1`.
- pytest fails on Python 3.12 (a typing error unrelated to our code), so tests run with Python 3.14.
- Same-seed GNN runs are not exactly repeatable (probably summation order in the edge aggregation), so differences below about 1 point are unresolved.

---

## Part 8. What is still open
1. Your guide's decision on the dataset switch.
2. Whether the larger encoder (bge-base) beats MiniLM. If yes, score test once.
3. Fine-tuning with the graph head end to end, and more seeds with a lower learning rate.
4. Why the paper's agent-level scores sit below the trivial floor, and why our zero-shot agent micro-F1 is 49.2.
5. Writing everything up with the final table next to the published rows.
6. The branch `aegis-finetuned-encoder` on the remote does not include the last local commits. By your instruction nothing is committed
   or pushed from now on until you say so.

---

## Update log (newest last)

### 2026-11-12, after the notes were first written
- **Are we beating the paper?** Numerically yes on AEGIS-Bench, with caveats. Our best so far (fine-tuned MiniLM, flat MLP, grouped test, 5 seeds):
  pair micro/macro 29.1/17.8, agent 79.1/59.7, error 38.8/38.0. Paper's small model DCL: 8.3/5.3, 22.9/20.2, 24.7/27.7. Paper's 14B fine-tuned
  Aegis-SFT: 16.6/10.0, 76.5/48.0, 27.5/27.7. Our average over the paper's 12 numbers is about 29.4 (DCL 12.6, Aegis-SFT 26.5).
- **But be careful:** the trivial "predict everything" rule averages about 25.6 on the same 12 numbers, so the headline average is mostly the
  agent-level floor. The real gains are at pair level (29 versus a floor of 7) and error level (39 versus a floor of 27). On Who&When we are
  at or below the floor, and the paper's own agent/error rows sit below the floor (unexplained). So the safe claim is about pair-level and
  error-level scores on the grouped split, always shown next to the floor.
- The session was renamed `comeback_main`.
- The larger-encoder run (`aegis-ft-bge`, bge-base-en-v1.5) is still running on the Kaggle GPU (about 75+ minutes in). Nothing new to report yet.
- Standing rule from you: no commits or pushes until you say so; these two files are never committed. You asked that I keep both files
  updated as we go. This update log is where new progress gets added.

### 2026-11-12, loop plan
- You asked the 5-minute loop to run on a plan and to **stop after step 2**: wait for the bge-base fine-tune, compare it with MiniLM on validation
  (`ft_eval`), report, then stop the loop. No test scoring and no further experiment unless you ask. The loop now also updates these notes every round.

### 2026-11-12, notes are now mandatory in the loop
- You asked that both notes files be updated **every single loop round, always**. The loop prompt now says so with no exceptions: even if nothing changed,
  a one-line status entry is added to both files each round.

### 2026-11-12, Kaggle GPU status check (you asked)
- `aegis-ft-bge` (larger encoder): still RUNNING, about 90 minutes after launch (my estimate was about 1 hour; a base-size encoder with gradient checkpointing and batch size 2 is slower than I assumed).
- `aegis-ft-full` (MiniLM fine-tune): COMPLETE, output saved. `aegis-ft-mpnet`: ERROR (the two failed attempts; logs saved).

### 2026-11-12, loop tick
- Kernel `aegis-ft-bge` still RUNNING. Nothing else changed. You asked whether extra Kaggle accounts would help: yes for repeating the fine-tune with different seeds and training settings, but not needed for the current step.

### 2026-11-12, loop tick
- Kernel `aegis-ft-bge` still RUNNING. No other change.

### 2026-11-12, loop tick
- Kernel `aegis-ft-bge` still RUNNING. No other change.

### 2026-11-12, loop tick
- Kernel `aegis-ft-bge` still RUNNING (about 2 hours). No other change.

### 2026-11-12, loop tick
- Kernel `aegis-ft-bge` still RUNNING (about 2 h 10 min). No other change.

### 2026-11-12, loop tick
- Kernel `aegis-ft-bge` still RUNNING (about 2 h 15 min). No other change.

### 2026-11-12, loop tick
- Kernel `aegis-ft-bge` still RUNNING (about 2 h 20 min). No other change.

### 2026-11-12, loop tick
- Kernel `aegis-ft-bge` still RUNNING (about 2 h 25 min). No other change.

### 2026-11-12, bge-base fine-tune finished
- `aegis-ft-bge` COMPLETE after about 2.5 hours (about 760 seconds per epoch). Log and vectors are saved in `aegis/results/kaggle_logs/ft_bge/`. 4 epochs ran; best was epoch 2.
- Validation loss of the training head: bge best 0.1395 (epoch 2), against 0.1387 for the MiniLM fine-tune (epoch 3). So on this measure the bigger encoder is **not** better.
- Now running the real comparison (flat MLP, 0-layer control, 2-layer GNN on the bge vectors, 3 seeds, grouped validation) on CPU. Results next round.

### 2026-11-12, loop tick
- bge comparison running: 2 of 3 seeds done (about 100 s each). Result table next round, then the loop stops.

### 2026-11-12, bge-base versus MiniLM result (validation only) — loop stops here
Grouped validation, 3 seeds, micro-F1 pair / agent / error (the fine-tuned encoders only):

| Model | MiniLM fine-tuned | bge-base fine-tuned |
|---|---|---|
| flat MLP | 36.93 / 80.22 / 44.29 | 34.89 / 83.57 / 43.77 |
| 0-layer control | 33.40 / 85.33 / 43.02 | 35.40 / 83.24 / 43.51 |
| 2-layer GNN | 35.60 / 86.08 / 44.82 | 33.02 / 87.58 / 43.42 |

- **Verdict: the bigger encoder is not clearly better.** Pair-level scores are lower for the flat MLP and the GNN and higher for the control; all differences are within the seed noise (pair std 2.8, 3.4 and 0.6).
  Only agent-level for the GNN is a bit higher (87.6 versus 86.1). It also took 5 times longer to train (about 760 seconds per epoch against 148). So MiniLM stays our best and cheapest encoder.
- Test was **not** scored for bge, as instructed. The loop is stopped as you asked. Nothing is committed or pushed.
- Ideas left, only if you ask: repeat the MiniLM fine-tune with other seeds and settings (could use your extra Kaggle accounts), fine-tune with the graph head end to end, and a cleaner early-stopping slice.

- Paper review (2026-11-12): we read the whole paper and wrote the findings in `local-notes/paper_review.md`. The paper has a few inconsistencies, and it never compares against the trivial "predict everything" floor. Fine-tuning the small encoder end to end is what the paper itself does, so our gain replicates its practice. We also fixed the old "paper says 127 Who&When" note (the paper says 184). No code changed; nothing committed.

- Code change (2026-11-12): `aegis/src/eval/final_test.py` now prints `avg12` (the paper Table 1 "Avg." = mean of the 12 F1 numbers) for floor, frequency and each model summary. Checked: paper DCL row gives 12.61. Ledger rows unchanged; no GPU needed; uncommitted.

### 2026-11-12, fixing gaps in our own work: step 1, GNN reproducibility
- Tested why same-seed GNN runs differed. On a 400-train-trajectory subset, 3 epochs, 4 CPU threads: max abs difference between same-seed runs was 1.8e-7; with 1 thread it was exactly 0.0. Cause: multi-threaded float summation order in `index_add_`.
- Fix: `train_gnn` in `aegis/src/gnn/turn_gnn.py` now runs on one thread by default (`deterministic=True`) and restores the thread count afterwards. Verified bit-identical on the subset. Uncommitted. Older GNN numbers were produced without this, so differences under about 1 point in them remain unresolved; new GNN runs are repeatable (slower, one thread).
- Remaining own gaps to fix, in order: (2) clean early-stopping slice for the encoder fine-tune, (3) end-to-end graph-head fine-tune, (4) more seeds, (5) error analysis from 18 to 30 trajectories. Who&When stays out of scope.

### 2026-11-12, own-gap fixes: steps 2 and 5
- Error analysis completed to 30 trajectories (12 more read, 2 per framework, seed 7); addendum written at the end of `aegis/docs/error_analysis.md`.
  Main findings: injected faults often leave tell-tale phrases ("incorrect formula", "# Incorrect: Should return False"), so part of our score may be recognising the injector's style (also a likely reason all models fail on Who&When);
  mode labels often do not match the visible symptom, which fits agent/error micro-F1 being far above pair; some faults show nothing in the text; symptoms can sit on a different agent than the label (fault propagation).
- Kaggle fine-tune script `aegis/scripts/kaggle_finetune_minilm.py` now has `--es-holdout <fraction>`: holds out a query-grouped slice of the grouped TRAIN set for the encoder's early stopping, so grouped val is never seen by the encoder
  and downstream validation numbers become clean. Default 0 keeps old behaviour. Syntax checked only; not run yet. No Kaggle job launched.

### 2026-11-12, Kaggle launch: clean early-stopping fine-tune (own-gap fix 2)
- Token given by you in chat this turn; used only as env var `KAGGLE_API_TOKEN`, not written anywhere.
- Launched two kernels (MiniLM, `--es-holdout 0.15`, epochs 6, patience 2, bs 8, seeds 0 and 1): `aryanmahajan7/aegis-ft-es-s0`, `aegis-ft-es-s1`, export dirs `ft_shards_es_s0/s1`. No smoke test first (the script change is small); if they error, the log shows it.
- Logs are not readable while RUNNING. On COMPLETE: pull log+output into `aegis/results/kaggle_logs/ft_es_s0/` and `ft_es_s1/` first, then run `ft_eval` on grouped validation. Do not stop or delete running kernels.


---

## Session 2026-11-12 (evening): learning notes and the status of the gap-fixing work

### A. Corrections to older notes
- Who&When size: the paper (v6) says 184 annotated errors, which matches our processed file (184). The old "127" note was wrong.
- The bge-base fine-tune is FINISHED (not running): not clearly better than MiniLM on validation, and 5x slower. MiniLM stays our best encoder.
- Loop: the old 5-minute loop ended. A NEW loop (id a9794407, every 5 minutes, session-only, expires after 7 days) now monitors the two new Kaggle kernels and updates these notes every tick.

### B. What the dataset is (explained to the user in plain words)
- One entry = one failed multi-agent run. Fields: `id`; `metadata` (framework, benchmark, model, num_agents, num_injected_agents, task_type); `input` (query, conversation_history = list of turns {step, agent_name, content, phase, sometimes agent_role or is_injected}, final_output); `output.faulty_agents` (the label: list of {agent_name, error_type, injection_strategy}); `ground_truth` (correct_answer, injected_agents, is_injection_successful).
- Never used as inputs because they leak the label: num_agents, num_injected_agents, ground_truth.*, injection_strategy, per-turn is_injected (macnet only, 17,496 train turns). Allowed but not loaded: phase, agent_role.
- Counted on the train file (7,146 records, 84,758 turns): final_output is missing for all 1,198 dylan records; is_injection_successful is True for every train record; phase values are llm_debate "discussion", macnet/dylan/smoagents/magentic_one "reasoning", agentverse "initialization"/"reasoning"/"evaluation".
- Made by taking successful runs and injecting faults (14 MAST modes). Two injection strategies: `prompt_injection` (an LLM rewrites the agent's input; about 7.3k train labels) and `response_corruption` (an LLM rewrites the agent's output; about 8.4k), plus a few plain `prompt` labels (about 200, not explained in the code read).
- Agent LLM: gpt-4o-mini for 9,260 of 9,533 records (the authors' config names gpt-4o-mini-2024-07-18); magentic_one records say `unknown-model` (273). The authors' magentic script also uses gpt-4o-mini-2024-07-18, so probably the same (inference). The injection rewriter's model was not found (the configs also list gpt-4o, gemini, llama).
- Injection prompts live in `aegis/third_party/AEGIS/aegis_core/malicious_factory/fm_malicious_system.py` (some prompts are tailored for math, programming, multiple choice, science).
- Per-framework structure (train): llm_debate always 12 turns (3 assistants, each: question, answer, others' opinions, revision, with a helper `User`); agentverse 4-27 turns, median 12 (RoleAssigner, Solver, Critic, Evaluator loop); macnet 10-16, usually 10 (nodes Node-1..Node4, Node-1/Node-2 never faulty); dylan 3-18, median 10 (layered debate, 4 assistants, some records with named roles); magentic_one 3-44, median 11 (Orchestrator + WebSurfer/Coder/ComputerTerminal/FileSurfer, exactly 1 faulty); smoagents 7-45, median 16 (Manager + Searcher, JSON-like content, exactly 1 faulty).
- No agent-to-agent edges are stored; our graph is our own guess (consecutive turns, same-agent turns, query to every turn).
- Who&When record (checked): framework/benchmark "who_when", free-form agent names (for example Tickets_Pricing_Expert, "Orchestrator (-> WebSurfer)", human), mostly 5-8 turns, label produced by mapping human notes to the 14 modes with Gemini-2.5-Flash.
- Data files: `Aegis-Training/train.jsonl` (7,146) and `val.jsonl` (1,787) are for training and tuning; `Aegis-Bench/test.jsonl` (600) is the final in-domain test ("AEGIS-Bench" means only this file); `whowhen.jsonl` is the out-of-domain test. Train only on train; tune only on val; test sets are scored once.

### C. What the paper did (explained to the user)
- Built the dataset; defined the task and 3 scoring levels (pair, agent, error), each with micro and macro F1; two benchmarks (AEGIS-Bench, Who&When); headline "Avg" = mean of the 12 numbers (checked: the paper DCL row sums to 151.30, /12 = 12.61).
- Approaches evaluated: Random 4.08; zero-shot LLMs Qwen2.5-7B 12.43, Qwen2.5-14B 13.99, o3 20.24; Aegis-SFT (14B fine-tuned, verl script `run_aegis_sft.sh`) 26.51 best; Aegis-GRPO (RL, `run_aegis_grpo.sh`) 18.41; DCL (small model, 23-35M parameters, MiniLM encoder trained end to end, bag of turns, MIL attention on top-3 turns, two prototype banks, losses: classification + supervised contrastive + hierarchy consistency) 12.61. Extra studies: DCL ablations (removing the consistency loss: 12.61 to 9.52), SFT epochs (Who&When peaks at 2 epochs), 3 seeds for DCL.
- Metric explanations given: micro-F1 pools all guesses (common classes dominate); macro-F1 averages per-class F1 (rare classes count equally). In this project agent-level classes are raw agent-name strings (many rare names), so agent macro is far below agent micro; macro is over the union of classes in true or predicted sets.

### D. Gaps: what we handled, what is still open (summary given to the user)
- Handled in our own evaluation: grouped split (leakage), trivial floor in every table, our own DCL-style model, leak-proof loader, graph invented by us.
- Still open from the paper: agent-level scores below the floor (unexplained), our zero-shot agent micro 49.2 vs paper 27.6, two different test files, count mismatches, DCL details unspecified, paper inconsistencies (4090 vs 3090, BCE vs ASL).
- Still open in our own work: Who&When at or below the floor (out of scope), graph gain unproven after fine-tuning, validation optimism of encoder early stopping, seed noise, guide approval.

### E. Own-gap fixes done this session
1. GNN reproducibility: same-seed runs differed because of multi-threaded float sum order (max abs diff 1.8e-7 with 4 threads after 3 epochs, 0.0 with 1 thread). Fixed: `train_gnn` now defaults to one thread (`deterministic=True`, wrapper around `_train_gnn`) in `aegis/src/gnn/turn_gnn.py`. Verified identical on a 400-trajectory subset. Older GNN numbers remain unresolved below about 1 point. Uncommitted.
2. Error analysis finished (30 trajectories): addendum in `aegis/docs/error_analysis.md`: tell-tale injector phrases (so part of the score may be injector-style recognition, likely also why Who&When fails), mode labels often do not match the visible symptom, some faults invisible, symptoms can sit on a different agent than the labelled one (fault propagation). Qualitative only (first 9 turns, 130 characters).
3. `aegis/scripts/kaggle_finetune_minilm.py` got `--es-holdout <fraction>`: holds out a query-grouped slice of the grouped TRAIN set for the encoder's early stopping so grouped val stays untouched; `run_info.json` records the `es` size. Uncommitted.

### F. GPU run in progress (own-gap fix: clean early stopping)
- Kernels `aryanmahajan7/aegis-ft-es-s0` and `aegis-ft-es-s1` (MiniLM, `--es-holdout 0.15`, epochs 6, patience 2, bs 8, encoder seeds 0 and 1, export dirs `ft_shards_es_s0` and `ft_shards_es_s1`), launched about 17:30, both RUNNING at first check. Build dirs are in the session scratchpad. The Kaggle token was given by the user in chat and is used only as an env var.
- On completion: pull log+output into `aegis/results/kaggle_logs/ft_es_s0/` and `ft_es_s1/` first, then `ft_eval --ft-dir ... --skip-frozen --seeds 3` on grouped validation; compare with the earlier 36.93 / 33.40 / 35.60 (pair micro, flat / control / GNN, early-stopped on val). The new numbers should be slightly LOWER and honest.
- Still to build after that: end-to-end graph-head fine-tune (GNN inside the Kaggle script, needs new code) and more seeds for final test scoring.
- Rules in force: no commits or pushes until the user says; never delete a RUNNING kernel before its output is saved; keep all logs; local notes are never committed.


### 2026-11-12 17:3x, loop created
- Loop a9794407 created (every 5 min). es-s0 and es-s1 RUNNING.

### 2026-11-12, loop tick
- aegis-ft-es-s0 and aegis-ft-es-s1 still RUNNING. No other change.

### 2026-11-12, Who&When work started (own-gap fix: Who&When at/below floor)
- Facts found: Who&When has EXACTLY 1 faulty agent per record (184/184), about 3.7 agents per record, 27% of agents faulty, 20 labels name an agent absent from the turns (max agent recall about 89%), mode mix skewed (FM-2.6 26%, FM-3.3 21%, FM-2.2 12.5%, FM-1.1 11%) unlike AEGIS (balanced); names are free-form (WebSurfer 31 faulty, Verification_Expert 18, Orchestrator 18...).
- New script `aegis/src/eval/whowhen_study.py` (zero-shot: no Who&When label is used for training or tuning; ledger phase "7"). Flat MLP on grouped train, fine-tuned MiniLM vectors (`ft_full`), 5 seeds. Log: `aegis/results/runs/whowhen_study_ft_full.log`.
- Result 1 (micro pair / agent / error, Who&When, floor 3.42 / 38.32 / 13.33): base (names + framework one-hot) thresholds 3.71 / 40.18 / 10.20; **name-free (zero out framework + agent-name one-hot) 5.80±0.60 / 48.63±1.45 / 13.28±0.96**: above the floor at pair and agent, at the floor on error. Cost on AEGIS grouped test: pair 29.05 -> 25.93, error 38.80 -> 38.08, agent unchanged (79.1).
- Result 2: top-1 decoding (one pair per record, from the one-faulty-agent prior) is WORSE than thresholds (pair 2.50, agent 36.85 for base; 3.59 / 39.89 name-free). Dropped. Negative result kept.
- Compare paper Who&When rows: DCL 1.60 / 8.40 / 14.67; Aegis-SFT 4.03 / 51.14 / 9.87; o3 7.41 / 53.10 / 14.88. Name-free model: pair above SFT, below o3; agent below SFT and o3; error between SFT and DCL/o3.
- Running now: variant `nameemb` (name-free plus MiniLM embedding of the agent-name text), log `whowhen_study_ft_full_v2.log`.
- Result 3 (`whowhen_study_ft_full_v2.log`, needed `USE_TF=0 TRANSFORMERS_NO_TF=1 USE_TORCH=1` because local TensorFlow/protobuf is broken): variant `nameemb` (name-free one-hots + 384-d MiniLM embedding of the agent-name text): AEGIS grouped test pair **31.31±1.99** / agent 80.02 / error 39.80 (macro 19.17 / 62.11 / 38.74) = best in-domain so far (base 29.05 / 79.08 / 38.80); Who&When 4.87±0.82 / 41.73 / 13.75 = above the floor on all three (3.42 / 38.32 / 13.33) but a smaller Who&When gain than name-free.
- Trade-off table (micro pair / agent / error): base test 29.05 / 79.08 / 38.80, W&W 3.71 / 40.18 / 10.20; namefree test 25.93 / 79.09 / 38.08, W&W 5.80 / 48.63 / 13.28; nameemb test 31.31 / 80.02 / 39.80, W&W 4.87 / 41.73 / 13.75. Honest caveat: the variants were chosen after seeing Who&When results (selection on the evaluation set, mild); seed std on test pair is about 2, so the nameemb vs base gain (+2.3) is about one std.
- Not yet done for Who&When: combine nameemb + name-free regimes, per-record error analysis of failures, fine-tuning the encoder with name-free inputs, using the clean-early-stopping encoders (es-s0/s1) once they finish, scoring these variants with 5 seeds on the paper protocol. Ledger rows: phase "7".

### 2026-11-12, loop tick
- es-s0/es-s1 still RUNNING (checked after Who&When study). No other change.

### 2026-11-12, clean early-stopping fine-tune FINISHED (es-s0, es-s1)
- Both kernels COMPLETE; logs + outputs saved FIRST in `aegis/results/kaggle_logs/ft_es_s0/` and `ft_es_s1/` (log, ft_log.json, run_info.json, ft_shards_es_s*; 164 MB each).
- Setup: MiniLM, `--es-holdout 0.15`: 3,234 train / 631 early-stop (115 held-out queries) / 940 val (never seen by the encoder). 6 epochs ran (patience 2 hit its limit only at the end); best early-stop loss 0.1414 (s0, epoch 4) and 0.1433 (s1, epoch 4). Epoch losses s0 (train/es): .1917/.1625, .1549/.1486, .1401/.1448, .1287/.1414, .1179/.1434, .1076/.1480. Overfits after epoch 4.
- Running now: `ft_eval --ft-dir ... --seeds 3 --skip-frozen` on grouped validation for each encoder seed (logs `aegis/results/runs/ft_eval_es_s0.log`, `_s1.log`). Compare against the earlier (val-early-stopped, slightly optimistic) 36.93 / 33.40 / 35.60 pair micro (flat / control / GNN).

### 2026-11-12, clean early-stopping result (grouped validation, encoder never saw val; 3 downstream seeds per encoder; micro pair / agent / error)
| Model | enc seed 0 | enc seed 1 | mean of the two | old val-early-stopped encoder (optimistic) |
|---|---|---|---|---|
| flat MLP | 32.70 / 85.29 / 42.89 | 30.62 / 87.09 / 42.09 | 31.66 / 86.19 / 42.49 | 36.93 / 80.22 / 44.29 |
| 0-layer control | 32.32 / 85.51 / 42.43 | 30.37 / 88.04 / 41.69 | 31.35 / 86.78 / 42.06 | 33.40 / 85.33 / 43.02 |
| 2-layer GNN | 33.12 / 87.46 / 43.86 | 32.97 / 88.19 / 43.83 | 33.05 / 87.83 / 43.85 | 35.60 / 86.08 / 44.82 |
- Pair macro: flat 24.84 / 24.21, control 25.16 / 25.79, GNN 26.80 / 28.39. Logs: `aegis/results/runs/ft_eval_es_s0.log`, `_s1.log`; ledger rows written by ft_eval.
- Reading: (1) the earlier validation numbers WERE optimistic: flat MLP drops 36.9 -> 31.7 pair (-5.3), GNN 35.6 -> 33.1 (-2.5); but this also has 16% less encoder training data (3,234 vs 3,865), so optimism and data size cannot be separated here. (2) With a clean encoder the 2-layer GNN is best on pair, agent and error in BOTH encoder seeds (pair +1.4 over flat, +1.7 over control on average; agent macro +3 to +5), reversing the earlier "flat = GNN after fine-tuning" reading on validation. Differences are small relative to seed spread (flat std up to 2.6 in one run), so it is suggestive, not established. (3) Test not scored for these encoders (test is scored once per final configuration).
- Loop a9794407 deleted (both kernels done, evals done). No further GPU needed for this step. Nothing committed or pushed.

### 2026-11-12, new 5-minute work loop (cron 049588a8)
- You asked for a loop to complete the remaining work with notes updated side by side. Plan in the loop prompt: (1) final test scoring on the clean encoders es_s0, es_s1 (5 seeds, once; `final_test --ft-dir ... --tag es_s*`), (2) Who&When study on the clean encoders, (3) write the final tables into `aegis/docs/final_results.md`, (4) write (not launch) the end-to-end graph fine-tune Kaggle script `aegis/scripts/kaggle_finetune_graph.py`, (5) stop the loop and summarise.
- Code: `final_test.py` got `--tag` (ledger split-name suffix, so encoders do not mix in the ledger). Steps 1 and 2 run as ONE background chain (scratchpad `chain.sh`, sequential to limit memory): final_test s0, s1, then whowhen_study s0, s1. Progress marker: `aegis/results/runs/chain_status.log` (lines `CHAIN_DONE ...`, `CHAIN_ALL_DONE`). Logs: `final_test_es_s{0,1}.log`, `whowhen_study_es_s{0,1}.log`. Do not start these again.

### 2026-11-12, loop step 4 done: end-to-end graph fine-tune script written
- Cron job is now `4bfcdadd` (the first one, 049588a8, was replaced because its prompt would have restarted the already-running chain).
- New file `aegis/scripts/kaggle_finetune_graph.py` (standalone, syntax-checked only; NOT run, no GPU launched): trains MiniLM and the turn-graph GNN together. Per batch: encode every turn (max 40 nodes per trajectory, evenly spaced plus the first turn of each agent) and the query; project to 128; `--layers` relational mean-aggregation layers (0 = control) over the same 4 edge types as `aegis/src/graph.py`; mean-pool per agent; head with metadata (`--name-free` zeroes framework + agent-name one-hot); BCE on 14 modes + agent. Early stopping on `--es-holdout` (default 0.15, query-grouped slice of grouped train), grouped val never seen by the encoder. Exports `preds.json` (probabilities for es/val/test/who_when per trajectory and agent) plus `ft_log.json`, `run_info.json`; scoring is then done on CPU (thresholds tuned on val only). Encoder lr 2e-5, rest 1e-3, bs 8, epochs 6, patience 2, fp16 autocast on CUDA.
- Not yet verified: no CPU/GPU smoke test (free RAM was 4.4 GB of 15.8 GB while the scoring chain runs; a smoke test loads the whole train file). Plan: smoke test locally with `--limit 16` after the chain finishes (needs an input dir with train/val/test/whowhen/grouped_ids), then a Kaggle smoke kernel, then full runs (layers 2 and 0, name-free and not) only with the user's go-ahead.
- Still need a CPU scorer for `preds.json` (not written yet): load preds, `tune()` thresholds on val probabilities, decode test and Who&When, evaluate; reuse `aegis/src/baselines/flat_mlp.py` `decode`/`tune` and `metrics.evaluate`.

### 2026-11-12, loop updated (cron now with the full plan)
- You approved adding the remaining work to the loop. The loop (replaces 4bfcdadd) now covers: (B) CPU scorer for graph fine-tune predictions, (C1) clean-encoder tables into `aegis/docs/final_results.md`, (C2) local CPU smoke test of `kaggle_finetune_graph.py`, (C3) Kaggle smoke kernel `aegis-ftg-smoke`, (C4) full kernels `aegis-ftg-l2` and `aegis-ftg-l0` (6 epochs, patience 2, bs 8, es-holdout 0.15), (C5) scoring and comparison with flat/GNN, (C6) optional name-free pair, (D) finish and summarise. GPU use is now approved for these Kaggle kernels. Chain still running.

### 2026-11-12, loop rule change
- You asked that the notes be updated after every single completed step (not only each tick). Loop re-created (now cron id in CronList; old 6269cbb2 deleted) with that rule: after each finished step or sub-step, update both notes files immediately before starting the next. Time estimate given to you: about 2 hours from 18:10 (until about 20:10), up to 2.5 h with failures, plus 30-40 min for the optional name-free pair.

### 2026-11-12 18:2x, chain item done: final_test on clean encoder es_s0 (grouped protocol, 5 seeds, micro/macro; log `aegis/results/runs/final_test_es_s0.log`, ledger tag `_es_s0`)
- floor: test pair 7.35/6.35, agent 68.58/61.98, error 27.14/26.99; Who&When 3.42/2.85, 38.32/38.41, 13.33/12.34; avg12 25.59. frequency avg12 18.90.
- flat MLP: test pair 27.05±1.96 / 17.36, agent 81.03±1.86 / 63.40, error 38.68±0.35 / 39.07; Who&When 3.78 / 43.71 / 9.92; avg12 30.73±0.49.
- 0-layer control: test pair 27.91±3.19 / 18.19, agent 79.79 / 65.75, error 39.11 / 39.07; Who&When 1.55 / 37.09 / 8.45; avg12 29.85±0.35.
- 2-layer GNN: test pair 27.14±0.45 / 17.94, agent 82.67±0.81 / 66.68, error 39.23±0.39 / 40.14; Who&When 2.64 / 35.78 / 8.73; avg12 30.18±0.77.
- Reading: with the clean encoder (3,234 train) the three models are within noise on test pair (27.1-27.9), GNN best on agent and error macro; older ft_full test pair was 28.5-29.7 (validation-early-stopped encoder, 3,865 train). So the clean test numbers are about 1.5-2.5 pair lower. Who&When still at/below floor with the base features.
- Next in the chain: final_test es_s1 (running), then whowhen_study es_s0 and es_s1.

### 2026-11-12 18:3x, loop step B done: CPU scorer for the graph fine-tune predictions
- Wrote `aegis/src/eval/score_graph_ft.py` (syntax-checked, not yet run on real predictions). Loads `preds.json` (splits es/val/test/who_when), aligns by trajectory id and agent order (asserts they match), tunes thresholds on grouped val only (`flat_mlp.tune`), scores test and Who&When once, prints micro/macro pair/agent/error, floors, avg12 (paper definition) and a per-framework test breakdown, and logs to the ledger as phase "8" with `--tag`. Handles smoke slices (fewer predicted records than the full splits). Usage: `python -m aegis.src.eval.score_graph_ft --preds <preds.json> --tag ftg_l2`.
- Still running: chain (final_test es_s1, then whowhen_study es_s0/es_s1). Next after CHAIN_ALL_DONE: C1 tables, C2 local smoke test of the graph script.

- 18:2x tick: chain still running (final_test es_s1, 2-layer GNN seed 3 of 5). Prepared the smoke input dir in the scratchpad (ftg_smoke_in: train/val/test/whowhen jsonl copied; grouped_ids.json still to be generated once the chain is done). No GPU running or needed.

### 2026-11-12 18:2x, chain item done: final_test on clean encoder es_s1 (grouped, 5 seeds, micro/macro; log `aegis/results/runs/final_test_es_s1.log`, ledger tag `_es_s1`)
- flat MLP: test pair 25.80±0.20 / 16.54, agent 81.34 / 64.22, error 38.45 / 38.70; Who&When 4.53 / 42.28 / 12.37; avg12 30.92±0.32.
- 0-layer control: test pair 25.21±0.67 / 17.41, agent 80.98 / 66.75, error 38.21 / 39.07; Who&When 3.10 / 37.87 / 11.21; avg12 30.36±0.42.
- 2-layer GNN: test pair 27.46±1.40 / 18.06, agent 81.11 / 63.94, error 39.09 / 39.16; Who&When 3.12 / 37.10 / 10.41; avg12 30.23±0.70.
- Two encoder seeds on test (pair micro, es_s0 / es_s1): flat 27.05 / 25.80, control 27.91 / 25.21, GNN 27.14 / 27.46. GNN best in s1 (+1.7 over flat), tied in s0; the encoder seed moves test pair by about 1 point, as large as the model differences. Averages of the two encoder seeds: flat 26.43, control 26.56, GNN 27.30 (pair micro); error micro 38.57 / 38.66 / 39.16; agent micro 81.19 / 80.39 / 81.89. Suggestive GNN edge (+0.9 pair, +0.6 error over flat), not established.
- Who&When with base features: all three at/below floor again (pair 1.6-4.5 vs floor 3.42). Now running: whowhen_study es_s0 (nameemb variant seed 2) then es_s1.

### 2026-11-12 18:3x, chain COMPLETE (CHAIN_ALL_DONE) and step C1 done
- whowhen_study on clean encoders finished (logs `whowhen_study_es_s0.log`, `_es_s1.log`; ledger phase 7, tags es_s0/es_s1). Mean of the two encoders, flat MLP, 5 seeds, micro pair / agent / error:
  - base: Who&When 4.16 / 43.00 / 11.15, AEGIS test 26.43 / 81.19 / 38.57
  - name-free: Who&When 5.29 / 45.48 / 12.74, test 25.21 / 77.29 / 37.84
  - name-embedding: Who&When 5.04 / 40.85 / 15.04, test **29.41 / 79.78 / 39.09** (+3.0 pair over base; +2.2 and +3.8 per encoder)
  - Who&When floor 3.42 / 38.32 / 13.33. Per encoder: es_s0 base 3.78/43.71/9.92, namefree 5.74/47.58/12.52, nameemb 4.39/40.44/13.61; es_s1 base 4.53/42.28/12.37, namefree 4.83/43.37/12.96, nameemb 5.69/41.26/16.46.
  - Top-1 decoding worse everywhere (not used).
- Conclusion: name-embedding is the best overall and its gain replicates on both clean encoders: in-domain test pair +3.0, Who&When error 15.04 (above floor 13.33, DCL 14.67, o3 14.88) and pair 5.04 (above floor and Aegis-SFT 4.03, below o3 7.41); agent level on Who&When still near the floor (40.85 vs 38.32; Aegis-SFT/o3 51-53). Name-free helps Who&When agent most (45.48) but costs about 4 pair points in-domain.
- C1: wrote a new section "Clean-encoder results" at the end of `aegis/docs/final_results.md` (validation table, grouped test table with floor, Who&When table with published DCL/Aegis-SFT/o3 rows, caveats). Local edit, uncommitted.
- Next: C2 local CPU smoke test of the graph script (generate grouped_ids.json first).

### 2026-11-12 18:4x, step C2 done: local CPU smoke test of the graph fine-tune script PASSED
- Generated `grouped_ids.json` (3,865 train / 940 val ids) into the scratchpad smoke input dir, ran `aegis/scripts/kaggle_finetune_graph.py --limit 16 --epochs 1 --bs 4 --es-holdout 0.25` on CPU: trained 1 epoch (12 train / 4 early-stop / 8 val trajectories; about 103 s on CPU, so GPU is needed for the full run), exported es/val/test/who_when predictions (8 each for the smoke slice), wrote `ft_log.json`, `preds.json`, `run_info.json`. No bugs found; no code change needed.
- `score_graph_ft.py --no-ledger` ran on the smoke `preds.json` without errors (id and agent-order alignment passed; thresholds, per-framework breakdown, floors printed). Smoke numbers are meaningless (1 epoch on 12 trajectories, predictions equal the floor).
- Next: C3, Kaggle smoke kernel `aegis-ftg-smoke` (--limit 64 --epochs 1 --bs 8).

### 2026-11-12 18:4x, step C3 started: Kaggle smoke kernel launched
- `aryanmahajan7/aegis-ftg-smoke` pushed (graph script, --limit 64 --epochs 1 --bs 8, GPU, datasets aegis-raw-data + aegis-grouped-ids). GPU RUNNING. On COMPLETE: save log+output to `aegis/results/kaggle_logs/ftg_smoke/` first, check, then launch aegis-ftg-l2 and aegis-ftg-l0.

### 2026-11-12 18:4x, step C3 done: Kaggle smoke PASSED; step C4 launching
- `aegis-ftg-smoke` COMPLETE; log/output saved FIRST in `aegis/results/kaggle_logs/ftg_smoke/` (log, ft_log.json, preds.json, run_info.json). 55 train / 9 early-stop / 16 val trajectories, 1 epoch took 4 s on the GPU (es loss 0.2331), predictions exported for es/val/test/who_when (16 each). No errors. Speed estimate for the full run: 3,234 train trajectories is about 59x more, so about 4 min/epoch, 6 epochs about 25 min plus export.
- Launching full kernels `aegis-ftg-l2` (layers 2) and `aegis-ftg-l0` (layers 0), both: epochs 6, patience 2, bs 8, es-holdout 0.15, seed 0.

- 18:5x: launched aegis-ftg-l2 and aegis-ftg-l0 (both RUNNING on GPU). Logs unreadable until COMPLETE; then save to aegis/results/kaggle_logs/ftg_l2 and ftg_l0 first, score with score_graph_ft.py.

- 18:3x tick: aegis-ftg-l2 and aegis-ftg-l0 QUEUED on Kaggle (not yet running). No other change.

- 18:41 tick: aegis-ftg-l2 and aegis-ftg-l0 now RUNNING on Kaggle GPU (started about 18:40). Logs unreadable until COMPLETE; expected about 25-35 min. Not touching them.

- 18:46 tick: aegis-ftg-l2 and aegis-ftg-l0 still RUNNING (about 6 min in). No other change.

- 18:51 tick: aegis-ftg-l2 and aegis-ftg-l0 still RUNNING (about 11 min in). No other change.

### 2026-11-12 18:5x, step C4 done: end-to-end graph fine-tune kernels COMPLETE, logs saved FIRST
- `aegis-ftg-l2` and `aegis-ftg-l0` finished (about 15 min each, about 130 s/epoch on the T4, faster than my 25-35 min estimate). Logs/outputs saved in `aegis/results/kaggle_logs/ftg_l2/` and `ftg_l0/` (log, ft_log.json, preds.json, run_info.json; 1.6 MB each).
- Setup: layers 2 / 0, epochs 6, patience 2 (never triggered), bs 8, es-holdout 0.15 (3,234 train / 631 early-stop / 940 val), seed 0, encoder lr 2e-5, rest 1e-3.
- Early-stop loss per epoch: l2 .1641 .1606 .1559 .1431 .1469 .1459 (best epoch 4, 0.1431); l0 .1666 .1547 .1497 .1441 .1453 .1440 (best epoch 6, 0.1440). Train loss l2 .1911 -> .1213, l0 .1955 -> .1214. The two are essentially equal on this loss (l2 slightly lower best).
- Next: C5, score both with `score_graph_ft.py` (tags ftg_l2, ftg_l0), one at a time.

### 2026-11-12 19:0x, step C5 done: end-to-end graph fine-tune scored (CPU, thresholds from grouped val only; logs `aegis/results/runs/score_ftg_l2.log`, `score_ftg_l0.log`; ledger phase 8)
| Model (micro pair / agent / error) | Val | Test | Who&When (floor 3.42 / 38.32 / 13.33) |
|---|---|---|---|
| E2E 0-layer control | 30.74 / 86.56 / 42.15 | 26.00 / 80.88 / 38.38 | 2.40 / 41.18 / 9.76 |
| E2E 2-layer graph | 29.59 / 87.74 / 39.88 | 23.69 / 80.15 / 35.22 | 2.90 / 42.55 / 7.54 |
| two-stage flat (mean of 2 encoders) | 31.66 / 86.19 / 42.49 | 26.43 / 81.19 / 38.57 | 4.16 / 43.00 / 11.15 |
| two-stage GNN | 33.05 / 87.83 / 43.85 | 27.30 / 81.89 / 39.16 | see final_test logs |
- avg12: E2E control 30.72, E2E graph 29.06 (floor 25.59). Per-framework test pair (graph vs control): agentverse 25.2 vs 29.1, dylan 18.1 vs 18.3, llm_debate 28.8 vs 28.9, macnet 25.9 vs 28.6, magentic_one 19.2 vs 18.4 (n=28), smoagents 4.7 vs 7.8 (n=48).
- RESULT: end-to-end graph training does NOT help: the 2-layer model is 2.3 pair below its 0-layer control on test and 1.2 below on validation, and both E2E models are at or below the two-stage pipeline. One seed and one encoder run, noise about 1-3 points, so "graph is worse" is suggestive; what is supported is "no sign of a graph gain". Defensible claim: a graph over turns adds at most a small unproven gain; most of the improvement is from fine-tuning the encoder.
- Written as a new section "End-to-end graph fine-tune" at the end of `aegis/docs/final_results.md` (local edit, uncommitted).
- Next: C6 optional name-free E2E pair on the GPU (Who&When).

### 2026-11-12 19:0x, step C6 started: name-free end-to-end pair launched
- `aegis-ftg-l2nf` and `aegis-ftg-l0nf` pushed (same as ftg_l2/ftg_l0 plus `--name-free`: framework and agent-name one-hot columns zeroed). GPU RUNNING/queued. On COMPLETE: save logs to aegis/results/kaggle_logs/ftg_l2nf and ftg_l0nf FIRST, score with score_graph_ft.py (tags ftg_l2nf, ftg_l0nf), compare with the two-stage name-free variant (Who&When 5.29 / 45.48 / 12.74 mean; test 25.21 / 77.29 / 37.84).

- 19:00 tick: aegis-ftg-l2nf and aegis-ftg-l0nf RUNNING on Kaggle GPU (about 15-20 min total expected). No other change.

- 19:05 tick: aegis-ftg-l2nf and aegis-ftg-l0nf still RUNNING (about 5 min in). No other change.

- 19:10 tick: aegis-ftg-l2nf and aegis-ftg-l0nf still RUNNING (about 10 min in). No other change.

### 2026-11-12 19:1x, step C6: name-free E2E kernels COMPLETE, logs saved FIRST
- `aegis-ftg-l2nf` and `aegis-ftg-l0nf` finished; logs/outputs saved in `aegis/results/kaggle_logs/ftg_l2nf/` and `ftg_l0nf/`. Next: score both with score_graph_ft.py (tags ftg_l2nf, ftg_l0nf).

### 2026-11-12 19:2x, step C6 done: name-free end-to-end pair scored (logs `aegis/results/runs/score_ftg_l2nf.log`, `score_ftg_l0nf.log`; ledger phase 8)
| Model (micro pair / agent / error) | Val | Test | Who&When (floor 3.42 / 38.32 / 13.33) |
|---|---|---|---|
| E2E name-free 0-layer control | 23.72 / 83.77 / 35.88 | 20.19 / 77.77 / 33.38 | 4.41 / 47.62 / 9.97 |
| E2E name-free 2-layer graph | 27.86 / 87.00 / 40.81 | 22.90 / 81.95 / 35.72 | 5.04 / 39.18 / 9.90 |
- avg12: control 27.97, graph 29.47. Best early-stop loss: graph 0.1477 (epoch 5), control 0.1544 (epoch 4).
- Here the graph beats its control (+2.7 test pair, +4.1 val pair), the OPPOSITE of the name-dependent pair (graph -2.3 vs control). Single seed per model: the sign of the graph effect is unstable across the two comparisons, consistent with noise; no claim either way. Name-free E2E is clearly worse in-domain than name-dependent E2E (control 20.2 vs 26.0); Who&When pair 4.4-5.0 above floor, error 9.9-10.0 below the 13.33 floor; the two-stage name-embedding flat MLP stays the best Who&When setting (pair 5.04, error 15.04).
- Added to the end of `aegis/docs/final_results.md` (local edit, uncommitted).
- Remaining: step D, final summary and deleting the loop.

### 2026-11-12 19:2x, LOOP FINISHED (step D)
- Cron 118a6b8f deleted. Every planned step completed: chain (final_test + whowhen_study on es_s0/es_s1), CPU scorer, final tables, local + Kaggle smoke, E2E graph pair (l2, l0), name-free E2E pair (l2nf, l0nf), notes. No kernel is running. Nothing committed or pushed; the working tree holds many uncommitted files (modified: aegis docs, final_test.py, turn_gnn.py, kaggle_finetune_minilm.py, metrics_history.jsonl, root context.md, new-plan.md; new: kaggle_logs ft_bge, ft_es_s*, ftg_*, runs logs, whowhen_study.py, score_graph_ft.py, kaggle_finetune_graph.py). Awaiting the user for: commit/push, guide approval, more seeds, write-up.

### 2026-11-12 evening, brainstorm for the next round saved
- Full idea list (16 ideas, sources, which are from papers vs my own, ranking, agreed order: diagnostic first, then pair-level head + consistency loss, peer-contrast, attention pooling, per-class thresholds, sibling contrast, content-flow edges, label-shift for Who&When, ensembling) is in `local-notes/ideas.md`, together with the project state summary needed to resume after a context reset. Related-work links: AgenTracer 2509.03312, GraphTracer 2510.10581, AFANet 2608.18575, Who&When 2505.00212, FALAT 2606.00765 (snippets only). The loop is deleted; no kernel running; nothing committed.

### 2026-11-12 evening, loop information saved
- Loop design, standing rules, plan steps A-D, kernel names, commands (token via env var only, not stored) and output locations are in `local-notes/loop.md`. No loop or kernel is active now.

### 2026-11-12 evening, consolidated record saved
- `local-notes/approaches_tried.md`: all approaches beyond the paper with results, the GNN-improvement table, and what led to improvement (previously only in chat answers).

### 2026-11-12 evening, master results record saved
- `local-notes/results_master.md`: every approach and result for AEGIS-Bench and Who&When, reference rows, setup facts, caveats and evidence locations, checked against `aegis/docs/final_results.md` and `context.md` section 7.

- 2026-11-12 night: second brainstorm saved as section 5 of `local-notes/ideas.md` (papers matched to our situation; novelty candidates A sibling pseudo-clean reference, B deviation-then-de-propagation, C leave-one-framework-out validation).

### 2026-11-12 night: STEP 0 DIAGNOSTIC DONE (new `aegis/src/eval/diagnostic.py`, log `aegis/results/runs/diagnostic_es_s0.log`)
Setup: name-embedding flat MLP on clean encoder es_s0 vectors, 3 seeds, grouped test, thresholds tuned on val; ORACLE conditions used only to measure ceilings (not results).
| Condition (micro/macro) | pair | agent | error |
|---|---|---|---|
| model | 27.79 / 18.18 | 81.37 / 67.84 | 38.66 / 39.02 |
| oracle agents (true faulty-agent indicator, modes from model) | 31.12 / 20.53 | 92.20 / 89.25 | 40.09 / 40.57 |
| oracle modes (model mode probs masked to the trajectory's true modes, agents from model) | 56.06 / 44.54 | 72.05 / 60.95 | 79.90 / 79.67 |
- Knowing the faulty agents would add only +3.3 pair; knowing which modes occur would add +28.3 pair. MODE identification is the bottleneck, not agent detection.
- Of the faulty agents the model does find, the exact mode set is right only 21.2% of the time (323/1008, 183/1150, 183/1166 per seed).
- Caveat: the oracle-modes ceiling uses trajectory-level truth (a strong hint), so 56 is an upper bound, not a target. The `both` row is not 100 because masking multiplies the model's mode probability and thresholds still apply.
- Consequence for the plan: prioritise ideas that sharpen WHICH mode, not WHO: attention pooling over turns (find the symptomatic turn), mode-specific detectors, label graph / hard-constraint pair head, sibling deviation (shows what changed), de-propagation. Agent-level ideas (peer contrast for who) matter less for pair F1. Who&When agent level is a separate problem (name shift).

- 2026-11-12 night: plain-language guide to every approach and the novel ones added as section 6 of `local-notes/ideas.md` (it also holds the step-0 diagnostic: mode identification is the bottleneck).

- The 27-versus-49 puzzle (2026-11-12): we re-scored our saved zero-shot answers using the authors' own scoring code and their way of reading the model's answer. Nothing moved (49.24), so the way we read and score answers is not the cause. The other 600-record test file has the same statistics, so that alone does not explain it either. What is left is how the model was run: the authors used full-precision weights, random sampling and a 256-token answer limit; we used 4-bit weights, greedy decoding and 512 tokens. The script is ready to repeat the run their way on a 200-record sample; it needs the Kaggle login to be re-established.

- 2026-11-12 ~20:15: IMPROVEMENT PUSH started. Six wave-1 Kaggle kernels launched on three accounts (A aryanmahajan665: aegis-es-s2, aegis-es-s3; B aryanmahajan7: aegis-es-s4, aegis-len512; C amahaj55: aegis-l12, aegis-bgesmall). Input datasets copied into A and C. New notes: local-notes/timeline.md (approaches in simple words) and local-notes/results.md (numbers). Cron 3ac42719 every 5 min monitors. Added --max-len to kaggle_finetune_minilm.py (uncommitted).

- 2026-11-12 ~20:25: user asked for bullet points, no tables: timeline.md and results.md converted; memory feedback saved; cron recreated with the bullet rule.

- 20:12 tick: all six wave-1 kernels RUNNING. improve_cpu es_s0 still running (namedrop seed 2 left). Early per-seed read (prod decoder): peer-contrast looks better than base: val pair 39.2/39.1/38.0 vs base 37.2/31.0/30.6; test pair 32.0/31.7/31.0 vs base 31.8/26.6/25.0; Who&When agent lower for peer (36.7-38.3 vs 39.6-41.9). namedrop: test pair 24.8/27.1 (lower), Who&When pair 5.2/6.3, agent 44.2/49.3 (higher). Final table next tick.

- 2026-11-12 night: `local-notes/approaches_vs_paper.md` written: every approach, what it is, its result, gain in points versus DCL and Aegis-SFT, AEGIS-Bench and Who&When, caveats. Bullets, no tables.

- 20:16 tick: all six wave-1 kernels still RUNNING. improve_cpu es_s0 FINISHED: peer-contrast test pair 31.56 vs base 27.79 (+3.8, std 0.43), name-dropout Who&When agent 47.04 vs 41.00 (+6.0) but in-domain pair -1.8, pm/min decoder mixed, per-class thresholds +1.6 on base but hurt peer. Verdicts in results.md section 3. Started chain2.sh (es_s0 peer+namedrop, then es_s1 base/peer/namedrop/peernd).

- 2026-11-12 night: `local-notes/approaches_vs_paper.md` rewritten: only implemented approaches, each compared against every paper approach (Random, DCL, Qwen-7B/14B, Aegis-SFT, Aegis-GRPO, o3) with exact point gains, AEGIS-Bench and Who&When. Bullets, no tables.

- 2026-11-12 night: Who&When detail section (Part 3) appended to `local-notes/approaches_vs_paper.md`: history incl. the bug, all variants, gains over floor and every paper row, what helped, caveats.

- 20:23 tick: es_s2 and es_s4 COMPLETE and saved (best epoch 4, losses 0.1415 and 0.1398, similar to es_s0 0.1414). Launched aegis-es-s5 (A) and aegis-es-s6 (B). Still RUNNING: es_s3, len512, l12, bgesmall. CPU: chain2 running es_s1 improve_cpu; downstream scoring of es_s2/es_s4 waits for it (one heavy job at a time).

- 20:40 tick: es_s3, len512, bgesmall COMPLETE and saved (losses: es_s3 0.1403 best epoch 5; len512 0.1378 best epoch 3, the lowest; bgesmall 0.1415 best epoch 4). es_s1 improve_cpu replication DONE: peer +4.4 test pair, namedrop Who&W agent +4.2 (both replicate). chain3 started (improve_cpu on len512, bgesmall, es_s2, es_s3, es_s4). User asked for ALL values: full-values appendix (every seed, micro+macro) now generated into approaches_vs_paper.md and results.md by local-notes/gen_full_values.py (re-run each tick). Running kernels: es_s5 (A, queued), es_s6 (B), l12 (C). Account C has one idle slot.

- 20:41: launched aegis-len512-s1 on account C (len512 with seed 1) to check whether the lowest-loss encoder (len512) replicates. Short name for saving: len512_s1.

- 20:46 tick: aegis-l12 COMPLETE and saved (best epoch 4, loss 0.1391, second lowest after len512 0.1378). Running: es_s5 (A, queued), es_s6 (B), len512-s1 (C). CPU: chain3 improve_cpu on len512 in progress; l12 to be queued after chain3.

- 20:50 tick: improve_cpu on len512 (3 seeds): len512 is clearly better in-domain. base test pair 34.9/33.9/34.5 (mean 34.4 vs MiniLM-256 base 27.8), peer test pair 35.5/37.3/36.0 (mean 36.3), peernd 34.5/34.4/35.0; but Who&W is lower for base/peer (error 9-13, below floor) while peernd Who&W agent 42.0/46.4/46.9. Added chunking to kaggle_finetune_minilm.py (--chunk-chars, --max-chunks; CPU-tested, short text identical) and launched aegis-chunk on account B. Running: es_s5 (A), es_s6 (B), chunk (B), len512-s1 (C).

- 20:56 tick: all four remaining kernels RUNNING (es_s5 A, es_s6 B, chunk B, len512-s1 C). chain3: len512 done and recorded (peer test pair 36.27, base 34.41); bgesmall next. Full-values appendix refreshed.

- 20:52 tick: es_s6 COMPLETE and saved (best epoch 4, loss 0.1392). bgesmall scored: peer test pair 32.28 vs base 26.26 (+6.0), same as MiniLM-256: no gain from encoder family; peer gain replicates on a third encoder. Launched aegis-l12len512 on B (L12 + 512 tokens). Queued chain4 (l12, es_s6 improve_cpu). Running: es_s5 (A), chunk (B), l12len512 (B), len512-s1 (C).

- 20:55 tick: es_s5 COMPLETE and saved (best epoch 4, loss 0.1438; epochs 1-6: 0.1651, 0.152, 0.1445, 0.1438, 0.1445, 0.149). Launched aegis-len512-s2 on account A (third len512 seed). Running: chunk (B), l12len512 (B, queued), len512-s1 (C), len512-s2 (A). CPU chain3 on es_s2 in progress.

- 22:10: USER TARGET CHANGE: improve Who&When most, without hurting AEGIS. Saved to timeline.md. All kernels A len512-s2 running; l12len512-s1 (B), l12len512-s2 (C) launched. chunk, l12len512, len512_s1 saved. Results es_s2..s6, l12 recorded in results.md. ww_focus.py written; chain5 (improve_cpu on new encoders) running, chain6 (ww_focus on 7 encoders) queued.

- 22:09 tick: kernels len512-s2 (A), l12len512-s1 (B), l12len512-s2 (C) RUNNING. chain5 on l12len512 in progress; chain6 (ww_focus) queued behind it. No new result.

- 22:16 tick: no kernel finished (len512-s2 A, l12len512-s1 B, l12len512-s2 C RUNNING). l12len512 scored: peer test pair 37.26 (best AEGIS), WW 3.77/39.51/11.98 (no Who&W gain). chain5 now on len512_s1.

- 22:19 tick: kernels len512-s2, l12len512-s1, l12len512-s2 RUNNING; chain5 finished len512_s1, now chunk then es_s5; chain6 (ww_focus) after.

- 22:25 USER DECISION: skip LoRA side track for now; focus on cheap CPU/encoder Who&W ideas (ensemble, name-dropout, label-shift, attention pooling, LOFO validation, name-to-role mapping).

- 22:28 tick: 3 kernels RUNNING. chain5 on chunk. ww_attn.py (attention pooling) written; one-seed smoke running on es_s0 (2 threads). ww_focus path bug fixed (es_s0/es_s1 live in kaggle_logs/ft_es_s0 etc).

- 22:33 tick: len512-s2 COMPLETE and saved. ww_attn smoke (es_s0, 1 seed, attn): test 27.65/82.00/38.88, WW 7.04/42.20/16.80 (single seed, noise high). chain5 finished chunk, on es_s5. Queued chain7 (ww_attn es_s0 mean vs attn, improve_cpu len512_s2, ww_attn len512_s1) after chain6.

- 22:37 tick: account A idle after len512-s2; launched aegis-len512-s3 on A (short name len512_s3). Running: l12len512-s1 (B), l12len512-s2 (C), len512-s3 (A). chain5 on es_s5; chain6 then chain7 queued.

- 22:45 tick: chain5 done (chunk, es_s5 recorded). chunk: peer test 36.52 but WW error 10.4 (worse). ww_focus (ensemble/nd/label-shift on 7 encoders) now running; chain7 queued after it. Kernels len512-s3 (A), l12len512-s1 (B), l12len512-s2 (C) RUNNING.

- 22:55: 4th GPU account D = aryanomgh65 (kaggle.txt line 7; private datasets aegis-raw-data + aegis-grouped-ids created there). Added --attn (end-to-end gated attention pooling over turns) to kaggle_finetune_minilm.py (att params in the head lr group); launched aryanomgh65/aegis-attn (MiniLM 256, seed 0, es-holdout 0.15; short name attn). Not CPU-smoke-tested (no CUDA here): watch the log at COMPLETE.

- 22:59: notes updated (timeline.md attention-pooling section, results.md running list). ww_focus has trained es_s0..es_s3, es_s4..es_s6 next.

- 22:53 tick (clock shows 22:53): all 4 kernels RUNNING (len512-s3 A, l12len512-s1 B, l12len512-s2 C, attn D). ww_focus trained es_s0..es_s3 of 7; no new result.

- 23:00 USER: entire focus on Who&When from now; freed GPUs only get Who&W jobs. Memory saved.

- 23:10: brainstorm round 3 saved (ideas.md section 7, leads A to G; escalation ladder in timeline.md). User also asked: keep brainstorming papers repeatedly; stronger model ok.

- 23:15: loop.md section 7 (current loop), approaches_tried.md section L written. Notes are now updated each tick as part of the loop (step 3).

- 23:20 tick: saved l12len512_s1, l12len512_s2 (len512_s2 earlier). Accounts B and C idle. chain8 queued. Notes updated this step (results.md, timeline.md).

- 23:27: launched Who&W-only GPU jobs: aryanmahajan7/aegis-nonames (MiniLM 256 seed 0, framework+agent-name columns zeroed in the head during fine-tuning; short name nonames) and amahaj55/aegis-nometa (also count scalars zeroed = targeted augmentation, lead B; short name nometa). Both compare with es_s0. Added --no-names/--no-scalars flags to kaggle_finetune_minilm.py (also guard sd<1e-3). Running: len512-s3 (A), attn (D), nonames (B), nometa (C). Account A idle after len512-s3.

- 23:02 tick: len512-s3 (A), attn (D), nonames (B), nometa (C) all RUNNING. ww_focus has trained all 7 encoders and is now computing the ensemble per seed (results print at the end). No new result.

- 23:12 tick: attn COMPLETE saved. ww_focus DONE and recorded (nd ensemble WW 6.80/45.78/16.12; +shift 8.46/46.42/17.08; peernd 5.37/42.27/12.17). Trade-off AEGIS vs Who&W noted. Launched aegis-nonames-s1 on D. chain7 (attention pooling CPU) running; chain8, chain9 queued.

- 23:14 tick: kernels len512-s3 (A), nonames (B), nometa (C), nonames-s1 (D) RUNNING. chain7: ww_attn es_s0 done: attention pooling no Who&W gain (agent 44.99 vs 47.04). improve_cpu len512_s2 running.

- 23:20: wrote ww_block.py (block-dropout: 4 training copies with name/peer blocks on/off; Who&W evaluated under 4 masks, +label-shift, ensemble over encoders; AEGIS test with all blocks on). Queued chain10 (7 256-token encoders) after chain9. Post-hoc Who&W mask choice flagged in the docstring.

- 23:35: ideas.md section 8 (data checks: siblings share 7.5% text, AEGIS 2-4 faulty agents per record vs Who&W 1). Ladder rung 1 dropped in literal form.

- 23:45 tick: saved len512_s3, nonames, nometa. Launched nometa-s1 (A) and nonames512 (B). nonames-s1 (D) running. C idle. chain11 queued (improve_cpu base,namedrop,peernd on nonames, nometa, len512_s3 after chain10).

- 23:32 tick: nonames-s1 COMPLETE saved. len512_s2 scored (peernd WW 4.36/42.43/11.89: len512_s1 7.09 was seed luck). ww_attn len512_s1: attention gives AEGIS +2.9 pair (std 0.30) with Who&W neutral. chain7 done; chain8 running (improve_cpu l12len512_s1/_s2), chain9, chain10, chain11 queued. Accounts C and D idle.

- 23:35 tick: nometa-s1 (A) and nonames512 (B) RUNNING; C and D idle (waiting on no-names scoring). chain8 (improve_cpu l12len512_s1/_s2) running; chain9-11 queued. No new result.

- 23:41 tick: nometa-s1 (A), nonames512 (B) RUNNING; C, D idle. chain8 scored l12len512_s1 (peernd test 35.69, WW 4.73/43.31/11.84); l12len512_s2 in progress.

- 23:55 USER: try the complex ideas first, then LoRA. Launched aegis-advfw (C) and aegis-supcon (D). Added --adv-fw / --supcon to the fine-tune script. A: nometa-s1 and B: nonames512 still running.

- 00:05: wrote ww_complex.py (hierarchical mode heads, kNN/prototype classifier, kNN+MLP mix; flat name-dropout control) and queued chain12 on es_s0..es_s3 after chain11.

- 23:50 tick: l12len512_s2 scored (stable: AEGIS 35.5-36.4, WW error 10.7-11.9). USER: explore all complex ideas before LoRA, assign as GPUs free, then LoRA. 4 kernels RUNNING (nometa-s1 A, nonames512 B, advfw C, supcon D). chain9 (attn scoring) running.

- 23:58: added --seq flag + launch.sh helper. All four accounts busy (nometa-s1, nonames512, advfw, supcon). Next free account gets seq, then advsup.

- 23:50: nometa-s1 COMPLETE saved. Launched aegis-seq on A (order-aware small transformer over turn vectors, end-to-end; MiniLM 256 seed 0; compare with es_s0). NOT GPU-smoke-tested.

- 00:10: LoRA reader script + scorer written (not yet launched; goes on the first GPU that frees after the complex encoder jobs).

- 23:50 tick: seq (A), nonames512 (B), advfw (C), supcon (D) all RUNNING. chain9 (attn scoring) in progress. No new result.

- 00:00 tick: wrote local-notes/loop_restart.md (full loop prompt, accounts, queue, results, notes map) and memory pointer. attn scoring recorded (not better than es_s0). 4 kernels RUNNING (seq A, nonames512 B, advfw C, supcon D). chain10 (ww_block) running.
- 2026-11-13 early: advfw+supcon saved, relaunched advsup (D) and advfw1 (C); chain13 queued.
- 2026-11-13: seq saved, chain14 queued, aegis-noscal launched on A.
- 2026-11-13: block-dropout (ww_block) names+peer: AEGIS 32.93, WW 6.10/44.57/12.65, +shift 8.79/48.54/17.86 = new best. advfw1/advsup saved; supcon5, seqsup launched; chain15 queued.
- 2026-11-13: noscal, nonames512 saved; attnsup, l12sup512 launched; chain16 queued.
- 2026-11-13: nonames scored; chain11 on nometa.
- 2026-11-13: nometa scored (no gain); supcon5, seqsup saved; advfw-s1, supcon-s1 launched; chain17 queued.
- 2026-11-13 00:45: attnsup saved, attnadvsup launched, chain18 queued.
- 2026-11-13 ~00:55: len512_s3 scored; chain11 done; chain12 running.
- 2026-11-13: chain12 done (hier, kNN dead for Who&W; knnmix AEGIS ok, WW lower); advfw_s1, supcon_s1 saved; advsup-s1, noscal-s1 launched; chain19 queued.
- 2026-11-13 01:05: attnadvsup saved; LoRA kernel aegis-lora launched on A.
- 2026-11-13 06:05: batch scoring recorded (supcon/supcon5 best Who&W gains; adv dead; seq trade-off); LoRA relaunched after torchao fix; supcon5-s1, supcon10, attnsup5 launched; chain21 running.
- 2026-11-13 06:02: queued chain22 = ww_block (block-dropout) on supcon, supcon5, attnsup after chain21; kernels (lora, supcon5-s1, supcon10, attnsup5) running.
- 2026-11-13 06:16: supcon5_s1, supcon10 saved; supcon5-s2, seqsup5 launched; chain23 queued.
- 2026-11-13 06:20: supcon_s1 does not replicate seed 0 (WW pair 4.2-4.8); attnsup5 saved; supcon10-s1 launched; chain24 queued.
- 2026-11-13 06:35: supcon5_s2 (B) saved clean; scoring queued chain25; launched supcon5-s3 (B, seed 3) to build a multi-seed supcon5 ensemble for block-dropout.
- 2026-11-13 06:37: seqsup5 (C) saved clean; scoring queued chain26; launched supcon5-s4 (C). Fixed chain25 dir-name bug (supcon5_s2 -> supcon5-s2).
- 2026-11-13 06:40: supcon10_s1 (D) saved clean; scoring queued chain27; launched supcon5-s5 (D, seed 5).
- 2026-11-13 06:55: supcon5_s3 (B), supcon5_s4 (C) saved clean; scoring queued chain28; launched aegis-l12sup256 (B, L12 256 supcon 0.5) and aegis-attnsup5-s1 (C).
- 2026-11-13 07:00: supcon5_s5 (D) saved clean; scoring queued chain29; launched supcon3-s6 (D).
- 2026-11-13 07:05: CPU scoring is the bottleneck; started chain22 (ww_block on supcon,supcon5,attnsup) NOW in parallel with chain21 (2 CPU jobs at once, deliberate; user said take any permission and keep going) because it is the decisive experiment.
- 2026-11-13 07:13: attnsup5_s1 (C) saved clean (scoring to queue); launched supcon5-s7 (C). advsup_s1 scored (adv+supcon seed 1: see improve_cpu_advsup_s1.log).
- 2026-11-13 07:18: supcon3_s6 (D) saved clean (scoring to queue); launched supcon5-s8 (D). Block-dropout on contrastive encoders (chain22): supcon, supcon5 trained, attnsup next.
- 2026-11-13 07:23: l12sup256 (B) saved clean (scoring to queue); launched l12sup512-s1 (B, L12 512 supcon 0.5 seed 1).
- 2026-11-13 07:30: wbsup done (AEGIS 34.87, WW 5.35/47.03/10.04; no new WW best); enc_dir fixed; wbsc5 and wbmix block-dropout ensembles started (chain30).
- 2026-11-13 07:33: supcon5_s7 (C) saved clean; launched l12sup256-s1 (C).
- 2026-11-13 07:37: supcon5_s8 (D) saved clean; launched len512sup (D, L6 512 tokens supcon 0.5).
- 2026-11-13 07:45: LoRA scored: AEGIS test 8.22/74.92/19.59, WW 3.10/44.25/9.09 -> not competitive; A relaunched with seqsup5-s1.
- 2026-11-13 07:48: chain21 done and recorded (l12sup512 strong AEGIS/weak WW; noscal_s1 no gain; adversarial dead x3).
- 2026-11-13 07:55: supcon5_s1 scored: WW pair 4-5 (seed 0 gain not replicated); waiting for wbsc5/wbmix.
- 2026-11-13 08:02: seqsup5_s1 (A), l12sup256_s1 (C) saved clean (scoring to queue); launched plain-recipe es-s7 (A) and es-s8 (C) to enlarge the es block-dropout ensemble (current best WW row).
- 2026-11-13 08:11: len512sup (D) saved clean (scoring to queue); launched plain es-s9 (D).
- 2026-11-13 08:15: supcon10 scored (no gain, contrastive weight not a lever); l12sup512_s1 saved; es-s10 launched on B.
- 2026-11-13 08:20: es_s7 (A), es_s8 (C) saved clean; launched es-s11 (A), es-s12 (C). wbsc5 trained all 6 encoders; summary pending.
- 2026-11-13 08:25: wbsc5 done: AEGIS test 36.15, WW 6.01/47.39/10.07 (shift hurts); best WW row still es block-dropout+shift 8.79/48.54/17.86; wbmix next.
- 2026-11-13 08:30: es_s9 (D) saved clean; launched es-s13 (D).
- 2026-11-13 08:35: user: stop launching new GPU jobs, just complete running ones (es-s10..s13); CPU scoring/ensembles continue.
- 2026-11-13 08:35: es_s10 (B) saved clean; B left idle per user. attnsup5 scored (see improve_cpu_attnsup5.log). Not launching new GPU jobs.
- 2026-11-13 08:40: es_s11 (A), es_s12 (C) saved clean; A, B, C idle per user; only es-s13 (D) still running. Final block-dropout ensemble with es_s7..s13 to queue when es_s13 lands.
- 2026-11-13 08:50: es_s13 (D) saved clean; ALL GPU jobs finished and saved, all 4 GPUs idle (per user). Queued chain31 = ww_block on es_s0..es_s13 (14 plain-recipe encoders, tag wbes14) after wbmix.
- 2026-11-13 08:53: supcon5_s2 scored, same pattern (WW pair 4-5).
- 2026-11-13 09:16: wbmix done: all-on AEGIS 35.45, WW 5.53/46.16/10.85 (+shift 7.32/47.01/16.76); names+shift 9.07/47.79/18.63 (post-hoc, AEGIS 31.23). No clear new winner; wbes14 next.
- 2026-11-13 09:21: seqsup5 scored: AEGIS pair ~22 (-10), WW no gain: sequence encoder dead. wbes14 training 1/14.
- 2026-11-13 09:40: user allows GPUs again; launched aegis-lora-3b (B) and aegis-lora-long (C). A, D idle.
- 2026-11-13 09:40: supcon10_s1 scored (see improve_cpu_supcon10_s1.log); wbes14 5/14 trained; lora-3b, lora-long running.
- 2026-11-13 09:42: supcon10_s1: agent F1 50.2-50.4 (best single encoder), AEGIS 31.8-32.8, pair 4-6.
- 2026-11-13 09:57: CPU-offload to Kaggle: dataset aegis-cpujob uploaded to A (code + data + es_s7..s13 vectors); pushed kernel aegis-cpu-es7 (A; ww_block wbes7b, output ww_block_wbes7b.log + metrics_history_wbes7b.jsonl; save to aegis/results/kaggle_logs/cpu_es7). D dataset upload in progress, then push aegis-cpu-scb (wbscb). Staging dir D:/aryan/cpujob_stage (delete after). C: drive full (615MB free), told user.
- 2026-11-13 10:02: aegis-cpu-scb (D) first push failed (dataset not attached yet; log saved in kaggle_logs/cpu_scb_fail); re-pushed v2 with dataset. aegis-cpu-es7 (A) running. wbes14 9/14 trained.
- 2026-11-13 10:20: cpu_es7 (A) saved; wbes7b: es_s7..s13 all-on AEGIS 33.23, WW 5.24/43.03/11.52, +shift 7.76/46.82/17.99 -> best es row reproducible (~8 pair). A idle. wbes14 11/14.
- 2026-11-13 10:25: cpu_scb (D) saved; wbscb: all-on AEGIS 36.85, WW 5.51/46.98/9.99 (replicates wbsc5). A and D idle. wbes14 13/14 trained.
- 2026-11-13 10:32: wbes14 done: all-on AEGIS 32.96, WW 5.56/43.75/11.76; +shift 8.24/47.72/17.12 = converged es-ensemble estimate (halves 8.79 / 7.76). chain28 done. A, D idle; B,C LoRA running.
- 2026-11-13 10:50: chain29 done (all scoring chains finished). New CPU idea launched (chain32, tag wbtop1): per-record fault-count prior for Who&W: one-faulty-agent-per-record decode (top1: modes >= threshold; top1m: single best mode), flag --top1 added to ww_block.py; es_s0..s6 block-dropout, reports it as a separate benchmark-structure-prior row (no Who&W labels used; Who&W 'exactly one faulty agent' is a known dataset property).
- 2026-11-13 11:15: wbtop1 done: one-agent-per-record decode DEAD (pair +0.7 without shift, 0 with shift, agent -4..-8). Reproduces 8.79/48.54/17.86 exactly (deterministic pipeline). CPU free.
- 2026-11-13 11:21: brainstorm round (ideas.md section 9): nothing cheap beyond a content-flow originator feature.
- 2026-11-13 12:00: lora_3b (B) saved+scored: AEGIS 7.69/77.42/16.00, WW 5.91/45.93/10.42 (better than 1.5B LoRA, not competitive with ensembles). B idle; lora-long (C) running.
- 2026-11-13 13:20: lora_long (C) saved+scored: AEGIS 10.36/78.91/18.84, WW 3.23/41.94/13.48. LoRA closed: no run beats encoder ensembles (best LoRA WW = 3B 5.91). All 4 GPUs idle, CPU idle; nothing running.
- 2026-11-13 13:30: CPU idea launched: test-time adaptation / self-training on unlabeled Who&W (--tta in ww_block.py; confident own predictions (top agent p>=0.5, modes >=0.4) become extra training rows, retrain; separate 'tta_*' rows, transductive, no Who&W label). Smoke test on es_s0, 1 seed (ww_block_ttasmoke.log); full run on es_s0..s6 next if it works.
- 2026-11-13 13:40: TTA smoke promising for pair (+1.4, single enc/seed; agent -1.9). Launched full test: local ww_block --tta es_s0..s6 1 seed (ttaA, log ww_block_ttaA.log) and Kaggle A aegis-cpu-tta (es_s7..s13, 1 seed, ttaB; output via --file-pattern 'ttaB|\.log').
- 2026-11-13 13:46: ttaB (Kaggle A) saved: TTA lifts WW pair 4.73->7.21 (all-on, no shift), AEGIS flat; same level as label-shift, not stacking. ttaA (local es_s0..s6) summary pending.
- 2026-11-13 13:51: ttaA done: all-on pair 6.50->7.98 (AEGIS 33.68->34.41); names+tta+shift 9.60/50.45/16.35 but not replicated in ttaB (7.06/47.64/13.93). TTA ~ label-shift equivalent, no new best. Idle: all GPUs, CPU.
- 2026-11-13 14:00: launched leave-one-framework-out validation of block-dropout masks (new aegis/src/eval/ww_lofo.py; es_s0..s2, 1 seed; log ww_lofo_lofo1.log): which mask (names+peer/names/peer/neither) transfers best to an unseen AEGIS framework, a Who&W-free way to pick the Who&W mask. Held-out frameworks with enough rows: agentverse, dylan, llm_debate, macnet (magentic_one/smoagents tiny).
- 2026-11-13 14:11: LOFO done: all-on best/tied on pair (18.2 vs 17.4-17.7), agent +6-7 with names; no mask justified beyond all-on. Post-hoc names-only rows stay post hoc. Next CPU idea: per-record fault-count correction / de-propagation.
- 2026-11-13 14:20: brainstorm round (ideas.md sec 10): nothing new/cheap; fault-count correction == top1 (already dead). Remaining untested low-value: de-propagation, name-to-role mapping. Queue effectively exhausted; all GPUs+CPU idle.
- 2026-11-13 14:40: launched noise-tightening replicates of TTA (3 head seeds each, same code): Kaggle A aegis-cpu-tta3a (es_s7..s13, tag ttaB3) and Kaggle D aegis-cpu-tta3d (es_s0..s6, tag ttaA3). Save with --file-pattern 'tta[AB]3|\.log' into kaggle_logs/cpu_tta3a, cpu_tta3d. B, C idle, local CPU idle.
- 2026-11-13 14:35: aegis-cpu-tta3d (D) failed in 90 s: D's dataset holds only the supcon/attnsup encoders, not es_s0..s6 (FileNotFoundError es_s0; log saved kaggle_logs/cpu_tta3d). Re-pushed v2 with the 7 supcon/attnsup encoders (wbscb set), tag ttaS3: TTA on the best-joint contrastive ensemble. tta3a (A, es_s7..s13 x3 seeds) still running.
- 2026-11-13 14:56: ttaB3 saved+scored: TTA +1.3 pair no-shift (6.55 vs 5.24), but plain+shift 7.76 > tta; stacking hurts. TTA = weaker alternative to shift. D ttaS3 pending.
- 2026-11-13 15:21: ttaS3 (D) saved+scored: TTA on contrastive ensemble +0.6 pair (noise), AEGIS 36.85 unchanged. TTA closed: stable +1-1.5 pair no-shift on plain ensembles only; shift remains better. All GPUs + CPU idle, queue exhausted.
- 2026-11-13 ~17:20: brainstorm round 11 (ideas.md sec 11): CatchBench, Who&When Pro, AgenTracer; nothing cheap/new to test. Idle: all GPUs, CPU.
- 2026-11-13: wrote local-notes/approach.md (every approach explained with numbers, glossary, pipeline, novelty vs paper sources). Citations from memory; verify before thesis.
- 2026-11-13: loop STOPPED by user request (cron fb04818e deleted). All kernels saved, nothing running. Deliverables: approach.md, results.md, ideas.md.
- 2026-11-14: rr scripts written and smoke-tested (see rr_runbook.md section 1-2). Ledger hygiene: moved 24 smoke/fake rows (tags rr_smoke, rr_fuse_tt_) out of aegis/results/metrics_history.jsonl into aegis/results/rr/smoke_ledger_rows_moved_out.jsonl (backup of full ledger in scratchpad). Nothing running: all GPUs and CPU idle. Next (by the user, at medium effort): runbook section 5 step 1 (rr_models --retr 0, tag rr_base), then Gate 0.
- 2026-11-14: user approved the Who&When agent-name normalisation as a SEPARATE extra row only; implemented in rr_common/rr_models/rr_gate0/rr_fuse (raw-name rows unchanged and stay the headline; normalised rows labelled post-hoc). Tested on tiny/synthetic inputs only.
