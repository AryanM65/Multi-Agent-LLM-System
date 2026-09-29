# Dataset Expansion Guide — Instructions for a Second Generation Run

> Give this file, together with the `btp-pipeline/` folder and `dataset/topology_pool.json` from the original project, to whoever is running the second generation job. If they're using Claude Code (or any coding agent), paste this whole file as the task description — it's written to be followed step by step, either by a human or by an agent.

## 0. What this is for

We're expanding a research dataset for a fault-localization study on multi-agent LLM pipelines. The dataset is a set of JSONL trial records, each one a pipeline run (question → Retriever → Reasoner → Writer, or a more complex graph) with a deliberately injected fault (or none) and measured per-node uncertainty. Full background isn't needed to run this — just follow the steps below.

**Your job**: run the existing generation script on a **disjoint set of questions** (guaranteed not to overlap with any question already used), producing a new trials file that can be safely merged into the main dataset afterward.

**Do not** modify the pipeline code (`src/`), the topology pool (`dataset/topology_pool.json`), or generate into the same log file as any existing dataset. Treat this as an isolated, additive run.

---

## 1. What you should have received

```
btp-pipeline/
  src/                             # pipeline code — do not modify
  scripts/run_study_vllm.py        # the generation script you will run
  scripts/run_study2.py            # imported by run_study_vllm.py, needed as-is
  scripts/verify_results.py        # for checking your own output before sending it back
  scripts/sample_examples.py       # generates your disjoint question set
  scripts/download_data.py         # caches HotpotQA locally
  requirements.txt
  data/
    all_questions_used_so_far.json # every question already used anywhere in the dataset so far
dataset/
  topology_pool.json               # the fixed 20-topology pool — must stay identical to the sender's copy
DATASET_EXPANSION_GUIDE.md         # this file
```

If `data/all_questions_used_so_far.json` is missing, stop and ask for it — it is required for step 3 and must not be regenerated or guessed.

---

## 2. Environment setup

Requirements: a CUDA GPU with **at least 16GB VRAM** (the model is 4-bit quantized but still needs headroom for KV-cache during batched generation). CPU-only will not work for this step.

```bash
cd btp-pipeline
python -m venv btp-env
source btp-env/bin/activate        # Windows: btp-env\Scripts\activate
pip install -r requirements.txt
pip install vllm
python scripts/download_data.py    # downloads ~50MB, caches HotpotQA locally, run once
```

Verify the GPU is visible before doing anything else:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
If this prints `False`, stop — do not proceed to generation on CPU, it will not work in any reasonable time.

---

## 3. Generate your disjoint question set (no GPU needed for this step)

```bash
cd btp-pipeline
python scripts/sample_examples.py \
  --existing data/all_questions_used_so_far.json \
  --add 60 \
  --seed 1337 \
  --out data/friend_new_questions.json
```

- `--add 60` — how many new questions to sample. Change this number if a different amount was agreed on, but do not lower it silently if the script warns it found fewer qualifying questions than requested (see step 3.1).
- `--seed 1337` — use this exact seed unless told otherwise, so the run is reproducible.
- This script **guarantees zero overlap by construction**: it builds the set of questions already in `--existing` and skips any HotpotQA question that matches one of them exactly. You do not need to manually check for duplicates at this step (you will still verify at the end, in step 6).

### 3.1 If the script warns "only found N/60 qualifying new questions"

This means the quality filters (gold-context titles must be present in the retrieved context; total context length capped) rejected too many candidates. In that case:
- Re-run with a larger `--add` value and let it find as many as it can, or
- Report back the actual number obtained rather than proceeding with fewer than expected — don't pad the pool with lower-quality questions to hit a target count.

---

## 4. Run the generation

Recommended: **k=10** self-consistency samples per node (this is the current direction of the project — a k=5 run would need a separate merge path and doesn't help the current effort). Confirm this with whoever sent you this guide before starting if you're unsure.

```bash
cd btp-pipeline
BTP_BACKEND=vllm python scripts/run_study_vllm.py \
  --topology-pool ../dataset/topology_pool.json \
  --examples-json data/friend_new_questions.json \
  --k 10 \
  --batch-size 15 \
  --examples-per-topology-full-grid 4 \
  --examples-per-topology-partial 7 \
  --log-path logs/trials_friend.jsonl \
  --skip-log-path logs/skipped_friend.jsonl \
  --resume
```

**Important — do not change these:**
- `--log-path` / `--skip-log-path` must use the `_friend` (or similarly distinct) names shown above. Never write into a file also used by the original dataset, and never overwrite an existing `trials.jsonl`.
- `--topology-pool` must point at the exact `dataset/topology_pool.json` you were given, unmodified. If this file differs at all from the sender's copy, the two datasets will not merge cleanly (topology_id references would mean different things).

**If you hit an out-of-memory error**: lower `--batch-size` first (try 8, then 4) before changing anything else. Do not change `--k` or the model to work around memory issues — ask first, since that would produce data that isn't directly comparable to the rest of the dataset.

**`--resume` is safe to rely on**: if the run crashes or you need to stop it, re-running the exact same command will skip any topology that already has complete records in `--log-path` and continue from where it left off. It does not redo completed work or duplicate records.

**Expect this to take a while.** Per-topology time scales with how many Reasoner-role nodes run in the same batched round (Reasoner uses a larger token budget), not simple node count — expect roughly 7–20+ minutes per topology, plus a one-time ~8–10 minute model-load cost at the start. With k=10 (double the samples of the original k=5 runs), expect each topology to take meaningfully longer than that range.

---

## 5. Monitor the run properly

Do **not** run this command in a way that buffers all output until the process exits (e.g. don't pipe through something that captures output silently) — a multi-hour run with no visible progress is easy to mistake for a hang. If running as a background job, stream output line-by-line to a log file and check it periodically rather than waiting for the process to exit before looking at anything.

The script prints one line per topology as it starts and finishes, and a final summary line with the total record count when done. If you don't see any new output for more than ~25 minutes on a single topology, something may be stuck — check GPU utilization before killing and restarting.

---

## 6. Verify your output before sending it back (do not skip this)

```bash
cd btp-pipeline
python scripts/verify_results.py --trials logs/trials_friend.jsonl --skipped logs/skipped_friend.jsonl
```

This recomputes record counts, label distributions, and basic structural checks directly from the file. Report the output of this command back along with the data.

**Also do a manual content spot-check** — this project has a history of runs that completed cleanly (no crashes, correct-looking counts) while still containing real bugs that only showed up when someone actually read the generated text. Open `logs/trials_friend.jsonl`, pick 3–4 records with different `true_label` values (`noise`, `contamination`, `ceiling`, `clean`), and read their `samples` field. For a faulted trial, the fault should visibly show up somehow — e.g. a `ceiling` trial's answers should look constrained/degraded compared to a `clean` trial on a similar question. If a fault type's trials look indistinguishable from clean ones, flag this before sending the data back rather than assuming it's fine.

---

## 7. What to send back

1. `logs/trials_friend.jsonl` — the generated trial records.
2. `logs/skipped_friend.jsonl` — the skip log (every skipped trial is logged with a reason; this file matters too, don't discard it).
3. The exact output of `verify_results.py` from step 6.
4. `data/friend_new_questions.json` — the question set you actually used (needed to update the shared "questions used so far" file on merge).
5. Confirmation of: which `src/` commit/version you ran (so we know it matches), which `k` you used, and total wall-clock time taken.

**Do not merge your file into the original `dataset/trials.jsonl` yourself** — send it back as a separate file. Merging, re-verifying the combined file, and updating the shared question-tracking file happens on the receiving end.

---

## 8. Quick troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `torch.cuda.is_available()` returns `False` | No GPU visible to the process — check CUDA drivers/toolkit install, don't proceed until this is `True`. |
| Import error on `vllm` install | Usually a NumPy version conflict — check `pip check` for `numpy<2` requirements from other packages and resolve before retrying, rather than downgrading vllm. |
| Out-of-memory during generation | Lower `--batch-size` (try 8, then 4). Do not change `--k` or switch models to fix this. |
| Run seems stuck / no output for 20+ min on one topology | Check GPU utilization first (`nvidia-smi`) before killing anything — some topologies with several Reasoner nodes in the same round genuinely take 20+ minutes. |
| `sample_examples.py` warns fewer questions found than requested | See step 3.1 — report the actual count, don't pad with lower-quality questions. |
