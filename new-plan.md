# PLAN.md: Aegis-Bench track (context and instructions for Claude Code)

Written 1 Nov 2026 from a long planning chat. Read this whole file before doing anything.
Tags used below:
- **[VERIFIED]** read directly in the Aegis paper (ICLR 2026, arXiv 2509.14295 v6).
- **[UNVERIFIED]** could not be checked. Verify it in the repo or data and update this file.
- **[DECISION]** a choice that needs the user's input.

---

## 0. One-paragraph summary

The team (Group 121, NSUT: Aryan Mahajan, Yogesh Kumar, Shubham Yadav) has been building a
BTP on fault localization in multi-agent LLM pipelines using self-consistency uncertainty and a
GNN, on a dataset they generated themselves (1,762 trials, 20 topologies). The requirement has
now changed: **they must use an existing dataset that was used in a published paper**, and
compare against **published papers only**. The chosen route is **AEGIS-Bench** (Kong et al.,
ICLR 2026). The task is multi-agent failure attribution: given a failed trajectory, predict
which agents are faulty and which of 14 error modes they committed. The plan is to reproduce
the paper's published baselines, then build a lightweight GNN and compare it with the
paper's small-model baseline (DCL) under the paper's exact protocol.

**Status gate [DECISION]:** the user's guide must approve this change of dataset and scope.
Do not invest in Phase 5 or later until the user confirms approval.

---

## 1. What carries over from the old project, and what does not

Old project (do not modify; keep as supplementary material):
- Repo: `D:\aryan\Projects\Multi agent LLM System`, git branch `model_v1`. Windows machine.
- `btp-pipeline/src/` holds the DAG pipeline executor and fault injection. `model/` holds the
  GNN code and `model/metrics_history.jsonl` (43 logged runs). `dataset/trials.jsonl` has
  1,762 trials.
- Best old result: OOD top-1 fault-node accuracy about 0.395 (naive max-uncertainty 0.145,
  flat RandomForest 0.249). That number was selected using the OOD set, so it is optimistic.
- **Hard rule from the old project: never put model code in `btp-pipeline/src/`.**

Carries over (as habits and code ideas):
- Baselines first (naive, then flat/no-graph, then GNN), so the value of graph structure is measured.
- Ablate one change at a time and log every run in a `metrics_history.jsonl`.
- Fit PCA or other transforms on train only. Split by group (task), not by row.
- Read real samples; do not trust counts alone. Three earlier bugs were found only by reading text.
- PyTorch Geometric code and a GraphSAGE and attention-pooling starting point. These are
  hypotheses from the old data and may not transfer.

Does NOT carry over:
- Self-consistency uncertainty. Aegis trajectories are single deterministic runs (agents are
  GPT-4o-mini at temperature 0 [VERIFIED]), so repeated-sample disagreement cannot be computed.
- The noise, contamination and ceiling taxonomy, and the held-out-topology split. Aegis has its
  own 14 error modes and its own splits.

---

## 2. Published-only rule

Compare only against, and rely on, published (peer-reviewed) work:

| Paper | Venue | Use |
|---|---|---|
| Aegis (Kong et al.) | ICLR 2026 | Dataset, splits, metrics, all baselines |
| Who&When (Zhang et al.) | ICML 2025 | Out-of-distribution test set |
| MAST (Cemri et al.) | NeurIPS 2025 (as listed in the team's references) | Source of the 14 error modes |
| AgenTracer (Zhang et al.) | ICLR 2026 | Optional second dataset (TracerTraj); related work |
| StepFinder (Zhu et al.) | KDD '26 | Related work; sequence-model baselines |
| MATU (Chen et al.) | ACL 2026 | Related work (reliability task, not this track) |
| SAUP (Zhao et al.) | ACL 2025 | Conceptual background |

Preprints. Related work only; never a baseline to beat: AFANet (arXiv 2608.18575), ASCon
(2608.10646), PropUQ-MAS (2608.22130), RUPA (2608.16002), GraphTracer (2510.10581),
VerifyMAS (2605.17467), AgentLocate (2607.07989). AFANet is the only GNN I found on this
data, and it is a preprint. If you cite it, say so.

---

## 3. Verified facts about Aegis-Bench

### 3.1 Dataset [VERIFIED]
- 9,533 trajectories, 24,843 injected error instances (about 2.6 per trajectory), so the task
  is **multi-label**.
- Built by injecting errors into successful runs (an LLM "manipulator" with prompt injection
  or response corruption), keeping only runs that actually fail. Labels are known by construction.
- Six MAS frameworks (trajectory counts): LLM Debate 2,404; MacNet 2,359; AgentVerse 1,995;
  DyLAN 1,845; SmolAgents 481; Magentic-One 449.
  - MacNet supports chain, star, tree and fully connected topologies.
  - DyLAN is a dynamic graph.
  - AgentVerse is hierarchical (solver, critic, evaluator).
  - Magentic-One is orchestrator-executor.
  - SmolAgents is multi-agent ReAct with tools.
- Agents per system (paper Table 3): Debate 3, Magentic-One 5, MacNet 4, AgentVerse 4, DyLAN 4.
  SmolAgents is not shown there. Average injected errors per trajectory: 1.9, 1.0, 3.6, 3.0, 2.9
  respectively. Injection success rate: 76.5%, 54.5%, 51.4%, 43.8%, 34.1%.
- Six tasks: MATH 2,048; SciBench 1,871; GSM8K 1,741; HumanEval 1,497; MMLU(-Pro) 1,446; GAIA 954.
- Label quality: program-vs-human kappa 0.81 against 0.85 human-human on 100 sampled trajectories.
- All agents were GPT-4o-mini at temperature 0, which means deterministic single runs.

### 3.2 Error modes (14, from MAST) [VERIFIED, wording from the paper's evaluation prompt]
- FM-1.x Specification: 1.1 task specification deviation; 1.2 role specification deviation;
  1.3 add redundant steps; 1.4 remove conversation history; 1.5 remove termination conditions.
- FM-2.x Inter-agent misalignment: 2.1 repeat handled tasks; 2.2 make request ambiguous;
  2.3 deviate from main goal; 2.4 hide important information; 2.5 ignore other agents;
  2.6 inconsistent reasoning.
- FM-3.x Verification: 3.1 premature termination; 3.2 remove verification steps; 3.3 incorrect verification.
- Mode counts are roughly balanced, 1.5K to 2.3K each. Macro-F1 still matters.
- Naming differs between the code and the paper (EM-x.y in the composition table, FM-x.y in the
  prompts). Normalize to one scheme in the loader.

### 3.3 Splits [VERIFIED]
- Test set "Aegis-Bench" = 100 trajectories sampled from each of the six task benchmarks
  (600 total). The remainder is split 80% train / 20% validation, with a fixed seed.
- Out-of-distribution test: Who&When (127 trajectories, human-annotated). Its free-text
  `mistake_reason` was mapped to the 14 modes with Gemini-2.5-Flash in the paper.
- **[UNVERIFIED] Leakage risk:** each successful run has several faulty variants. If test
  and train variants share the same base task, scores are inflated. Check this in the audit.
  If it leaks, also build a split grouped by base task and report both.

### 3.4 Metrics [VERIFIED]
- Three levels:
  - **Pair:** the set of (agent, error mode) pairs.
  - **Agent:** the set of faulty agents, ignoring the mode.
  - **Error:** the set of error modes, ignoring the agent.
- Micro-F1 and Macro-F1 at each level. Micro pools counts across samples. Macro averages
  per-class F1 equally.
- "Avg." in the paper's Table 1 is the mean of all 12 numbers (3 levels x micro/macro x 2 benchmarks).
- **[UNVERIFIED]:** which classes macro-F1 uses at agent level. Use the authors' evaluation
  script if it exists, and record the definition in `docs/decisions.md`.

### 3.5 Published reference numbers (paper Table 1, percent) [VERIFIED]
Columns: Aegis-Bench (Pair, Agent, Error: micro then macro F1), then Who&When (same), then Avg.

| Model | AB Pair µ/M | AB Agent µ/M | AB Error µ/M | W&W Pair µ/M | W&W Agent µ/M | W&W Error µ/M | Avg |
|---|---|---|---|---|---|---|---|
| Random | 0.33/0.21 | 4.54/3.56 | 11.23/11.15 | 0.11/0.05 | 1.06/0.83 | 8.74/7.14 | 4.08 |
| DCL (small, ours to beat) | 8.33/5.30 | 22.93/20.23 | 24.73/27.70 | 1.60/0.77 | 8.40/6.07 | 14.67/10.57 | 12.61 |
| Qwen2.5-7B-Instruct (zero-shot) | 5.02/2.52 | 27.55/14.49 | 14.96/11.36 | 2.31/1.14 | 40.92/23.50 | 3.64/1.77 | 12.43 |
| Qwen2.5-14B-Instruct (zero-shot) | 5.47/2.20 | 35.78/12.71 | 20.24/5.91 | 0.00/0.00 | 49.88/33.19 | 1.56/1.35 | 13.99 |
| Aegis-SFT (14B fine-tuned) | 16.62/9.99 | 76.53/47.97 | 27.53/27.66 | 4.03/2.08 | 51.14/36.94 | 9.87/7.77 | 26.51 |
| Aegis-GRPO (14B) | 6.84/2.55 | 49.74/18.38 | 21.19/16.10 | 2.45/1.49 | 54.43/40.88 | 4.15/2.67 | 18.41 |
| o3 (zero-shot) | 7.86/2.27 | 40.31/23.27 | 22.37/16.76 | 7.41/3.98 | 53.10/42.55 | 14.88/8.63 | 20.24 |

DCL over 3 seeds (paper Table 10, mean ± std):
- Aegis-Bench: Pair µF1 8.33 ± 1.84, Pair MF1 5.30 ± 0.91; Agent µF1 22.93 ± 0.78, Agent MF1 20.23 ± 0.69;
  Error µF1 24.73 ± 0.59, Error MF1 27.70 ± 1.04.
- Who&W: Pair µF1 1.60 ± 0.95; Agent µF1 8.40 ± 1.40; Error µF1 14.67 ± 1.27.

### 3.6 DCL (the published lightweight baseline) [VERIFIED]
- The model:
  - Encoder: all-MiniLM-L6-v2 (about 23–35M parameters in total), trained end to end.
  - A trajectory is a **bag of turns**. Multiple-instance attention picks the top-K=3 evidence turns.
  - Turn representations are aligned with two learnable prototype banks, one for agents and one
    for error modes. They are initialized from text definitions.
  - A pair head combines the two (product/bilinear).
  - **No graph and no edges.**
- Loss: asymmetric BCE (classification at agent, error and pair level), supervised contrastive
  loss (temperature 0.07), and hierarchical consistency (a pair's probability must not exceed
  min(agent probability, error probability), as a squared hinge).
- Hyperparameters (paper Table 7): AdamW, lr 1e-4, cosine schedule with linear warmup (ratio 0.1),
  weight decay 0.01, batch 128, 2 epochs of 500 steps, gradient clip 1.0, bf16, encoder dim 384,
  projection dim 128, bilinear rank 64, gate strength 1.0, loss weights λcls 1.0, λcon 0.4, λhier 0.6,
  3 seeds.
- Training used 4 GPUs (the paper's main text and appendix disagree on the model: 4090 vs 3090).
- Aegis-SFT and GRPO need much more compute (4 x A800; 7B–14B models). **Do not rerun them.**
  Quote their numbers.

### 3.7 Training-protocol lesson from the paper [VERIFIED]
In-domain scores kept rising across SFT epochs, but Who&When peaked at 2 epochs and then fell.
Use early stopping on validation. Do not tune on the test set.

---

## 4. What is NOT verified (resolve in Phase 1)

I could not open the GitHub repo (`github.com/kfq20/AEGIS`) from the planning chat.
1. Which scripts exist: data loader, evaluation script, DCL training code, checkpoints.
   The paper says it releases all code, data and models. The only usage example I saw (on the
   dataset card) is for **generating** error data, not training.
2. The record schema. Only the field names were seen: `id`, `metadata`, `input`, `output`,
   `ground_truth`. One example id: `HumanEval_gpt-4o-mini_agentverse_1_af64acb7`, which seems
   to encode benchmark, model, framework, index and a hash.
3. Whether the communication topology (edges) is stored.
4. How `input` stores the conversation (a string, or a list of turns with agent names).
5. Whether a processed Who&When (already mapped to the 14 modes) is distributed.
6. Whether `metadata` contains injection information (a leakage risk; see rule 2).
7. Whether train and test share base tasks (section 3.3).

---

## 5. Operating rules for Claude Code

1. **Isolation.** Do not edit `btp-pipeline/src/`, existing `model/` code, `dataset/` or old
   results. Create the new work under `aegis/` on a new branch `aegis_v1` (branch from `model_v1`).
2. **No label leakage.** `ground_truth` and any `metadata` field that reveals injected errors,
   plans or flags must never be a model input. Audit `metadata` keys first. Treat unknown keys as
   forbidden until reviewed.
3. **Test-set discipline.** Tune only on validation. Touch the 600-trajectory test set only for
   reproducing published rows (no tuning) and for final reporting.
4. **Log every run** to `aegis/results/metrics_history.jsonl` (schema in section 9).
5. **Read real samples**, not only statistics, at each stage (loader, turn parsing, labels, predictions).
6. **One change at a time** in ablations.
7. **Naming honesty.** Anything that simplifies a published method is named "X-style". Never
   claim a reproduction unless it matches the published number within its spread.
8. **Ask before** spending money or using a paid API, downloading anything over about 5 GB,
   or committing data files. Keep data and embedding caches out of git.
9. **Windows first.** Use `pathlib`, no bash-only scripts. Provide `.py` entry points. The GNN
   trains on CPU. vLLM runs only on the Kaggle notebooks.
10. **Update this file** when an [UNVERIFIED] item is resolved, and put each decision in
    `aegis/docs/decisions.md` with its date and reason.

---

## 6. Compute plan

- **Local Windows CPU:** data loading, auditing, GNN training on cached embeddings, evaluation.
- **Kaggle T4 (free):** (a) one-time MiniLM embedding of all turns, cached to disk (faster than CPU);
  (b) the zero-shot Qwen2.5-7B-Instruct reproduction using the existing vLLM + AWQ setup;
  (c) DCL training if its code is available (MiniLM-sized; I estimate it fits a T4 but have not tested it).
- Embedding cache: key on a hash of the turn text; store as `.npy` or parquet; never recompute.
- Note: DCL fine-tunes its encoder end to end. A frozen-encoder GNN may underperform it.
  Unfreezing the last encoder layers is an optional ablation, run on Kaggle.
- Context length: the paper trained on 8,192 tokens and evaluated Who&When at 32,768.
  A 7B-AWQ model on a T4 may need a reduced maximum length. Measure the trajectory token length
  distribution in the audit and record how many exceed the limit.

---

## 7. Directory layout (proposed)

```
aegis/
  README.md
  configs/                 # yaml, one per experiment
  data/                    # gitignored: raw/, cache/ (embeddings)
  src/
    schema.py              # record dataclasses; error-mode normalization
    loaders.py             # HF loading, split handling
    turns.py               # turn segmentation, agent-name normalization
    embed.py               # MiniLM embedding with cache
    graphs.py              # graph construction (PyG Data objects)
    metrics.py             # pair/agent/error micro+macro F1 (+ unit tests)
    baselines/             # random.py, frequency.py, flat_mlp.py, zeroshot_llm.py, dcl.py
    gnn/                   # model.py, train.py, ablate.py
    eval/                  # evaluate.py, bootstrap.py, breakdowns.py
  scripts/                 # Kaggle runner scripts (vLLM, embedding)
  results/                 # metrics_history.jsonl, predictions/, figures/
  docs/                    # dataset_audit.md, reproduction_log.md, decisions.md
  tests/
```

---

## 8. Phases, tasks, acceptance criteria

Time estimates are rough guesses, not commitments.

### Phase 0: Setup (1–2 days)
- Create branch `aegis_v1`, the `aegis/` tree and `requirements-aegis.txt`
  (torch CPU, torch_geometric, sentence-transformers, scikit-learn, pandas, numpy, datasets,
  pyyaml, tqdm, pytest; vllm only in Kaggle scripts).
- Seeds: a single `set_seed()` helper used everywhere.
- **Done when:** `pytest` runs and an empty `metrics_history.jsonl` writer works.

### Phase 1: Data acquisition and audit (3–5 days)
Tasks:
1. Download `Fancylalala/AEGIS` from Hugging Face. Clone `kfq20/AEGIS` into `aegis/third_party/` (gitignored).
2. Resolve every item in section 4 and write the answers into `docs/dataset_audit.md`.
3. Report: records per split; per framework, per task, per error mode; faulty agents per
   trajectory; number of agents and turns per trajectory; token-length distribution; agent-name formats per framework.
4. Print and read at least 5 raw trajectories per framework next to their `ground_truth`.
   Note any label that looks wrong.
5. Leakage checks: `metadata` keys; base-task overlap between train/val and test; any text in the
   log that literally reveals the injection (e.g., an injected-instruction string).
6. Decide the graph construction (section 8, Phase 5) based on whether edges exist.

**Done when:** `docs/dataset_audit.md` answers every item in section 4 and a loader returns
`(trajectory_id, framework, task, turns[{agent, text}], labels{(agent, mode)})` for all splits.
**Decision gates [DECISION]:** if turns cannot be parsed, or no per-agent labels exist, stop and
report to the user.

### Phase 2: Evaluation harness (3–4 days)
1. Prefer the authors' evaluation script. If it is missing, implement `metrics.py` as defined in
   section 3.4, with unit tests on hand-built examples (perfect, empty, partial, duplicates).
2. Prediction format: JSON mapping trajectory id to a list of `{agent_name, error_type}`.
   Normalize agent names (case, whitespace) consistently.
3. Sanity check: run a random predictor and compare with the paper's Random row. Use the same
   random definition as the paper if the code shows it; otherwise note the definition you used.

**Done when:** the metric code reproduces the Random row to within sampling noise, or the
difference is explained in `docs/decisions.md`.

### Phase 3: Reproduce published rows (1–2 weeks)
3a. **Zero-shot Qwen2.5-7B-Instruct on the 600 test trajectories.**
- Use the paper's Appendix E.3 standard evaluation prompt (copy the text exactly). Output is a JSON
  object of faulty agents and error codes.
- Run with vLLM on Kaggle (AWQ, temperature 0). Parse robustly; count unparseable outputs as empty and report that rate.
- Compare with the paper's row: AB Pair µF1 5.02, Agent 27.55, Error 14.96.
- The AWQ model differs from the paper's full-precision model, so a few points of difference is
  plausible. This tolerance is a guess; explain any larger gap.

3b. **DCL.**
- If training code exists: run with the section 3.6 hyperparameters, 3 seeds.
- If only checkpoints exist: evaluate them.
- If neither exists: re-implement from Appendix C and Table 7 and call it **DCL-style**. Document
  every guess, such as how the agent prototypes are created from text definitions.
- Target: within about one to two standard deviations of the published mean (section 3.5).

3c. Do not rerun Aegis-SFT, GRPO or the proprietary models. Quote them.

**Done when:** `docs/reproduction_log.md` lists each row with published vs reproduced values and
a verdict ("matched", "close", "not matched, because ...").

### Phase 4: Own simple baselines (3–5 days)
1. **Frequency baseline:** always predict the most common modes and agent roles.
2. **Flat model (no graph):** for each agent, mean-pool its turn embeddings, then logistic regression
   or a small MLP with 14 outputs, plus an agent-faulty head. This is the analogue of the
   old RandomForest: it shows whether graph structure helps.
3. Evaluate on validation, then report on test once.

**Done when:** all baselines have validation numbers in `metrics_history.jsonl`.

### Phase 5: The GNN (2–3 weeks)
**Graph options (choose in Phase 1, record in decisions.md):**
- **A. Agent graph.** Nodes are agents; edges come from the stored framework topology
  (preferred if available [UNVERIFIED]). Node features come from pooled turns.
- **B. Turn graph (fallback).** Nodes are turns; edges link consecutive turns (both directions)
  and turns by the same agent; pool turns to agents for prediction. This is the AFANet-style construction.
- Report which one was used and why.

**Features:** MiniLM turn embeddings (frozen, cached) optionally PCA-compressed (fit on train
only), role or agent-type one-hot, normalized turn position, turn length. Optional,
flagged as experimental: teacher-forced token entropy from a local Qwen model on each turn's text.
This measures surprise under Qwen, not the generating model's uncertainty. Label it that way.

**Targets:** per agent, a faulty-agent logit and 14 error-mode logits (multi-label sigmoid).
Pair score = agent probability x mode probability, or the direct mode logits for that agent.
Agent-level prediction = faulty-agent head; error-level = union of modes over predicted-faulty agents.

**Loss:** BCE or asymmetric BCE; optionally the DCL hierarchical-consistency term
(pair <= min(agent, error)). Class imbalance: macro-F1 matters, so test class weighting.
**Thresholds:** tune per head on validation only.

**Ablation order (one change at a time; each logged):**
1. Flat baseline (Phase 4) is the floor.
2. GNN forward edges, then bidirectional edges.
3. Convolution type: SAGE vs GAT vs GCN. (SAGE won on the old data; test it, do not assume it.)
4. Readout/pooling: mean vs attention.
5. Depth (2 vs 3 layers; the old project saw over-smoothing at 3) and hidden size (start 64).
6. Embedding compression: none vs PCA vs learned linear projection.
7. Loss variants and the consistency term.
8. Optional features (entropy, framework one-hot).
9. Optional: unfreeze the last encoder layers (Kaggle).

Starting config: GraphSAGE, 2 layers, hidden 64, dropout 0.3, AdamW lr 1e-3, early stopping on
validation macro-F1 with patience 10 (starting values, not tuned).

**Done when:** a best configuration is chosen on validation only, and its config file is frozen in git
before any test-set evaluation.

### Phase 6: Final evaluation and analysis (1–2 weeks)
1. Run the frozen config with at least 3 seeds (5 preferred). Report mean ± std in the same format
   as the paper's Table 1 (12 numbers plus Avg).
2. Compare with DCL (paired by seed and with bootstrap confidence intervals resampled by base
   task, or by trajectory if no task id exists), and cite the published SFT/GRPO/LLM rows as references.
3. Breakdowns: per framework, per task, per error mode, by number of agents, by number of faulty agents.
   Include the two frameworks the paper found hard (DyLAN, MacNet).
4. Out-of-distribution: Who&When.
   - Agent-level needs only the `mistake_agent` label. That is the safest metric.
   - Pair and error level need the 14-mode mapping. If a processed file is not released, either use a
     local Qwen model to map `mistake_reason` (state this deviates from the paper's Gemini-2.5-Flash) or report agent-level only.
5. Extra generalization test: leave-one-framework-out, if framework ids are available.
6. Cost accounting: trainable parameters, training time, inference time per trajectory versus the LLM baselines.
7. Error analysis: where the GNN fails, with examples.

**Done when:** results are in `results/`, with figures and a short `docs/results_summary.md`.

### Phase 7: Write-up (1–2 weeks)
- Rewrite the report around this track: new title [DECISION], Introduction, Related Work
  (published vs preprint labeled), Dataset, Method, Experiments, Results, Limitations, Future Work.
- Update the slide deck and the timeline slide.
- Keep the old pipeline as a "supplementary controlled study" only if the guide allows it.
- Update the reference list (statuses in section 2).

---

## 9. Experiment log schema (`results/metrics_history.jsonl`, one JSON per line)

```json
{
  "run_id": "2026-11-05T10:30:00_flat_mlp_seed1",
  "git_commit": "abc1234",
  "phase": "4",
  "model": "flat_mlp",
  "config": {"...": "full config dict"},
  "config_hash": "sha1",
  "seed": 1,
  "split": "val",
  "metrics": {
    "aegis_bench": {
      "pair":  {"micro_f1": 0.0, "macro_f1": 0.0, "micro_p": 0.0, "micro_r": 0.0},
      "agent": {"micro_f1": 0.0, "macro_f1": 0.0},
      "error": {"micro_f1": 0.0, "macro_f1": 0.0}
    },
    "who_and_when": null,
    "avg12": null
  },
  "n_params": 0,
  "train_seconds": 0,
  "notes": ""
}
```

---

## 10. Allowed and disallowed claims

Allowed (if the numbers support them):
- "Our lightweight GNN reaches X avg on Aegis-Bench, compared with DCL's 12.61 under the same split and metrics."
- "Graph structure helps/does not help, per the flat-baseline comparison."
- "Reproduced row R within tolerance."

Not allowed:
- "We beat Aegis-SFT" unless it is literally true on the same metrics. (Unlikely at agent level: 76.53.)
- "First GNN for failure attribution." Use "to our knowledge", and mention the AFANet preprint.
- Comparing any score against numbers from a different dataset or metric.
- Presenting a DCL-style re-implementation as DCL.

Realistic target: the best small-scale model (above DCL on the average and at agent and error level), at a tiny
fraction of the cost of the fine-tuned LLMs. Pair-level scores are low and noisy for every method, so
rely on seeds there.

---

## 11. Risks

- Dataset approval from the guide (section 0).
- Repo may lack training code or metrics, so reproduction effort could grow (3b fallback).
- Train/test leakage through shared base tasks would inflate everything. Check early.
- A frozen-encoder GNN may not beat DCL's end-to-end encoder. Plan the unfreezing ablation.
- Who&When labels need an LLM mapping that may not match the paper's.
- Long trajectories may exceed a T4 context budget for the zero-shot reproduction.
- The crowded area: other groups work on the same benchmark. Keep claims modest.

---

## 12. Open questions for the user [DECISION]

1. Has the guide approved switching to Aegis-Bench? Which base paper is registered for the project?
2. Keep the old pipeline and dataset as a supplementary section, or drop it?
3. New project title? (A suggestion for discussion: "Lightweight Graph Neural Networks for
   Failure Attribution in LLM Multi-Agent Systems"; the old title no longer fits.)
4. Remaining Kaggle GPU quota, and is any paid API allowed (it would only be needed to mimic the paper's Who&When label mapping)?
5. Deadline for the next evaluation.

---

## 13. First session checklist for Claude Code

1. Read this file; confirm the repo path and branch; create `aegis_v1`.
2. Create the directory tree and `requirements-aegis.txt`; add `aegis/data/` and `aegis/third_party/` to `.gitignore`.
3. Download the dataset and clone the Aegis repo; list the repo's files and report what exists (loader, eval, DCL, checkpoints).
4. Load one record per framework; print the raw fields; write the schema findings into `docs/dataset_audit.md`.
5. Report back to the user with the answers to section 4 before building anything else.