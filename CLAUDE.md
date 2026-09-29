# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project instructions

- Permissions: maximal autonomy for commands (commits, pushes, installs run without asking) AND for file writes/edits (Edit/Write tool run without asking, per explicit user instruction 2026-09-28). Still always ask first for: force push, reset --hard, branch delete, rm -rf, anything touching credentials/secrets files. See .claude/settings.json for exact allow/ask lists.

- **Always keep logs — never let them be ephemeral.** Kaggle kernel logs are NOT readable while a kernel is RUNNING (confirmed repeatedly: `kaggle kernels logs`/`kernels output` both return empty until COMPLETE) and are lost forever if the kernel is deleted before COMPLETE (this cost a full dataset-gen run's worth of data once already). Rules:
  - The moment ANY kernel (or any long-running job) reaches COMPLETE, pull its full log/output and save it into the repo (or another durable location) before doing anything else with it — don't just read it and move on.
  - Never delete/stop a RUNNING kernel unless its output has already been pulled and saved, or the loss is explicitly acceptable and confirmed. If a kernel must be interrupted, prefer batching it into smaller resumable chunks (via `--topology-limit`-style flags or equivalent) specifically so there's always a recent completed checkpoint to fall back on.
  - Per-epoch / per-run training logs are real research artifacts, not scratch files — do not delete them during "cleanup" passes. If unsure whether a log is disposable, keep it or ask first.
  - When generating data or training across multiple parallel jobs, back up each one's output to the repo (e.g. `dataset/k10_partial/`) as soon as it lands, rather than waiting to collect everything at the end.

## What this project is

BTech thesis (BTP-I). A multi-agent LLM pipeline (retriever → reasoner → writer, generalized to arbitrary DAG topologies) is run with a fault deliberately injected at one node. Per-node self-consistency uncertainty is measured over k samples. A GNN then tries to **localize which node was faulted**, from the graph alone — including on **held-out topologies it never trained on** (the OOD split). The headline metric is `ood_top1` accuracy.

Two halves, deliberately separated:

- **`btp-pipeline/`** — data *generation*. Runs the LLM pipeline, injects faults, measures uncertainty, writes JSONL trials. Imports rooted at `btp-pipeline/` (`from src.x import y`), so run its scripts from that directory.
- **`model/`** — everything GNN: feature engineering, training, evaluation, baselines, ablations. Imports rooted at the **repo root** (`python -m model.x`). **All model code goes here, never in `btp-pipeline/src/`.**

`dataset/` holds the generated JSONL trial records plus `topology_pool.json`. `docs/` holds the research write-ups that explain *why* each design choice was made — `docs/model/model.md`, `docs/model/model-plan.md`, and `docs/model/futurework.md` are the ones worth reading before changing model code.

## Commands

Run model code from the **repo root**:

```bash
python -m model.improve_v2 dataset/trials_k10_disc.jsonl   # train a recipe (arg = trials file)
python -m model.data.split                                 # sanity-check split, asserts no question leakage
python -m model.data.build_graph_dataset                   # inspect one built graph
python -m model.data.enrich_discrepancy <in.jsonl> <out.jsonl>   # add discrepancy/length features
python -m model.diagnostics.oracle_probe                   # zero-training feature-signal check
```

Run pipeline code from **`btp-pipeline/`**:

```bash
python -m tests.test_uncertainty                 # the only test file in the repo
python scripts/run_study_vllm.py --mock          # generation, no LLM (deterministic stubs)
python scripts/run_study_vllm.py --topology-limit 5 --resume   # chunked, resumable real run
BTP_MOCK=1 ...                                   # env equivalent of --mock
BTP_BACKEND=ollama|mlx|vllm                      # backend selection (config.py)
```

Kaggle (from repo root, Windows):

```bash
export PYTHONUTF8=1   # kaggle CLI crashes with charmap UnicodeEncodeError otherwise
python -c "import kaggle; a=kaggle.KaggleApi(); a.authenticate(); print(a.kernels_status('<owner>/<slug>'))"
```

## Architecture notes that aren't obvious from one file

**Per-graph variable-length output.** `FaultLocalizerGNN` emits one logit per node plus a "no fault" logit, so a graph with N nodes is a softmax over N+1 classes. Graphs are 3–7 nodes; label spaces are **local to each graph**, never batched into one shared space. Clean/control trials get class index `len(node_ids)`.

**The split is by topology, then by question.** `model/data/split.py`: OOD test = every graph whose topology's `split` field is `"ood_test"` — never touched during training or tuning. Train/val is split by **question**, not by trial, because the same question recurs across many topologies and fault conditions. The `__main__` block asserts zero question overlap; run it after any split change.

**Feature vector layering.** `node_feature_vector()` in `model/data/build_graph_dataset.py` builds, in order: 12 base scalars (3 uncertainty measures + has-flags, `inference_gaps`, `item_frequencies`, role one-hot) → 14 discrepancy scalars (7 fields × value+has-flag, `DISCREPANCY_FIELDS`) → optional 384-dim mean embedding → optional 384-dim embedding std. Every optional scalar uses the same `(value, has_value)` convention, so `in_dim` stays stable whether or not a trial was enriched. PCA-128 is fit on the **train split only** and applied to all graphs (`reduce_embeddings.py`).

**Why the discrepancy features exist.** Every original feature was an *uncertainty* measure, and uncertainty only moves for one of the three fault types. Against the clean controls: contamination `Δsemantic_unc = +0.235`, noise `+0.062`, ceiling **`−0.015` (inverted)** — ceiling truncates a node, and shorter output is *more* self-consistent, so uncertainty falls. `node_len_z` (length vs the node's own clean-control baseline) localizes ceiling at 0.716 where every uncertainty signal scored 0.03–0.28. Read `model/data/enrich_discrepancy.py`'s module docstring before adding features.

**Clean controls are calibration, not labels.** `compute_length_baselines()` derives per-`(topology_id, node_id)` length stats from control trials only. This is legitimate, but it means a deployed model needs clean baseline runs for its own topology — a real operational requirement, flagged deliberately.

**Every run must be logged.** `model/metrics_log.py` appends config + metrics to `model/metrics_history.jsonl`. Training entrypoints call `log_run(...)` at the end. This file is how "did that tweak help?" gets answered — it is the results ledger, not a cache.

**`retriever_c` is never a fault target** in the `star` and `triple_retriever_fanin` topologies. Graphs are still built normally; `KNOWN_UNREACHABLE_TARGETS` in `model/evaluate.py` exists so per-class reports can mask it. Top-1/top-2 accuracy is unaffected.

**Experiment scripts are one-file ablations.** `model/improve_v2.py` … `improve_v9.py`, plus the `*_sweep.py` files, each duplicate the same load → enrich → build → split → PCA → train-N-seeds → ensemble → `log_run` flow with one axis changed, and each carries a docstring stating the hypothesis. Add a new numbered file rather than mutating an old one — the old ones are the record of what was tried.

## Results ledger (as of 2026-09-28)

Best on the full 1785-record dataset: **`ood_top1` = 0.4035** (`final-train`, commit d0ab4d5). `improve_v2` scored 0.438 but on the smaller 1278-record set, so it is not directly comparable. Target is 0.6.

**Do not re-run these — already ruled out:** regularization, GAT conv, PCA-64, focal loss + 7 seeds, 2-layer, hidden_dim=128 (all 0.40–0.44); dataset scale-up 1278→1785 (no gain); dropping embeddings (they help: 0.4035 vs 0.325 scalars-only); within-graph z-score/margin encoding (**hurt** OOD, 0.312→0.276 — plain within-graph *rank* is what shipped).

**Known unfixed issues:** `btp-pipeline/src/diagnose.py` `verify_against_baseline()` still uses a hardcoded `assumed_std=0.10`; the empirical per-node std is 0.165 and 253 controls now exist. `inference_gaps`, `item_frequencies`, `per_node_diagnoses` and `diagnosed_label` are `None`/empty in 100% of records but still fed to the model as `0.0` + a has-flag.

---

Optional: OpenAI Codex (`~/.codex`) and Gemini CLI (`~/.gemini`) configs exist on this machine. Reply `/import` to scan and list what's importable (MCP servers, slash commands, subagents, skills, instructions), then `/import --yes=<digest>` with the digest from the scan output to apply the user-level items.
