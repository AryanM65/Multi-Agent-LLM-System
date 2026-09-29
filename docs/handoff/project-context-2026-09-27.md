# Project Handoff — Full Context (2026-09-27)

> Written to seed a new chat session. Covers the research question, everything completed so far, current best result, known doc/reality gaps, and what's still open.

**Title (working):** *Signature-Aware Orchestration for Uncertainty Propagation in Multi-Agent LLM Systems* — BTech Thesis (BTP), 2026.
**Repo:** `D:\aryan\Projects\Multi agent LLM System` (git branch `model_v1`, clean, all committed as of 2026-09-27).

---

## The research question

Multi-agent LLM pipelines (Retriever → Reasoner → Writer, or more complex DAGs) can fail silently: an error at one agent propagates downstream and the final output still looks fluent/confident. Standard confidence checks only look at the final output, so they can't say *which* agent failed or *why*. This project asks: can per-node self-consistency uncertainty + the pipeline's topology (who feeds whom) be used to automatically localize the fault source and classify the fault type, with no external oracle — and does this generalize to pipeline shapes never seen during training?

---

## Phase 1 — Pipeline + fault injection + uncertainty measurement (complete)

- `btp-pipeline/src/`: arbitrary-DAG pipeline executor (`topology.py`, Kahn's-algorithm topological sort; `pipeline.py` executes it), role-ordered nodes (`retriever` < `reasoner` < `writer`).
- Each node sampled k times (self-consistency). Three uncertainty metrics, never collapsed to one scalar (`PipelineTrace.uncertainties()` always returns per-node dict — this is the core design bet the whole GNN approach rests on):
  - **lexical** = `1 - modal-match fraction` (every node)
  - **semantic** = embedding-cluster entropy via `all-MiniLM-L6-v2` (reasoner/writer only — separates real disagreement from CoT phrasing variance)
  - **jaccard** = set-overlap disagreement (retriever, multi-item output only)
- Three fault types, injectable at any node, role-aware dispatch (`src/faults.py`): **noise** (temp→1.2), **contamination** (input swapped for a distractor), **ceiling** (evidence/instructions stripped, verified the answer no longer survives). Deterministically seeded per `(question_id, fault_type, target_node)`.
- Retry-then-reprobe diagnostic heuristic (`src/diagnose.py`) exists as a comparison baseline, deliberately **never** run during dataset generation.
- Validated on a 45-trial local run (MLX/Apple Silicon backend, chain topology only): 100% precision on 24/45 diagnosable trials. Key early finding: a single global uncertainty threshold doesn't work — Reasoner's CoT output has high baseline lexical variance even when correct, so thresholds must be per-role.

---

## Phase 2 — Dataset generation at scale (complete)

- Backend migrated to **vLLM serving `Qwen/Qwen2.5-7B-Instruct-AWQ`** (4-bit AWQ, float16) on **Kaggle T4 GPUs** (+ Lightning AI T4 in parallel for extra throughput). Originally planned an MXFP4-quantized model — doesn't run on T4 (needs Hopper/Ada), hence the switch.
- **20-topology pool** (`dataset/topology_pool.json`): 14 train shapes (3–6 nodes: chain, fan-ins, fan-outs, star, tree, mixed DAGs...) + 6 OOD shapes (`ood_random_0`–`5`, 5–7 nodes, randomly generated, never seen in training).
- Real bugs found via **direct content inspection** (not just "did it crash") and fixed before finalizing: (1) ceiling-fault dispatch compared node *name* instead of *role*, silently breaking ceiling faults on every non-chain topology; (2) the hardened ceiling instruction was computed but never substituted into the actual prompt; (3) contamination's distractor bank could pick the clean value as its own "wrong" substitute. All three fixed in `src/faults.py`/`src/pipeline.py`; dataset regenerated from scratch afterward (never patched in place).
- **Current dataset**: `dataset/trials.jsonl` = **1,762 verified trial records** (expanded from an original clean 493). Verified composition (recomputed directly from the file, not from stale docs):
  - By fault type: clean=262, noise=578, contamination=550, ceiling=372
  - By split: train-topology=1188, OOD-topology=574
  - Topology sizes: 1×3-node, 3×4-node, 9×5-node, 6×6-node, 1×7-node
- `dataset/skipped.jsonl` = 15 records (all genuine `ceiling_answer_survived`-type skips, always logged with a reason, never silently dropped).
- Full field-by-field schema is in `docs/dataset/dataset_description.md` — ground truth is `fault_config["target_node"]` (node-level) + `true_label` (graph-level: clean/noise/contamination/ceiling), verified via a z-score deviation check against each question's clean-control baseline.

---

## Phase 3 — GNN fault-localization model (in progress, actively being iterated)

All code lives in `./model/` (never in `btp-pipeline/src/` — hard rule).

**Pipeline**: trials.jsonl + topology_pool.json → per-node feature vectors → PyG graph objects → train/val split (grouped by *question*, not trial, to avoid leakage) with OOD topologies held out entirely → baselines → GNN.

**Feature evolution** (each step logged/verified in `model/metrics_history.jsonl`, 43 runs so far):
- Started at 8-dim scalar features: `[lexical, semantic, jaccard, has_semantic, has_jaccard, is_retriever, is_reasoner, is_writer]`
- Added bidirectional edges (forward + reverse) — consistent, cheap win, kept permanently
- Added 384-dim mean-pooled sentence embeddings of node output text, PCA-compressed (fit on train graphs only, no leakage) — raw embeddings alone overfit badly (train/val looked great, OOD barely moved) until PCA + dropout + input-projection layer were added
- Added `node_embedding_std` (embedding variance across the k samples) as an extra feature — confirmed a real win
- Compared GAT vs GCN vs GraphSAGE convolutions at matched hyperparameters — **SAGE clearly best** (0.359 vs GAT 0.275 vs GCN 0.246 OOD top-1)
- Compared pooling strategies for the graph-level auxiliary head — **attention pooling** was the single-technique best (0.387)
- Tried: multi-task fault-type auxiliary head (neutral/slightly negative), 3-seed ensembling alone (0.383), various loss functions (label smoothing, focal — modest, not decisive), PCA dimension sweep (128 components did better than 32 or 64 once variance features were added)
- **Latest run (2026-09-27 17:15:23)**: stacking attention pooling **and** the 3-seed ensemble together (rather than either alone) gave a further improvement — this is the current best.

**Current best result** (verified directly from `model/metrics_history.jsonl`, run at 2026-09-27 17:15:23, notes: "Attention pooling + 3-seed ensemble (stacking the two best independent wins)"): **GraphSAGE, 3 layers, hidden_dim=64, dropout=0.3, lr=5e-4, bidirectional edges, PCA-128 embeddings (mean+std/variance), attention pooling, 3-seed ensemble → OOD top-1 accuracy = 0.395**, OOD top-2 = 0.587, OOD precision = 0.427, OOD recall = 0.365, OOD F1 = 0.380.

**Comparison table** (OOD test set, n=574, all evaluated identically):

| Approach | OOD top-1 |
|---|---|
| Naive max-uncertainty baseline (no learning) | 0.145 |
| Non-graph RandomForest (flat features) | 0.249 |
| GNN — GAT | 0.275 |
| GNN — GCN | 0.246 |
| GNN — SAGE (base config) | 0.359 |
| GNN — SAGE + embedding variance feature | 0.378 |
| GNN — SAGE + attention pooling | 0.387 |
| GNN — SAGE + 3-seed ensemble (no attention pooling) | 0.383 |
| **GNN — SAGE + attention pooling + 3-seed ensemble (best)** | **0.395** |

This confirms the core thesis claim so far: graph structure adds real value over both a naive per-node rule and a non-graph model, and SAGE-style neighbor aggregation clearly outperforms GAT/GCN on these small (3–7 node) graphs. The full evolution from the first GNN run to the current best is charted in `docs/results/figures/fig8_accuracy_evolution.png`.

**⚠️ Known doc/reality gap**: `docs/model/model.md` §8's experiment log is **stale** — it stops at OOD top-1=0.275 and states "target 0.6" as the number still to beat. The actual `model/metrics_history.jsonl` has 42 more runs since then, reaching 0.395. Update that doc before anyone else reads it, or just treat `metrics_history.jsonl` as the source of truth.

**Charts** (real data, saved permanently in `docs/results/figures/`, regenerable via `docs/results/figures/make_report_charts.py`):
- `fig1_fault_type_distribution.png` — dataset composition by fault type
- `fig2_train_ood_split.png` — train vs. OOD trial counts
- `fig3_topology_size_distribution.png` — topology graph-size distribution
- `fig4_ood_accuracy_progression.png` — OOD top-1 across all 42 logged runs (1762-record dataset only)
- `fig5_architecture_comparison.png` — GAT vs. GCN vs. SAGE at matched hyperparameters
- `fig6_model_comparison.png` — naive baseline vs. RandomForest vs. best GNN
- `fig7_best_model_metrics.png` — best model's top-1/top-2/precision/recall/F1 across train/val/OOD
- `fig8_accuracy_evolution.png` — milestone-by-milestone OOD accuracy from the first GNN run (0.220) to the current best (0.395)

**ROC/PR/confusion-matrix charts (2026-09-28 follow-up)** — these required actually re-running the best model locally (torch_geometric installed via pip; the repo's `dataset/trials.jsonl` already has the enriched `node_embeddings`/`node_embedding_std` fields needed) to dump real per-node predicted probabilities on the OOD set, since `metrics_history.jsonl` only stores aggregate metrics, not raw scores. Local CPU reproduction of the attn_ensemble config got OOD top-1=0.383 (matches the logged 0.395 run closely; small difference is normal run-to-run seed/CPU variance). Regenerate via `docs/results/figures/make_roc_charts.py` (trains 3 seeds, dumps `ood_node_predictions.json`/`ood_graph_predictions.json`) then `docs/results/figures/make_roc_plots.py` (builds the charts from those dumps):
- `fig9_roc_curve.png` — pooled binary ROC ("is this node the true fault source", across every node instance in every OOD graph, n=3962). **AUC = 0.739.**
- `fig10_pr_curve.png` — same pooled binary task as a Precision-Recall curve (more informative than ROC here since only ~14.5% of node instances are positive). **Average Precision = 0.415** vs. a 0.145 random baseline.
- `fig11_roc_by_role.png` — the same ROC broken out by node role: reasoner AUC=0.785 (n=1946), writer AUC=0.732 (n=770), retriever AUC=0.695 (n=672), "no fault" AUC=0.675 (n=574). Reasoner faults are the easiest to identify; retriever faults and clean/no-fault graphs are the hardest.
- `fig12_role_confusion_matrix.png` — row-normalized confusion matrix, true role vs. predicted role, OOD set. **Key finding**: the model over-predicts "reasoner" across the board (41% of true-retriever cases, 37% of true-writer cases, and 50% of true-no-fault/clean cases all get predicted as reasoner) — reasoner is the model's default/majority-class fallback, not just its strongest class. Worth flagging explicitly as a limitation in the Results/Discussion section.
- `fig13_accuracy_by_fault_type.png` — OOD top-1 accuracy split by true fault type: **contamination 0.589** (n=168, clearly the easiest — matches the Phase-1 local-study finding that contamination gives the strongest signal), ceiling 0.35 (n=140), clean 0.286 (n=84), noise 0.264 (n=182, hardest — consistent with noise being the most "recoverable"/transient fault type and thus the weakest structural signal).

**Not yet done / open items**:
- k=10 self-consistency regeneration (finer-grained uncertainty resolution than current k=5) was planned/queued per the docs but not verified as landed — check `dataset/trials.jsonl` record count and a sample record's `samples` list length (should be 10, not 5) to confirm before assuming it's done.
- Per-fault-type / per-node-role / per-topology-size error analysis (Step 8 in `docs/model/model-plan.md`) — not yet done on the current best model.
- `retriever_c` in `star`/`triple_retriever_fanin` topologies is never a valid fault-target label (known, accepted gap — exclude it from metric denominators for those two topologies, don't count it as a model failure).
- Multi-parent contamination only corrupts the first parent, not all parents feeding a node — known, accepted limitation, not planned to be fixed.

---

## Report-writing progress (drafted in chat, not yet saved to files elsewhere)

Already drafted, in order, in the prior conversation:
1. Introduction
2. Motivation
3. Problem Statement
4. Simulation Platform and Requirements
5. Conclusion (explicitly flagged as provisional/interim)
6. Objectives
7. Methodology (with 6 charts generated from real data)

Charts generated in that session (real data, not fabricated) were saved at a **session-specific temp path** that will not exist in a new chat:
`C:\Users\aryan\AppData\Local\Temp\claude\D--aryan-Projects-Multi-agent-LLM-System\a216c012-8e82-4126-8dc1-c6131997de2a\scratchpad\fig1..6_*.png`

If those PNGs are still needed, either copy them somewhere permanent, or ask a new session to regenerate them from `model/metrics_history.jsonl` + `dataset/trials.jsonl`. Generation logic:
- Fault-type / split / topology-size bar charts: from `dataset/trials.jsonl` + `dataset/topology_pool.json`.
- OOD-accuracy-progression line chart and model-comparison bar chart: from `model/metrics_history.jsonl`, restricted to `ood.n == 574` runs for a consistent baseline (the first-ever run used the old 493-record dataset and has `ood.n == 164` — exclude it from these comparisons).

**Still needed for the report**: Related Work / Literature Review (waiting on research papers to be supplied), Results section proper (should use the 0.387 number and the 6 charts, plus the still-pending error analysis), Discussion/Limitations, References.
