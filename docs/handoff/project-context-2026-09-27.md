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
- Compared pooling strategies for the graph-level auxiliary head — **attention pooling** currently best
- Tried: multi-task fault-type auxiliary head (neutral/slightly negative), 3-seed ensembling (helped a bit, 0.383, but attention pooling alone beat it at 0.387), various loss functions (label smoothing, focal — modest, not decisive), PCA dimension sweep (128 components did better than 32 or 64 once variance features were added)

**Current best result** (verified directly from `model/metrics_history.jsonl`, run at 2026-09-27 16:35:43): **GraphSAGE, 3 layers, hidden_dim=64, dropout=0.3, lr=5e-4, bidirectional edges, PCA-128 embeddings (mean+std/variance), attention pooling → OOD top-1 accuracy = 0.387**, OOD top-2 = 0.552, OOD macro-F1 ≈ 0.353.

**Comparison table** (OOD test set, n=574, all evaluated identically):

| Approach | OOD top-1 |
|---|---|
| Naive max-uncertainty baseline (no learning) | 0.145 |
| Non-graph RandomForest (flat features) | 0.249 |
| GNN — GAT | 0.275 |
| GNN — GCN | 0.246 |
| GNN — SAGE (base config) | 0.359 |
| **GNN — SAGE + attention pooling (best)** | **0.387** |
| GNN — SAGE + 3-seed ensemble | 0.383 |

This confirms the core thesis claim so far: graph structure adds real value over both a naive per-node rule and a non-graph model, and SAGE-style neighbor aggregation clearly outperforms GAT/GCN on these small (3–7 node) graphs.

**⚠️ Known doc/reality gap**: `docs/model/model.md` §8's experiment log is **stale** — it stops at OOD top-1=0.275 and states "target 0.6" as the number still to beat. The actual `model/metrics_history.jsonl` has 41 more runs since then, reaching 0.387. Update that doc before anyone else reads it, or just treat `metrics_history.jsonl` as the source of truth.

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
