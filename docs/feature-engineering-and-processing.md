# Feature Engineering and Data Processing — k=5 Dataset Era

> Scope note: this covers exactly how one raw trial record was turned into a GNN training example **for the finished k=5, 1762-record dataset and its best result (OOD top-1 = 0.395)** — the code as it existed at commit `2880d27` (`attn_ensemble.py`, the run that combined SAGE + attention pooling + 3-seed ensemble). It deliberately stops before the later feature-engineering work done for the k=10 dataset track (discrepancy features, edge-based features, structural/topological features, calibrated z-scores, directed dual-edge message passing, mixture heads) — none of that existed yet at this point and is out of scope here. Written 2026-09-29.

---

## 1. The processing pipeline, end to end (k=5 era)

```
dataset/trials.jsonl  (raw generation output: uncertainties, samples, fault_config, ...)
        │
        ▼
enrich_features.py      → inference_gaps, item_frequencies, node_embeddings, node_embedding_std
        │                  (all 4 ARE consumed as model features at this stage — see §2, §3)
        ▼
build_graph_dataset.py  → one PyG Data object per trial: node feature matrix `x`,
        │                  `edge_index` (forward-only or bidirectional-merged), label `y`
        ▼
reduce_embeddings.py    → PCA-compresses the raw 384-dim embedding columns
        │                  (fit on TRAIN graphs only, applied to all)
        ▼
model/gnn.py's FaultLocalizerGNN.forward(x, edge_index)
```

Enrichment is additive — `enrich_features.py` adds new dict fields to each trial record without touching the original generation fields. `build_graph_dataset.py` handles a not-yet-enriched trial gracefully (see §4's "value + has-flag" convention), so nothing breaks if a trial hasn't been through `enrich_features.py` yet.

---

## 2. The per-node scalar feature vector — 12 dimensions

This is the entire scalar (non-embedding) portion of a node's feature vector at this stage of the project, built by `node_feature_vector()` in `model/data/build_graph_dataset.py`:

```
[lexical_unc, semantic_unc, jaccard_unc, has_semantic, has_jaccard,
 inference_gap, has_inference_gap, item_frequency, has_item_frequency,
 is_retriever, is_reasoner, is_writer]
```

| # | Feature | Source field | Present for | Notes |
|---|---|---|---|---|
| 1 | `lexical_uncertainty` | `trial["uncertainties"][node_id]` | every node | `1 − (fraction of k samples matching the modal normalized answer)`. Defaults to `0.0` if missing. |
| 2 | `semantic_uncertainty` | `trial["semantic_uncertainties"][node_id]` | reasoner, writer | Entropy over embedding-cluster assignment of the k samples — see the project's separate note on the exact formula. `0.0` placeholder for retriever. |
| 3 | `jaccard_uncertainty` | `trial["jaccard_uncertainties"][node_id]` | retriever, multi-item output only | `1 − mean pairwise Jaccard set-overlap` across the k samples' parsed item sets. `0.0` placeholder otherwise. |
| 4 | `has_semantic` | derived | every node | `1.0` if #2 is a real measurement, `0.0` if it's a meaningless placeholder |
| 5 | `has_jaccard` | derived | every node | Same idea for #3 |
| 6 | `inference_gap` | `trial["inference_gaps"][node_id]` (computed by `enrich_features.py`) | every node with at least one sample | `1 − cosine_similarity(embed(parent's output), embed(this node's output))` — "how much did this node change the information it was given." Root nodes (no parents) compare against the trial's question instead of a parent's output. |
| 7 | `has_inference_gap` | derived | every node | `1.0` if #6 was computable |
| 8 | `item_frequency` | `trial["item_frequencies"][node_id]` (computed by `enrich_features.py`) | retriever, multi-item output only | Mean per-item inclusion rate across the node's own k samples — "on average, how consistently was each retrieved item actually included" |
| 9 | `has_item_frequency` | derived | every node | `1.0` if #8 was computable |
| 10 | `is_retriever` | `topology["nodes"][node_id]["role"]` | every node | One-hot role encoding — exactly one of #10-12 is `1.0` |
| 11 | `is_reasoner` | same | every node | |
| 12 | `is_writer` | same | every node | |

**Why the has-flags exist**: `semantic_uncertainty` is structurally `None` for every retriever (the metric doesn't apply to that role at all, not "wasn't computed this time"). Silently replacing `None` with `0.0` would make the model unable to tell "measured, genuinely 0" apart from "not applicable" — the has-flag makes that distinction explicit and learnable. The exact same reasoning applies to `inference_gap` (undefined if a node has no samples) and `item_frequency` (only meaningful for multi-item Retriever output, which is the less common case — see the project's data-quality notes on the Retriever tending to echo whole paragraphs rather than distinct sentences).

---

## 3. Node embeddings — 384 (or 768, mean+variance) raw dimensions, PCA-compressed

Source: `enrich_features.py`'s `compute_node_embeddings()`, consumed via the `use_embeddings=True` / `use_embedding_variance=True` flags on `build_graph_dataset.py`'s `node_feature_vector()`.

- **Mean embedding** (384-dim, `all-MiniLM-L6-v2` sentence embeddings): every one of a node's k raw sample texts is embedded, and the k embedding vectors are **averaged element-wise**. This captures the *central meaning* of what the node tends to say — a signal a single scalar uncertainty number throws away entirely (two nodes can have identical uncertainty scores while saying completely different things).
- **Std embedding** (384-dim, only when `use_embedding_variance=True`): the **per-dimension standard deviation** across the same k embedding vectors, instead of their mean. This is explicitly *not* redundant with the scalar uncertainty metrics in §2 — it's "how much do the k samples disagree, expressed as a 384-dimensional spread in meaning-space" rather than one collapsed scalar number. This was tried and **confirmed as a real, independent improvement** in the ablation history (OOD top-1 0.359 → 0.378 when added on top of the SAGE switch).
- A node with only 1 sample has an undefined standard deviation — treated as `0` (a genuine "no disagreement observed," not a missing-data placeholder).
- Missing embeddings (a trial not yet run through `enrich_features.py`) are zero-padded to the full 384 dimensions, so the feature vector's total width (`in_dim`) stays identical whether or not a given trial has been enriched.

### 3.1 PCA compression (`reduce_embeddings.py`)

Feeding the full 384 (or 768, with variance) raw embedding dimensions straight into a small GNN caused **severe overfitting** — this is a directly-observed, logged finding, not a theoretical concern: train top-1 accuracy jumped from 0.30 to 0.54 while OOD accuracy stayed essentially flat around 0.23–0.24. The fix: PCA-compress the embedding columns down to a much smaller number of components (**128** was the configuration used for the best k=5 result) **fit only on the training-split graphs**, then applied to every graph (train, validation, and OOD) — this avoids leaking any validation/OOD-specific structure into the compression itself.

When both mean and std embeddings are present, **two entirely independent PCA models** are fit — one for the mean block, one for the std block — since they represent genuinely different kinds of signal (central tendency vs. spread across samples) and there's no principled reason to force them through a shared projection.

**Ablation history for the PCA dimension** (all on the 1762-record dataset, SAGE architecture, with the variance/std block enabled): PCA-16 (explained variance 0.40) underperformed PCA-32 (0.60), which underperformed **PCA-128** (0.94 explained variance) — the configuration that produced the best k=5 result. A PCA-64 point was also tried (0.80 explained variance) and fell between the two. Higher explained-variance retention consistently tracked with better OOD accuracy in this range, up to the 128-dimension setting actually used.

---

## 4. The "value + has-flag" convention

Every optional scalar at this stage of the project — `semantic_uncertainty`, `jaccard_uncertainty`, `inference_gap`, `item_frequency` — follows the same rule, applied consistently in `node_feature_vector()`:

```python
value = trial.get(field, {}).get(node_id)
has_value = 1.0 if value is not None else 0.0
value = value if value is not None else 0.0
```

i.e. every such feature contributes **two** columns to the final vector: the value itself (or `0.0` if absent/not-applicable) and a flag saying whether that value is real. This is deliberate and load-bearing: a silent `0.0` alone would be ambiguous between "measured, genuinely zero" and "not applicable to this node's role," and the model would have no way to learn the difference.

---

## 5. Full feature vector layout and dimension count (k=5, best-result configuration)

Assembled in exactly this order by `node_feature_vector()`:

| Block | Dimensions | Cumulative |
|---|---|---|
| §2 Base scalars (lexical/semantic/jaccard/inference-gap/item-frequency + their has-flags + role one-hot) | 12 | 12 |
| §3 Mean node embedding, PCA-compressed to 128 | 128 | 140 |
| §3 Std node embedding, PCA-compressed to 128 | 128 | **268** |

**`in_dim = 268`** is exactly the value logged in `model/metrics_history.jsonl` for every run using this full configuration (bidirectional edges + embeddings + embedding variance + PCA-128), including the 0.395 best result — confirming this table matches the code that actually produced the headline number.

A scalar-only feature vector (no embeddings at all, `use_embeddings=False`) is **12-dimensional**. The very first GNN run in the project's history (the 493-record dataset, OOD top-1 = 0.220) used an even smaller, earlier 8-dimensional vector — `[lexical, semantic, jaccard, has_semantic, has_jaccard, is_retriever, is_reasoner, is_writer]`, without `inference_gap`/`item_frequency` at all, since those two fields were only added by a later `enrich_features.py` pass.

---

## 6. Edge representation

Built in `trial_to_graph()`, two modes existed at this stage:

1. **Forward-only** (`bidirectional=False`): `edge_index` = the topology's edges exactly as stored, `[src_list, dst_list]` — information can only flow the same direction it actually executed in the pipeline (parent → child).
2. **Bidirectional/merged** (`bidirectional=True`, the mode used for the best result): forward and reverse edges are concatenated into **one** `edge_index` tensor — `[src+dst, dst+src]`. A single GNN conv layer then treats parent→child and child→parent connections identically; it cannot distinguish "upstream" from "downstream" purely from edge direction, but this does let a fault's symptom propagate backward through message passing toward its true upstream cause. This was tested against forward-only and adopted permanently after a small, consistent OOD improvement (0.220 → 0.232 on an otherwise-matched comparison).

There is no separate reverse-edge tensor and no per-direction convolution stack at this stage — that is later work, out of scope here.

---

## 7. Labels

- **Node-level target** (`g.y`, the primary training signal): an integer class index into `[0 .. num_nodes]` for that specific graph — index `i` means "node `node_ids[i]` was the true fault target," and index `num_nodes` (one past the last real node) is the dedicated **"no fault"** class for clean/control trials. This is a `(num_nodes + 1)`-way softmax, computed per-graph inside `FaultLocalizerGNN.forward()`: `node_out = torch.cat([node_logits, no_fault_logit])` — never an all-zero vector, which would be ill-defined under softmax.
- **Graph-level target** (optional, `multi_task=True`): `true_label` string (`clean`/`noise`/`contamination`/`ceiling`) mapped to one of 4 classes, predicted from the pooled graph representation (`graph_repr`, see §8) via a single linear layer (`fault_type_head`), trained jointly with the node-level loss. **Tried once in the ablation history and found neutral-to-slightly-negative** (OOD top-1 0.275 → 0.267 when added) — not part of the best (0.395) configuration.
- **Bookkeeping fields carried on the graph object, never fed to the model**: `topology_id` (used only to join split membership and structure), `question` (used only for the by-question train/val split, never as a feature), `node_ids` (index↔name lookup for error analysis), `true_label` (the graph-level bookkeeping copy — the actual target used in training is `g.y`).

---

## 8. Graph-level pooling (feeds the "no fault" head, and the multi-task head if enabled)

`FaultLocalizerGNN` reduces all of a graph's node embeddings (after message passing) down to one single graph-level vector, `graph_repr`, three interchangeable strategies were compared:

| Pool type | How `graph_repr` is computed | OOD top-1 (matched config) |
|---|---|---|
| `mean` (default) | Simple element-wise average of every node's final embedding | 0.375 |
| `max` | Element-wise maximum across every node's final embedding | 0.361 |
| **`attention`** (used for the best result) | A learned linear layer scores each node, scores are softmax-normalized into per-node weights, and the graph representation is their weighted sum — the model learns *which nodes matter most* for the graph-level "is there a fault at all" decision, rather than treating every node equally | **0.387** |

This `graph_repr` feeds two places: the `no_fault_head` (always) and the `fault_type_head` (only if `multi_task=True`).

---

## 9. What is explicitly out of scope for this document

The following existed only in later work, done for the k=10 dataset track, and are **not** part of what's described above:
- Discrepancy/novelty features (`node_novel_ratio`, `node_dropped_ratio`, `node_sibling_disagree`, `node_child_novel`, `node_len_mean`, `node_len_std`)
- Leave-one-out calibrated z-scores against clean-control baselines
- Edge-based disagreement features (`node_in_edge_disagree_*`, `node_out_edge_disagree_*`)
- Structural/topological position features (`in_degree`, `out_degree`, normalized depth, descendants ratio, source/sink flags)
- Directed dual-edge-tensor message passing (`edge_index` + `edge_index_rev`, separate forward/reverse convolution stacks)
- The fault-type mixture head (`mixture_head=True`)

These were added specifically because uncertainty-based features (everything in this document) were found to be unable to localize ceiling faults at all — a finding made *after* the k=5 era described here concluded. They're documented separately.
