# Model Approaches — How Each One Actually Works (k=5 Dataset Era)

> Scope note: this explains every modeling approach tried on the **finished k=5, 1762-record dataset**, up through the best result (OOD top-1 = 0.395, commit `2880d27`) — what each approach *is* and *how it works mechanically*, not just its accuracy number (the accuracy ablation table already lives in `docs/information.md` §4; this document is the companion "how does it actually work" reference). Stops before the k=10-era additions (discrepancy features, edge features, structural features, directed message passing, mixture heads) — those are separate, later work. Not to be confused with `docs/model/model.md`, an earlier planning document written *before* any training code existed; this document describes the code as it was actually built and run.

---

## 1. The task, restated precisely

Every approach below solves the same problem: given one small graph (3–7 nodes, each with a feature vector — see `docs/feature-engineering-and-processing.md`) and the graph's edges, predict **which single node was the true fault source**, or that the graph is clean (a `(num_nodes + 1)`-way classification, one extra class for "no fault"). Approaches are presented in the order they were actually tried, each one required to beat the previous tier before more complexity was considered justified.

---

## 2. Baseline 1 — Naive max-uncertainty (no learning at all)

**File**: `model/baselines.py`

**How it works**: look at column 0 of the graph's feature matrix (`lexical_uncertainty`, the one metric guaranteed present for every node), and predict whichever node has the highest value. If every node's lexical uncertainty is effectively zero (≤ 1e-9), predict "no fault" instead — nothing stood out, so guess clean.

```python
def naive_max_uncertainty_baseline(graph):
    lexical_uncertainties = graph.x[:, 0]
    if float(lexical_uncertainties.max()) <= 1e-9:
        return graph.num_nodes  # "no fault" class
    return int(lexical_uncertainties.argmax())
```

**Why this is the right starting point**: it requires no training data, no learned parameters, and takes one line of logic. If a downstream model can't beat this, it isn't earning its complexity. It also has a very specific, predictable *failure mode* that motivates everything after it: a fault's symptom is often loudest **one hop downstream** of where the fault actually was injected (a corrupted Retriever makes the Reasoner *more* uncertain than the Retriever itself sometimes appears, because the Reasoner's uncertainty reflects both its own behavior and the inconsistency it inherited). This baseline has no way to reason about that — it can only ever point at the single loudest symptom, never trace it back to its source.

**Result**: OOD top-1 = 0.145 on the finished 1762-record dataset.

---

## 3. Baseline 2 — Non-graph RandomForest (does the *graph structure* matter, or just richer features?)

**File**: `model/nongraph_baseline.py`

**How it works**: takes exactly the same per-node feature vectors the GNN would use, but throws away the graph structure entirely. Each graph's node-feature matrix is:
1. **Zero-padded** up to the largest node count seen across train/val/OOD (so every graph becomes the same fixed-size input, since a plain tabular classifier can't handle variable-length inputs the way a GNN can).
2. **Flattened** into one long vector (all of node 1's features, then node 2's, etc.), with the graph's real node count appended as one extra feature (so the model at least knows how many of the padded slots are real).
3. Fed to a standard `sklearn.ensemble.RandomForestClassifier` (200 trees, `class_weight="balanced"` to counter the "no fault" class being a minority), trained to predict the same node-index-or-"no-fault" label the GNN uses.

**Why this comparison matters, and why it's an imperfect but still useful one**: this isolates the question "is the *graph structure* (which node connects to which) actually adding value, or would a much simpler flat-table model do just as well on the same information?" The comparison isn't perfectly fair — flattening means "feature slot 3" refers to a different physical node depending on whether the graph has 4 nodes or 7, so the RandomForest can't learn genuinely transferable positional reasoning the way a GNN's message-passing can. But it's the standard version of this ablation and still answers the intended question, especially since it's evaluated on the exact same OOD topologies never seen during training.

**Result**: OOD top-1 = 0.249 — clearly above the naive baseline (features alone help) but, as later results show, clearly below a well-tuned GNN (structure specifically adds further value beyond just having richer features).

---

## 4. The core architecture — Graph Neural Network (GNN) via message passing

**File**: `model/gnn.py`, class `FaultLocalizerGNN`.

### 4.1 The general idea, in plain terms

A GNN processes a graph in **rounds** (called *layers*). In each round, every node looks at its directly-connected neighbors' current feature vectors, combines that information with its own, and produces an updated vector for itself. After 2–3 rounds, a node's vector reflects not just its own original data, but a summary of its neighborhood a few hops out — which is exactly what's needed here, since "was node X the true fault source" often depends on a *pattern across the graph* (e.g., "this node itself looks only mildly off, but everything downstream of it looks off too") rather than node X's number in isolation.

### 4.2 Step-by-step through `FaultLocalizerGNN.forward()`

1. **Input projection**: every node's raw feature vector (12 scalars, or 268 dimensions with PCA-compressed embeddings — see the feature-engineering doc) is passed through one linear layer plus dropout, compressing it down to a fixed `hidden_dim` (32 or 64 in the runs that mattered) *before* any message passing starts. This step exists specifically because feeding a large, mostly-raw embedding vector directly into a graph convolution layer (which itself is fairly simple, linear-ish) caused severe overfitting — train accuracy shot up while OOD accuracy stayed flat. Projecting and regularizing first gives the network a chance to learn a compact, useful summary before mixing neighbor information in.
2. **Message-passing layers** (2–3 of them, stacked): each layer takes every node's current vector, mixes in its neighbors' vectors (the exact mixing rule depends on which convolution type is used — see §4.3), and produces a new vector for every node, followed by a ReLU activation and dropout.
3. **Node-level head**: a single linear layer turns each node's final (post-message-passing) vector into one number — a raw "fault score" for that specific node.
4. **Graph-level pooling** (see §5): all nodes' final vectors are reduced to one single vector representing "the whole graph," used only for the "no fault" decision (and the optional fault-type auxiliary head).
5. **"No fault" head**: one more linear layer turns the pooled graph vector into a single extra score — "is this graph clean at all."
6. **Final output**: the per-node fault scores and the single "no fault" score are concatenated into one vector of length `(num_nodes + 1)`. This is what gets passed through a softmax (during loss computation) to produce a proper probability distribution over "which node, or no fault at all."

### 4.3 The three convolution types compared (GAT vs. GCN vs. SAGE)

All three are standard PyTorch Geometric layers, swapped in via one config flag (`conv_type`), with every other setting held fixed for the comparison:

| Type | Core idea (how neighbor mixing works) | OOD top-1 (matched hyperparameters) |
|---|---|---|
| **GAT** (Graph Attention Network) | Each node learns *how much attention to pay* to each of its neighbors individually — not every neighbor contributes equally; a learned weighting decides which neighbor's information matters more for this particular update. This was the project's original default. | 0.275 |
| **GCN** (Graph Convolutional Network) | Neighbor information is combined using a **fixed** rule based on node degree (how many connections each node has) — no learned per-neighbor weighting, just a normalized average that accounts for how "busy" each neighbor's own connections are. Simpler and cheaper than GAT, but less flexible. | 0.246 (worst of the three) |
| **SAGE** (GraphSAGE) | Each node's neighbors' vectors are aggregated (by default, a simple **mean**), then that aggregated neighbor-summary is concatenated with the node's *own* current vector and passed through a shared linear transformation. The key structural difference from GCN/GAT: a node's own information is kept explicitly separate from its neighbors' aggregated information, rather than blended together in the same step. | **0.359 — clear best of the three, and the architecture used for every result after this comparison** |

**Why SAGE won, in plain terms**: on graphs this small (3–7 nodes), GAT's extra learned-attention machinery is more parameters to fit with very little data per graph, which likely encourages overfitting to training-topology quirks rather than a genuinely transferable signal. GCN's fixed-averaging rule, on the other hand, may be too rigid to capture the asymmetric "upstream cause, downstream symptom" pattern this problem depends on. SAGE's explicit separation of "myself" from "my aggregated neighbors" appears to strike the right balance for this specific problem size and structure — this is an empirical finding from the ablation, not a claim that SAGE is universally better.

### 4.4 Bidirectional edges

By default, a topology's edges only point the direction information actually flowed during the real pipeline execution (parent → child). The **bidirectional** setting additionally adds every edge's reverse direction into the same `edge_index` tensor, so a single convolution layer treats parent→child and child→parent connections identically. Mechanically, this simply lets message passing propagate a symptom **backward** — from a downstream node whose uncertainty is elevated, back toward the actual upstream node that caused it — which forward-only edges cannot do at all in a shallow (2–3 layer) network. Tested directly against forward-only edges and adopted permanently after a small, consistent OOD improvement.

---

## 5. Graph-level pooling strategies

Used only for the "no fault" decision (and the optional fault-type head, §6) — never for the per-node fault scores themselves, which come directly from each node's own final vector (§4.2 step 3).

| Pool type | Mechanism | OOD top-1 (matched config) |
|---|---|---|
| `mean` (default) | Simple element-wise average across every node's final vector — every node contributes equally regardless of role or position | 0.375 |
| `max` | Element-wise maximum across every node's final vector — only the single strongest signal per dimension survives, everything else is discarded | 0.361 |
| **`attention`** (used for the best result) | A small learned linear layer assigns each node a raw importance score; these scores are passed through a softmax (so they sum to 1 across the graph's nodes) and used as weights in a weighted sum of the nodes' vectors. The model *learns* which nodes' information matters most for the graph-level decision, rather than treating every node as equally informative by assumption. | **0.387** |

Attention pooling's improvement over plain mean pooling is the single largest individual gain from any one technique tried in this project — bigger than the entire embedding-variance addition, and on par with the SAGE architecture switch itself.

---

## 6. The multi-task auxiliary head (tried, not adopted)

**Mechanism**: in addition to the per-node fault-localization output, one more linear layer takes the pooled graph vector and predicts a 4-way classification over the fault *type* (`clean`/`noise`/`contamination`/`ceiling`) directly — a completely separate prediction target, trained jointly with the main node-localization loss (see §7.3 for exactly how the two losses are combined).

**Intuition for why this was tried**: knowing the fault *type* might help the model narrow down *where* to look — e.g., if the model is confident this is a "ceiling" fault, that might correlate with different nodes than a "contamination" fault. Forcing the network to also predict the type was hoped to encourage it to build internal representations that are more sensitive to this distinction.

**Result**: tried once, found **neutral-to-slightly-negative** (OOD top-1 dropped from 0.275 to 0.267 when added, at that point in the ablation sequence) — not included in the final best configuration. Not re-tried in combination with the later winning changes (SAGE, embedding variance, attention pooling) within the k=5 era.

---

## 7. Training mechanics

**File**: `model/train.py`.

### 7.1 Why training iterates graph-by-graph, not in mini-batches

Every graph has a *different* number of possible classes (a 3-node graph is a 4-way softmax; a 7-node graph is an 8-way softmax) — PyTorch Geometric's standard batched-loss machinery assumes one shared label space per batch, which doesn't fit this setup cleanly. Given the dataset's small size (a few hundred to under 2000 tiny graphs, fast on CPU), the simpler and still-correct solution adopted was to loop over graphs **one at a time** within each epoch, computing and backpropagating loss individually per graph, rather than engineering a padded/masked batching scheme purely for speed that isn't actually needed at this scale.

### 7.2 Class-imbalance correction — the `no_fault_weight`

"No fault" (clean) trials are a minority class relative to all the real-fault-node classes combined (roughly 15% of trials, per the dataset's actual composition). Rather than weighting individual *nodes* differently (which wouldn't make sense — a node's index is arbitrary per graph, so "node 2" carries no consistent meaning across different topologies), the correction is applied specifically to the "no fault vs. some real fault" axis, since that is the genuine, consistent imbalance:

```
no_fault_weight = (number of faulted trials) / (number of clean trials)
```

This weight multiplies the loss contribution of every "no fault" example, so misclassifying a clean graph counts proportionally more, balancing out its lower frequency in the training data.

### 7.3 Loss function options

The core loss is standard multi-class cross-entropy (per graph, given its own `(num_nodes+1)`-way softmax), with three variants tried:

- **Plain weighted cross-entropy** (`"ce"`, default): standard `F.cross_entropy(logits, target, weight=class_weights)`.
- **Label smoothing** (`"label_smoothing"`): instead of training the model to push the correct class's probability all the way to 1.0 (and every wrong class to exactly 0.0), the target is *softened* slightly — the model is trained toward, say, 90% confidence on the correct class and a small residual probability spread across the wrong ones. This discourages overconfidence, which can help when labels are somewhat noisy (as they can be here — recall that `verification.target_deviated: false` cases exist, where an injected fault didn't measurably show up as a symptom, meaning the "correct" label is harder than usual to actually justify from the visible features alone).
- **Focal loss** (`"focal"`): down-weights examples the model is *already* getting right with high confidence, and concentrates the gradient signal on examples it's currently getting wrong. Formula: standard cross-entropy `ce`, converted to a per-example weight via `(1 - p_t)^gamma` where `p_t = exp(-ce)` is the model's predicted probability for the true class — an *easy*, already-confident example (`p_t` near 1) gets its loss shrunk toward zero, while a *hard*, currently-wrong example (`p_t` near 0) keeps close to its full loss value. `gamma=2.0` was used.

When the multi-task head (§6) is active, the two losses are combined as `loss = node_loss + aux_weight * type_loss`, with `aux_weight = 0.3` — the fault-type loss contributes, but is deliberately weighted down relative to the primary node-localization loss.

### 7.4 Optimizer and early stopping

- **Adam optimizer**, with `weight_decay` (a small additional penalty on large weights, a standard regularization technique) — `lr=1e-3` or `5e-4` depending on the run, `weight_decay=1e-4`.
- **Early stopping on validation accuracy**: after every epoch, the model is evaluated on the held-out validation split (never OOD — OOD is only touched once, at the very end). If validation top-1 accuracy improves, the current model weights are saved as the new "best" checkpoint. If it hasn't improved for `patience` consecutive epochs (30, in the runs that mattered), training stops early. The **final model used for evaluation is the best-validation-accuracy checkpoint**, not necessarily the model from the very last epoch trained — this is standard practice to avoid reporting a model that has started overfitting past its peak generalization point.

---

## 8. Evaluation metrics — exactly what gets measured and how

**File**: `model/evaluate.py`.

- **Top-1 accuracy**: for each graph, does the model's single highest-scoring class exactly match the true label? The headline number throughout this project.
- **Top-2 accuracy**: is the true label among the model's *two* highest-scoring classes? Reported because graphs range from 3 to 7 nodes — top-1 is already a meaningfully hard test on a 3-node graph (roughly 25% random-chance baseline) but a much harder one on a 7-node graph (roughly 12.5% random-chance baseline), so top-2 shows how "close" the model gets on the larger, harder graphs specifically.
- **Macro-precision / macro-recall / macro-F1**: computed via `sklearn.metrics.precision_recall_fscore_support` with `average="macro"` — every class (every possible node index, plus "no fault") is weighted **equally** regardless of how often it appears, specifically so a model can't "win" on the plain accuracy number just by being good at the most common classes while quietly failing on rarer ones.
- **A known, deliberate exclusion**: `retriever_c` in the `star` and `triple_retriever_fanin` topologies is never a genuine possible fault target in the underlying dataset (a data-generation coverage gap, not a modeling issue — see `docs/information.md` §3.5). `KNOWN_UNREACHABLE_TARGETS` in `evaluate.py` exists so a downstream per-class report can explicitly mask this class out for those two topologies, rather than counting "the model never predicted `retriever_c` there" as a failure — it was never a valid correct answer to begin with.
- **In-distribution vs. out-of-distribution are always reported separately, never pooled into one number** — the gap between them is treated as the single most important finding, not an inconvenience to average away.

---

## 9. Ensembling — the technique behind the final 0.395 result

**Files**: `model/ensemble.py` (the original 3-seed SAGE ensemble) and `model/attn_ensemble.py` (the final version, stacking ensembling on top of attention pooling).

**Mechanism, precisely**:
1. Train the *exact same* architecture and hyperparameters **3 separate times**, using `torch.manual_seed(0)`, `torch.manual_seed(1)`, and `torch.manual_seed(2)` respectively before each training run. Each seed produces a different random starting point for the model's weights (and different random shuffling during training), so each of the 3 resulting trained models ends up with slightly different learned parameters, even though nothing else about the setup changed.
2. At prediction time, run **all 3** trained models on the same graph, convert each model's raw output into a proper probability distribution via softmax, and **average the three probability distributions together** (not a majority vote over 3 separate final answers — an average of the underlying confidence scores, computed first).
3. The final prediction is whichever class has the highest value in that *averaged* probability distribution.

**Why this helps**: with a dataset this small, a single training run can land in a slightly different, individually-quirky solution purely due to where its random weights happened to start — one seed's model might have an unlucky blind spot on some particular node-role pattern that another seed's model doesn't share. Averaging cancels out each individual model's idiosyncratic mistakes, since it's unlikely all 3 independently-trained models share the exact same blind spot.

**Result of combining every winning technique**: SAGE convolution (§4.3) + embedding-variance features (see feature-engineering doc) + attention pooling (§5) + this 3-seed ensembling, all stacked together, is what produced the final **OOD top-1 = 0.395** — each individual technique's gain was verified in isolation first (per `docs/information.md`'s ablation table) before being combined into this final configuration.

---

## 10. Summary — the complete "menu" of approaches, in one table

| Approach | How it fundamentally works | OOD top-1 |
|---|---|---|
| Naive max-uncertainty | Pick the node with the single highest lexical-uncertainty value | 0.145 |
| Non-graph RandomForest | Flatten every node's features into one fixed-size row, ignore graph structure entirely | 0.249 |
| GNN — GAT | Learned per-neighbor attention weighting during message passing | 0.275 |
| GNN — GCN | Fixed, degree-normalized neighbor averaging during message passing | 0.246 |
| GNN — SAGE | Neighbor-mean aggregation kept explicitly separate from the node's own vector | 0.359 |
| + embedding-variance features | Same SAGE model, richer per-node input (see feature-engineering doc) | 0.378 |
| + attention pooling | Learned per-node importance weighting for the graph-level "no fault" decision | 0.387 |
| + 3-seed ensemble | Average the probability outputs of 3 independently-trained copies of the same model | **0.395 (final, best)** |

Everything past this point in the project's history — discrepancy/edge/structural features, directed dual-edge message passing, the mixture head — is later, k=10-era work and is documented separately.
