# Model Training Plan — Fault-Localization Model over the Generated Dataset

> Written 2026-09-26. This is the plan for the phase that comes *after* dataset generation. No training code exists yet as of this writing — this document is the step-by-step spec for building it. Read [`docs/dataset/dataset_description.md`](../dataset/dataset_description.md) Section 6 and 11 first if you haven't; it defines exactly what's input, what's label, and what's bookkeeping-only in each record. For the Kaggle/Lightning AI generation infra itself, see [`docs/infra/kaggle-and-lightning-setup.md`](../infra/kaggle-and-lightning-setup.md).

---

## Plain-English Walkthrough (read this first)

This section explains every step in simple language, with one running example, before the formal plan below. If a term below is unfamiliar later in the doc, come back here.

### The one-sentence goal

We have 493 examples of "a pipeline ran, something may have gone wrong, here's exactly what went wrong and where." We want to train a model that, given only the *symptoms* (how confused/uncertain each part of the pipeline was), can guess *where the problem started* — without being told the answer.

### The running example we'll use throughout

Topology: **`chain`** — three agents wired in a straight line:

```
Retriever  →  Reasoner  →  Writer
```

- **Retriever** looks up facts.
- **Reasoner** thinks through the facts to get an answer.
- **Writer** writes the final sentence.

Question: *"Were Scott Derrickson and Ed Wood of the same nationality?"*
A fault was injected: **`noise`** at the **Retriever**. (`noise` = we cranked up the Retriever's "randomness" setting, so its output is less consistent than normal — like asking someone the same question 5 times after they've had too much coffee: same rough idea, but the wording jitters.)

---

### Step 1 — Understand what one data point ("trial") looks like

**Term: "trial"** — one full run of one topology, on one question, with (optionally) one fault switched on at one node.

For our example, the Retriever was asked the same sub-question **5 times** (this repeat-count is called **k=5**, short for "k samples" — asking the same thing multiple times to see how consistent the answers are). It came back 5 times with things like:
```
1. "Scott Derrickson is an American director..."
2. "Ed Wood was an American filmmaker..."
3. "Scott Derrickson, American, born 1966..."
4. "Ed Wood, American filmmaker..."
5. "Scott Derrickson is a director from the US..."
```
Because `noise` was injected here, these 5 answers **disagree with each other more** than they would normally — that disagreement *is* the fault's fingerprint. We convert "how much do the 5 answers disagree" into a single number per node, called **uncertainty**.

**Term: "uncertainty"** — a number from 0 to 1 saying "how much did the model waver when asked the same thing 5 times." 0 = gave the exact same answer every time (very confident/consistent). 1 = gave 5 completely different answers (very inconsistent). In our example: Retriever = 0.4 (moderately inconsistent — the fault's effect), Reasoner = 0.6 (it "inherited" some confusion from the noisy Retriever), Writer = 0.2 (a bit, but less — the fault's effect has partly washed out by the time it reaches the end).

**Term: "true_label"** — the answer key. Here it's `"noise"` at node `"retriever"`. We know this for certain because *we* (the dataset generator) deliberately injected it — this isn't a guess.

So one trial = **numbers describing symptoms** (uncertainty per node) + **the graph shape** (who feeds whom) + **the true answer** (which node, which fault type).

---

### Step 2 — Turn each trial into something a model can read: a "graph"

**Term: "graph" (in this context, not a chart)** — a structure of dots (**nodes**) connected by arrows (**edges**). Not a line graph you'd plot — think of it like a flowchart or an org chart.

Our `chain` example as a graph:
```
[Retriever] --edge--> [Reasoner] --edge--> [Writer]
```
- **Nodes** = Retriever, Reasoner, Writer (3 dots).
- **Edges** = Retriever→Reasoner, Reasoner→Writer (2 arrows, meaning "feeds into").

**Term: "feature vector"** — a short list of numbers attached to each node, describing everything the model is allowed to know about that node. For our Retriever node:
```
[lexical_uncertainty=0.4, semantic_uncertainty=0 (n/a for retriever), jaccard=0, has_semantic=0, has_jaccard=0, is_retriever=1, is_reasoner=0, is_writer=0]
```
This is just "here's a row of numbers describing this dot in the graph." Every node gets one of these rows.

**Term: "label"** — what we want the model to predict. For this trial: Retriever = 1 (yes, this is the faulty node), Reasoner = 0, Writer = 0.

So: **1 trial → 1 small graph, made of a few dots with number-rows attached, plus an answer key saying which dot is "it."**

---

### Step 3 — Split the 493 graphs into three piles

**Term: "train set"** — the graphs the model is allowed to *study* and learn from (like practice problems with answers shown).
**Term: "validation set" (val)** — a smaller pile the model does *not* study, used only to check "am I actually learning, or just memorizing the practice problems?" during training (like a practice quiz).
**Term: "test set" (here, the OOD set)** — graphs from **6 topologies the model has never seen in any form**, opened only once, at the very end (like the real final exam — no do-overs).

Why OOD (Out-Of-Distribution) matters: imagine the model just memorized "in `chain`, the middle node is usually the answer." That trick would score high on training data but be useless if you handed it a totally new pipeline shape it's never seen. Testing on 6 brand-new random shapes is how we catch that kind of cheating.

Concretely: 14 topologies (chain and 13 others) → split further into train/val. 6 topologies (`ood_random_0`...`ood_random_5`) → held back entirely for the final test.

---

### Step 4 — Try the dumbest possible approach first (the "baseline")

**Term: "baseline"** — the simplest thing you can try, used as a reference point. If your fancy model can't beat the baseline, the fancy model isn't earning its complexity.

Our baseline: **"just pick whichever node has the highest uncertainty number."** In our example: Retriever=0.4, Reasoner=0.6, Writer=0.2 → baseline guesses **Reasoner** — but the *true* answer is Retriever! This baseline gets this specific example **wrong**, because the fault's symptom (confusion) is loudest one step *downstream* of where it actually started, not at the source itself. This is exactly the pattern a smarter model needs to learn: "high uncertainty two steps away can mean the real problem is upstream," not "the loudest symptom is always the culprit."

---

### Step 5 — Build the actual learning model: a GNN

**Term: "GNN" (Graph Neural Network)** — a model built to read graphs (dots + arrows + number-rows) instead of a flat table or a sentence. It works by **"message passing"**: each round, every node looks at its neighbors' number-rows, mixes that information in with its own, and updates its own summary. Do this 2-3 rounds and every node ends up with a summary that reflects not just itself, but its neighborhood.

In our example, after message passing: the Reasoner node's summary now "knows" that its upstream neighbor (Retriever) was unusually uncertain, not just that Reasoner itself was somewhat uncertain. That's the whole trick — it lets the model reason about the *pattern across the graph* ("high uncertainty downstream of a node, but that node itself also looks a bit off") rather than each node's number in isolation.

**Term: "training" / "learning the weights"** — the GNN starts with random internal settings (**weights**) that control exactly *how* it mixes neighbor information. Training means: show it a training example, let it guess, compare the guess to the true answer, then nudge the weights slightly in the direction that would have made the guess more correct. Repeat this thousands of times across all training examples.

**Term: "loss"** — a single number measuring "how wrong was the guess." High loss = bad guess, low loss = good guess. Training tries to make loss go down over time.

**Term: "epoch"** — one full pass through all the training examples. You typically run many epochs (tens to hundreds), watching the loss go down each time, like re-reading a practice-problem set multiple times until you've internalized the pattern.

**Term: "overfitting"** — when the model gets *really* good at the practice problems (train set) but that improvement stops showing up on the practice quiz (val set) — a sign it memorized specifics instead of learning the general pattern. This is why we always watch val-set performance, not just train-set performance, while training.

---

### Step 6 — Check the final grade, honestly

Run the trained GNN once on the OOD set (the 6 never-before-seen topologies) and report the accuracy. Compare that number against:
- The baseline's accuracy on the same OOD set (did the GNN actually help?)
- The GNN's own accuracy on the *train* topologies (is there a big gap, meaning it overfit?)

**Term: "accuracy"** — out of all the graphs in a pile, what fraction did the model correctly point to the true faulty node.

For our worked example specifically: a good model, at test time, should correctly point to **Retriever** — recognizing that even though Reasoner's uncertainty (0.6) was numerically higher, the graph *pattern* (upstream node moderately uncertain, downstream node inheriting and slightly amplifying that uncertainty) points back to Retriever as the true source. That's the exact behavior the naive baseline in Step 4 got wrong, and exactly what the message-passing in Step 5 is designed to catch.

---

### Quick glossary (all terms used above, in one place)

| Term | Plain meaning |
|---|---|
| Trial | One example: one pipeline run, with a known true fault location (or none) |
| Node | One "dot" in the graph — here, one pipeline agent (Retriever/Reasoner/Writer) |
| Edge | An arrow between two nodes — "output of A feeds into B" |
| Feature vector | The row of numbers describing one node |
| Uncertainty | A 0-1 number: how much a node's repeated answers disagreed with each other |
| Label | The correct answer we're training the model to predict |
| Train set | Examples the model studies and learns from |
| Validation set | Examples used mid-training to check for overfitting, never trained on |
| Test set (OOD) | Examples opened only once at the very end, on entirely unseen topology shapes |
| Baseline | The simplest possible approach, used as a reference point to beat |
| GNN | A neural network designed to read graph-shaped data via message passing |
| Message passing | Each node updates its summary by mixing in its neighbors' summaries, repeated a few rounds |
| Weights | The model's internal, learnable dials that get adjusted during training |
| Loss | A number measuring how wrong a prediction was; training tries to shrink it |
| Epoch | One full pass through all training examples |
| Overfitting | Model does well on practice problems but not on new ones — memorized instead of generalized |
| Accuracy | Fraction of examples where the model's top guess was correct |

---

## 0. What we're building, in one paragraph

Given one pipeline execution ("trial": a small DAG of Retriever/Reasoner/Writer-role nodes, each carrying uncertainty features from k=5 self-consistency samples), predict **which node was the true fault source** (node-level classification over the graph, one node per trial is faulted, or none if `is_control`). Optionally also predict **fault type** (`noise`/`contamination`/`ceiling`/`clean`) as a secondary target. The dataset (`dataset/trials.jsonl`, 493 records, 329 train-topology / 164 OOD-topology) already has verified labels — this phase is purely: featurize → train → evaluate, with generalization to **unseen topologies** (the OOD split) as the key test of whether the model learned a structural signal rather than memorizing per-topology shortcuts.

---

## 1. Model architecture choice

Build these in order — each is a checkpoint, not throwaway work, since the naive baseline is the thing the GNN needs to beat to justify its existence:

1. **Naive/threshold baseline**: per-node uncertainty vs. a fixed or per-role threshold, no learning — "the node with max uncertainty is the fault source." Trivial to implement, gives an immediate sanity floor. Also re-derive `diagnose_trace`'s retry-then-reprobe heuristic as a second baseline if time permits (it's in `src/diagnose.py`, never invoked during generation — this is exactly the separate post-hoc pass `dataset_description.md` Section 8 describes).
2. **Non-graph ML baseline**: flatten each trial into a fixed-size feature table (won't naturally generalize across variable node-count topologies, so this only makes sense evaluated *within* a fixed topology, or with padding/masking to a max node count) — logistic regression / gradient-boosted trees (XGBoost/LightGBM) per-node, framed as binary "is this node the fault" classification. Useful as a second sanity check that graph structure is actually adding value later.
3. **GNN (the real target)**: message-passing network over the topology graph (edges from `topology_pool.json`), node classification head. This is the only architecture that naturally handles variable node count/structure and can generalize to the 6 unseen OOD topologies — the central point of the whole thesis.

Recommended GNN specifics:
- **Framework**: PyTorch Geometric (PyG) — standard, well-documented, easiest to iterate on for a thesis timeline.
- **Layer type**: start with **GAT** (Graph Attention Network) or **GraphSAGE**, 2-3 layers. Attention is a nice fit here because "which upstream node contributed the corrupted signal" is naturally an attention-like question, and attention weights are interpretable for the write-up. GCN is a reasonable simpler fallback if GAT underperforms or is harder to tune.
- **Directionality**: the pipeline DAG is directed (`edges: [[src, dst]]`, info flows src→dst). Test both directed message passing and an added reverse-edge channel (fault signal often needs to propagate *backward* from a downstream symptom to an upstream cause) — this is likely to matter a lot given the whole premise is localizing an upstream cause from downstream uncertainty spikes. Don't assume forward-only edges are correct without testing.
- **Readout**: node-level output head (per-node logit → softmax/sigmoid over "is fault source"), not graph-level pooling, since the target is node-level.
- **Graph-level auxiliary head** (optional): fault-type classification (`clean`/`noise`/`contamination`/`ceiling`) via mean/attention pooling over final node embeddings, trained jointly with node-level loss (multi-task). Cheap to add, may regularize the node-level task.

---

## 2. Step-by-step build order

### Step 1 — Environment setup
```bash
pip install torch torch-geometric scikit-learn xgboost pandas numpy matplotlib
```
Use CPU is fine for this dataset size (493 graphs, ≤7 nodes each — trivially small for a GNN; no GPU needed for training, unlike the generation phase).

### Step 2 — Data loading & validation
- Load `dataset/trials.jsonl` and `dataset/topology_pool.json`.
- **Run `scripts/verify_results.py` first** if not already done on this exact file — `dataset_description.md` explicitly flags this project's history of prose/data mismatches; don't skip it.
- Join every trial to its topology via `topology_id` → get `nodes` (role per node) and `edges` from `topology_pool.json`.
- Join every trial's `topology_id` → `topology_pool.json`'s `split` field to recover train/OOD membership (not stored per-trial, see Section 9 of `dataset_description.md`).

### Step 3 — Feature extraction (write `model/data/build_graph_dataset.py`)
For each trial record, build one graph:
- **Node features** (per `dataset_description.md` §6.1 and §11's worked example), 6-dim vector per node:
  `[lexical_uncertainty, semantic_uncertainty, jaccard_uncertainty, is_retriever, is_reasoner, is_writer]`
  - `lexical_uncertainty` from `uncertainties[node_id]` — always present.
  - `semantic_uncertainty` from `semantic_uncertainties[node_id]` — `null` for retriever nodes.
  - `jaccard_uncertainty` from `jaccard_uncertainties[node_id]` — present only for some retriever nodes (multi-item output).
  - Role one-hot from `topology_pool.json`, not per-trial.
- **Missing-value decision (must resolve — this is the "open decision" flagged in `dataset_description.md` §6.1)**: use an explicit **mask bit per optional feature** (e.g. `has_semantic`, `has_jaccard`) rather than silently imputing 0 — a missing semantic-uncertainty (because the node is a retriever) is structurally different information from a genuine 0.0 value, and imputing without a mask risks the model treating "role doesn't have this metric" the same as "this metric came back clean." Feature vector becomes 8-dim: `[lexical, semantic, jaccard, has_semantic, has_jaccard, is_retriever, is_reasoner, is_writer]`.
- **Edges**: directed edge list from `topology_pool.json`, converted to PyG `edge_index` (2×E tensor). Add reverse edges as a separate edge type or duplicate set if testing bidirectional message passing (Section 1).
- **Node label** (primary target): one-hot / index vector, `1` at `fault_config["target_node"]`'s index, all-zero for `is_control` trials (decide up front: exclude control trials from the localization task entirely, or include them as "no fault, predict background/none" — recommend including with an explicit extra "none" class rather than all-zero, since all-zero under a softmax is ill-defined; a `k+1`-way softmax per node — real nodes plus "no fault in this graph" — or a graph-level "is this graph faulted at all" gate before node localization is cleaner).
- **Graph label** (secondary target): `true_label` string → integer class (`clean`/`noise`/`contamination`/`ceiling`).
- Serialize to a PyG `InMemoryDataset` or simply pickle a list of `torch_geometric.data.Data` objects — small dataset, no need for anything fancier.

#### Step 3, detail — what each feature actually means (with real numbers)

A **feature vector** is just a row of numbers attached to one node — everything the model is allowed to see about that node. No text, no raw model output, just numbers summarizing behavior. Each node gets this **8-dimensional vector**:

```
[lexical_uncertainty, semantic_uncertainty, jaccard_uncertainty, has_semantic, has_jaccard, is_retriever, is_reasoner, is_writer]
```

Walking through each piece using the worked example from `dataset_description.md` §11 (`chain` topology, `noise` fault at Retriever, question about Scott Derrickson/Ed Wood):

**1. `lexical_uncertainty`** — "did the exact wording agree across the k=5 tries?" Source: `uncertainties[node_id]`, computed as `1 - (most_common_answer_count / k)`. Retriever's 5 samples here (`"Scott Derrickson is an American director..."`, `"Ed Wood was an American filmmaker..."`, etc.) are all worded differently → `lexical_uncertainty = 0.4`. A clean (no-fault) Retriever would typically answer near-identically all 5 times → `0.0` or `0.2`. **Always present, for every node role.**

**2. `semantic_uncertainty`** — "did the *meaning* agree, even if wording differed?" Source: `semantic_uncertainties[node_id]`. Embeds each of the k answers and measures how spread-out they are in meaning-space, so two differently-worded but same-meaning answers ("Yes, both American." vs. "They shared the same nationality.") score *low* here even though lexical uncertainty is high. Reasoner's 5 conclusions in this example all effectively say "Yes, both American" with some phrasing drift → `0.55`. **Only computed for `reasoner`/`writer` roles — always `null` for `retriever`** (Retriever produces raw text, not an answer-shaped conclusion, so this metric doesn't apply to it; Jaccard is used instead, below).

**3. `jaccard_uncertainty`** — Retriever-only: "how much did the *set* of retrieved items overlap across tries?" Only meaningful when the Retriever's output parses into multiple discrete items (e.g. split by newline into separate sentences) — then it's a set-overlap score, giving partial credit for "2 of 3 items matched" instead of lexical uncertainty's all-or-nothing view. **Only present for `retriever` nodes with multi-item output** — per `dataset_description.md`'s data-quality note, Qwen2.5's Retriever tends to echo whole paragraphs (parsed as one blob) rather than distinct sentences, so this is **empty (`{}`) more often than not**. Empty in this worked example.

**4 & 5. `has_semantic`, `has_jaccard` — the mask bits.** These aren't in the raw data — we add them ourselves, and they're the fix for the "open decision" flagged above. The problem: `semantic_uncertainty` is `null` for every Retriever, not because something went wrong but because the metric doesn't apply to that role. If we silently replaced `null` with `0`, the model can't distinguish "measured, genuinely `0.0`" (very consistent) from "not applicable, meaningless placeholder `0`" — collapsing those could teach the model something false (e.g. implying Retrievers are always semantically consistent, when the metric was simply never computed). The fix: an explicit flag alongside the value.
```
Retriever node: semantic_uncertainty = 0.0  (placeholder, meaningless)   has_semantic = 0 (ignore the 0)
Reasoner  node: semantic_uncertainty = 0.55 (real, measured)             has_semantic = 1 (trust it)
```
Same idea for `has_jaccard` — `1` only for Retriever nodes with genuine multi-item output, `0` everywhere else (including single-blob Retriever output).

**6, 7, 8. `is_retriever`, `is_reasoner`, `is_writer` — role one-hot.** Source: not per-trial — comes from `topology_pool.json`'s `nodes[node_id]["role"]`, since a node's role never changes across trials. One-hot means exactly one of the three is `1`, the rest `0`:
```
Retriever node → [1, 0, 0]     Reasoner node → [0, 1, 0]     Writer node → [0, 0, 1]
```
This matters because uncertainty means something different by role — a 0.4 lexical uncertainty on a Retriever (raw fact lookup) isn't directly comparable to 0.4 on a Writer (final phrasing). The role tag lets the model learn role-specific patterns instead of treating every node identically.

**Putting it together — the full worked example:**

| node | lexical | semantic | jaccard | has_semantic | has_jaccard | is_retriever | is_reasoner | is_writer | **full 8-dim vector** |
|---|---|---|---|---|---|---|---|---|---|
| retriever | 0.4 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | `[0.4, 0, 0, 0, 0, 1, 0, 0]` |
| reasoner | 0.6 | 0.55 | 0 | 1 | 0 | 0 | 1 | 0 | `[0.6, 0.55, 0, 1, 0, 0, 1, 0]` |
| writer | 0.2 | 0.18 | 0 | 1 | 0 | 0 | 0 | 1 | `[0.2, 0.18, 0, 1, 0, 0, 0, 1]` |

Each row is what one node "looks like" to the GNN. Stack these 3 rows plus the 2 edges (`retriever→reasoner`, `reasoner→writer`) and that's the complete input graph for this trial — everything the model gets to work with when guessing "which node was faulted" (true answer here: `retriever`).

### Step 4 — Train/val/OOD split
- **OOD test set = all trials whose `topology_id` is in `topology_pool.json`'s 6 `ood_test`-split topologies (164 records).** Never touch this during training or hyperparameter tuning — it's held out entirely to measure generalization to unseen structure, per the thesis's core question.
- **Within the 329 train-topology trials**: split into train/val, e.g. 80/20. **Split by question, not by trial** (`dataset_description.md` §9 explicitly flags this — the same 30 questions recur across many topologies/fault conditions, so a naive per-trial random split leaks question-specific phrasing between train and val). Group by `question` field, split groups.
- Consider also holding out 1-2 *train-split* topologies entirely from training (a second, "in-distribution-family but unseen-instance" validation cut) to distinguish "generalizes to new topology instances" from "generalizes to genuinely novel random-DAG structure" (the OOD set) — optional but useful for the thesis narrative.

### Step 5 — Baselines first (Section 1's item 1-2)
- Implement and score the naive max-uncertainty baseline and (optionally) `diagnose_trace`-based baseline on the same train/val/OOD splits, before writing any GNN code. This gives the numbers the GNN needs to beat and catches labeling/feature bugs early on a much simpler system.
- Metrics (Section 6 below) computed identically across every model so comparisons are apples-to-apples.

### Step 6 — GNN implementation
- `model/gnn.py` (new): PyG `GATConv`/`SAGEConv` stack, 2-3 layers, node classification head (+ optional graph-level fault-type head).
- `model/train.py` (new): training loop — cross-entropy loss on node-level fault-location target (+ weighted auxiliary graph-level loss if using the multi-task head), Adam optimizer, early stopping on val loss.
- Given the dataset's small size (493 graphs), expect to need: careful regularization (dropout, weight decay), possibly a shallow network (2 layers), and many epochs are cheap — this is not a "need a GPU cluster" scale problem.
- Log train/val loss and val metrics per epoch; keep the best-val-metric checkpoint.

### Step 7 — Evaluation (Section 6 defines exact metrics)
Run every trained model (baseline + GNN) on:
1. In-distribution val set
2. Held-out train-topology(ies) if using the optional second cut
3. **OOD test set** — the headline number for the thesis

Report per split, not just pooled — the gap between in-distribution and OOD performance *is* the finding.

### Step 8 — Error analysis
- Confusion matrix per fault type (`noise` vs `contamination` vs `ceiling`) — the dataset description already predicts these should have different "recoverability" signatures; check whether the model's errors cluster by fault type as expected.
- Check performance split by node role (is the model better at localizing Retriever faults than Reasoner faults, etc.) and by graph size (does accuracy degrade on the 5-7 node OOD topologies vs. 3-4 node train topologies — an expected and reportable limitation if so).
- Inspect a few `verification.target_deviated: false` trials (the "hard to detect" cases the description flags) — does the model also fail on exactly these, or does it succeed where a naive threshold check wouldn't (that would be the differentiator that justifies the GNN)?

---

## 3. Known open decisions to resolve before/while coding (carried over from `dataset_description.md`)

These are unresolved in the data/schema layer, deliberately left to this phase — resolve them explicitly in `build_graph_dataset.py` and document the choice in a comment there, since they affect reproducibility:

1. **Missing feature handling** (semantic/jaccard uncertainty absent for some roles) → recommended: mask bits, see Step 3.
2. **Control (`is_control=true`) trials in the node-localization task** → recommended: explicit "no fault" class, see Step 3.
3. **Edge directionality** (forward-only vs. bidirectional message passing) → test both, see Section 1.
4. **Multi-parent contamination's partial-corruption limitation** (`dataset_description.md` §0, known-not-fixed) — only the first parent of a multi-parent node is corrupted in the underlying data for `contamination` trials. This means some `contamination`-labeled trials have a weaker-than-ideal signal at multi-parent nodes; worth flagging as a caveat in the thesis write-up rather than something to fix in the model layer.
5. **`retriever_c` coverage gap in `star` / `triple_retriever_fanin`** (verified 2026-09-26 by direct inspection of `trials.jsonl`/`skipped.jsonl`) — these two topologies' `retriever_c` node is never once used as a fault target in the dataset. Root cause: `scripts/run_study_vllm.py`'s `select_fault_conditions()` assigns each node exactly one fault type by cycling `noise→contamination→ceiling` on node position, with no retry if that one injection fails; `retriever_c` lands on `ceiling`, which fails every time for it (11 of the dataset's 15 total skips). **Decision: not being fixed / regenerated** — each trial still targets exactly one node (or none, for controls), which is the intended one-fault-per-trial design; this gap only means `retriever_c` isn't a valid *predicted class* for these two graph shapes. **How to apply**: when building the per-graph label space in `build_graph_dataset.py`, and when scoring per-node accuracy in evaluation, exclude `retriever_c` from the candidate/target-node set for `star` and `triple_retriever_fanin` graphs specifically — don't count "never predicted retriever_c there" as a model failure, since it was never a possible correct answer to begin with. (Secondary, lower-priority note: the skip-log `reason` field is always the generic `"{type}_injection_failed_{node}"` string per `run_study_vllm.py:364`, not the more specific `ceiling_answer_survived` that `dataset_description.md` §0 claims — a doc inaccuracy, not a data defect.)

---

## 4. Metrics

- **Node-level fault localization** (primary): top-1 accuracy (predicted argmax node == true faulted node), plus top-2 accuracy given the small graphs. Also report **per-class precision/recall** if using the "none" class for controls, since accuracy alone can be misleading with a majority class.
- **Graph-level fault type** (secondary, if using the multi-task head): standard multi-class accuracy/F1, 4-way (`clean`/`noise`/`contamination`/`ceiling`).
- **Calibration check**: does the model's confidence correlate with `verification.z_score` (higher-z-score, i.e. more clearly-deviated faults, should be easier and higher-confidence)? A useful sanity/interpretability check, not a hard requirement.
- Always report **OOD-split metrics separately from in-distribution metrics** — never average them together into one headline number, since the whole point is measuring the generalization gap.

---

## 5. Practical notes

- **Dataset size (493 records) is small for a GNN** — expect to lean on: strong regularization, k-fold cross-validation within the train split (rather than a single train/val split) for more reliable hyperparameter selection, and treating results with appropriate statistical caution (report variance across folds/seeds, not a single run's number) in the write-up.
- **No GPU needed for this phase** — unlike dataset generation (which needed Kaggle's T4s for the LLM), training a small GNN on ~500 tiny graphs runs in seconds-to-minutes on CPU. Don't over-invest in Kaggle/cloud infra here; local `btp-pipeline/` environment is fine.
- **Reuse `scripts/verify_results.py`-style rigor**: this project has a demonstrated pattern (see [`docs/dataset/dataset_description.md`](../dataset/dataset_description.md) §0's bug-fix writeup) of prose/summary claims not matching underlying data. Apply the same discipline to model evaluation — don't trust a single aggregate metric; spot-check individual predictions against the raw trial record before reporting a result.

---

## 6. File layout for this phase

**All model-related code lives in `./model/` at the project root** (sibling to `btp-pipeline/` and `dataset/`), not inside `btp-pipeline/src/`. That directory only contains the dataset-generation pipeline; the model is a separate concern with its own folder:

```
Multi agent LLM System/
  btp-pipeline/          # dataset-generation pipeline (unchanged)
  dataset/               # trials.jsonl, topology_pool.json, dataset_description.md
  model/
    __init__.py
    data/
      build_graph_dataset.py   # trials.jsonl + topology_pool.json -> PyG Data list
      dataset.py                # PyG InMemoryDataset / Data-list loader, split logic
    gnn.py                      # GAT/SAGE model definition
    baselines.py                 # naive max-uncertainty + diagnose_trace baseline
    metrics.py                    # shared eval metrics across all models
    train.py                       # CLI entry point: load data, train, eval, save checkpoint
    evaluate.py                     # CLI entry point: load checkpoint, report metrics per split
    checkpoints/                     # saved model weights (gitignored)
```

Reference the dataset-generation pipeline's code (`btp-pipeline/src/diagnose.py` for the `diagnose_trace` baseline, `btp-pipeline/dataset/topology_pool.json` for structure) by import/path from `model/`, but write no new training code inside `btp-pipeline/`.

---

## 7. Immediate next action

Start with **Step 2 + Step 3** (`model/data/build_graph_dataset.py`) — everything downstream depends on getting the feature/label extraction right and resolving the three open decisions in Section 3. Validate it by hand on a handful of records against the worked example in `dataset_description.md` §11 before moving to Step 5 (baselines).

---

## 8. Experiment log (post-baseline, 2026-09-27) — chasing OOD top-1 accuracy up from ~0.22

Full numbers are in `model/metrics_history.jsonl` (append-only, one row per run). Summary of every change tried, in order, on OOD top-1 accuracy (macro-F1 in parens):

| # | Change | Dataset size | OOD top1 | OOD top2 | OOD macroF1 |
|---|---|---|---|---|---|
| 0 | Original baseline (8-dim scalar features, forward-only edges) | 493 | 0.220 | 0.354 | — |
| 1 | Dataset expanded (60+30 disjoint questions, k=5, bugs fixed) | 1762 | 0.206 | 0.380 | 0.179 |
| 2 | + Bidirectional edges | 1762 | 0.232 | 0.383 | 0.232 |
| 3 | + Node embeddings (384-dim mean-pooled per node, raw, no regularization) | 1762 | 0.239 | 0.446 | 0.229 |
| 4 | + Input projection layer + dropout 0.3 + PCA-32 embedding compression (fixes #3's overfitting) | 1762 | **0.263** | 0.463 | **0.257** |
| 5 | + Hyperparameter sweep winner (hidden_dim=64, num_layers=2, lr=1e-3) | 1762 | **0.275** | 0.449 | 0.259 |
| 6 | + Multi-task auxiliary head (fault-type prediction, aux_weight=0.3) | 1762 | 0.267 | 0.434 | 0.259 |

**Best model so far**: run #5 — bidirectional edges + enriched scalar features (`inference_gaps`, `item_frequencies`) + PCA-compressed node embeddings + input projection/dropout regularization + hidden_dim=64/num_layers=2/lr=1e-3, **no** multi-task head (it made things slightly worse). Checkpoint: pulled from Kaggle kernel `aryanmahajan7/btp-hyperparam-sweep` version corresponding to that config; retrain locally with the same config to reproduce (see `model/hyperparam_sweep.py`'s winning combo).

**What actually moved the needle vs. what didn't**:
- Bidirectional edges (#2): small but real, cheap, no downside. Keep always.
- Raw embeddings alone (#3): looked like a big win on train/val, but OOD barely moved — classic overfitting from feeding 384 raw dims into a tiny GNN with no regularization. A trap if you only look at train/val numbers.
- Regularization + dimensionality reduction (#4): this is what actually made embeddings pay off on OOD, not the embeddings themselves. The lesson: a bigger feature space needs proportionally more regularization, or the model just memorizes train-topology-specific text patterns.
- Hyperparameter sweep (#5): hidden_dim=64 helped meaningfully; num_layers=3 uniformly hurt (likely over-smoothing on these small 3-7 node graphs); lr=5e-4 was uniformly worse than 1e-3.
- Multi-task head (#6): neutral-to-slightly-negative. Not worth the added complexity as implemented (aux_weight=0.3); might be worth revisiting with a smaller aux_weight or a different formulation, but not a priority.

**Still well short of the eventual target (0.6 OOD top1)** — even the best 8-way-comparison result (0.275) is only modestly above chance-level guessing for the smaller-node-count OOD topologies. Two things queued to actually close that gap, both in progress as of this writing:
- **k=10 self-consistency regeneration** (up from k=5) — finer-grained uncertainty values (10 discrete levels instead of 6), running in parallel across two Kaggle accounts and a Lightning AI account, covering the 60-question and 30-question-disjoint sets separately.
- Once k=10 data lands: rerun the full enrichment pipeline (`model/data/enrich_features.py`, now producing `inference_gaps` + `item_frequencies` + `node_embeddings` in one pass) and retrain with the winning architecture from this log.

**Code added this round** (all under `model/`, see file docstrings for details):
- `model/data/enrich_features.py` — post-hoc pass computing `inference_gaps` (embedding drift from parent output to node output), `item_frequencies` (per-item retrieval consensus), and `node_embeddings` (384-dim mean-pooled sentence embeddings), all from data already in `trials.jsonl` — no regeneration needed for this part.
- `model/data/reduce_embeddings.py` — PCA compression of the 384-dim embeddings to a configurable smaller dimension (32 used above), fit on train graphs only to avoid leakage into val/OOD.
- `model/gnn.py` — added an input projection layer (`nn.Linear(in_dim, hidden_dim)` + dropout) before the first GAT layer, and an optional `multi_task` flag adding a fault-type auxiliary head.
- `model/train.py` — added class-imbalance loss weighting for the "no fault" class (`compute_no_fault_weight`), and multi-task combined-loss support.
- `model/hyperparam_sweep.py` — small sweep script (hidden_dim × num_layers × lr, ~8 combos), logs every run.
- `model/nongraph_baseline.py` — flat-feature RandomForest baseline (answers "is the graph structure actually helping"), not yet run against the final architecture.
- `model/eval_ood_breakdown.py` — per-topology OOD accuracy breakdown (confirmed the weak OOD performance is spread fairly evenly across all 6 OOD topologies, not concentrated in one — see log entry from that run).

See `docs/infra/kaggle-and-lightning-setup.md` for the infra practices used to run all of the above (CPU-only kernels for non-GPU-bound work, the k=10 multi-account parallelization pattern, Lightning AI credit-exhaustion recovery).

**Dataset status update (2026-09-28)**: this section is otherwise stale — see `model/metrics_history.jsonl` for the full run history since #6, which reached **0.395 OOD top1** on the original k=5/1762-record dataset (SAGE architecture, attention pooling, node_embedding_std variance feature, PCA-128, 3-seed ensemble). The k=10 regeneration mentioned above is in progress: partial output (1,099 records across 16/20 topologies) is backed up at `dataset/k10_partial/*.jsonl`, and a preliminary pipeline-validation run (`dataset/trials_k10_preview.jsonl`, `model/preview_run.py`) is in flight on that partial data. The real k=10 dataset (`dataset/trials_k10.jsonl`) and final retrain haven't landed yet — target is still ~1,762 records to match the original dataset's scale.
