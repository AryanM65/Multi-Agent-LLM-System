# Model Plan — Final, Step-by-Step (Simple Language + Code)

> This is the one file to follow to actually build the fault-localization model. It assumes the dataset is done (`dataset/trials.jsonl`, 493 verified trials — see `btp-pipeline/dataset_description.md` and `btp-pipeline/model.md` for the deep-dive background). This file is the short, practical version: what to do, in what order, with just enough code to see how each step actually works.
>
> **All code for this phase goes in `./model/`** at the project root (sibling to `btp-pipeline/` and `dataset/`), never inside `btp-pipeline/src/`.

---

## The goal, in one line

Given the "symptoms" of a pipeline run (how confused each agent was), guess **which agent caused the problem** — without being told the answer. We already have 493 examples where we *do* know the true answer (because we deliberately broke things when generating the data), so this is normal supervised learning.

---

## Step 0 — Set up your tools

```bash
pip install torch torch-geometric scikit-learn pandas numpy matplotlib
```

No GPU needed. The dataset is tiny (493 small graphs) — this trains on a laptop CPU in minutes, not hours.

---

## Step 1 — Load the data and make sure it's trustworthy

**What this step does**: read the raw files, join them together, and double check nothing is broken before we build anything on top of it.

Two files matter:
- `dataset/trials.jsonl` — one line per trial (the actual data)
- `dataset/topology_pool.json` — describes the *shape* of each of the 20 pipeline layouts (which nodes exist, who feeds whom, train vs. OOD)

```python
# model/data/load_raw.py
import json

def load_topologies(path="dataset/topology_pool.json"):
    topos = json.load(open(path, encoding="utf-8"))
    return {t["topology_id"]: t for t in topos}

def load_trials(path="dataset/trials.jsonl"):
    return [json.loads(line) for line in open(path, encoding="utf-8")]

topologies = load_topologies()
trials = load_trials()
print(f"{len(trials)} trials across {len(topologies)} topologies")
```

We already verified this data by hand (2026-09-26): 493 trials, no missing node data, no duplicates, label counts match documentation exactly (`noise=162, contamination=154, ceiling=103, clean=74`). One known, accepted quirk: two topologies (`star`, `triple_retriever_fanin`) have a node called `retriever_c` that's never the fault target — that's fine, it's just never a valid "correct answer" for those two shapes (see Step 3).

---

## Step 2 — Turn each trial into a "graph" a model can read

**Simple explanation**: A graph here just means "dots connected by arrows." Each dot (**node**) is one pipeline agent (Retriever/Reasoner/Writer). Each arrow (**edge**) means "this agent's output feeds into that one." Every dot gets a short list of numbers describing it (its **feature vector**) — that's all the model gets to see.

### The 8 numbers per node

| # | Feature | Plain meaning |
|---|---|---|
| 1 | `lexical_uncertainty` | Did the exact wording agree across 5 repeated tries? (0 = always the same, 1 = always different) |
| 2 | `semantic_uncertainty` | Did the *meaning* agree, even if wording differed? (Reasoner/Writer only) |
| 3 | `jaccard_uncertainty` | For Retriever: did the *set* of facts returned overlap across tries? |
| 4 | `has_semantic` | 1 if #2 is a real measured value, 0 if it's just a meaningless placeholder |
| 5 | `has_jaccard` | Same idea, for #3 |
| 6-8 | `is_retriever` / `is_reasoner` / `is_writer` | Which role this node is (exactly one of these three is 1) |

**Why the "has_..." flags exist**: `semantic_uncertainty` is `null` for every Retriever node, because that metric literally doesn't make sense for it. If we just replaced `null` with `0`, the model couldn't tell "this is a genuinely perfect score of 0" apart from "this metric was never computed." The flag tells it which case it is.

### Example, made concrete

Topology: `Retriever → Reasoner → Writer`. A `noise` fault was injected at the Retriever.

| node | vector `[lex, sem, jac, has_sem, has_jac, is_R, is_Re, is_W]` |
|---|---|
| retriever | `[0.4, 0, 0, 0, 0, 1, 0, 0]` |
| reasoner | `[0.6, 0.55, 0, 1, 0, 0, 1, 0]` |
| writer | `[0.2, 0.18, 0, 1, 0, 0, 0, 1]` |

Edges: `retriever → reasoner`, `reasoner → writer`.
Label: retriever = 1 (the true fault), reasoner = 0, writer = 0.

### Code

```python
# model/data/build_graph_dataset.py
import json
import torch
from torch_geometric.data import Data

ROLE_LIST = ["retriever", "reasoner", "writer"]

def node_features(trial, node_id, role):
    lex = trial["uncertainties"].get(node_id, 0.0) or 0.0
    sem = trial["semantic_uncertainties"].get(node_id)
    jac = trial.get("jaccard_uncertainties", {}).get(node_id)

    has_sem = 1.0 if sem is not None else 0.0
    has_jac = 1.0 if jac is not None else 0.0
    sem = sem or 0.0
    jac = jac or 0.0

    role_onehot = [1.0 if role == r else 0.0 for r in ROLE_LIST]
    return [lex, sem, jac, has_sem, has_jac] + role_onehot

def trial_to_graph(trial, topology):
    node_ids = list(topology["nodes"].keys())
    idx_of = {n: i for i, n in enumerate(node_ids)}

    x = [node_features(trial, n, topology["nodes"][n]["role"]) for n in node_ids]

    edges = topology["edges"]
    edge_index = [[idx_of[src] for src, dst in edges],
                  [idx_of[dst] for src, dst in edges]]

    target = trial.get("fault_config", {}).get("target_node")
    y = idx_of[target] if target else -1  # -1 = "clean" / no fault (handled specially, see below)

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor([y], dtype=torch.long),
        topology_id=trial["topology_id"],
    )
```

**Two decisions baked into this code** (documented in `btp-pipeline/model.md` Section 3):
1. Missing values get a `0` **plus** an explicit `has_...` flag — never a silent `0` alone.
2. `is_control` (clean, no fault) trials get `y = -1`, a placeholder for "no fault node" — decide at training time whether to use a `k+1`-way class ("no fault" is its own answer option) or drop controls from the localization task and use them only as negatives elsewhere. Recommended: `k+1`-way, it's cleaner.

---

## Step 3 — Split into train / validation / test (OOD)

**Simple explanation**: three piles.
- **Train** — the model studies these, answers included (like practice problems).
- **Validation** — a quiz during practice, to check it's not just memorizing.
- **Test (OOD)** — 6 topologies the model has *never seen in any form*, opened only once at the very end (the real final exam).

```python
# model/data/split.py
import random

def split_trials(trials, topologies, val_fraction=0.2, seed=0):
    train_topo_trials = [t for t in trials if topologies[t["topology_id"]]["split"] == "train"]
    ood_trials       = [t for t in trials if topologies[t["topology_id"]]["split"] == "ood_test"]

    # group by question so the same question never appears in both train and val
    by_question = {}
    for t in train_topo_trials:
        by_question.setdefault(t["question"], []).append(t)

    questions = list(by_question.keys())
    random.Random(seed).shuffle(questions)
    n_val_q = int(len(questions) * val_fraction)
    val_questions = set(questions[:n_val_q])

    train_set = [t for q, ts in by_question.items() if q not in val_questions for t in ts]
    val_set   = [t for q, ts in by_question.items() if q in val_questions for t in ts]

    return train_set, val_set, ood_trials
```

**Reminder**: `retriever_c` in `star`/`triple_retriever_fanin` is never a true fault target — when scoring accuracy later, exclude it from the "possible correct answers" for graphs of those two topologies, so the model isn't penalized for something it was never shown an example of.

---

## Step 4 — Baseline first (the thing the real model has to beat)

**Simple explanation**: before building anything clever, try the dumbest possible rule: "guess whichever node has the highest uncertainty number." If this already works great, we don't need a fancy model. If it fails in specific ways, that tells us what the fancy model needs to fix.

```python
# model/baselines.py
def naive_max_uncertainty_baseline(graph):
    lexical_uncertainties = graph.x[:, 0]  # column 0 = lexical_uncertainty
    return int(lexical_uncertainties.argmax())
```

Run this over the val set and the OOD set, record accuracy. Keep these numbers — they're the bar the GNN must clear.

---

## Step 5 — Build the real model: a small GNN

**Simple explanation of a GNN**: a neural network built to read "dots + arrows + numbers" instead of a flat table. It works by **message passing**: each round, every node looks at its neighbors' numbers, mixes them in with its own, and updates its own summary. After 2-3 rounds, a node's summary reflects not just itself but its neighborhood — which is exactly what's needed here, since "who caused this" often depends on a pattern *across* connected nodes, not just one node's number in isolation.

```python
# model/gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class FaultLocalizerGNN(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=32, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim))
        self.node_head = nn.Linear(hidden_dim, 1)  # one "fault score" per node

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return self.node_head(h).squeeze(-1)  # shape: [num_nodes] — one logit per node
```

- **"Layer"** = one round of message passing. 2-3 is enough for these small graphs (3-7 nodes).
- **`GATConv`** = "Graph Attention" layer — lets each node decide *how much* to weight each neighbor's info, rather than averaging everything equally. Good fit here since "how much did upstream confusion actually matter" is naturally a weighting question.
- **Output**: one number per node ("how likely is this node the fault"), turned into a probability via softmax across the graph's nodes when computing loss/predictions.

---

## Step 6 — Train it

**Simple explanation of training**: show the model an example, let it guess, compare the guess to the true answer (this comparison is the **loss** — a single number, lower = better), then nudge the model's internal settings (**weights**) slightly toward a better guess next time. Do this for every example, many times over (each full pass = one **epoch**).

```python
# model/train.py
import torch
from torch_geometric.loader import DataLoader

def train(model, train_graphs, val_graphs, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            # mask out control (-1) graphs, or handle as extra class -- see Step 2 note
            loss = torch.nn.functional.cross_entropy(
                logits_per_graph(logits, batch), batch.y
            )
            loss.backward()
            optimizer.step()

        val_acc = evaluate(model, val_graphs)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "model/checkpoints/best.pt")

        print(f"epoch {epoch}: val_acc={val_acc:.3f}")
```

(`logits_per_graph` is a small helper needed because PyG batches multiple small graphs together — it splits the flat logits tensor back into per-graph slices before computing loss. Left as a real implementation detail for when you write this file.)

**Watch for overfitting**: if train accuracy keeps climbing but val accuracy stalls or drops, the model is memorizing the practice problems instead of learning the general pattern. Stop training at the epoch with the best *val* accuracy (this is what `best_val_acc`/checkpoint-saving above does — this pattern is called **early stopping**).

---

## Step 7 — Grade it, honestly

### What metric are we actually using? Short answer: yes, accuracy — but not *only* accuracy

**Primary metric: top-1 node accuracy.** For each graph, the model predicts one node as "the fault." Accuracy = what fraction of graphs it got exactly right (predicted node == true faulted node). This is the headline number and the simplest to explain, so it's the main one to report.

But accuracy alone can be misleading here, for reasons specific to this dataset — so a few more numbers are reported alongside it, each answering a question plain accuracy can't:

| Metric | Plain-language question it answers | Why it's needed here |
|---|---|---|
| **Top-1 accuracy** (primary) | "How often is its single best guess exactly right?" | The headline number, easiest to explain in the thesis. |
| **Top-2 accuracy** | "How often is the true node one of its top 2 guesses?" | Graphs have as few as 3 nodes (so top-1 is already a real test) up to 7 nodes (OOD topologies) — top-2 shows whether the model is "close" on cases it doesn't nail exactly, which matters more on bigger graphs where random guessing is much harder. |
| **Per-class precision/recall** (one row per node-role, or per specific node) | "When it says 'Retriever', is it usually right? And does it catch most of the *actual* Retriever-fault cases, or miss a lot of them?" | Plain accuracy can hide a model that's great at spotting one role (e.g. Writer faults are "loud" and easy) while quietly failing on another (e.g. Reasoner faults). Precision/recall per class exposes this; accuracy alone doesn't. |
| **Macro-F1** (average of per-class F1 scores, each class weighted equally) | "How well does it do on rare or hard cases, not just the big obvious ones?" | The dataset isn't perfectly balanced — some fault types have more examples (`noise=162` vs `ceiling=103`), some nodes are fault targets far less often than others (see the `retriever_c` gap in `model.md`/`futurework.md`). Macro-F1 stops a model from "winning" by just being good at the most common cases. |
| **Confusion matrix** (which wrong node it picked, when wrong) | "When it's wrong, *what* does it confuse the true node with?" | A model that always confuses "reasoner" with its immediate neighbor "writer" is failing in a very different (and more explainable) way than one that guesses randomly among all nodes. This is qualitative, not a single number, but essential for the error-analysis in Step 8. |

**One dataset-specific rule, not optional**: for graphs from `star` and `triple_retriever_fanin` (the two topologies where `retriever_c` is never a true fault target — see `btp-pipeline/model.md` Section 3, item 5), **exclude `retriever_c` from the set of "correct answer" candidates** when computing any of the metrics above for those graphs. It was never a valid ground-truth label there, so a low score involving it isn't a real model failure — it's a data-coverage artifact, and reporting it as a failure would understate the model unfairly.

### Code

Run the best checkpoint once on the OOD set (topologies never seen in any form):

```python
# model/evaluate.py
def evaluate(model, graphs, k_for_top_k=2):
    model.eval()
    top1_correct, top2_correct, total = 0, 0, 0
    all_true, all_pred = [], []  # for precision/recall/F1/confusion matrix, e.g. via sklearn

    with torch.no_grad():
        for g in graphs:
            logits = model(g.x, g.edge_index)
            true_node = int(g.y)

            ranked = logits.argsort(descending=True)
            pred = int(ranked[0])

            if pred == true_node:
                top1_correct += 1
            if true_node in ranked[:k_for_top_k].tolist():
                top2_correct += 1

            all_true.append(true_node)
            all_pred.append(pred)
            total += 1

    return {
        "top1_accuracy": top1_correct / total,
        "top2_accuracy": top2_correct / total,
        "y_true": all_true,   # feed these into sklearn.metrics.classification_report /
        "y_pred": all_pred,   # confusion_matrix for precision/recall/F1 per class
    }
```

Report **three sets of numbers side by side**, never averaged together:
1. Metrics on the train-topology val set
2. Metrics on the OOD test set
3. Baseline (Step 4)'s metrics on the same OOD set

The gap between (1) and (2) tells you how much the model generalizes vs. memorizes. The gap between (2) and (3) tells you whether the GNN is actually earning its complexity — top-1 accuracy is the number to lead with in each comparison, with top-2/macro-F1/confusion matrix as supporting detail.

---

## Step 8 — Look at what it got wrong

Don't stop at one accuracy number. Break it down:
- By fault type (`noise` vs `contamination` vs `ceiling`) — the data description predicts these have different "difficulty" signatures; confirm the model's mistakes match that pattern.
- By node role — is it worse at localizing Reasoner faults than Retriever faults?
- By graph size — does it get worse on the bigger 5-7 node OOD topologies?
- Manually check a few `verification.target_deviated: false` trials (cases where the fault was injected but barely showed up as a symptom) — these are the genuinely hard cases; a good model's mistakes should cluster here, not spread randomly.

---

## File layout (where each piece of code above actually lives)

```
Multi agent LLM System/
  btp-pipeline/       # dataset-generation pipeline (already done)
  dataset/            # trials.jsonl, topology_pool.json
  model/
    data/
      load_raw.py
      build_graph_dataset.py
      split.py
    baselines.py
    gnn.py
    train.py
    evaluate.py
    checkpoints/       # saved model weights (not committed to git)
```

---

## Order to actually do this in

1. `model/data/load_raw.py` — load + sanity check (Step 1)
2. `model/data/build_graph_dataset.py` — turn trials into graphs (Step 2)
3. `model/data/split.py` — train/val/OOD split (Step 3)
4. `model/baselines.py` — naive baseline, get a number to beat (Step 4)
5. `model/gnn.py` + `model/train.py` — build and train the real model (Steps 5-6)
6. `model/evaluate.py` — final honest grading on OOD (Step 7)
7. Error analysis, by hand, using the trained model + the raw trial records (Step 8)

Start at step 1, get each one fully working before moving to the next — don't jump ahead to the GNN before the baseline number exists, since without it you can't tell if the GNN is actually helping.
