"""Reproduce the current best run (model/attn_ensemble.py: SAGE, hd=64, nl=3,
dropout=0.3, lr=5e-4, bidirectional edges, PCA-128 mean+std embeddings,
attention pooling, 3-seed ensemble) and dump REAL per-node probabilities on
the OOD set, to compute genuine ROC/PR curves and a role-level confusion
matrix -- not estimated numbers.
"""
import sys, os, json
sys.path.insert(0, r"D:\aryan\Projects\Multi agent LLM System")

import torch
import numpy as np

from model.data.load_raw import load_topologies, load_trials, validate
from model.data.build_graph_dataset import build_dataset
from model.data.split import split_graphs
from model.data.reduce_embeddings import fit_and_reduce_embeddings_with_variance
from model.gnn import FaultLocalizerGNN
from model.train import train
from model.evaluate import evaluate as evaluate_single, macro_prf1
from model.ensemble import evaluate_ensemble

HIDDEN_DIM = 64
NUM_LAYERS = 3
LR = 5e-4
DROPOUT = 0.3
CONV_TYPE = "sage"
POOL_TYPE = "attention"
SEEDS = [0, 1, 2]
EPOCHS = 150
PATIENCE = 30
PCA_COMPONENTS = 128

OUT = os.path.dirname(os.path.abspath(__file__))

topologies = load_topologies()
trials = load_trials()
validate(trials, topologies)
graphs = build_dataset(trials, topologies, bidirectional=True, use_embeddings=True,
                        use_embedding_variance=True)
train_set, val_set, ood_set = split_graphs(graphs, topologies, seed=0)
fit_and_reduce_embeddings_with_variance(train_set, graphs, n_components=PCA_COMPONENTS, seed=0)
print(f"in_dim={graphs[0].x.shape[1]}  train={len(train_set)} val={len(val_set)} ood={len(ood_set)}")

models = []
for seed in SEEDS:
    print(f"--- training seed={seed} ---")
    torch.manual_seed(seed)
    in_dim = graphs[0].x.shape[1]
    model = FaultLocalizerGNN(in_dim=in_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                              dropout=DROPOUT, conv_type=CONV_TYPE, pool_type=POOL_TYPE)
    model, best_val_acc, best_epoch, epochs_run = train(
        model, train_set, val_set, epochs=EPOCHS, lr=LR, patience=PATIENCE, verbose=False,
    )
    single_ood = evaluate_single(model, ood_set)
    print(f"seed={seed} ood_top1={single_ood['top1_accuracy']:.3f} (best_epoch={best_epoch}, epochs_run={epochs_run})")
    models.append(model)

ood_metrics = evaluate_ensemble(models, ood_set)
print(f"ENSEMBLE ood_top1={ood_metrics['top1_accuracy']:.3f}  ood_top2={ood_metrics['top2_accuracy']:.3f}")

# ---------------------------------------------------------------------------
# Dump real per-node ensembled probabilities for every OOD graph, plus role
# labels, for ROC/PR curves and a role-level confusion matrix.
# ---------------------------------------------------------------------------
for m in models:
    m.eval()

records = []
KNOWN_UNREACHABLE = {"star": "retriever_c", "triple_retriever_fanin": "retriever_c"}

graph_records = []

with torch.no_grad():
    for gi, g in enumerate(ood_set):
        probs_sum = None
        for m in models:
            out = m(g.x, g.edge_index)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits, dim=0)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        avg_probs = (probs_sum / len(models)).numpy()  # [num_nodes + 1] (+1 = "no fault")

        true_y = int(g.y.item())
        pred_y = int(avg_probs.argmax())
        topo = topologies[g.topology_id]
        unreachable = KNOWN_UNREACHABLE.get(g.topology_id)

        true_role = "no_fault" if true_y == len(g.node_ids) else topo["nodes"][g.node_ids[true_y]]["role"]
        pred_role = "no_fault" if pred_y == len(g.node_ids) else topo["nodes"][g.node_ids[pred_y]]["role"]
        graph_records.append({
            "graph_idx": gi, "topology_id": g.topology_id, "true_label": g.true_label,
            "true_role": true_role, "pred_role": pred_role, "correct": true_y == pred_y,
        })

        for i, node_id in enumerate(g.node_ids):
            if node_id == unreachable:
                continue  # never a valid fault target for this topology -- exclude, not a model failure
            role = topo["nodes"][node_id]["role"]
            records.append({
                "graph_idx": gi,
                "topology_id": g.topology_id,
                "true_label": g.true_label,
                "node_id": node_id,
                "role": role,
                "is_true_fault_node": 1 if i == true_y else 0,
                "prob_is_fault": float(avg_probs[i]),
            })
        # also record the "no fault" pseudo-node's score against its own binary target
        records.append({
            "graph_idx": gi,
            "topology_id": g.topology_id,
            "true_label": g.true_label,
            "node_id": "__no_fault__",
            "role": "no_fault",
            "is_true_fault_node": 1 if true_y == len(g.node_ids) else 0,
            "prob_is_fault": float(avg_probs[-1]),
        })

with open(os.path.join(OUT, "ood_node_predictions.json"), "w", encoding="utf-8") as f:
    json.dump(records, f)
with open(os.path.join(OUT, "ood_graph_predictions.json"), "w", encoding="utf-8") as f:
    json.dump(graph_records, f)

print(f"Dumped {len(records)} per-node OOD prediction records.")

# also stash macro metrics for reference
val_metrics = evaluate_ensemble(models, val_set)
train_metrics = evaluate_ensemble(models, train_set)
train_prf1 = macro_prf1(train_metrics["y_true"], train_metrics["y_pred"])
val_prf1 = macro_prf1(val_metrics["y_true"], val_metrics["y_pred"])
ood_prf1 = macro_prf1(ood_metrics["y_true"], ood_metrics["y_pred"])
print("train_prf1", train_prf1)
print("val_prf1", val_prf1)
print("ood_prf1", ood_prf1)
