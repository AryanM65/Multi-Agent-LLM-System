"""Architecture comparison -- GAT (default) vs GCN vs SAGE, at the winning
hyperparams from model/hyperparam_sweep.py (hidden_dim=64, num_layers=2,
lr=1e-3). Answers whether GAT's learned attention is actually earning its
extra complexity over simpler fixed-aggregation conv layers. See
docs/model/futurework.md Section 6.
"""
import torch

from model.data.load_raw import load_topologies, load_trials, validate
from model.data.build_graph_dataset import build_dataset
from model.data.split import split_graphs
from model.data.reduce_embeddings import fit_and_reduce_embeddings
from model.gnn import FaultLocalizerGNN
from model.train import train
from model.evaluate import evaluate, macro_prf1
from model.baselines import evaluate_baseline
from model.metrics_log import log_run

HIDDEN_DIM = 64
NUM_LAYERS = 2
LR = 1e-3
DROPOUT = 0.3
SEED = 0
EPOCHS = 150
PATIENCE = 30
PCA_COMPONENTS = 32
CONV_TYPES = ["gat", "gcn", "sage"]


def run_one(graphs, train_set, val_set, ood_set, conv_type):
    torch.manual_seed(SEED)
    in_dim = graphs[0].x.shape[1]
    model = FaultLocalizerGNN(in_dim=in_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                              dropout=DROPOUT, conv_type=conv_type)

    model, best_val_acc, best_epoch, epochs_run = train(
        model, train_set, val_set, epochs=EPOCHS, lr=LR, patience=PATIENCE, verbose=False,
    )

    train_metrics = evaluate(model, train_set)
    val_metrics = evaluate(model, val_set)
    ood_metrics = evaluate(model, ood_set)
    baseline_ood = evaluate_baseline(ood_set)

    train_prf1 = macro_prf1(train_metrics["y_true"], train_metrics["y_pred"])
    val_prf1 = macro_prf1(val_metrics["y_true"], val_metrics["y_pred"])
    ood_prf1 = macro_prf1(ood_metrics["y_true"], ood_metrics["y_pred"])

    config = {
        "seed": SEED, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT,
        "lr": LR, "epochs_requested": EPOCHS, "patience": PATIENCE, "in_dim": in_dim,
        "bidirectional_edges": True, "use_embeddings": True, "pca_components": PCA_COMPONENTS,
        "conv_type": conv_type, "arch_sweep": True,
    }
    metrics = {
        "train": {"top1_accuracy": train_metrics["top1_accuracy"], "top2_accuracy": train_metrics["top2_accuracy"], **train_prf1, "n": len(train_set)},
        "val": {"top1_accuracy": val_metrics["top1_accuracy"], "top2_accuracy": val_metrics["top2_accuracy"], **val_prf1, "n": len(val_set)},
        "ood": {"top1_accuracy": ood_metrics["top1_accuracy"], "top2_accuracy": ood_metrics["top2_accuracy"], **ood_prf1, "n": len(ood_set)},
        "baseline_ood_top1_accuracy": baseline_ood["accuracy"],
        "best_epoch": best_epoch, "epochs_run": epochs_run,
    }
    log_run(config=config, metrics=metrics, notes=f"Architecture sweep run (conv_type={conv_type})")
    return config, metrics


if __name__ == "__main__":
    topologies = load_topologies()
    trials = load_trials()
    validate(trials, topologies)
    graphs = build_dataset(trials, topologies, bidirectional=True, use_embeddings=True)
    train_set, val_set, ood_set = split_graphs(graphs, topologies, seed=SEED)

    pca = fit_and_reduce_embeddings(train_set, graphs, n_components=PCA_COMPONENTS, seed=SEED)
    print(f"PCA explained variance ratio (sum): {pca.explained_variance_ratio_.sum():.3f}")
    print(f"in_dim={graphs[0].x.shape[1]}  train={len(train_set)} val={len(val_set)} ood={len(ood_set)}")

    results = []
    for conv_type in CONV_TYPES:
        print(f"\n--- conv_type={conv_type} ---")
        config, metrics = run_one(graphs, train_set, val_set, ood_set, conv_type)
        ood_top1 = metrics["ood"]["top1_accuracy"]
        print(f"ood_top1={ood_top1:.3f}  ood_top2={metrics['ood']['top2_accuracy']:.3f}  ood_macroF1={metrics['ood']['f1']:.3f}")
        results.append((ood_top1, conv_type, metrics))

    results.sort(key=lambda r: -r[0])
    print("\n=== Architecture sweep ranked by OOD top1 ===")
    for ood_top1, conv_type, metrics in results:
        print(f"ood_top1={ood_top1:.3f}  conv_type={conv_type}")
