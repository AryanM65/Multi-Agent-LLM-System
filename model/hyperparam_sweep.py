"""Small hyperparameter sweep -- hidden_dim x num_layers x lr, ~8 combos
(not a full grid). Trains each, evaluates OOD, logs every run to
model/metrics_history.jsonl, prints a ranked summary at the end. See
docs/model/futurework.md Section 6.
"""
import itertools

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

HIDDEN_DIMS = [32, 64]
NUM_LAYERS = [2, 3]
LRS = [1e-3, 5e-4]
SEED = 0
EPOCHS = 150
PATIENCE = 30


def run_one(graphs, train_set, val_set, ood_set, hidden_dim, num_layers, lr, use_embeddings):
    torch.manual_seed(SEED)
    in_dim = graphs[0].x.shape[1]
    # dropout=0.3 matches the regularization fix that fixed the embeddings
    # overfit (see model/metrics_history.jsonl run comparison) -- keep it
    # consistent across the sweep rather than reverting to the old 0.2.
    model = FaultLocalizerGNN(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.3)

    model, best_val_acc, best_epoch, epochs_run = train(
        model, train_set, val_set, epochs=EPOCHS, lr=lr, patience=PATIENCE, verbose=False,
    )

    train_metrics = evaluate(model, train_set)
    val_metrics = evaluate(model, val_set)
    ood_metrics = evaluate(model, ood_set)
    baseline_ood = evaluate_baseline(ood_set)

    train_prf1 = macro_prf1(train_metrics["y_true"], train_metrics["y_pred"])
    val_prf1 = macro_prf1(val_metrics["y_true"], val_metrics["y_pred"])
    ood_prf1 = macro_prf1(ood_metrics["y_true"], ood_metrics["y_pred"])

    config = {
        "seed": SEED, "hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": 0.3,
        "lr": lr, "epochs_requested": EPOCHS, "patience": PATIENCE, "in_dim": in_dim,
        "bidirectional_edges": True, "use_embeddings": use_embeddings, "sweep": True,
    }
    metrics = {
        "train": {"top1_accuracy": train_metrics["top1_accuracy"], "top2_accuracy": train_metrics["top2_accuracy"], **train_prf1, "n": len(train_set)},
        "val": {"top1_accuracy": val_metrics["top1_accuracy"], "top2_accuracy": val_metrics["top2_accuracy"], **val_prf1, "n": len(val_set)},
        "ood": {"top1_accuracy": ood_metrics["top1_accuracy"], "top2_accuracy": ood_metrics["top2_accuracy"], **ood_prf1, "n": len(ood_set)},
        "baseline_ood_top1_accuracy": baseline_ood["accuracy"],
        "best_epoch": best_epoch, "epochs_run": epochs_run,
    }
    log_run(config=config, metrics=metrics, notes="Hyperparameter sweep run")
    return config, metrics


if __name__ == "__main__":
    import sys

    use_embeddings = "--embeddings" in sys.argv

    topologies = load_topologies()
    trials = load_trials()
    validate(trials, topologies)
    graphs = build_dataset(trials, topologies, bidirectional=True, use_embeddings=use_embeddings)
    train_set, val_set, ood_set = split_graphs(graphs, topologies, seed=SEED)

    if use_embeddings:
        pca = fit_and_reduce_embeddings(train_set, graphs, n_components=32, seed=SEED)
        print(f"PCA explained variance ratio (sum): {pca.explained_variance_ratio_.sum():.3f}")

    print(f"in_dim={graphs[0].x.shape[1]}  train={len(train_set)} val={len(val_set)} ood={len(ood_set)}")

    results = []
    for hidden_dim, num_layers, lr in itertools.product(HIDDEN_DIMS, NUM_LAYERS, LRS):
        print(f"\n--- hidden_dim={hidden_dim} num_layers={num_layers} lr={lr} ---")
        config, metrics = run_one(graphs, train_set, val_set, ood_set, hidden_dim, num_layers, lr, use_embeddings)
        ood_top1 = metrics["ood"]["top1_accuracy"]
        print(f"ood_top1={ood_top1:.3f}  ood_top2={metrics['ood']['top2_accuracy']:.3f}  ood_macroF1={metrics['ood']['f1']:.3f}")
        results.append((ood_top1, config, metrics))

    results.sort(key=lambda r: -r[0])
    print("\n=== Sweep ranked by OOD top1 ===")
    for ood_top1, config, metrics in results:
        print(f"ood_top1={ood_top1:.3f}  hidden_dim={config['hidden_dim']} num_layers={config['num_layers']} lr={config['lr']}")
