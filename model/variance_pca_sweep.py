"""PCA-dim sweep specifically for the node_embedding_std variance config
(model/variance_run.py used pca_components=128 straight from the mean-only
sweep winner, never re-swept with variance features on). Sweeps
[16, 32, 64, 128] at the winning SAGE hd64/nl3/lr5e-4 config with
use_embedding_variance=True. See docs/model/futurework.md.
"""
import torch

from model.data.load_raw import load_topologies, load_trials, validate
from model.data.build_graph_dataset import build_dataset
from model.data.split import split_graphs
from model.data.reduce_embeddings import fit_and_reduce_embeddings_with_variance
from model.gnn import FaultLocalizerGNN
from model.train import train
from model.evaluate import evaluate, macro_prf1
from model.baselines import evaluate_baseline
from model.metrics_log import log_run

HIDDEN_DIM = 64
NUM_LAYERS = 3
LR = 5e-4
DROPOUT = 0.3
CONV_TYPE = "sage"
SEED = 0
EPOCHS = 150
PATIENCE = 30
PCA_COMPONENT_OPTIONS = [16, 32, 64, 128]


def run_one(raw_trials, topologies, pca_components):
    graphs = build_dataset(raw_trials, topologies, bidirectional=True, use_embeddings=True,
                            use_embedding_variance=True)
    train_set, val_set, ood_set = split_graphs(graphs, topologies, seed=SEED)
    pca_mean, pca_std = fit_and_reduce_embeddings_with_variance(
        train_set, graphs, n_components=pca_components, seed=SEED,
    )

    torch.manual_seed(SEED)
    in_dim = graphs[0].x.shape[1]
    model = FaultLocalizerGNN(in_dim=in_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                              dropout=DROPOUT, conv_type=CONV_TYPE)
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
        "bidirectional_edges": True, "use_embeddings": True, "use_embedding_variance": True,
        "pca_components": pca_components, "conv_type": CONV_TYPE, "variance_pca_sweep": True,
        "pca_mean_explained_variance": float(pca_mean.explained_variance_ratio_.sum()),
        "pca_std_explained_variance": float(pca_std.explained_variance_ratio_.sum()),
    }
    metrics = {
        "train": {"top1_accuracy": train_metrics["top1_accuracy"], "top2_accuracy": train_metrics["top2_accuracy"], **train_prf1, "n": len(train_set)},
        "val": {"top1_accuracy": val_metrics["top1_accuracy"], "top2_accuracy": val_metrics["top2_accuracy"], **val_prf1, "n": len(val_set)},
        "ood": {"top1_accuracy": ood_metrics["top1_accuracy"], "top2_accuracy": ood_metrics["top2_accuracy"], **ood_prf1, "n": len(ood_set)},
        "baseline_ood_top1_accuracy": baseline_ood["accuracy"],
        "best_epoch": best_epoch, "epochs_run": epochs_run,
    }
    log_run(config=config, metrics=metrics, notes=f"Variance-config PCA sweep run (pca_components={pca_components})")
    return config, metrics


if __name__ == "__main__":
    topologies = load_topologies()
    trials = load_trials()
    validate(trials, topologies)

    results = []
    for pca_components in PCA_COMPONENT_OPTIONS:
        print(f"\n--- pca_components={pca_components} ---")
        config, metrics = run_one(trials, topologies, pca_components)
        ood_top1 = metrics["ood"]["top1_accuracy"]
        print(f"ood_top1={ood_top1:.3f}  ood_top2={metrics['ood']['top2_accuracy']:.3f}  ood_macroF1={metrics['ood']['f1']:.3f}")
        results.append((ood_top1, pca_components, metrics))

    results.sort(key=lambda r: -r[0])
    print("\n=== Variance-config PCA sweep ranked by OOD top1 ===")
    for ood_top1, pca_components, metrics in results:
        print(f"ood_top1={ood_top1:.3f}  pca_components={pca_components}")
