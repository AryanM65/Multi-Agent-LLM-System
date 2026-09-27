"""Multi-seed SAGE ensemble -- averages logits from 3 independently trained
SAGE models (winning arch from model/arch_sweep.py, hidden_dim=64,
num_layers=2, lr=1e-3) at seeds 0/1/2, and evaluates the ensembled
prediction against each individual seed. See docs/model/futurework.md.
"""
import torch

from model.data.load_raw import load_topologies, load_trials, validate
from model.data.build_graph_dataset import build_dataset
from model.data.split import split_graphs
from model.data.reduce_embeddings import fit_and_reduce_embeddings
from model.gnn import FaultLocalizerGNN
from model.train import train
from model.evaluate import macro_prf1
from model.baselines import evaluate_baseline
from model.metrics_log import log_run

HIDDEN_DIM = 64
NUM_LAYERS = 2
LR = 1e-3
DROPOUT = 0.3
CONV_TYPE = "sage"
SEEDS = [0, 1, 2]
EPOCHS = 150
PATIENCE = 30
PCA_COMPONENTS = 32


def train_one(graphs, train_set, val_set, seed):
    torch.manual_seed(seed)
    in_dim = graphs[0].x.shape[1]
    model = FaultLocalizerGNN(in_dim=in_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                              dropout=DROPOUT, conv_type=CONV_TYPE)
    model, best_val_acc, best_epoch, epochs_run = train(
        model, train_set, val_set, epochs=EPOCHS, lr=LR, patience=PATIENCE, verbose=False,
    )
    return model, best_epoch, epochs_run


def evaluate_ensemble(models, graphs, k_for_top_k=2):
    """Average softmax probabilities across models (not raw logits -- models
    can differ in scale), then compute top1/top2 same as model/evaluate.py."""
    for m in models:
        m.eval()
    top1_correct, top2_correct, total = 0, 0, 0
    all_true, all_pred = [], []

    with torch.no_grad():
        for g in graphs:
            probs_sum = None
            for m in models:
                out = m(g.x, g.edge_index)
                logits = out[0] if isinstance(out, tuple) else out
                probs = torch.softmax(logits, dim=0)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            avg_probs = probs_sum / len(models)
            true_y = int(g.y.item())

            ranked = avg_probs.argsort(descending=True)
            pred_y = int(ranked[0])

            if pred_y == true_y:
                top1_correct += 1
            if true_y in ranked[:k_for_top_k].tolist():
                top2_correct += 1

            all_true.append(true_y)
            all_pred.append(pred_y)
            total += 1

    return {
        "top1_accuracy": top1_correct / total,
        "top2_accuracy": top2_correct / total,
        "y_true": all_true,
        "y_pred": all_pred,
    }


if __name__ == "__main__":
    from model.evaluate import evaluate as evaluate_single

    topologies = load_topologies()
    trials = load_trials()
    validate(trials, topologies)
    graphs = build_dataset(trials, topologies, bidirectional=True, use_embeddings=True)
    train_set, val_set, ood_set = split_graphs(graphs, topologies, seed=0)

    pca = fit_and_reduce_embeddings(train_set, graphs, n_components=PCA_COMPONENTS, seed=0)
    print(f"PCA explained variance ratio (sum): {pca.explained_variance_ratio_.sum():.3f}")
    print(f"in_dim={graphs[0].x.shape[1]}  train={len(train_set)} val={len(val_set)} ood={len(ood_set)}")

    models = []
    for seed in SEEDS:
        print(f"\n--- training seed={seed} ---")
        model, best_epoch, epochs_run = train_one(graphs, train_set, val_set, seed)
        single_ood = evaluate_single(model, ood_set)
        print(f"seed={seed} ood_top1={single_ood['top1_accuracy']:.3f} (best_epoch={best_epoch}, epochs_run={epochs_run})")
        models.append(model)

    ood_metrics = evaluate_ensemble(models, ood_set)
    val_metrics = evaluate_ensemble(models, val_set)
    train_metrics = evaluate_ensemble(models, train_set)
    baseline_ood = evaluate_baseline(ood_set)

    train_prf1 = macro_prf1(train_metrics["y_true"], train_metrics["y_pred"])
    val_prf1 = macro_prf1(val_metrics["y_true"], val_metrics["y_pred"])
    ood_prf1 = macro_prf1(ood_metrics["y_true"], ood_metrics["y_pred"])

    print(f"\n=== Ensemble ({len(SEEDS)} SAGE seeds, avg softmax) ===")
    print(f"ood_top1={ood_metrics['top1_accuracy']:.3f}  ood_top2={ood_metrics['top2_accuracy']:.3f}  ood_macroF1={ood_prf1['f1']:.3f}")

    config = {
        "seeds": SEEDS, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT,
        "lr": LR, "epochs_requested": EPOCHS, "patience": PATIENCE, "in_dim": graphs[0].x.shape[1],
        "bidirectional_edges": True, "use_embeddings": True, "pca_components": PCA_COMPONENTS,
        "conv_type": CONV_TYPE, "ensemble": True, "ensemble_size": len(SEEDS),
    }
    metrics = {
        "train": {"top1_accuracy": train_metrics["top1_accuracy"], "top2_accuracy": train_metrics["top2_accuracy"], **train_prf1, "n": len(train_set)},
        "val": {"top1_accuracy": val_metrics["top1_accuracy"], "top2_accuracy": val_metrics["top2_accuracy"], **val_prf1, "n": len(val_set)},
        "ood": {"top1_accuracy": ood_metrics["top1_accuracy"], "top2_accuracy": ood_metrics["top2_accuracy"], **ood_prf1, "n": len(ood_set)},
        "baseline_ood_top1_accuracy": baseline_ood["accuracy"],
    }
    log_run(config=config, metrics=metrics, notes=f"{len(SEEDS)}-seed SAGE ensemble (avg softmax over seeds {SEEDS})")
