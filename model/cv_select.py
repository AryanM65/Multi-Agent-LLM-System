"""Honest model selection (model_improvement.md Section 5.4): score the
improve_v2 recipe by 5-fold leave-topology-out CV instead of by the
question-only val split, then train the final ensemble on all 14 train
topologies and score OOD exactly once.

Today's ledger entries (0.395 -> 0.403 -> 0.411) all picked winners by
looking at OOD across many runs -- that slowly overfits the "held-out" set.
This entrypoint's config carries "selection_method": "leave_topology_out_cv"
so future entries can be told apart from that older practice. The CV mean is
the number to trust for "did this recipe actually generalize"; the single
final OOD score is reported alongside it, never used to pick anything after
the fact.
"""
import statistics

import torch

from model.data.load_raw import load_topologies, load_trials, validate
from model.data.enrich_features import enrich
from model.data.enrich_discrepancy import enrich_discrepancy
from model.data.enrich_edges import enrich_edges
from model.data.build_graph_dataset import build_dataset
from model.data.split import split_graphs
from model.data.split_cv import grouped_kfold_by_topology, inner_val_split, cv_summary
from model.data.reduce_embeddings import fit_and_reduce_embeddings_with_variance
from model.gnn import FaultLocalizerGNN
from model.train import train
from model.evaluate import evaluate as evaluate_single, macro_prf1
from model.baselines import evaluate_baseline
from model.metrics_log import log_run
from model.ensemble import evaluate_ensemble

HIDDEN_DIM = 64
NUM_LAYERS = 3
LR = 5e-4
DROPOUT = 0.3
CONV_TYPE = "sage"
POOL_TYPE = "hybrid"
EPOCHS = 150
PATIENCE = 30
PCA_COMPONENTS = 128

CV_FOLDS = 5
CV_SEEDS = [0, 1, 2]       # reduced from 5 for the 5-fold CV loop (5x3=15 trainings) -- runtime
FINAL_SEEDS = [0, 1, 2, 3, 4]  # full ensemble for the single final OOD score


def _train_one(graphs, train_set, val_set, seed):
    torch.manual_seed(seed)
    in_dim = graphs[0].x.shape[1]
    model = FaultLocalizerGNN(in_dim=in_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                              dropout=DROPOUT, conv_type=CONV_TYPE, pool_type=POOL_TYPE)
    model, best_val_acc, best_epoch, epochs_run = train(
        model, train_set, val_set, epochs=EPOCHS, lr=LR, patience=PATIENCE, verbose=False,
    )
    return model


if __name__ == "__main__":
    import sys
    in_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/trials_k10_disc.jsonl"

    topologies = load_topologies()
    trials = load_trials(path=in_path)
    validate(trials, topologies)
    print(f"Loaded {len(trials)} trials from {in_path}")

    if "node_embeddings" not in trials[0]:
        print("Enriching (mean+variance node embeddings)...")
        trials = enrich(trials, topologies, with_embeddings=True)
    print("Enriching (leave-one-out calibrated discrepancy scalars)...")
    enrich_discrepancy(trials, topologies)
    print("Enriching (per-edge discrepancy features)...")
    enrich_edges(trials, topologies)

    graphs = build_dataset(trials, topologies, bidirectional=True, use_embeddings=True,
                           use_embedding_variance=True)
    in_dim = graphs[0].x.shape[1]
    print(f"in_dim={in_dim}")

    # === Step 1: 5-fold leave-topology-out CV (selection signal) ===
    folds = grouped_kfold_by_topology(graphs, topologies, k=CV_FOLDS, seed=0)
    fold_scores = []
    for fi, (train_subset, held_out_subset, held_out_ids) in enumerate(folds):
        inner_train, inner_val = inner_val_split(train_subset, val_fraction=0.15, seed=fi)
        # PCA fit on this fold's inner-train only -- no leakage of the held-out
        # topologies' embedding distribution into the fold's model.
        fold_graphs = inner_train + inner_val + held_out_subset
        fit_and_reduce_embeddings_with_variance(inner_train, fold_graphs, n_components=PCA_COMPONENTS, seed=0)

        models = [_train_one(fold_graphs, inner_train, inner_val, seed) for seed in CV_SEEDS]
        held_out_metrics = evaluate_ensemble(models, held_out_subset)
        fold_scores.append(held_out_metrics["top1_accuracy"])
        print(f"fold {fi} (held out {held_out_ids}): "
              f"top1={held_out_metrics['top1_accuracy']:.3f}  n={len(held_out_subset)}")

    cv = cv_summary(fold_scores)
    print(f"\nCV mean +/- std: {cv['mean']:.3f} +/- {cv['std']:.3f}  (folds={[round(s,3) for s in cv['folds']]})")

    # === Step 2: final recipe, trained on ALL 14 train topologies, OOD scored ONCE ===
    train_set, val_set, ood_set = split_graphs(graphs, topologies, seed=0)
    fit_and_reduce_embeddings_with_variance(train_set, graphs, n_components=PCA_COMPONENTS, seed=0)

    final_models = []
    for seed in FINAL_SEEDS:
        print(f"\n--- final ensemble: training seed={seed} ---")
        model = _train_one(graphs, train_set, val_set, seed)
        single_ood = evaluate_single(model, ood_set)
        print(f"seed={seed} ood_top1={single_ood['top1_accuracy']:.3f}")
        final_models.append(model)

    ood_metrics = evaluate_ensemble(final_models, ood_set)
    val_metrics = evaluate_ensemble(final_models, val_set)
    train_metrics = evaluate_ensemble(final_models, train_set)
    baseline_ood = evaluate_baseline(ood_set)

    train_prf1 = macro_prf1(train_metrics["y_true"], train_metrics["y_pred"])
    val_prf1 = macro_prf1(val_metrics["y_true"], val_metrics["y_pred"])
    ood_prf1 = macro_prf1(ood_metrics["y_true"], ood_metrics["y_pred"])

    print(f"\n=== leave-topology-out CV selection, final OOD score ({len(trials)} records) ===")
    print(f"cv_mean={cv['mean']:.3f}  cv_std={cv['std']:.3f}  final_ood_top1={ood_metrics['top1_accuracy']:.3f}")
    gap = ood_metrics["top1_accuracy"] - cv["mean"]
    if gap > 0.05:
        print(f"WARNING: final OOD ({ood_metrics['top1_accuracy']:.3f}) exceeds CV mean "
              f"({cv['mean']:.3f}) by {gap:.3f} -- suspicious, OOD may be easier than the CV "
              f"folds by chance rather than the recipe generalizing better than CV suggests.")

    config = {
        "seeds": FINAL_SEEDS, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT,
        "lr": LR, "epochs_requested": EPOCHS, "patience": PATIENCE, "in_dim": in_dim,
        "bidirectional_edges": True, "use_embeddings": True, "use_embedding_variance": True,
        "pca_components": PCA_COMPONENTS, "conv_type": CONV_TYPE, "pool_type": POOL_TYPE,
        "ensemble": True, "ensemble_size": len(FINAL_SEEDS), "dataset_size": len(trials),
        "selection_method": "leave_topology_out_cv", "cv_folds": CV_FOLDS, "cv_seeds": CV_SEEDS,
        "cv_select": True, "discrepancy_features": True, "edge_features": True,
    }
    metrics = {
        "train": {"top1_accuracy": train_metrics["top1_accuracy"], "top2_accuracy": train_metrics["top2_accuracy"], **train_prf1, "n": len(train_set)},
        "val": {"top1_accuracy": val_metrics["top1_accuracy"], "top2_accuracy": val_metrics["top2_accuracy"], **val_prf1, "n": len(val_set)},
        "ood": {"top1_accuracy": ood_metrics["top1_accuracy"], "top2_accuracy": ood_metrics["top2_accuracy"], **ood_prf1, "n": len(ood_set)},
        "baseline_ood_top1_accuracy": baseline_ood["accuracy"],
        "cv_mean_top1": cv["mean"], "cv_std_top1": cv["std"], "cv_fold_scores": cv["folds"],
    }
    log_run(config=config, metrics=metrics,
            notes=f"Leave-topology-out CV selection ({CV_FOLDS} folds) + single final OOD score, "
                  f"discrepancy z-scores + edge features + structural position features, {len(trials)}-record dataset")
    print("\nLogged to model/metrics_history.jsonl")
