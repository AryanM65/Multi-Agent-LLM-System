"""Temporary fast variant of model/cv_select.py: skips node-embedding
enrichment (the per-trial encoder calls are impractically slow on local CPU
-- 200/1785 trials in the time available, ~24h+ projected for the full
enrichment pass alone). Everything else (leave-topology-out CV, final OOD
score, log_run) is identical. Scalar + discrepancy + edge + structural
features only -- config flags this clearly. Not committed; run once for a
real number under time pressure, then delete.
"""
import statistics
import torch

from model.data.load_raw import load_topologies, load_trials, validate
from model.data.enrich_discrepancy import enrich_discrepancy
from model.data.enrich_edges import enrich_edges
from model.data.build_graph_dataset import build_dataset
from model.data.split import split_graphs
from model.data.split_cv import grouped_kfold_by_topology, inner_val_split, cv_summary
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

CV_FOLDS = 5
CV_SEEDS = [0, 1, 2]
FINAL_SEEDS = [0, 1, 2, 3, 4]


def _train_one(graphs, train_set, val_set, seed, in_dim):
    torch.manual_seed(seed)
    model = FaultLocalizerGNN(in_dim=in_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                              dropout=DROPOUT, conv_type=CONV_TYPE, pool_type=POOL_TYPE)
    model, *_ = train(model, train_set, val_set, epochs=EPOCHS, lr=LR, patience=PATIENCE, verbose=False)
    return model


if __name__ == "__main__":
    import sys
    in_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/trials_k10_disc.jsonl"

    topologies = load_topologies()
    trials = load_trials(path=in_path)
    validate(trials, topologies)
    print(f"Loaded {len(trials)} trials from {in_path}", flush=True)

    enrich_discrepancy(trials, topologies)
    enrich_edges(trials, topologies)

    graphs = build_dataset(trials, topologies, bidirectional=True, use_embeddings=False)
    in_dim = graphs[0].x.shape[1]
    print(f"in_dim={in_dim} (no embeddings)", flush=True)

    folds = grouped_kfold_by_topology(graphs, topologies, k=CV_FOLDS, seed=0)
    fold_scores = []
    for fi, (train_subset, held_out_subset, held_out_ids) in enumerate(folds):
        inner_train, inner_val = inner_val_split(train_subset, val_fraction=0.15, seed=fi)
        models = [_train_one(graphs, inner_train, inner_val, seed, in_dim) for seed in CV_SEEDS]
        held_out_metrics = evaluate_ensemble(models, held_out_subset)
        fold_scores.append(held_out_metrics["top1_accuracy"])
        print(f"fold {fi} (held out {held_out_ids}): top1={held_out_metrics['top1_accuracy']:.3f}  n={len(held_out_subset)}", flush=True)

    cv = cv_summary(fold_scores)
    print(f"\nCV mean +/- std: {cv['mean']:.3f} +/- {cv['std']:.3f}  (folds={[round(s,3) for s in cv['folds']]})", flush=True)

    train_set, val_set, ood_set = split_graphs(graphs, topologies, seed=0)
    final_models = []
    for seed in FINAL_SEEDS:
        model = _train_one(graphs, train_set, val_set, seed, in_dim)
        single_ood = evaluate_single(model, ood_set)
        print(f"seed={seed} ood_top1={single_ood['top1_accuracy']:.3f}", flush=True)
        final_models.append(model)

    ood_metrics = evaluate_ensemble(final_models, ood_set)
    val_metrics = evaluate_ensemble(final_models, val_set)
    train_metrics = evaluate_ensemble(final_models, train_set)
    baseline_ood = evaluate_baseline(ood_set)

    train_prf1 = macro_prf1(train_metrics["y_true"], train_metrics["y_pred"])
    val_prf1 = macro_prf1(val_metrics["y_true"], val_metrics["y_pred"])
    ood_prf1 = macro_prf1(ood_metrics["y_true"], ood_metrics["y_pred"])

    print(f"\n=== leave-topology-out CV selection, NO EMBEDDINGS, final OOD score ({len(trials)} records) ===", flush=True)
    print(f"cv_mean={cv['mean']:.3f}  cv_std={cv['std']:.3f}  final_ood_top1={ood_metrics['top1_accuracy']:.3f}", flush=True)
    gap = ood_metrics["top1_accuracy"] - cv["mean"]
    if gap > 0.05:
        print(f"WARNING: final OOD exceeds CV mean by {gap:.3f} -- suspicious.", flush=True)

    config = {
        "seeds": FINAL_SEEDS, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT,
        "lr": LR, "epochs_requested": EPOCHS, "patience": PATIENCE, "in_dim": in_dim,
        "bidirectional_edges": True, "use_embeddings": False,
        "ensemble": True, "ensemble_size": len(FINAL_SEEDS), "dataset_size": len(trials),
        "selection_method": "leave_topology_out_cv", "cv_folds": CV_FOLDS, "cv_seeds": CV_SEEDS,
        "cv_select": True, "discrepancy_features": True, "edge_features": True,
        "note": "embeddings skipped -- local CPU encoder too slow, see model/cv_select.py for the full recipe with embeddings",
    }
    metrics = {
        "train": {"top1_accuracy": train_metrics["top1_accuracy"], "top2_accuracy": train_metrics["top2_accuracy"], **train_prf1, "n": len(train_set)},
        "val": {"top1_accuracy": val_metrics["top1_accuracy"], "top2_accuracy": val_metrics["top2_accuracy"], **val_prf1, "n": len(val_set)},
        "ood": {"top1_accuracy": ood_metrics["top1_accuracy"], "top2_accuracy": ood_metrics["top2_accuracy"], **ood_prf1, "n": len(ood_set)},
        "baseline_ood_top1_accuracy": baseline_ood["accuracy"],
        "cv_mean_top1": cv["mean"], "cv_std_top1": cv["std"], "cv_fold_scores": cv["folds"],
    }
    log_run(config=config, metrics=metrics,
            notes=f"Leave-topology-out CV selection ({CV_FOLDS} folds), NO EMBEDDINGS (local CPU too slow), "
                  f"discrepancy z-scores + edge features + structural position features, {len(trials)}-record dataset")
    print("\nLogged to model/metrics_history.jsonl", flush=True)
