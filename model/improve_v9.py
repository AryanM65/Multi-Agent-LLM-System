"""Multi-task retest: fault-type head alongside node localization.

multi_task was tried once early (OOD 0.2665) and dropped. It is worth one
retest now for a specific reason: back then every feature was an uncertainty
measure, so knowing the fault type bought nothing -- there was no
type-specific signal to route to. After enrich_discrepancy there is:

    ceiling        node_len_z (argmin)        oracle 0.716
    contamination  semantic_uncertainties     oracle 0.444
    noise          node_child_novel           oracle 0.371

An oracle that knew the type and picked the best single feature scores
0.485, well above the 0.4035 single-task baseline -- so type conditioning
now has real headroom. The auxiliary head pushes the shared representation
to encode which signal matters for this trial.

Otherwise identical to improve_v2's winning recipe.
"""
import torch

from model.data.load_raw import load_topologies, load_trials, validate
from model.data.enrich_features import enrich
from model.data.build_graph_dataset import build_dataset
from model.data.split import split_graphs
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
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 150
PATIENCE = 30
PCA_COMPONENTS = 128
AUX_WEIGHT = 0.3


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

    graphs = build_dataset(trials, topologies, bidirectional=True, use_embeddings=True,
                           use_embedding_variance=True)
    train_set, val_set, ood_set = split_graphs(graphs, topologies, seed=0)
    print(f"in_dim={graphs[0].x.shape[1]}  train={len(train_set)} val={len(val_set)} ood={len(ood_set)}")

    fit_and_reduce_embeddings_with_variance(train_set, graphs, n_components=PCA_COMPONENTS, seed=0)

    models = []
    for seed in SEEDS:
        print(f"\n--- training seed={seed} ---")
        torch.manual_seed(seed)
        model = FaultLocalizerGNN(in_dim=graphs[0].x.shape[1], hidden_dim=HIDDEN_DIM,
                                  num_layers=NUM_LAYERS, dropout=DROPOUT, conv_type=CONV_TYPE,
                                  pool_type=POOL_TYPE, multi_task=True)
        model, best_val_acc, best_epoch, epochs_run = train(
            model, train_set, val_set, epochs=EPOCHS, lr=LR, patience=PATIENCE,
            verbose=False, aux_weight=AUX_WEIGHT,
        )
        single = evaluate_single(model, ood_set)
        print(f"seed={seed} ood_top1={single['top1_accuracy']:.3f} (best_epoch={best_epoch}, ran={epochs_run})")
        models.append(model)

    ood_metrics = evaluate_ensemble(models, ood_set)
    val_metrics = evaluate_ensemble(models, val_set)
    train_metrics = evaluate_ensemble(models, train_set)
    baseline_ood = evaluate_baseline(ood_set)

    train_prf1 = macro_prf1(train_metrics["y_true"], train_metrics["y_pred"])
    val_prf1 = macro_prf1(val_metrics["y_true"], val_metrics["y_pred"])
    ood_prf1 = macro_prf1(ood_metrics["y_true"], ood_metrics["y_pred"])

    print(f"\n=== multi-task + discrepancy features ({len(trials)} records) ===")
    print(f"train_top1={train_metrics['top1_accuracy']:.3f}  val_top1={val_metrics['top1_accuracy']:.3f}  "
          f"ood_top1={ood_metrics['top1_accuracy']:.3f}  ood_top2={ood_metrics['top2_accuracy']:.3f}")

    config = {
        "seeds": SEEDS, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT,
        "lr": LR, "epochs_requested": EPOCHS, "patience": PATIENCE, "in_dim": graphs[0].x.shape[1],
        "bidirectional_edges": True, "use_embeddings": True, "use_embedding_variance": True,
        "pca_components": PCA_COMPONENTS, "conv_type": CONV_TYPE, "pool_type": POOL_TYPE,
        "ensemble": True, "ensemble_size": len(SEEDS), "dataset_size": len(trials),
        "multi_task": True, "aux_weight": AUX_WEIGHT, "discrepancy_features": True,
        "improve_v9": True,
    }
    metrics = {
        "train": {"top1_accuracy": train_metrics["top1_accuracy"], "top2_accuracy": train_metrics["top2_accuracy"], **train_prf1, "n": len(train_set)},
        "val": {"top1_accuracy": val_metrics["top1_accuracy"], "top2_accuracy": val_metrics["top2_accuracy"], **val_prf1, "n": len(val_set)},
        "ood": {"top1_accuracy": ood_metrics["top1_accuracy"], "top2_accuracy": ood_metrics["top2_accuracy"], **ood_prf1, "n": len(ood_set)},
        "baseline_ood_top1_accuracy": baseline_ood["accuracy"],
    }
    log_run(config=config, metrics=metrics,
            notes=f"Multi-task fault-type head + discrepancy/length features, {len(trials)}-record dataset")
