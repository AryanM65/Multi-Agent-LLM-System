"""Preliminary check on the partial k=10 dataset (1099 records, 16/20
topologies -- missing mixed_asymmetric, star, wide_fanin, wide_fanout_reasoner,
Track F's still-running final batch) BEFORE the full k=10 generation
finishes. Runs enrichment (mean+variance embeddings) in-process, then trains
the winning config (SAGE hd64/nl3/lr5e-4, PCA-128, attention pooling,
3-seed ensemble). Purpose: validate the pipeline early and get an early
accuracy signal, NOT the final reported number -- 4 train topologies are
still completely absent from this data. See docs/model/futurework.md.
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
POOL_TYPE = "attention"
SEEDS = [0, 1, 2]
EPOCHS = 150
PATIENCE = 30
PCA_COMPONENTS = 128


if __name__ == "__main__":
    import sys
    in_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/trials_k10_preview.jsonl"

    topologies = load_topologies()
    trials = load_trials(path=in_path)
    validate(trials, topologies)
    print(f"Loaded {len(trials)} k=10-partial trials from {in_path}")

    print("Enriching (mean+variance node embeddings, inference_gaps, item_frequencies)...")
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
    val_metrics = evaluate_ensemble(models, val_set)
    train_metrics = evaluate_ensemble(models, train_set)
    baseline_ood = evaluate_baseline(ood_set)

    train_prf1 = macro_prf1(train_metrics["y_true"], train_metrics["y_pred"])
    val_prf1 = macro_prf1(val_metrics["y_true"], val_metrics["y_pred"])
    ood_prf1 = macro_prf1(ood_metrics["y_true"], ood_metrics["y_pred"])

    print(f"\n=== PREVIEW (partial k=10, {len(trials)} records, 16/20 topologies) ===")
    print(f"ood_top1={ood_metrics['top1_accuracy']:.3f}  ood_top2={ood_metrics['top2_accuracy']:.3f}  ood_macroF1={ood_prf1['f1']:.3f}")

    config = {
        "seeds": SEEDS, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT,
        "lr": LR, "epochs_requested": EPOCHS, "patience": PATIENCE, "in_dim": graphs[0].x.shape[1],
        "bidirectional_edges": True, "use_embeddings": True, "use_embedding_variance": True,
        "pca_components": PCA_COMPONENTS, "conv_type": CONV_TYPE, "pool_type": POOL_TYPE,
        "ensemble": True, "ensemble_size": len(SEEDS), "dataset_size": len(trials),
        "preview_partial_k10": True,
    }
    metrics = {
        "train": {"top1_accuracy": train_metrics["top1_accuracy"], "top2_accuracy": train_metrics["top2_accuracy"], **train_prf1, "n": len(train_set)},
        "val": {"top1_accuracy": val_metrics["top1_accuracy"], "top2_accuracy": val_metrics["top2_accuracy"], **val_prf1, "n": len(val_set)},
        "ood": {"top1_accuracy": ood_metrics["top1_accuracy"], "top2_accuracy": ood_metrics["top2_accuracy"], **ood_prf1, "n": len(ood_set)},
        "baseline_ood_top1_accuracy": baseline_ood["accuracy"],
    }
    log_run(config=config, metrics=metrics, notes=f"PREVIEW run on partial k=10 dataset ({len(trials)} records, 16/20 topologies, 4 train topologies still missing) -- not the final reported number")
