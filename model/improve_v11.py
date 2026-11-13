"""Filter/down-weight inert fault labels on TRAIN/VAL only (model_improvement.md
Section 4.3 / roadmap step 7).

~40% of contamination/ceiling trials and ~80% of noise trials show no
measurable change at the target node against their paired same-question
clean control (every faulted trial has one). Training on them teaches the
model to guess "which node is the fault target" from a label that isn't
actually observable in the node's own output -- pure noise for the loss.

Distance metric: for a faulted trial's target node, take
max(|z|) over every leave-one-out calibrated scalar computed by
model/data/enrich_discrepancy.py (lexical/semantic/jaccard uncertainty,
node_novel_ratio, node_dropped_ratio, node_sibling_disagree, node_len_std,
node_len_mean/node_len_z) at that node. This directly reuses the calibrated
"how deviated is this node from its own clean-control history" scores
already built for Section 4.1, rather than introducing a second ad hoc
distance measure. |z| < INERT_THRESHOLD (1.0, per the spec's suggestion) =
inert: the target node doesn't look different from its own clean baseline,
so the label is present but not learnable from the features.

Two variants, both trained and logged separately:
  (a) drop    -- remove inert trials from TRAIN/VAL entirely.
  (b) downweight -- keep them, but scale their loss by
      max(0.15, min(1.0, |z| / 2)) so gradient isn't zero, just reduced.
      This needs a per-graph loss weight the shared model/train.py loop
      doesn't support, so it uses a small local training loop instead of
      importing model.train (kept local to this file only).

OOD is NEVER filtered or reweighted -- ood_top1 stays comparable to the
0.4114 baseline on the full unfiltered set. A secondary ood_top1_effective
is also reported, restricted to OOD trials whose target node IS measurably
deviated by the same distance metric -- labeled secondary throughout, never
the headline.
"""
import copy

import torch
import torch.nn.functional as F

from model.data.load_raw import load_topologies, load_trials, validate
from model.data.enrich_discrepancy import enrich_discrepancy, CALIBRATED_FIELDS
from model.data.enrich_features import enrich as enrich_embeddings
from model.data.build_graph_dataset import build_dataset
from model.data.split import split_graphs
from model.data.reduce_embeddings import fit_and_reduce_embeddings_with_variance
from model.gnn import FaultLocalizerGNN
from model.train import train, compute_no_fault_weight
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

INERT_THRESHOLD = 1.0
DOWNWEIGHT_FLOOR = 0.15
DOWNWEIGHT_SCALE = 2.0

# Every leave-one-out calibrated z field enrich_discrepancy.py produces.
Z_FIELDS = list(CALIBRATED_FIELDS.values())


def target_distance(trial: dict) -> float | None:
    """max(|z|) at the fault target node across every calibrated scalar.
    None for clean controls (never filtered) and for faulted trials with no
    calibrated value at the target node (too few controls to calibrate --
    left alone rather than assumed inert)."""
    fc = trial.get("fault_config") or {}
    target = fc.get("target_node")
    if not target:
        return None
    zs = []
    for field in Z_FIELDS:
        v = trial.get(field, {}).get(target)
        if v is not None:
            zs.append(abs(v))
    return max(zs) if zs else None


def is_inert(trial: dict) -> bool:
    d = target_distance(trial)
    return d is not None and d < INERT_THRESHOLD


def annotate_graphs_with_distance(trials: list, graphs: list) -> None:
    """trials and graphs are built in the same order by build_dataset -- zip
    to attach each trial's distance/inert flag onto its graph."""
    assert len(trials) == len(graphs)
    for t, g in zip(trials, graphs):
        g.inert_distance = target_distance(t)
        g.is_inert = is_inert(t)


def breakdown_by_fault_type(trials: list) -> dict:
    counts = {}
    for t in trials:
        ft = (t.get("fault_config") or {}).get("type")
        if ft is None:
            continue
        counts.setdefault(ft, {"total": 0, "inert": 0})
        counts[ft]["total"] += 1
        if is_inert(t):
            counts[ft]["inert"] += 1
    return counts


def downweight(g) -> float:
    if g.inert_distance is None:
        return 1.0  # clean control, or uncalibrated -- full weight
    return max(DOWNWEIGHT_FLOOR, min(1.0, g.inert_distance / DOWNWEIGHT_SCALE))


def train_with_weights(model, train_graphs, val_graphs, epochs=EPOCHS, lr=LR, patience=PATIENCE):
    """Local copy of model/train.py's loop, extended with a per-graph loss
    weight (downweight()). Kept local rather than editing the shared
    model/train.py, which every other improve_vN.py also imports."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    no_fault_weight = compute_no_fault_weight(train_graphs)

    best_val_acc, best_epoch, best_state = -1.0, -1, None
    epochs_since_improvement, epochs_run = 0, 0

    for epoch in range(epochs):
        epochs_run = epoch + 1
        model.train()
        for g in train_graphs:
            logits = model(g.x, g.edge_index)
            class_weight = torch.ones(g.num_nodes + 1)
            class_weight[g.num_nodes] = no_fault_weight
            loss = F.cross_entropy(logits.unsqueeze(0), g.y, weight=class_weight)
            loss = loss * downweight(g)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_single(model, val_graphs)
        val_acc = val_metrics["top1_accuracy"]
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc, best_epoch, epochs_run


def run_variant(name, train_set, val_set, ood_set, graphs, trials, effective_ood_set, use_weighted_loop):
    models = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = FaultLocalizerGNN(in_dim=graphs[0].x.shape[1], hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                                  dropout=DROPOUT, conv_type=CONV_TYPE, pool_type=POOL_TYPE)
        if use_weighted_loop:
            model, best_val_acc, best_epoch, epochs_run = train_with_weights(model, train_set, val_set)
        else:
            model, best_val_acc, best_epoch, epochs_run = train(
                model, train_set, val_set, epochs=EPOCHS, lr=LR, patience=PATIENCE, verbose=False,
            )
        single = evaluate_single(model, ood_set)
        print(f"[{name}] seed={seed} ood_top1={single['top1_accuracy']:.3f} (best_epoch={best_epoch}, ran={epochs_run})")
        models.append(model)

    ood_metrics = evaluate_ensemble(models, ood_set)
    val_metrics = evaluate_ensemble(models, val_set)
    train_metrics = evaluate_ensemble(models, train_set)
    ood_effective_metrics = evaluate_ensemble(models, effective_ood_set) if effective_ood_set else None
    baseline_ood = evaluate_baseline(ood_set)

    train_prf1 = macro_prf1(train_metrics["y_true"], train_metrics["y_pred"])
    val_prf1 = macro_prf1(val_metrics["y_true"], val_metrics["y_pred"])
    ood_prf1 = macro_prf1(ood_metrics["y_true"], ood_metrics["y_pred"])

    print(f"\n=== {name} ({len(train_set)} train graphs after filtering) ===")
    print(f"train_top1={train_metrics['top1_accuracy']:.3f}  val_top1={val_metrics['top1_accuracy']:.3f}  "
          f"ood_top1={ood_metrics['top1_accuracy']:.3f}  ood_top2={ood_metrics['top2_accuracy']:.3f}")
    if ood_effective_metrics:
        print(f"ood_top1_effective={ood_effective_metrics['top1_accuracy']:.3f} (n={len(effective_ood_set)}, secondary metric)")

    return {
        "train_metrics": train_metrics, "val_metrics": val_metrics, "ood_metrics": ood_metrics,
        "ood_effective_metrics": ood_effective_metrics, "baseline_ood": baseline_ood,
        "train_prf1": train_prf1, "val_prf1": val_prf1, "ood_prf1": ood_prf1,
    }


if __name__ == "__main__":
    import sys
    in_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/trials_k10_disc.jsonl"

    topologies = load_topologies()
    trials = load_trials(path=in_path)
    validate(trials, topologies)
    print(f"Loaded {len(trials)} trials from {in_path}")

    if "lexical_unc_z" not in trials[0]:
        print("Running enrich_discrepancy (leave-one-out calibrated z-scores)...")
        enrich_discrepancy(trials, topologies)
    if "node_embeddings" not in trials[0]:
        print("Enriching (mean+variance node embeddings)...")
        trials = enrich_embeddings(trials, topologies, with_embeddings=True)

    breakdown = breakdown_by_fault_type(trials)
    print("\nInert-label breakdown by fault type (|z| < {:.1f} at target node):".format(INERT_THRESHOLD))
    for ft, c in sorted(breakdown.items()):
        print(f"  {ft}: {c['inert']}/{c['total']} inert ({100 * c['inert'] / c['total']:.1f}%)")

    graphs = build_dataset(trials, topologies, bidirectional=True, use_embeddings=True, use_embedding_variance=True)
    annotate_graphs_with_distance(trials, graphs)
    print(f"in_dim={graphs[0].x.shape[1]}")

    train_set, val_set, ood_set = split_graphs(graphs, topologies, seed=0)
    fit_and_reduce_embeddings_with_variance(train_set, graphs, n_components=PCA_COMPONENTS, seed=0)

    n_inert_train = sum(1 for g in train_set if g.is_inert)
    n_inert_val = sum(1 for g in val_set if g.is_inert)
    print(f"\ntrain: {len(train_set)} graphs, {n_inert_train} inert")
    print(f"val:   {len(val_set)} graphs, {n_inert_val} inert")
    print(f"ood:   {len(ood_set)} graphs (never filtered)")

    # Secondary OOD subset: only trials whose target node IS measurably
    # deviated -- reported alongside, never instead of, the full-OOD headline.
    ood_effective_set = [g for g in ood_set if not g.is_inert]
    print(f"ood_effective (secondary, target node measurably deviated): {len(ood_effective_set)}/{len(ood_set)}")

    # --- variant (a): hard drop -------------------------------------------
    train_drop = [g for g in train_set if not g.is_inert]
    val_drop = [g for g in val_set if not g.is_inert]
    result_drop = run_variant("drop", train_drop, val_drop, ood_set, graphs, trials, ood_effective_set,
                              use_weighted_loop=False)

    log_run(
        config={
            "seeds": SEEDS, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT, "lr": LR,
            "epochs_requested": EPOCHS, "patience": PATIENCE, "in_dim": graphs[0].x.shape[1],
            "bidirectional_edges": True, "use_embeddings": True, "use_embedding_variance": True,
            "pca_components": PCA_COMPONENTS, "conv_type": CONV_TYPE, "pool_type": POOL_TYPE,
            "ensemble": True, "ensemble_size": len(SEEDS), "dataset_size": len(trials),
            "discrepancy_features": True, "improve_v11": True,
            "inert_filter": "drop", "inert_threshold": INERT_THRESHOLD,
            "n_dropped_train": len(train_set) - len(train_drop), "n_dropped_val": len(val_set) - len(val_drop),
            "inert_breakdown_by_fault_type": breakdown,
        },
        metrics={
            "train": {"top1_accuracy": result_drop["train_metrics"]["top1_accuracy"], "top2_accuracy": result_drop["train_metrics"]["top2_accuracy"], **result_drop["train_prf1"], "n": len(train_drop)},
            "val": {"top1_accuracy": result_drop["val_metrics"]["top1_accuracy"], "top2_accuracy": result_drop["val_metrics"]["top2_accuracy"], **result_drop["val_prf1"], "n": len(val_drop)},
            "ood": {"top1_accuracy": result_drop["ood_metrics"]["top1_accuracy"], "top2_accuracy": result_drop["ood_metrics"]["top2_accuracy"], **result_drop["ood_prf1"], "n": len(ood_set)},
            "ood_top1_effective": result_drop["ood_effective_metrics"]["top1_accuracy"] if result_drop["ood_effective_metrics"] else None,
            "ood_effective_n": len(ood_effective_set),
            "baseline_ood_top1_accuracy": result_drop["baseline_ood"]["accuracy"],
        },
        notes=f"Inert-label HARD DROP (|z|<{INERT_THRESHOLD} at target node) from train/val only, {len(trials)}-record dataset. "
              f"ood_top1 is on the full unfiltered OOD set (headline, comparable to 0.4114 baseline); "
              f"ood_top1_effective is secondary, restricted to OOD trials with a measurable target deviation.",
    )

    # --- variant (b): down-weight ------------------------------------------
    result_dw = run_variant("downweight", train_set, val_set, ood_set, graphs, trials, ood_effective_set,
                            use_weighted_loop=True)

    log_run(
        config={
            "seeds": SEEDS, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "dropout": DROPOUT, "lr": LR,
            "epochs_requested": EPOCHS, "patience": PATIENCE, "in_dim": graphs[0].x.shape[1],
            "bidirectional_edges": True, "use_embeddings": True, "use_embedding_variance": True,
            "pca_components": PCA_COMPONENTS, "conv_type": CONV_TYPE, "pool_type": POOL_TYPE,
            "ensemble": True, "ensemble_size": len(SEEDS), "dataset_size": len(trials),
            "discrepancy_features": True, "improve_v11": True,
            "inert_filter": "downweight", "inert_threshold": INERT_THRESHOLD,
            "downweight_floor": DOWNWEIGHT_FLOOR, "downweight_scale": DOWNWEIGHT_SCALE,
            "n_downweighted_train": n_inert_train, "n_downweighted_val": n_inert_val,
            "inert_breakdown_by_fault_type": breakdown,
        },
        metrics={
            "train": {"top1_accuracy": result_dw["train_metrics"]["top1_accuracy"], "top2_accuracy": result_dw["train_metrics"]["top2_accuracy"], **result_dw["train_prf1"], "n": len(train_set)},
            "val": {"top1_accuracy": result_dw["val_metrics"]["top1_accuracy"], "top2_accuracy": result_dw["val_metrics"]["top2_accuracy"], **result_dw["val_prf1"], "n": len(val_set)},
            "ood": {"top1_accuracy": result_dw["ood_metrics"]["top1_accuracy"], "top2_accuracy": result_dw["ood_metrics"]["top2_accuracy"], **result_dw["ood_prf1"], "n": len(ood_set)},
            "ood_top1_effective": result_dw["ood_effective_metrics"]["top1_accuracy"] if result_dw["ood_effective_metrics"] else None,
            "ood_effective_n": len(ood_effective_set),
            "baseline_ood_top1_accuracy": result_dw["baseline_ood"]["accuracy"],
        },
        notes=f"Inert-label DOWN-WEIGHT (loss *= max({DOWNWEIGHT_FLOOR}, min(1, |z|/{DOWNWEIGHT_SCALE}))) on train only, "
              f"{len(trials)}-record dataset. Local weighted training loop (not model/train.py). "
              f"ood_top1 on full unfiltered OOD set; ood_top1_effective secondary.",
    )

    print(f"\nDone. drop ood_top1={result_drop['ood_metrics']['top1_accuracy']:.3f}  "
          f"downweight ood_top1={result_dw['ood_metrics']['top1_accuracy']:.3f}  baseline (0.4114) for comparison.")
