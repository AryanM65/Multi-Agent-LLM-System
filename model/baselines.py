"""Baseline predictors -- the numbers the GNN needs to beat.

naive_max_uncertainty_baseline: "guess whichever node has the highest lexical
uncertainty." No learning, no training data needed. If this already scores
well, a GNN isn't earning its complexity; if it fails in specific ways
(see model-plan.md Step 4/8), that's exactly what the GNN needs to fix.
"""
import torch


def naive_max_uncertainty_baseline(graph) -> int:
    """Returns a predicted class index: either a node index (0..num_nodes-1)
    or num_nodes itself (the "no fault" class) when every node's uncertainty
    is ~0, which we treat as "looks clean"."""
    lexical_uncertainties = graph.x[:, 0]
    if float(lexical_uncertainties.max()) <= 1e-9:
        return graph.num_nodes  # nothing stood out -> guess "no fault"
    return int(lexical_uncertainties.argmax())


def evaluate_baseline(graphs, predict_fn=naive_max_uncertainty_baseline):
    correct, total = 0, 0
    y_true, y_pred = [], []
    for g in graphs:
        true_y = int(g.y.item())
        pred_y = predict_fn(g)
        y_true.append(true_y)
        y_pred.append(pred_y)
        if pred_y == true_y:
            correct += 1
        total += 1
    return {"accuracy": correct / total, "y_true": y_true, "y_pred": y_pred}


if __name__ == "__main__":
    from model.data.load_raw import load_topologies, load_trials, validate
    from model.data.build_graph_dataset import build_dataset
    from model.data.split import split_graphs

    topologies = load_topologies()
    trials = load_trials()
    validate(trials, topologies)
    graphs = build_dataset(trials, topologies)
    train_set, val_set, ood_set = split_graphs(graphs, topologies)

    for name, split in [("train", train_set), ("val", val_set), ("ood", ood_set)]:
        result = evaluate_baseline(split)
        print(f"naive max-uncertainty baseline — {name}: accuracy = {result['accuracy']:.3f} (n={len(split)})")
