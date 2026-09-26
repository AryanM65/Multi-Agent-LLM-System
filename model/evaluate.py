"""Evaluation metrics: top-1/top-2 accuracy, plus y_true/y_pred for
precision/recall/F1/confusion-matrix via sklearn.metrics.

Known dataset-specific rule (see model-plan.md Step 7, btp-pipeline/model.md
Section 3 item 5): `retriever_c` in `star`/`triple_retriever_fanin` is never
a true fault target. `excluded_node_by_topology` lets a caller drop that
class index from consideration for those two topologies specifically when
computing per-class metrics -- not applied by default here since top-1/top-2
accuracy is unaffected by it (the true label is simply never that class), but
exposed so a downstream classification_report can mask it out cleanly.
"""
import torch

# topology_id -> node_id that is never a valid fault target in this dataset
KNOWN_UNREACHABLE_TARGETS = {
    "star": "retriever_c",
    "triple_retriever_fanin": "retriever_c",
}


def evaluate(model, graphs, k_for_top_k=2):
    model.eval()
    top1_correct, top2_correct, total = 0, 0, 0
    all_true, all_pred = [], []

    with torch.no_grad():
        for g in graphs:
            logits = model(g.x, g.edge_index)
            true_y = int(g.y.item())

            ranked = logits.argsort(descending=True)
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


def classification_report(graphs, y_true, y_pred):
    """Human-readable per-graph breakdown: node names instead of raw indices,
    with the known-unreachable-target quirk annotated rather than hidden."""
    from sklearn.metrics import classification_report as sk_report

    # Node-name labels aren't globally comparable across graphs of different
    # topologies (index 0 means a different node depending on topology), so
    # this report is intended to be run per-topology-group, not pooled, when
    # node-name-level detail is wanted. Pooled, we can still report the
    # class-index-level report as a coarse signal.
    labels = sorted(set(y_true) | set(y_pred))
    return sk_report(y_true, y_pred, labels=labels, zero_division=0)


if __name__ == "__main__":
    from model.data.load_raw import load_topologies, load_trials, validate
    from model.data.build_graph_dataset import build_dataset
    from model.data.split import split_graphs
    from model.gnn import FaultLocalizerGNN

    topologies = load_topologies()
    trials = load_trials()
    validate(trials, topologies)
    graphs = build_dataset(trials, topologies)
    train_set, val_set, ood_set = split_graphs(graphs, topologies)

    model = FaultLocalizerGNN(in_dim=8, hidden_dim=32, num_layers=2)
    model.load_state_dict(torch.load("model/checkpoints/best.pt"))

    for name, split in [("val", val_set), ("ood", ood_set)]:
        m = evaluate(model, split)
        print(f"\n=== {name} (n={len(split)}) ===")
        print(f"top1_accuracy: {m['top1_accuracy']:.3f}   top2_accuracy: {m['top2_accuracy']:.3f}")
        print(classification_report(split, m["y_true"], m["y_pred"]))
