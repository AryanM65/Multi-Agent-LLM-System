"""Per-topology OOD breakdown -- pooled OOD accuracy can hide "great on 5
topologies, terrible on 1." This reports top-1/top-2/macro-F1 per OOD
topology individually, using a trained checkpoint.
"""
import torch

from model.data.load_raw import load_topologies, load_trials, validate
from model.data.build_graph_dataset import build_dataset
from model.data.split import split_graphs
from model.gnn import FaultLocalizerGNN
from model.evaluate import evaluate, macro_prf1


def breakdown_by_topology(model, graphs):
    by_topo = {}
    for g in graphs:
        by_topo.setdefault(g.topology_id, []).append(g)

    rows = []
    for topo_id, topo_graphs in sorted(by_topo.items()):
        m = evaluate(model, topo_graphs)
        prf1 = macro_prf1(m["y_true"], m["y_pred"])
        rows.append({
            "topology": topo_id,
            "n": len(topo_graphs),
            "top1": m["top1_accuracy"],
            "top2": m["top2_accuracy"],
            "precision": prf1["precision"],
            "recall": prf1["recall"],
            "f1": prf1["f1"],
        })
    return rows


if __name__ == "__main__":
    import sys

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "model/checkpoints/best_1762.pt"

    topologies = load_topologies()
    trials = load_trials()
    validate(trials, topologies)
    graphs = build_dataset(trials, topologies)
    _, _, ood_set = split_graphs(graphs, topologies, seed=0)

    model = FaultLocalizerGNN(in_dim=8, hidden_dim=32, num_layers=2)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    rows = breakdown_by_topology(model, ood_set)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'topology':<20} {'n':>4} {'top1':>6} {'top2':>6} {'prec':>6} {'rec':>6} {'f1':>6}")
    for r in rows:
        print(f"{r['topology']:<20} {r['n']:>4} {r['top1']:>6.3f} {r['top2']:>6.3f} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")

    pooled = evaluate(model, ood_set)
    print(f"\nPooled OOD: top1={pooled['top1_accuracy']:.3f}  top2={pooled['top2_accuracy']:.3f}  n={len(ood_set)}")
