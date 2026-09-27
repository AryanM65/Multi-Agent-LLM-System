"""Flat-feature non-graph baseline -- answers "is the graph structure/
message-passing doing real work, or would a much simpler tabular model do
just as well?" See docs/model/futurework.md Section 4.

Each graph's per-node feature matrix is flattened (zero-padded to the
dataset-wide max node count) into one fixed-size vector, plus num_nodes as
an extra feature. A single RandomForestClassifier (used instead of XGBoost
to avoid a new dependency -- same ablation intent) is trained on that flat
representation, over the same y label space (node index, or num_nodes for
"no fault") the GNN uses. This is not a perfectly fair comparison (index 3
means a different thing in a 4-node vs. 7-node graph), but it's the standard
version of this ablation and still answers the intended question.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def flatten_graphs(graphs, max_nodes: int):
    X, y = [], []
    for g in graphs:
        feat_dim = g.x.shape[1]
        padded = np.zeros((max_nodes, feat_dim), dtype=np.float32)
        n = g.x.shape[0]
        padded[:n] = g.x.numpy()
        flat = np.concatenate([padded.flatten(), [float(n)]])
        X.append(flat)
        y.append(int(g.y.item()))
    return np.array(X), np.array(y)


def evaluate_nongraph_baseline(train_graphs, val_graphs, ood_graphs, seed=0):
    max_nodes = max(g.x.shape[0] for g in train_graphs + val_graphs + ood_graphs)

    X_train, y_train = flatten_graphs(train_graphs, max_nodes)
    X_val, y_val = flatten_graphs(val_graphs, max_nodes)
    X_ood, y_ood = flatten_graphs(ood_graphs, max_nodes)

    clf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced")
    clf.fit(X_train, y_train)

    def top1(X, y):
        return float((clf.predict(X) == y).mean())

    return {
        "train_top1": top1(X_train, y_train),
        "val_top1": top1(X_val, y_val),
        "ood_top1": top1(X_ood, y_ood),
        "max_nodes": max_nodes,
    }


if __name__ == "__main__":
    from model.data.load_raw import load_topologies, load_trials, validate
    from model.data.build_graph_dataset import build_dataset
    from model.data.split import split_graphs

    topologies = load_topologies()
    trials = load_trials()
    validate(trials, topologies)
    graphs = build_dataset(trials, topologies, bidirectional=True)
    train_set, val_set, ood_set = split_graphs(graphs, topologies, seed=0)

    result = evaluate_nongraph_baseline(train_set, val_set, ood_set)
    print(f"Non-graph flat-feature baseline (RandomForest, max_nodes={result['max_nodes']}):")
    print(f"  train top1: {result['train_top1']:.3f}")
    print(f"  val   top1: {result['val_top1']:.3f}")
    print(f"  ood   top1: {result['ood_top1']:.3f}")
