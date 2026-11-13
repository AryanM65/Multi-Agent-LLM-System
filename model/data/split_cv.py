"""Grouped k-fold-by-topology cross-validation split (model_improvement.md
Section 5.4) -- an ADDITIVE alternative to model/data/split.py's single
question-based val split, not a replacement (split.py is left untouched;
other code still imports split_graphs from it unchanged).

Today's val split holds out QUESTIONS on the SAME 14 train topologies, so
early stopping and every hyperparameter/feature choice optimize
in-distribution accuracy, not the thing that actually matters -- accuracy on
topology SHAPES the model has never seen. This groups the 14 train
topologies into k folds and holds out ~3 whole topologies per fold, so the
held-out score in each fold is itself an OOD-style estimate (just computed
without touching the real, never-tuned-against ood_test topologies).
"""
import random
import statistics


def grouped_kfold_by_topology(graphs, topologies, k=5, seed=0):
    """Returns a list of (train_subset, held_out_subset, held_out_topology_ids)
    tuples, one per fold. Only ever partitions topologies with split=="train";
    ood_test topologies never appear in any fold."""
    train_topo_ids = sorted({tid for tid, t in topologies.items() if t["split"] == "train"})
    rng = random.Random(seed)
    rng.shuffle(train_topo_ids)

    folds = [train_topo_ids[i::k] for i in range(k)]  # near-equal groups, e.g. 14/5 -> 3,3,3,3,2

    result = []
    for held_out_topo_ids in folds:
        held_out = set(held_out_topo_ids)
        train_subset = [
            g for g in graphs
            if topologies[g.topology_id]["split"] == "train" and g.topology_id not in held_out
        ]
        held_out_subset = [g for g in graphs if g.topology_id in held_out]
        result.append((train_subset, held_out_subset, held_out_topo_ids))
    return result


def inner_val_split(train_subset, val_fraction=0.15, seed=0):
    """A small by-QUESTION val split carved out of one fold's training subset,
    for early stopping only -- the fold's held-out TOPOLOGIES are never used
    for this (that would leak the thing being measured back into training
    control). Mirrors model/data/split.py's question-grouping logic."""
    by_question = {}
    for g in train_subset:
        by_question.setdefault(g.question, []).append(g)

    questions = sorted(by_question.keys())
    random.Random(seed).shuffle(questions)
    n_val_q = max(1, int(len(questions) * val_fraction))
    val_questions = set(questions[:n_val_q])

    inner_train = [g for q, gs in by_question.items() if q not in val_questions for g in gs]
    inner_val = [g for q, gs in by_question.items() if q in val_questions for g in gs]
    return inner_train, inner_val


def cv_summary(fold_top1_scores: list) -> dict:
    mean = statistics.mean(fold_top1_scores)
    std = statistics.pstdev(fold_top1_scores) if len(fold_top1_scores) > 1 else 0.0
    return {"mean": mean, "std": std, "folds": fold_top1_scores}


if __name__ == "__main__":
    from model.data.load_raw import load_topologies, load_trials, validate
    from model.data.build_graph_dataset import build_dataset

    topologies = load_topologies()
    trials = load_trials()
    validate(trials, topologies)
    graphs = build_dataset(trials, topologies)

    folds = grouped_kfold_by_topology(graphs, topologies, k=5, seed=0)
    for i, (train_subset, held_out_subset, held_out_ids) in enumerate(folds):
        print(f"fold {i}: held_out_topologies={held_out_ids}  "
              f"train_graphs={len(train_subset)}  held_out_graphs={len(held_out_subset)}")

    all_held_out = {tid for _, _, ids in folds for tid in ids}
    all_train_topo = {tid for tid, t in topologies.items() if t["split"] == "train"}
    assert all_held_out == all_train_topo, "every train topology must appear in exactly one fold"
    print(f"\nAll {len(all_train_topo)} train topologies covered across {len(folds)} folds. OK.")
