"""Train / validation / OOD-test split.

- OOD test = every graph whose topology's `split` is "ood_test" (6 topologies,
  164 trials). Never touched during training or hyperparameter tuning.
- Train/val = the remaining 14 topologies' graphs (329 trials), split by
  QUESTION (not by trial) so the same question's phrasing never appears in
  both train and val -- dataset_description.md Section 9 flags this risk
  explicitly (all 30 questions recur across many topologies/fault conditions).
"""
import random


def split_graphs(graphs, topologies, val_fraction=0.2, seed=0):
    train_topo_graphs = [g for g in graphs if topologies[g.topology_id]["split"] == "train"]
    ood_graphs = [g for g in graphs if topologies[g.topology_id]["split"] == "ood_test"]

    by_question = {}
    for g in train_topo_graphs:
        by_question.setdefault(g.question, []).append(g)

    questions = sorted(by_question.keys())
    random.Random(seed).shuffle(questions)
    n_val_q = max(1, int(len(questions) * val_fraction))
    val_questions = set(questions[:n_val_q])

    train_set = [g for q, gs in by_question.items() if q not in val_questions for g in gs]
    val_set = [g for q, gs in by_question.items() if q in val_questions for g in gs]

    return train_set, val_set, ood_graphs


if __name__ == "__main__":
    from model.data.load_raw import load_topologies, load_trials, validate
    from model.data.build_graph_dataset import build_dataset

    topologies = load_topologies()
    trials = load_trials()
    validate(trials, topologies)
    graphs = build_dataset(trials, topologies)

    train_set, val_set, ood_set = split_graphs(graphs, topologies)

    print(f"train: {len(train_set)} graphs")
    print(f"val:   {len(val_set)} graphs")
    print(f"ood:   {len(ood_set)} graphs")

    train_qs = {g.question for g in train_set}
    val_qs = {g.question for g in val_set}
    overlap = train_qs & val_qs
    print(f"\ntrain questions: {len(train_qs)}, val questions: {len(val_qs)}, overlap: {len(overlap)} (should be 0)")
    assert not overlap, "train/val question leakage detected!"
    print("No question leakage between train and val. OK.")
