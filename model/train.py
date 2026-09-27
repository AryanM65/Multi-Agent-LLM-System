"""Training loop.

Graphs vary in size (3-7 nodes), and each graph's softmax is over a
DIFFERENT number of classes (num_nodes + 1 "no fault"). That makes PyG's
standard batched-DataLoader loss computation awkward (it assumes one shared
label space per batch). Given the dataset is tiny (a few hundred graphs,
CPU-fast), we simply iterate graph-by-graph per epoch instead -- simpler and
correct, at negligible speed cost for this dataset size.
"""
import torch
import torch.nn.functional as F

from model.gnn import FaultLocalizerGNN, FAULT_TYPES
from model.evaluate import evaluate, macro_prf1, classification_report


def compute_no_fault_weight(graphs) -> float:
    """Inverse-frequency weight for the "no fault" class, since it's a
    minority class (~15% of trials) relative to all real-fault-node classes
    combined. weight = n_faulted / n_clean, so clean examples' loss counts
    proportionally more -- balances the two without needing per-node
    weighting (which doesn't make sense: node identity is arbitrary per
    graph, "no fault vs. some fault" is the real class-imbalance axis)."""
    n_clean = sum(1 for g in graphs if int(g.y.item()) == g.num_nodes)
    n_faulted = len(graphs) - n_clean
    if n_clean == 0:
        return 1.0
    return n_faulted / n_clean


def run_epoch(model, graphs, optimizer=None, no_fault_weight=1.0, aux_weight=0.3):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    for g in graphs:
        out = model(g.x, g.edge_index)

        if model.multi_task:
            logits, fault_type_logits = out
            weight = torch.ones(g.num_nodes + 1)
            weight[g.num_nodes] = no_fault_weight
            node_loss = F.cross_entropy(logits.unsqueeze(0), g.y, weight=weight)

            fault_type_idx = FAULT_TYPES.index(g.true_label)
            type_target = torch.tensor([fault_type_idx], dtype=torch.long)
            type_loss = F.cross_entropy(fault_type_logits.unsqueeze(0), type_target)

            loss = node_loss + aux_weight * type_loss
        else:
            logits = out  # [num_nodes + 1]
            weight = torch.ones(g.num_nodes + 1)
            weight[g.num_nodes] = no_fault_weight
            loss = F.cross_entropy(logits.unsqueeze(0), g.y, weight=weight)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())

    return total_loss / len(graphs)


def train(model, train_graphs, val_graphs, epochs=100, lr=1e-3, weight_decay=1e-4,
          checkpoint_path=None, patience=20, verbose=True, class_weighted=True, aux_weight=0.3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    no_fault_weight = compute_no_fault_weight(train_graphs) if class_weighted else 1.0
    if verbose and class_weighted:
        print(f"no_fault_weight = {no_fault_weight:.3f} (class-imbalance correction)")

    best_val_acc = -1.0
    best_epoch = -1
    epochs_since_improvement = 0
    best_state = None
    epochs_run = 0

    for epoch in range(epochs):
        epochs_run = epoch + 1
        train_loss = run_epoch(model, train_graphs, optimizer, no_fault_weight=no_fault_weight, aux_weight=aux_weight)
        val_loss = run_epoch(model, val_graphs, optimizer=None, no_fault_weight=no_fault_weight, aux_weight=aux_weight)
        val_metrics = evaluate(model, val_graphs)
        val_acc = val_metrics["top1_accuracy"]

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_since_improvement += 1

        if verbose:
            print(f"epoch {epoch:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.3f}{'  *' if improved else ''}")

        if epochs_since_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc, best_epoch, epochs_run


if __name__ == "__main__":
    import os
    from model.data.load_raw import load_topologies, load_trials, validate
    from model.data.build_graph_dataset import build_dataset
    from model.data.split import split_graphs
    from model.baselines import evaluate_baseline
    from model.metrics_log import log_run

    SEED = 0
    HIDDEN_DIM = 32
    NUM_LAYERS = 2
    DROPOUT = 0.2
    LR = 1e-3
    EPOCHS = 150
    PATIENCE = 30

    topologies = load_topologies()
    trials = load_trials()
    validate(trials, topologies)
    graphs = build_dataset(trials, topologies, bidirectional=True)
    train_set, val_set, ood_set = split_graphs(graphs, topologies, seed=SEED)

    torch.manual_seed(SEED)
    in_dim = graphs[0].x.shape[1]
    model = FaultLocalizerGNN(in_dim=in_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT)

    os.makedirs("model/checkpoints", exist_ok=True)
    model, best_val_acc, best_epoch, epochs_run = train(
        model, train_set, val_set,
        epochs=EPOCHS, lr=LR, patience=PATIENCE,
        checkpoint_path="model/checkpoints/best.pt",
    )

    # Full metrics on every split, using the best (early-stopped) checkpoint
    train_metrics = evaluate(model, train_set)
    val_metrics = evaluate(model, val_set)
    ood_metrics = evaluate(model, ood_set)
    baseline_ood = evaluate_baseline(ood_set)

    train_prf1 = macro_prf1(train_metrics["y_true"], train_metrics["y_pred"])
    val_prf1 = macro_prf1(val_metrics["y_true"], val_metrics["y_pred"])
    ood_prf1 = macro_prf1(ood_metrics["y_true"], ood_metrics["y_pred"])

    print(f"\nBest val accuracy: {best_val_acc:.3f} (epoch {best_epoch}, ran {epochs_run} epochs total)")
    print(f"train: top1={train_metrics['top1_accuracy']:.3f}  top2={train_metrics['top2_accuracy']:.3f}  "
          f"macroP={train_prf1['precision']:.3f}  macroR={train_prf1['recall']:.3f}  macroF1={train_prf1['f1']:.3f}")
    print(f"val:   top1={val_metrics['top1_accuracy']:.3f}  top2={val_metrics['top2_accuracy']:.3f}  "
          f"macroP={val_prf1['precision']:.3f}  macroR={val_prf1['recall']:.3f}  macroF1={val_prf1['f1']:.3f}")
    print(f"ood:   top1={ood_metrics['top1_accuracy']:.3f}  top2={ood_metrics['top2_accuracy']:.3f}  "
          f"macroP={ood_prf1['precision']:.3f}  macroR={ood_prf1['recall']:.3f}  macroF1={ood_prf1['f1']:.3f}")
    print(f"baseline ood top1: {baseline_ood['accuracy']:.3f}")

    print("\nOOD classification report (class index -> per-class precision/recall/F1):")
    print(classification_report(ood_set, ood_metrics["y_true"], ood_metrics["y_pred"]))

    log_run(
        config={
            "seed": SEED, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
            "dropout": DROPOUT, "lr": LR, "epochs_requested": EPOCHS, "patience": PATIENCE,
            "in_dim": in_dim,
        },
        metrics={
            "train": {"top1_accuracy": train_metrics["top1_accuracy"], "top2_accuracy": train_metrics["top2_accuracy"], **train_prf1, "n": len(train_set)},
            "val": {"top1_accuracy": val_metrics["top1_accuracy"], "top2_accuracy": val_metrics["top2_accuracy"], **val_prf1, "n": len(val_set)},
            "ood": {"top1_accuracy": ood_metrics["top1_accuracy"], "top2_accuracy": ood_metrics["top2_accuracy"], **ood_prf1, "n": len(ood_set)},
            "baseline_ood_top1_accuracy": baseline_ood["accuracy"],
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
        },
    )
    print("\nLogged this run to model/metrics_history.jsonl")
