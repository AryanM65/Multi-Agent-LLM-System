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

from model.gnn import FaultLocalizerGNN
from model.evaluate import evaluate


def run_epoch(model, graphs, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    for g in graphs:
        logits = model(g.x, g.edge_index)  # [num_nodes + 1]
        loss = F.cross_entropy(logits.unsqueeze(0), g.y)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())

    return total_loss / len(graphs)


def train(model, train_graphs, val_graphs, epochs=100, lr=1e-3, weight_decay=1e-4,
          checkpoint_path=None, patience=20, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_epoch = -1
    epochs_since_improvement = 0
    best_state = None
    epochs_run = 0

    for epoch in range(epochs):
        epochs_run = epoch + 1
        train_loss = run_epoch(model, train_graphs, optimizer)
        val_loss = run_epoch(model, val_graphs, optimizer=None)
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
    graphs = build_dataset(trials, topologies)
    train_set, val_set, ood_set = split_graphs(graphs, topologies, seed=SEED)

    torch.manual_seed(SEED)
    model = FaultLocalizerGNN(in_dim=8, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT)

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

    print(f"\nBest val accuracy: {best_val_acc:.3f} (epoch {best_epoch}, ran {epochs_run} epochs total)")
    print(f"train: top1={train_metrics['top1_accuracy']:.3f}  top2={train_metrics['top2_accuracy']:.3f}")
    print(f"val:   top1={val_metrics['top1_accuracy']:.3f}  top2={val_metrics['top2_accuracy']:.3f}")
    print(f"ood:   top1={ood_metrics['top1_accuracy']:.3f}  top2={ood_metrics['top2_accuracy']:.3f}")
    print(f"baseline ood top1: {baseline_ood['accuracy']:.3f}")

    log_run(
        config={
            "seed": SEED, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
            "dropout": DROPOUT, "lr": LR, "epochs_requested": EPOCHS, "patience": PATIENCE,
        },
        metrics={
            "train": {"top1_accuracy": train_metrics["top1_accuracy"], "top2_accuracy": train_metrics["top2_accuracy"], "n": len(train_set)},
            "val": {"top1_accuracy": val_metrics["top1_accuracy"], "top2_accuracy": val_metrics["top2_accuracy"], "n": len(val_set)},
            "ood": {"top1_accuracy": ood_metrics["top1_accuracy"], "top2_accuracy": ood_metrics["top2_accuracy"], "n": len(ood_set)},
            "baseline_ood_top1_accuracy": baseline_ood["accuracy"],
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
        },
    )
    print("\nLogged this run to model/metrics_history.jsonl")
