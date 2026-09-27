"""Per-topology OOD breakdown for the current best config (SAGE hd64/nl3/
lr5e-4, PCA-128, use_embedding_variance=True, ood_top1=0.378 pooled).
Answers: is the 0.378 gain broad across OOD topologies, or concentrated in
1-2 -- tells us whether more model tuning or more OOD data coverage is the
better next lever. See model/eval_ood_breakdown.py, docs/model/futurework.md.
"""
import torch

from model.data.load_raw import load_topologies, load_trials, validate
from model.data.build_graph_dataset import build_dataset
from model.data.split import split_graphs
from model.data.reduce_embeddings import fit_and_reduce_embeddings_with_variance
from model.gnn import FaultLocalizerGNN
from model.train import train
from model.evaluate import evaluate
from model.eval_ood_breakdown import breakdown_by_topology

HIDDEN_DIM = 64
NUM_LAYERS = 3
LR = 5e-4
DROPOUT = 0.3
CONV_TYPE = "sage"
SEED = 0
EPOCHS = 150
PATIENCE = 30
PCA_COMPONENTS = 128


if __name__ == "__main__":
    topologies = load_topologies()
    trials = load_trials()
    validate(trials, topologies)
    graphs = build_dataset(trials, topologies, bidirectional=True, use_embeddings=True,
                            use_embedding_variance=True)
    train_set, val_set, ood_set = split_graphs(graphs, topologies, seed=SEED)
    fit_and_reduce_embeddings_with_variance(train_set, graphs, n_components=PCA_COMPONENTS, seed=SEED)

    torch.manual_seed(SEED)
    in_dim = graphs[0].x.shape[1]
    model = FaultLocalizerGNN(in_dim=in_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                              dropout=DROPOUT, conv_type=CONV_TYPE)
    model, best_val_acc, best_epoch, epochs_run = train(
        model, train_set, val_set, epochs=EPOCHS, lr=LR, patience=PATIENCE, verbose=False,
    )

    rows = breakdown_by_topology(model, ood_set)
    print(f"{'topology':<20} {'n':>4} {'top1':>6} {'top2':>6} {'prec':>6} {'rec':>6} {'f1':>6}")
    for r in rows:
        print(f"{r['topology']:<20} {r['n']:>4} {r['top1']:>6.3f} {r['top2']:>6.3f} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")

    pooled = evaluate(model, ood_set)
    print(f"\nPooled OOD: top1={pooled['top1_accuracy']:.3f}  top2={pooled['top2_accuracy']:.3f}  n={len(ood_set)}")
