"""Compress the 384-dim node embeddings (from enrich_features.py) down to a
smaller dimension via PCA, fit on TRAIN graphs only (no leakage into val/ood).

Full 384-dim embeddings fed straight into a small GNN caused severe
overfitting (see model/metrics_history.jsonl run comparing bidirectional-only
vs +embeddings: train top1 0.30->0.54, OOD top1 flat at ~0.23-0.24). This
keeps the same signal at much lower dimensionality, reducing overfitting
surface.
"""
import numpy as np
import torch
from sklearn.decomposition import PCA

from model.data.build_graph_dataset import EMBEDDING_DIM


def fit_and_reduce_embeddings(train_graphs, all_graphs, n_components=32, seed=0):
    """Mutates each graph's `x` in place: replaces the trailing EMBEDDING_DIM
    columns (raw embeddings) with `n_components` PCA-reduced columns. Returns
    the fitted PCA object (for reference/logging)."""
    train_emb = np.concatenate(
        [g.x[:, -EMBEDDING_DIM:].numpy() for g in train_graphs], axis=0
    )
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(train_emb)

    for g in all_graphs:
        scalar = g.x[:, :-EMBEDDING_DIM]
        emb = g.x[:, -EMBEDDING_DIM:].numpy()
        reduced = pca.transform(emb)
        g.x = torch.cat([scalar, torch.tensor(reduced, dtype=torch.float)], dim=1)

    return pca
