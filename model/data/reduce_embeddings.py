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


def fit_and_reduce_embeddings_with_variance(train_graphs, all_graphs, n_components=32, seed=0):
    """Same as fit_and_reduce_embeddings, but for the [scalar | mean_emb(384)
    | std_emb(384)] layout produced when use_embedding_variance=True. Fits
    two INDEPENDENT PCA models (mean block, std block), each on train only --
    they're different signals (central tendency vs. spread across k samples),
    no reason to force them through the same projection. Returns
    (pca_mean, pca_std)."""
    train_mean = np.concatenate(
        [g.x[:, -2 * EMBEDDING_DIM:-EMBEDDING_DIM].numpy() for g in train_graphs], axis=0
    )
    train_std = np.concatenate(
        [g.x[:, -EMBEDDING_DIM:].numpy() for g in train_graphs], axis=0
    )
    pca_mean = PCA(n_components=n_components, random_state=seed)
    pca_mean.fit(train_mean)
    pca_std = PCA(n_components=n_components, random_state=seed)
    pca_std.fit(train_std)

    for g in all_graphs:
        scalar = g.x[:, :-2 * EMBEDDING_DIM]
        mean_emb = g.x[:, -2 * EMBEDDING_DIM:-EMBEDDING_DIM].numpy()
        std_emb = g.x[:, -EMBEDDING_DIM:].numpy()
        reduced_mean = pca_mean.transform(mean_emb)
        reduced_std = pca_std.transform(std_emb)
        g.x = torch.cat([
            scalar,
            torch.tensor(reduced_mean, dtype=torch.float),
            torch.tensor(reduced_std, dtype=torch.float),
        ], dim=1)

    return pca_mean, pca_std
