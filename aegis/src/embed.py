"""MiniLM turn embeddings with an on-disk cache keyed by a hash of the turn text.

Cache: aegis/data/cache/minilm/<first 2 hex>/<sha1>.npy is avoided (too many files); instead one
shard file per run: aegis/data/cache/minilm_shard_*.npz holding {hash: vector}. get_embeddings()
loads all shards, embeds only the missing texts, and writes a new shard.

Run (embeds every turn of every split): python -m aegis.src.embed
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LEN = 256
SHARD_SIZE = 5000


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    cache: dict = {}
    for p in sorted(CACHE_DIR.glob("minilm_shard_*.npz")):
        z = np.load(p)
        keys, vecs = z["keys"], z["vecs"]
        cache.update(zip(keys.tolist(), vecs))
    return cache


def get_embeddings(texts: list, batch_size: int = 128, device: str | None = None, verbose: bool = True) -> np.ndarray:
    """Return (len(texts), 384) float32 embeddings, computing only the texts missing from the cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _load_cache()
    keys = [text_key(t) for t in texts]
    missing = {}
    for k, t in zip(keys, texts):
        if k not in cache:
            missing[k] = t
    if missing:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME, device=device)
        model.max_seq_length = MAX_SEQ_LEN
        mk = list(missing)
        # sort by length so batches pad little
        order = sorted(range(len(mk)), key=lambda i: len(missing[mk[i]]))
        mk = [mk[i] for i in order]
        if verbose:
            print(f"embedding {len(mk)} new texts ({len(cache)} cached)", flush=True)
        t0 = time.time()
        # Encode in chunks and write one shard per chunk, so an interrupted run keeps its progress
        # and peak memory stays bounded (a previous full-corpus local run was killed for low memory).
        for s in range(0, len(mk), SHARD_SIZE):
            part = mk[s: s + SHARD_SIZE]
            vecs = model.encode([missing[k] for k in part], batch_size=batch_size, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
            shard = CACHE_DIR / f"minilm_shard_{int(time.time() * 1000)}.npz"
            np.savez(shard, keys=np.array(part), vecs=vecs)
            cache.update(zip(part, vecs))
            if verbose:
                print(f"  {min(s + SHARD_SIZE, len(mk))}/{len(mk)}  {time.time() - t0:.0f}s", flush=True)
    return np.stack([cache[k] for k in keys]).astype(np.float32)


if __name__ == "__main__":
    from .loaders import load_split

    texts: list = []
    for sp in ("test", "who_when", "val", "train"):   # smaller splits first so partial progress is useful
        texts.extend(u.text for t in load_split(sp) for u in t.turns)
    uniq = list(dict.fromkeys(texts))
    print(f"{len(texts)} turns, {len(uniq)} unique texts")
    get_embeddings(uniq)
