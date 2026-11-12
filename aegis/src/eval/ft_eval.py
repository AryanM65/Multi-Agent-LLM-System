"""Validation-only comparison: frozen MiniLM vs fine-tuned MiniLM vectors, same downstream models (grouped split).

Usage: python -m aegis.src.eval.ft_eval --ft-dir <dir with minilm_shard_*.npz> [--seeds 3]
Runs flat MLP, 0-layer control and 2-layer GNN on (a) the frozen cache and (b) the fine-tuned shards. Test / Who&When are not read.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ..baselines.flat_mlp import train_one, tune
from ..features import EmbStore, agent_vocab, build
from ..graph import make_items
from ..gnn.turn_gnn import train_gnn
from ..loaders import load_grouped_splits
from ..metrics import evaluate

LV = ("pair", "agent", "error")


def run(tag: str, store: EmbStore, sp: dict, seeds: int) -> None:
    vocab = agent_vocab(sp["train"])
    qvec = {q: store.lookup(q) for q in sorted({t.query for k in ("train", "val") for t in sp[k]})}
    Xtr, Ytr, mtr = build(sp["train"], store, vocab)
    Xva, Yva, mva = build(sp["val"], store, vocab)
    items = {k: make_items(sp[k], store, qvec, *d) for k, d in (("train", (Xtr, Ytr, mtr)), ("val", (Xva, Yva, mva)))}
    out: dict = {}
    for s in range(seeds):
        t0 = time.time()
        _, probs, _, _ = train_one(Xtr, Ytr, Xva, Yva, s)
        out.setdefault("flat_mlp", []).append(tune(probs, mva, sp["val"])[3])
        for layers in (0, 2):
            probs, _, _ = train_gnn(items["train"], items["val"], s, layers)
            out.setdefault(f"gnn_L{layers}", []).append(tune(probs, mva, sp["val"])[3])
        print(f"[{tag}] seed {s} done ({time.time() - t0:.0f}s)", flush=True)
    for name, runs in out.items():
        parts = [f"{l} {np.mean([100 * r[l]['micro_f1'] for r in runs]):.2f}/{np.mean([100 * r[l]['macro_f1'] for r in runs]):.2f}" for l in LV]
        pstd = np.std([100 * r["pair"]["micro_f1"] for r in runs])
        print(f"[{tag}] {name:9s} val grouped: " + "  ".join(parts) + f"  (pair micro std {pstd:.2f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-dir", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--skip-frozen", action="store_true")
    a = ap.parse_args()
    sp = load_grouped_splits()
    if not a.skip_frozen:
        run("frozen", EmbStore(), sp, a.seeds)
    run("finetuned", EmbStore(a.ft_dir), sp, a.seeds)
