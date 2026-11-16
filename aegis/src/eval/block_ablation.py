"""Shortcut decomposition with record-level bootstrap intervals (paper protocol, frozen MiniLM, flat MLP).

Variants (feature blocks of features.build, see features.py):
  full            every block
  shortcut_only   content block zeroed: only framework one-hot, agent-name one-hot and turn-count scalars remain
  content_only    name, framework and count blocks zeroed: only the mean turn embedding remains
Why: a reviewer asks how much of the score is skill on the text and how much is a shortcut (who the agent is, which
framework produced the log). shortcut_only needs no reading of the conversation at all.

Protocols: released train/val ("paper") and the same with train/val records dropped whose conversation is a verbatim copy of
a test record ("dedup"). Test sets: authors' repo file (primary, decision D11), HF file, Who&When.
Intervals: 2,000 bootstrap resamples of RECORDS; per seed predictions are resampled with the same record indices and averaged.
Run: python -m aegis.src.eval.block_ablation [--seeds 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..baselines import frequency
from ..baselines.flat_mlp import decode, train_one, tune
from ..features import FRAMEWORKS, EmbStore, agent_vocab, build
from ..loaders import load_paper_splits, load_split
from ..metrics import evaluate
from ..schema import ERROR_MODES
from ..utils import log_run

LV = ("pair", "agent", "error")
OUT = Path(__file__).resolve().parents[2] / "results" / "block_ablation.json"
SETS = {"repo_test": "test_repo", "hf_test": "test", "who_when": "who_when"}
E = 384


def project(pairs, level):
    if level == "pair":
        return set(pairs)
    i = 0 if level == "agent" else 1
    return {p[i] for p in pairs}


def counts(truth, pred):
    """Per-record (tp, fp, fn) for each level, as an (n, 3, 3) array."""
    c = np.zeros((len(truth), 3, 3))
    for r, (t, p) in enumerate(zip(truth, pred)):
        for li, lv in enumerate(LV):
            a, b = project(t, lv), project(p, lv)
            c[r, li] = (len(a & b), len(b - a), len(a - b))
    return c


def micro(c):  # c (..., 3 levels, 3) summed over records
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def bootstrap(per_seed_counts, n_boot=2000, seed=0):
    """per_seed_counts: (seeds, records, 3, 3). Returns point estimate and 95% interval per level (percent)."""
    rng = np.random.default_rng(seed)
    n = per_seed_counts.shape[1]
    point = micro(per_seed_counts.sum(1)).mean(0)
    boots = np.empty((n_boot, 3))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = micro(per_seed_counts[:, idx].sum(1)).mean(0)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {lv: {"micro": 100 * point[i], "lo": 100 * lo[i], "hi": 100 * hi[i]} for i, lv in enumerate(LV)}


def mask_cols(X, variant, d_total):
    X = X.copy()
    if variant == "shortcut_only":
        X[:, :E] = 0.0
    elif variant == "content_only":
        X[:, E:] = 0.0
    return X


def floor_counts(ts):
    return counts([t.labels for t in ts], [{(a, m) for a in set(t.agents) for m in ERROR_MODES} for t in ts])


def main(n_seeds: int) -> None:
    from ...scripts.compare_test_files import convo_hash

    store = EmbStore()
    paper = load_paper_splits()
    sets = {k: load_split(v) for k, v in SETS.items()}
    banned = {convo_hash(t) for k in ("hf_test", "repo_test") for t in sets[k]}
    protos = {
        "paper": paper,
        "dedup": {**paper, **{k: [t for t in paper[k] if convo_hash(t) not in banned] for k in ("train", "val")}},
    }
    result = {"seeds": n_seeds, "rows": {}}
    for k, ts in sets.items():
        result["rows"][f"floor|{k}"] = bootstrap(floor_counts(ts)[None])
    for proto, sp in protos.items():
        vocab = agent_vocab(sp["train"])
        ranking = frequency.fit(sp["train"])
        tv = [t.labels for t in sp["val"]]
        kbest = max((sum(evaluate(tv, frequency.predict(sp["val"], ranking, k))[l]["micro_f1"] for l in LV), k) for k in range(1, 16))[1]
        for k, ts in sets.items():
            c = counts([t.labels for t in ts], frequency.predict(ts, ranking, kbest))
            result["rows"][f"frequency|{proto}|{k}"] = bootstrap(c[None])
        Xtr0, Ytr, _ = build(sp["train"], store, vocab)
        Xva0, Yva, mva = build(sp["val"], store, vocab)
        feats0 = {k: build(ts, store, vocab) for k, ts in sets.items()}
        for variant in ("full", "shortcut_only", "content_only"):
            Xtr, Xva = mask_cols(Xtr0, variant, Xtr0.shape[1]), mask_cols(Xva0, variant, Xva0.shape[1])
            per = {k: [] for k in sets}
            for s in range(n_seeds):
                model, probs, epochs, n_params = train_one(Xtr, Ytr, Xva, Yva, s)
                _, ta, tm, _ = tune(probs, mva, sp["val"])
                for k, ts in sets.items():
                    X = mask_cols(feats0[k][0], variant, Xtr0.shape[1])
                    with torch.no_grad():
                        p = torch.sigmoid(model(torch.tensor((X - model.mu) / model.sd, dtype=torch.float32))).numpy()
                    pred = decode(p, feats0[k][2], len(ts), ta, tm)
                    per[k].append(counts([t.labels for t in ts], pred))
                    log_run(phase="S3", model=f"flat_mlp_{variant}", seed=s, split=f"{proto}_{k}", n_params=n_params,
                            config={"protocol": proto, "variant": variant, "t_a": ta, "t_m": tm, "epochs_run": epochs},
                            metrics={k: {lv: evaluate([t.labels for t in ts], pred)[lv] for lv in LV}},
                            notes="S3 block ablation; embeddings recomputed locally 2026-11-16")
            for k in sets:
                result["rows"][f"{variant}|{proto}|{k}"] = bootstrap(np.stack(per[k]))
            print(f"done {proto} {variant}", flush=True)
    OUT.write_text(json.dumps(result, indent=1))
    for key, r in result["rows"].items():
        print(f"{key:38s} " + " | ".join(f"{lv} {r[lv]['micro']:.2f} [{r[lv]['lo']:.2f},{r[lv]['hi']:.2f}]" for lv in LV), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    main(ap.parse_args().seeds)
