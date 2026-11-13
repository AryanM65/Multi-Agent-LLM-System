"""Who&When-focused experiments that must not hurt AEGIS: ensembling across encoders, name-dropout, and label-shift correction.

For each encoder directory (fine-tuned vectors) and each seed it trains the flat MLP (name-embedding features; variant nd = training set doubled with a
name-hidden copy, peernd = nd + peer-contrast features), then:
  single    - each model alone (mean over encoders and seeds)
  ensemble  - probabilities averaged over the encoders (same seed index), thresholds tuned on AEGIS grouped val of the averaged probabilities
  shift     - ensemble plus label-shift correction for Who&When: the mode prior of Who&When is estimated WITHOUT labels by EM on the model's own
              per-record mode distribution (Saerens-style), starting from the AEGIS train mode prior; mode probabilities are multiplied by prior_ww / prior_src
              before decoding. Thresholds still from AEGIS val (unchanged). No Who&When label is used anywhere.
All numbers: micro F1 on AEGIS grouped test and Who&When, plus macro.

Run: python -m aegis.src.eval.ww_focus --encoders es_s0,es_s1,... --variants nd,peernd --seeds 3 --tag ww1
Encoder dirs are aegis/results/kaggle_logs/<name>/ft_shards_<name>.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ..baselines.flat_mlp import train_one
from ..baselines.flat_probe import relational_features
from ..embed import get_embeddings
from ..features import EmbStore, agent_vocab, build
from ..loaders import load_grouped_splits, load_split
from ..metrics import evaluate
from ..schema import ERROR_MODES
from ..utils import log_run
from .improve_cpu import decode2, peer_block, tune2
from .whowhen_study import mask_cols

LV = ("pair", "agent", "error")
NM = len(ERROR_MODES)


def estimate_prior(probs: np.ndarray, meta: list, n_traj: int, src_prior: np.ndarray, iters: int = 50) -> np.ndarray:
    """EM estimate of the target mode prior from unlabeled predictions (per-record soft mode distribution)."""
    q = np.zeros((n_traj, NM))
    for (ti, _a), p in zip(meta, probs):
        q[ti] += p[-1] * p[:-1]
    q = q / np.clip(q.sum(1, keepdims=True), 1e-9, None)
    pri = src_prior.copy()
    for _ in range(iters):
        post = q * (pri / src_prior)
        post = post / np.clip(post.sum(1, keepdims=True), 1e-9, None)
        pri = post.mean(0)
    return pri


def enc_dir(enc: str) -> str:
    import os
    import re
    hy = re.sub(r"_s([0-9])", lambda m: "-s" + m.group(1), enc)     # later-seed vector dirs are hyphenated (ft_shards_supcon5-s1)
    for d in (f"aegis/results/kaggle_logs/{enc}/ft_shards_{enc}", f"aegis/results/kaggle_logs/{enc}/ft_shards_{hy}", f"aegis/results/kaggle_logs/ft_{enc}/ft_shards_{enc}"):
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(enc)


def line(res) -> str:
    return " ".join(f"{l[:4]} {100*res[l]['micro_f1']:.2f}/{100*res[l]['macro_f1']:.2f}" for l in LV)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", required=True)
    ap.add_argument("--variants", default="nd,peernd")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="ww1")
    args = ap.parse_args()
    encs = args.encoders.split(",")
    sp = load_grouped_splits()
    test, ww = load_split("test"), load_split("who_when")
    trajs = {"train": sp["train"], "val": sp["val"], "test": test, "who_when": ww}
    vocab = agent_vocab(sp["train"])
    src_prior = None
    per = {}   # (variant, seed) -> {split: [probs per encoder]}
    meta = None
    names_cache = None
    for enc in encs:
        store = EmbStore(enc_dir(enc))
        feats = {k: build(ts, store, vocab) for k, ts in trajs.items()}
        meta = {k: v[2] for k, v in feats.items()}
        E = store.dim
        if names_cache is None:
            names = sorted({a for v in feats.values() for (_, a) in v[2]})
            names_cache = dict(zip(names, get_embeddings([n.replace("_", " ").replace("(", " ").replace(")", " ") for n in names], verbose=False)))
        base = {k: np.concatenate([mask_cols(v[0], E, len(vocab)), np.stack([names_cache[a] for (_, a) in v[2]])], axis=1) for k, v in feats.items()}
        nb0 = base["train"].shape[1] - E
        if src_prior is None:
            ytr = feats["train"][1]
            src_prior = ytr.sum(0) / ytr.sum()
        for variant in args.variants.split(","):
            if variant == "peernd":
                X = {k: np.hstack([base[k], peer_block(trajs[k], store), relational_features(trajs[k], store)]) for k in trajs}
            else:
                X = base
            Xtr, Ytr = X["train"], feats["train"][1]
            drop = Xtr.copy()
            drop[:, nb0:nb0 + E] = 0.0
            Xtr, Ytr = np.vstack([Xtr, drop]), np.vstack([Ytr, Ytr])
            for s in range(args.seeds):
                model, _, _, _ = train_one(Xtr, Ytr, X["val"], feats["val"][1], s)
                d = per.setdefault((variant, s), {})
                for k in ("val", "test", "who_when"):
                    with torch.no_grad():
                        p = torch.sigmoid(model(torch.tensor((X[k] - model.mu) / model.sd, dtype=torch.float32))).numpy()
                    d.setdefault(k, []).append(p)
        print(f"[{enc}] trained", flush=True)
    for variant in args.variants.split(","):
        rows = {"single": [], "ensemble": [], "shift": []}
        for s in range(args.seeds):
            d = per[(variant, s)]
            ens = {k: np.mean(v, axis=0) for k, v in d.items()}
            # thresholds from AEGIS val only
            _, ta, tm = tune2("prod", ens["val"], meta["val"], sp["val"])
            pri = estimate_prior(ens["who_when"], meta["who_when"], len(ww), src_prior)
            w = pri / src_prior
            w = w / w.mean()
            shifted = ens["who_when"].copy()
            shifted[:, :-1] = np.clip(shifted[:, :-1] * w, 0, 1)
            res = {
                "ensemble": {"test": evaluate([t.labels for t in test], decode2(ens["test"], meta["test"], len(test), "prod", ta, tm)),
                             "who_when": evaluate([t.labels for t in ww], decode2(ens["who_when"], meta["who_when"], len(ww), "prod", ta, tm))},
                "shift": {"test": None,
                          "who_when": evaluate([t.labels for t in ww], decode2(shifted, meta["who_when"], len(ww), "prod", ta, tm))},
            }
            rows["ensemble"].append(res["ensemble"])
            rows["shift"].append(res["shift"])
            print(f"[{variant}] seed {s} ens-size {len(encs)} ta {ta} tm {tm} | ensemble test {line(res['ensemble']['test'])} | WW {line(res['ensemble']['who_when'])} | WW +shift {line(res['shift']['who_when'])}", flush=True)
            for kind in ("ensemble", "shift"):
                for split in ("test", "who_when"):
                    if res[kind][split] is None:
                        continue
                    log_run(phase="10", model=f"ww_focus_{variant}_{kind}", seed=s, split=f"{split}_grouped_{args.tag}", n_params=0,
                            config={"variant": variant, "kind": kind, "encoders": encs, "t_a": ta, "t_m": float(tm)},
                            metrics={split: {l: res[kind][split][l] for l in LV}}, notes="ww_focus: ensemble over encoders and label-shift correction; no Who&When label used")
        print(f"\n=== {args.tag} {variant}: mean over seeds, ensemble of {len(encs)} encoders (micro/macro F1, percent) ===")
        for kind in ("ensemble", "shift"):
            for split in ("test", "who_when"):
                rs = [r[split] for r in rows[kind] if r[split] is not None]
                if rs:
                    print(f"{variant:7s} {kind:9s} {split:9s} " + " | ".join(
                        f"{l[:4]} {np.mean([100*r[l]['micro_f1'] for r in rs]):.2f}+-{np.std([100*r[l]['micro_f1'] for r in rs]):.2f}/{np.mean([100*r[l]['macro_f1'] for r in rs]):.2f}" for l in LV))


if __name__ == "__main__":
    main()
