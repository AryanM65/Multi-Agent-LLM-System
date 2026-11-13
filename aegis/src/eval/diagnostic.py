"""Diagnostic: where does pair F1 get lost, the agent half or the mode half?

Trains the best current flat model (name-embedding variant, clean fine-tuned vectors) and scores the grouped test set under four conditions,
using ORACLE information only to measure ceilings (never as a result):
  model         - ordinary decoding (thresholds tuned on grouped val)
  oracle_agents - agent probability replaced by the true "is faulty" indicator; modes from the model
  oracle_modes  - mode probabilities masked to the modes that truly occur in the trajectory; agents from the model
  both          - both oracles (sanity: must be 100)
Also: of the faulty agents the model finds, how often is the exact mode set right?

Run: python -m aegis.src.eval.diagnostic --ft-dir aegis/results/kaggle_logs/ft_es_s0/ft_shards_es_s0 [--seeds 3]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ..baselines.flat_mlp import decode, train_one, tune
from ..embed import get_embeddings
from ..features import EmbStore, agent_vocab, build
from ..loaders import load_grouped_splits, load_split
from ..metrics import evaluate
from ..schema import ERROR_MODES
from .whowhen_study import mask_cols

LV = ("pair", "agent", "error")


def oracle_probs(probs: np.ndarray, meta: list, trajs: list, agents: bool, modes: bool) -> np.ndarray:
    p = probs.copy()
    for r, (ti, a) in enumerate(meta):
        truth = trajs[ti].labels
        if agents:
            p[r, -1] = 1.0 if any(x == a for (x, _) in truth) else 0.0
        if modes:
            true_modes = {m for (_, m) in truth}
            p[r, :-1] = p[r, :-1] * np.array([1.0 if m in true_modes else 0.0 for m in ERROR_MODES])
    return p


def fmt(res) -> str:
    return " ".join(f"{l[:4]} {100*res[l]['micro_f1']:.1f}/{100*res[l]['macro_f1']:.1f}" for l in LV)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-dir", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    store = EmbStore(args.ft_dir)
    sp = load_grouped_splits()
    test = load_split("test")
    vocab = agent_vocab(sp["train"])
    feats = {k: build(ts, store, vocab) for k, ts in (("train", sp["train"]), ("val", sp["val"]), ("test", test))}
    E = store.dim
    names = sorted({a for v in feats.values() for (_, a) in v[2]})
    nvec = dict(zip(names, get_embeddings([n.replace("_", " ").replace("(", " ").replace(")", " ") for n in names], verbose=False)))
    X = {k: np.concatenate([mask_cols(v[0], E, len(vocab)), np.stack([nvec[a] for (_, a) in v[2]])], axis=1) for k, v in feats.items()}
    truth = [t.labels for t in test]
    allres: dict = {}
    for s in range(args.seeds):
        model, probs, _, _ = train_one(X["train"], feats["train"][1], X["val"], feats["val"][1], s)
        _, ta, tm, _ = tune(probs, feats["val"][2], sp["val"])
        with torch.no_grad():
            p = torch.sigmoid(model(torch.tensor((X["test"] - model.mu) / model.sd, dtype=torch.float32))).numpy()
        meta = feats["test"][2]
        for name, (oa, om) in {"model": (False, False), "oracle_agents": (True, False), "oracle_modes": (False, True), "both": (True, True)}.items():
            q = oracle_probs(p, meta, test, oa, om)
            res = evaluate(truth, decode(q, meta, len(test), ta, tm))
            allres.setdefault(name, []).append(res)
        # exact mode-set agreement on correctly found faulty agents
        pred = decode(p, meta, len(test), ta, tm)
        hit = tot = 0
        for tr, pr in zip(truth, pred):
            tm_by = {}
            for (a, m) in tr:
                tm_by.setdefault(a, set()).add(m)
            pm_by = {}
            for (a, m) in pr:
                pm_by.setdefault(a, set()).add(m)
            for a, ms in tm_by.items():
                if a in pm_by:
                    tot += 1
                    hit += ms == pm_by[a]
        allres.setdefault("exact_modeset_given_agent_found", []).append(hit / max(tot, 1))
        print(f"seed {s} ta {ta} tm {tm} exact mode-set on found agents {hit}/{tot}", flush=True)
    print("\n=== grouped test, mean over seeds (micro/macro F1, percent) ===")
    for name in ("model", "oracle_agents", "oracle_modes", "both"):
        rs = allres[name]
        print(f"{name:14s} " + " | ".join(
            f"{l[:4]} {np.mean([100*r[l]['micro_f1'] for r in rs]):.2f}/{np.mean([100*r[l]['macro_f1'] for r in rs]):.2f}" for l in LV))
    print(f"exact mode-set right on found faulty agents: {100*np.mean(allres['exact_modeset_given_agent_found']):.1f}%")


if __name__ == "__main__":
    main()
