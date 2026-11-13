"""CPU experiments on cached fine-tuned vectors: do small changes to features, training data or decoding improve the flat model?

Base = the current best setup (name-embedding flat MLP: framework/name one-hot zeroed + 384-d MiniLM embedding of the agent-name text).
Feature / training variants (all trained on grouped train, early stop and thresholds on grouped val ONLY):
  base      - as above
  peer      - base + peer-contrast: (agent mean - mean of the other agents' turns), (agent mean - query vector), + 4 cosine features
  peernd    - peer + namedrop together
  namedrop  - base, training set doubled with a copy whose name-embedding block is zeroed (the model must also work without the name)
Decoders applied to every variant (same trained model):
  prod      - current: pair (a,m) if p_agent >= ta and p_mode[m] >= tm  (two global thresholds)
  pm        - pair if p_mode[m] >= tm only (p_mode is already trained on pair targets, so the agent factor may double count)
  min       - pair if min(p_agent, p_mode[m]) >= t  (hard rule: a pair is never more likely than its agent or its mode)
  perclass  - prod with ta global and ONE threshold per mode (coordinate ascent on val)
Everything tuned on val; test and Who&When are scored with the val-tuned decoders for every variant and ALL rows are reported (no picking by test).

Run: python -m aegis.src.eval.improve_cpu --ft-dir <dir with minilm_shard_*.npz> --tag es_s0 [--seeds 3] [--variants base,peer,namedrop]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ..baselines.flat_mlp import GRID, train_one
from ..baselines.flat_probe import _unit, relational_features
from ..embed import get_embeddings
from ..features import EmbStore, agent_vocab, build
from ..loaders import load_grouped_splits, load_split
from ..metrics import evaluate
from ..schema import ERROR_MODES
from ..utils import log_run
from .whowhen_study import mask_cols

LV = ("pair", "agent", "error")
NM = len(ERROR_MODES)


def decode2(probs, meta, n_traj, kind: str, ta: float, tm) -> list:
    """kind in prod|pm|min|perclass; tm is a float, or a length-14 vector for perclass."""
    tmv = np.full(NM, tm) if np.isscalar(tm) else np.asarray(tm)
    pred = [set() for _ in range(n_traj)]
    best = [(-1.0, None)] * n_traj
    for (ti, a), p in zip(meta, probs):
        pa, pm = p[-1], p[:-1]
        for mi in range(NM):
            if kind == "min":
                s = min(pa, pm[mi]); ok = s >= tmv[mi]
            elif kind == "pm":
                s = pm[mi]; ok = pm[mi] >= tmv[mi]
            else:  # prod / perclass
                s = pa * pm[mi]; ok = pa >= ta and pm[mi] >= tmv[mi]
            if s > best[ti][0]:
                best[ti] = (s, (a, ERROR_MODES[mi]))
            if ok:
                pred[ti].add((a, ERROR_MODES[mi]))
    for ti in range(n_traj):
        if not pred[ti] and best[ti][1] is not None:
            pred[ti].add(best[ti][1])
    return pred


def objective(res) -> float:
    return sum(res[l]["micro_f1"] for l in LV) / 3


def tune2(kind: str, probs, meta, trajs):
    truth = [t.labels for t in trajs]
    best = (-1, None, None)
    if kind == "prod":
        for ta in GRID:
            for tm in GRID:
                s = objective(evaluate(truth, decode2(probs, meta, len(trajs), "prod", ta, tm)))
                if s > best[0]:
                    best = (s, ta, tm)
    elif kind in ("pm", "min"):
        for t in GRID:
            s = objective(evaluate(truth, decode2(probs, meta, len(trajs), kind, 0.0, t)))
            if s > best[0]:
                best = (s, 0.0, t)
    else:  # perclass: start from the prod optimum, then coordinate ascent per mode
        _, ta, tm0 = tune2("prod", probs, meta, trajs)
        tmv = np.full(NM, tm0)
        cur = best_s = objective(evaluate(truth, decode2(probs, meta, len(trajs), "perclass", ta, tmv)))
        for _round in range(2):
            for mi in range(NM):
                for g in GRID:
                    t2 = tmv.copy(); t2[mi] = g
                    s = objective(evaluate(truth, decode2(probs, meta, len(trajs), "perclass", ta, t2)))
                    if s > best_s + 1e-9:
                        best_s, tmv = s, t2
        best = (best_s, ta, tmv)
    return best


def peer_block(trajs, store: EmbStore) -> np.ndarray:
    """(agent mean - mean of other agents' turns) | (agent mean - query vec) | 4 cosines; row order = features.build."""
    qs = sorted({t.query for t in trajs})
    qvec = dict(zip(qs, get_embeddings(qs, verbose=False)))
    rows = []
    for t in trajs:
        by: dict = {}
        for u in t.turns:
            by.setdefault(u.agent, []).append(store.lookup(u.text))
        q = qvec[t.query]
        for a, vs in by.items():
            m = np.mean(vs, axis=0)
            others = [v for b, w in by.items() if b != a for v in w]
            o = np.mean(others, axis=0) if others else m
            rows.append(np.concatenate([m - o, m - q]))
    return np.stack(rows).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", default="base,peer,namedrop")
    args = ap.parse_args()
    store = EmbStore(args.ft_dir)
    sp = load_grouped_splits()
    test, ww = load_split("test"), load_split("who_when")
    trajs = {"train": sp["train"], "val": sp["val"], "test": test, "who_when": ww}
    vocab = agent_vocab(sp["train"])
    feats = {k: build(ts, store, vocab) for k, ts in trajs.items()}
    E = store.dim
    names = sorted({a for v in feats.values() for (_, a) in v[2]})
    nvec = dict(zip(names, get_embeddings([n.replace("_", " ").replace("(", " ").replace(")", " ") for n in names], verbose=False)))
    base = {k: np.concatenate([mask_cols(v[0], E, len(vocab)), np.stack([nvec[a] for (_, a) in v[2]])], axis=1) for k, v in feats.items()}
    nb0 = base["train"].shape[1] - E   # start column of the name-embedding block
    allres: dict = {}
    for variant in args.variants.split(","):
        if variant in ("peer", "peernd"):
            pb = {k: np.hstack([peer_block(trajs[k], store), relational_features(trajs[k], store)]) for k in trajs}
            X = {k: np.hstack([base[k], pb[k]]) for k in trajs}
        else:
            X = base
        Xtr, Ytr = X["train"], feats["train"][1]
        if variant in ("namedrop", "peernd"):
            drop = Xtr.copy()
            drop[:, nb0:nb0 + E] = 0.0   # zero the name-embedding block (it sits at the end of the base features)
            Xtr, Ytr = np.vstack([Xtr, drop]), np.vstack([Ytr, Ytr])
        for s in range(args.seeds):
            model, probs, epochs, n_params = train_one(Xtr, Ytr, X["val"], feats["val"][1], s)
            p = {}
            for k in ("val", "test", "who_when"):
                with torch.no_grad():
                    p[k] = torch.sigmoid(model(torch.tensor((X[k] - model.mu) / model.sd, dtype=torch.float32))).numpy()
            for kind in ("prod", "pm", "min", "perclass"):
                _, ta, tm = tune2(kind, p["val"], feats["val"][2], sp["val"])
                row = {}
                for split, ts in (("val", sp["val"]), ("test", test), ("who_when", ww)):
                    row[split] = evaluate([t.labels for t in ts], decode2(p[split], feats[split][2], len(ts), kind, ta, tm))
                    allres.setdefault((variant, kind, split), []).append(row[split])
                    log_run(phase="9", model=f"flat_mlp_{variant}_{kind}", seed=s, split=f"{split}_grouped_{args.tag}", n_params=n_params,
                            config={"variant": variant, "decoder": kind, "ft_dir": args.ft_dir, "epochs_run": epochs},
                            metrics={split: {l: row[split][l] for l in LV}}, notes="improve_cpu: val-tuned decoders; all variants reported")
                print(f"[{variant}/{kind}] seed {s} val pair {100*row['val']['pair']['micro_f1']:.1f} | test " +
                      " ".join(f"{l[:4]} {100*row['test'][l]['micro_f1']:.1f}" for l in LV) + " | WW " +
                      " ".join(f"{l[:4]} {100*row['who_when'][l]['micro_f1']:.1f}" for l in LV), flush=True)
    print(f"\n=== {args.tag}: mean over seeds (micro F1, percent) ===")
    for (variant, kind, split), rs in allres.items():
        print(f"{variant:9s} {kind:9s} {split:9s} " + " | ".join(f"{l[:4]} {np.mean([100*r[l]['micro_f1'] for r in rs]):.2f}+-{np.std([100*r[l]['micro_f1'] for r in rs]):.2f}" for l in LV))


if __name__ == "__main__":
    main()
