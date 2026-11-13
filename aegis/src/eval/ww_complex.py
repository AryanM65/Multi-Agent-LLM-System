"""CPU tests of two 'complex' ideas on saved encoder vectors (no Who&When label used anywhere; thresholds tuned on AEGIS grouped validation):

  flat  : control = name-embedding MLP trained with name-dropout (same as ww_focus nd)
  hier  : hierarchical mode heads. Extra outputs for the three error groups (FM-1.x, FM-2.x, FM-3.x); the final probability of a mode is
          P(mode) * P(group of that mode) (soft consistency between levels). Mode identification is the measured bottleneck.
  knn   : non-parametric. Each agent row is matched to its k nearest AEGIS-train agent rows (cosine on the mean turn vector + peer-contrast
          block); mode and agent probabilities are similarity-weighted neighbour label averages.
  knnmix: 50/50 average of the knn and flat probabilities.

Run: python -m aegis.src.eval.ww_complex --encoders es_s0,es_s1 --seeds 3 --methods flat,hier,knn,knnmix --tag cx1
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from ..baselines.flat_mlp import make_targets
from ..embed import get_embeddings
from ..features import EmbStore, agent_vocab, build
from ..loaders import load_grouped_splits, load_split
from ..metrics import evaluate
from ..schema import ERROR_MODES
from ..utils import log_run, set_seed
from .improve_cpu import decode2, peer_block, tune2
from .whowhen_study import mask_cols
from .ww_focus import LV, enc_dir

NM = len(ERROR_MODES)
GROUP = np.array([int(m.split("-")[1].split(".")[0]) - 1 for m in ERROR_MODES])   # FM-2.3 -> 1


class Net(nn.Module):
    def __init__(self, d_in, n_out, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def hier_targets(Y):
    base = make_targets(Y)                                  # 14 modes + agent
    grp = np.stack([(Y[:, GROUP == g].sum(1) > 0) for g in range(3)], axis=1).astype(np.float32)
    return np.concatenate([base, grp], axis=1)               # 14 + 1 + 3


def train(Xtr, Ytr, Xva, Yva, seed, hier, epochs=60, patience=8):
    set_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-3, 1.0, sd)
    f = lambda X: torch.tensor((X - mu) / sd, dtype=torch.float32)
    tg = (lambda Y: torch.tensor(hier_targets(Y))) if hier else (lambda Y: torch.tensor(make_targets(Y)))
    xtr, xva, ytr, yva = f(Xtr), f(Xva), tg(Ytr), tg(Yva)
    model = Net(xtr.shape[1], ytr.shape[1])
    model.mu, model.sd = mu, sd
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    best, state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            lossf(model(xtr[b]), ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(xva), yva).item()
        if vl < best - 1e-4:
            best, state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model, X, hier):
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor((X - model.mu) / model.sd, dtype=torch.float32))).numpy()
    if not hier:
        return p
    modes, agent, grp = p[:, :NM], p[:, NM:NM + 1], p[:, NM + 1:]
    return np.concatenate([modes * grp[:, GROUP], agent], axis=1)


def knn_probs(Ftr, Ytr, F, k=25):
    n = lambda A: A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-9, None)
    S = n(F) @ n(Ftr).T
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    w = np.take_along_axis(S, idx, axis=1)
    w = np.exp(w * 10.0)
    w = w / w.sum(1, keepdims=True)
    Yt = make_targets(Ytr)
    return np.einsum("rk,rkc->rc", w, Yt[idx])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--methods", default="flat,hier,knn,knnmix")
    ap.add_argument("--tag", default="cx")
    args = ap.parse_args()
    methods = args.methods.split(",")
    sp = load_grouped_splits()
    test, ww = load_split("test"), load_split("who_when")
    trajs = {"train": sp["train"], "val": sp["val"], "test": test, "who_when": ww}
    vocab = agent_vocab(sp["train"])
    P = {}       # (method, seed, split) -> list over encoders
    meta, nc = None, None
    for enc in args.encoders.split(","):
        store = EmbStore(enc_dir(enc))
        feats = {k: build(ts, store, vocab) for k, ts in trajs.items()}
        meta = {k: v[2] for k, v in feats.items()}
        E = store.dim
        if nc is None:
            names = sorted({a for v in feats.values() for (_, a) in v[2]})
            nc = dict(zip(names, get_embeddings([n.replace("_", " ").replace("(", " ").replace(")", " ") for n in names], verbose=False)))
        base = {k: np.concatenate([mask_cols(v[0], E, len(vocab)), np.stack([nc[a] for (_, a) in v[2]])], axis=1) for k, v in feats.items()}
        nb0 = base["train"].shape[1] - E
        pk = {k: np.hstack([feats[k][0][:, :E], peer_block(trajs[k], store)]) for k in trajs}     # knn space: mean vector + peer block
        Ytr = feats["train"][1]
        drop = base["train"].copy()
        drop[:, nb0:nb0 + E] = 0.0
        Xaug, Yaug = np.vstack([base["train"], drop]), np.vstack([Ytr, Ytr])
        kn = {k: knn_probs(pk["train"], Ytr, pk[k]) for k in ("val", "test", "who_when")} if ("knn" in methods or "knnmix" in methods) else None
        for s in range(args.seeds):
            fl = None
            for m in ("flat", "hier"):
                if m in methods or (m == "flat" and "knnmix" in methods):
                    model = train(Xaug, Yaug, base["val"], feats["val"][1], s, hier=(m == "hier"))
                    pr = {k: predict(model, base[k], m == "hier") for k in ("val", "test", "who_when")}
                    for k in pr:
                        P.setdefault((m, s, k), []).append(pr[k])
                    if m == "flat":
                        fl = pr
            if "knn" in methods:
                for k in kn:
                    P.setdefault(("knn", s, k), []).append(kn[k])
            if "knnmix" in methods:
                for k in kn:
                    P.setdefault(("knnmix", s, k), []).append(0.5 * kn[k] + 0.5 * fl[k])
        print(f"[{enc}] done", flush=True)
    out = {}
    for m in methods:
        rows = []
        for s in range(args.seeds):
            ens = {k: np.mean(P[(m, s, k)], axis=0) for k in ("val", "test", "who_when")}
            _, ta, tm = tune2("prod", ens["val"], meta["val"], sp["val"])
            r = {"test": evaluate([t.labels for t in test], decode2(ens["test"], meta["test"], len(test), "prod", ta, tm)),
                 "who_when": evaluate([t.labels for t in ww], decode2(ens["who_when"], meta["who_when"], len(ww), "prod", ta, tm))}
            rows.append(r)
            for split in ("test", "who_when"):
                log_run(phase="10", model=f"ww_complex_{m}", seed=s, split=f"{split}_grouped_{args.tag}", n_params=0,
                        config={"method": m, "encoders": args.encoders, "t_a": ta, "t_m": float(tm)}, metrics={split: {l: r[split][l] for l in LV}},
                        notes="ww_complex: hierarchical heads / kNN; no Who&W label used")
        out[m] = rows
    print(f"\n=== {args.tag}: complex ideas, encoders {args.encoders}, mean+-std over {args.seeds} seeds (micro F1; macro after slash) ===")
    for m, rows in out.items():
        for split in ("test", "who_when"):
            print(f"{m:7s} {split:9s} " + " | ".join(
                f"{l[:4]} {np.mean([100*r[split][l]['micro_f1'] for r in rows]):.2f}+-{np.std([100*r[split][l]['micro_f1'] for r in rows]):.2f}/{np.mean([100*r[split][l]['macro_f1'] for r in rows]):.2f}" for l in LV))


if __name__ == "__main__":
    main()
