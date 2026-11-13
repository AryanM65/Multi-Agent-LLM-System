"""Attention pooling over an agent's turns (CPU, on saved fine-tuned vectors), with name-dropout training.

Control ('mean'): the usual name-embedding flat MLP (mean of turn vectors) trained with name-dropout.
'attn'   : the agent's turn vectors are combined with learned gated attention weights (softmax over turns) and the pooled vector is
           concatenated to the same flat features (mean block kept, so AEGIS in-domain information is not lost).
'attn_only': pooled vector replaces the mean block.
Same decoders as improve_cpu (prod: two global thresholds tuned on AEGIS grouped val). No Who&When label used anywhere.

Run: python -m aegis.src.eval.ww_attn --ft-dir aegis/results/kaggle_logs/es_s0/ft_shards_es_s0 --tag es_s0 --seeds 3 --variants mean,attn
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from ..baselines.flat_mlp import make_targets
from ..embed import get_embeddings, text_key
from ..features import EmbStore, agent_vocab, build
from ..loaders import load_grouped_splits, load_split
from ..metrics import evaluate
from ..schema import ERROR_MODES
from ..utils import log_run, set_seed
from .improve_cpu import decode2, tune2
from .whowhen_study import mask_cols

LV = ("pair", "agent", "error")
MAXT = 24


def turn_index(trajs, store: EmbStore):
    """(rows, MAXT) indices into store.vecs (pad = -1); row order identical to features.build."""
    rows = []
    for t in trajs:
        by: dict = {}
        for u in t.turns:
            by.setdefault(u.agent, []).append(store.index[text_key(u.text)])
        for a, idx in by.items():
            if len(idx) > MAXT:   # keep evenly spread turns
                idx = [idx[int(i)] for i in np.linspace(0, len(idx) - 1, MAXT)]
            rows.append(idx + [-1] * (MAXT - len(idx)))
    return np.array(rows, dtype=np.int64)


class AttnNet(nn.Module):
    def __init__(self, d_flat: int, E: int, vecs: torch.Tensor, mode: str, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.mode, self.vecs = mode, vecs
        att = mode != "mean"
        self.V = nn.Linear(E, 128) if att else None
        self.U = nn.Linear(E, 128) if att else None
        self.w = nn.Linear(128, 1) if att else None
        d_in = d_flat + (E if att else 0)
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, len(ERROR_MODES) + 1))

    def forward(self, x, tidx):
        if self.mode == "mean":
            return self.net(x)
        mask = tidx >= 0
        t = self.vecs[tidx.clamp(min=0)]                       # B x T x E
        s = self.w(torch.tanh(self.V(t)) * torch.sigmoid(self.U(t))).squeeze(-1)
        s = s.masked_fill(~mask, -1e9)
        a = torch.softmax(s, dim=1).unsqueeze(-1)
        pooled = (a * t).sum(1)
        return self.net(torch.cat([x, pooled], dim=1))


def train_attn(mode, Xtr, Itr, Ytr, Xva, Iva, Yva, vecs, E, seed, epochs=60, patience=8):
    set_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-3, 1.0, sd)
    f = lambda X: torch.tensor((X - mu) / sd, dtype=torch.float32)
    xtr, xva = f(Xtr), f(Xva)
    itr, iva = torch.tensor(Itr), torch.tensor(Iva)
    ytr, yva = torch.tensor(make_targets(Ytr)), torch.tensor(make_targets(Yva))
    model = AttnNet(xtr.shape[1], E, vecs, mode)
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
            lossf(model(xtr[b], itr[b]), ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(xva, iva), yva).item()
        if vl < best - 1e-4:
            best, state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model, X, I):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor((X - model.mu) / model.sd, dtype=torch.float32), torch.tensor(I))).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", default="mean,attn")
    args = ap.parse_args()
    store = EmbStore(args.ft_dir)
    E = store.dim
    vecs = torch.tensor(store.vecs)
    sp = load_grouped_splits()
    test, ww = load_split("test"), load_split("who_when")
    trajs = {"train": sp["train"], "val": sp["val"], "test": test, "who_when": ww}
    vocab = agent_vocab(sp["train"])
    feats = {k: build(ts, store, vocab) for k, ts in trajs.items()}
    meta = {k: v[2] for k, v in feats.items()}
    names = sorted({a for v in feats.values() for (_, a) in v[2]})
    nc = dict(zip(names, get_embeddings([n.replace("_", " ").replace("(", " ").replace(")", " ") for n in names], verbose=False)))
    base = {k: np.concatenate([mask_cols(v[0], E, len(vocab)), np.stack([nc[a] for (_, a) in v[2]])], axis=1) for k, v in feats.items()}
    idx = {k: turn_index(ts, store) for k, ts in trajs.items()}
    nb0 = base["train"].shape[1] - E
    Xtr, Ytr, Itr = base["train"], feats["train"][1], idx["train"]
    drop = Xtr.copy()
    drop[:, nb0:nb0 + E] = 0.0
    Xtr2, Ytr2, Itr2 = np.vstack([Xtr, drop]), np.vstack([Ytr, Ytr]), np.vstack([Itr, Itr])
    out = {}
    for variant in args.variants.split(","):
        rows = []
        for s in range(args.seeds):
            m = train_attn(variant, Xtr2, Itr2, Ytr2, base["val"], idx["val"], feats["val"][1], vecs, E, s)
            pr = {k: predict(m, base[k], idx[k]) for k in ("val", "test", "who_when")}
            _, ta, tm = tune2("prod", pr["val"], meta["val"], sp["val"])
            res = {"test": evaluate([t.labels for t in test], decode2(pr["test"], meta["test"], len(test), "prod", ta, tm)),
                   "who_when": evaluate([t.labels for t in ww], decode2(pr["who_when"], meta["who_when"], len(ww), "prod", ta, tm))}
            rows.append(res)
            print(f"[{variant}] seed {s} | test " + " ".join(f"{l[:4]} {100*res['test'][l]['micro_f1']:.2f}" for l in LV)
                  + " | WW " + " ".join(f"{l[:4]} {100*res['who_when'][l]['micro_f1']:.2f}" for l in LV), flush=True)
            for split in ("test", "who_when"):
                log_run(phase="10", model=f"ww_attn_{variant}", seed=s, split=f"{split}_grouped_{args.tag}", n_params=0,
                        config={"variant": variant, "ta": ta, "tm": float(tm), "maxT": MAXT},
                        metrics={split: {l: res[split][l] for l in LV}}, notes="attention pooling over turns + name-dropout; no Who&When label used")
        out[variant] = rows
    print(f"\n=== {args.tag}: attention pooling vs mean (name-dropout training; micro F1 mean+-std over seeds; macro after slash) ===")
    for variant, rows in out.items():
        for split in ("test", "who_when"):
            print(f"{variant:9s} {split:9s} " + " | ".join(
                f"{l[:4]} {np.mean([100*r[split][l]['micro_f1'] for r in rows]):.2f}+-{np.std([100*r[split][l]['micro_f1'] for r in rows]):.2f}/{np.mean([100*r[split][l]['macro_f1'] for r in rows]):.2f}" for l in LV))


if __name__ == "__main__":
    main()
