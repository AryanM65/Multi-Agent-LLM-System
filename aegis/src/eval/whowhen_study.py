"""Who&When study: which changes help the out-of-domain benchmark, WITHOUT using any Who&When label for training or tuning.

Models: flat MLP trained on the grouped train split (early stopping and thresholds on grouped val), as in final_test.py.
Variants (all zero-shot on Who&When):
  base       - current features (text + framework one-hot + agent-name one-hot + counts), threshold decoding
  nameemb    - namefree plus a 384-d embedding of the agent-name TEXT (frozen MiniLM), so role words transfer to unseen names
  namefree   - the framework and agent-name one-hot columns are zeroed in train and everywhere else (names in Who&When are
               free-form, so those columns carry no usable signal there and only add shift)
Decoders:
  thr        - thresholds t_a, t_m tuned on AEGIS grouped val (as before)
  top1       - exactly one pair per trajectory (argmax of p_agent * p_mode). This uses a STATED PRIOR about the benchmark, not its
               labels: Who&When labels exactly one responsible agent per record (184 of 184). Reported separately from `thr`.
Also scores the same models on the AEGIS test set (thresholds decoder) to show what name-free costs in-domain.

Run: python -m aegis.src.eval.whowhen_study [--seeds 5] [--ft-dir PATH]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ..baselines.flat_mlp import decode, train_one, tune
from ..embed import get_embeddings
from ..features import FRAMEWORKS, EmbStore, agent_vocab, build
from ..loaders import load_grouped_splits, load_split
from ..metrics import evaluate
from ..schema import ERROR_MODES
from ..utils import log_run

LV = ("pair", "agent", "error")


def decode_top1(probs: np.ndarray, meta: list, n_traj: int) -> list:
    best = [(-1.0, None)] * n_traj
    for (ti, a), p in zip(meta, probs):
        pa, pm = p[-1], p[:-1]
        mi = int(np.argmax(pm))
        if pa * pm[mi] > best[ti][0]:
            best[ti] = (pa * pm[mi], (a, ERROR_MODES[mi]))
    return [{b[1]} if b[1] is not None else set() for b in best]


def line(res) -> str:
    return " ".join(f"{l[:4]} {100*res[l]['micro_f1']:.1f}/{100*res[l]['macro_f1']:.1f}" for l in LV)


def mask_cols(X: np.ndarray, emb_dim: int, vocab_len: int) -> np.ndarray:
    X = X.copy()
    X[:, emb_dim: emb_dim + len(FRAMEWORKS) + vocab_len + 1] = 0.0
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ft-dir", default="aegis/results/kaggle_logs/ft_full/ft_shards")
    ap.add_argument("--tag", default="ft_full")
    args = ap.parse_args()
    store = EmbStore(args.ft_dir) if args.ft_dir else EmbStore()
    sp = load_grouped_splits()
    test, ww = load_split("test"), load_split("who_when")
    vocab = agent_vocab(sp["train"])
    feats = {k: build(ts, store, vocab) for k, ts in (("train", sp["train"]), ("val", sp["val"]), ("test", test), ("who_when", ww))}
    trajs = {"train": sp["train"], "val": sp["val"], "test": test, "who_when": ww}
    E = store.dim
    # agent-name text embedded with the FROZEN MiniLM (names like "Verification_Expert" carry role meaning; "Assistant1" carries little)
    names = sorted({a for v in feats.values() for (_, a) in v[2]})
    nvec = dict(zip(names, get_embeddings([n.replace("_", " ").replace("(", " ").replace(")", " ") for n in names], verbose=False)))
    results: dict = {}
    for variant in ("base", "namefree", "nameemb"):
        if variant == "base":
            X = {k: v[0] for k, v in feats.items()}
        else:
            X = {k: mask_cols(v[0], E, len(vocab)) for k, v in feats.items()}
        if variant == "nameemb":
            X = {k: np.concatenate([X[k], np.stack([nvec[a] for (_, a) in feats[k][2]])], axis=1) for k in X}
        for s in range(args.seeds):
            model, probs, epochs, n_params = train_one(X["train"], feats["train"][1], X["val"], feats["val"][1], s)
            _, ta, tm, _ = tune(probs, feats["val"][2], sp["val"])
            p = {}
            for tag in ("test", "who_when"):
                with torch.no_grad():
                    p[tag] = torch.sigmoid(model(torch.tensor((X[tag] - model.mu) / model.sd, dtype=torch.float32))).numpy()
            out = {
                ("test", "thr"): evaluate([t.labels for t in test], decode(p["test"], feats["test"][2], len(test), ta, tm)),
                ("who_when", "thr"): evaluate([t.labels for t in ww], decode(p["who_when"], feats["who_when"][2], len(ww), ta, tm)),
                ("who_when", "top1"): evaluate([t.labels for t in ww], decode_top1(p["who_when"], feats["who_when"][2], len(ww))),
            }
            for key, res in out.items():
                results.setdefault((variant,) + key, []).append(res)
                log_run(phase="7", model=f"flat_mlp_{variant}_{key[1]}", seed=s, split=f"{key[0]}_grouped_{args.tag}", n_params=n_params,
                        config={"variant": variant, "decoder": key[1], "t_a": ta, "t_m": tm, "epochs_run": epochs, "ft_dir": args.ft_dir},
                        metrics={key[0]: {l: res[l] for l in LV}},
                        notes="Who&When study: zero-shot, no Who&When label used for training or tuning; top1 decoder uses the stated one-faulty-agent prior")
            print(f"[{variant}] seed {s} ta {ta} tm {tm} | WW thr {line(out[('who_when','thr')])} | WW top1 {line(out[('who_when','top1')])} | test thr {line(out[('test','thr')])}", flush=True)
    print("\n=== mean over seeds (micro/macro F1, percent) ===")
    for key, rs in results.items():
        parts = [f"{l[:4]} {np.mean([100*r[l]['micro_f1'] for r in rs]):.2f}+-{np.std([100*r[l]['micro_f1'] for r in rs]):.2f}/{np.mean([100*r[l]['macro_f1'] for r in rs]):.2f}" for l in LV]
        print(f"{key[0]:9s} {key[1]:8s} {key[2]:5s} " + " | ".join(parts))


if __name__ == "__main__":
    main()
