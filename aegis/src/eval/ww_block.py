"""Block-dropout: one model that keeps the AEGIS peer-contrast gain AND can run without peer/name features for Who&When.

Features per agent row: base (mean vector, scalars, framework/name one-hots zeroed) | name embedding block (E) | peer block (peer_block + relational_features).
Training set = 4 copies of the AEGIS grouped-train rows: all blocks on, name block off, peer block off, both off. So the same network is trained to work
whichever blocks are present.

Evaluation (no Who&When label used anywhere; thresholds always tuned on AEGIS grouped validation with THE SAME block mask as the evaluated split):
  AEGIS test      : all blocks on (the in-domain operating point) -> must stay near the peernd AEGIS level
  Who&When        : each of 4 masks (names on/off x peer on/off), optionally ensembled over encoders, optionally + label-shift (EM prior, as in ww_focus)
NOTE: choosing a Who&When mask after looking at Who&When is a post-hoc choice; report it as such (or justify it by leave-one-framework-out first).

Run: python -m aegis.src.eval.ww_block --encoders es_s0,es_s1,... --seeds 3 --tag wb256
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
from ..utils import log_run
from .improve_cpu import decode2, peer_block, tune2
from .whowhen_study import mask_cols
from .ww_focus import LV, enc_dir, estimate_prior, line

MASKS = {"names+peer": (1, 1), "names": (1, 0), "peer": (0, 1), "neither": (0, 0)}   # (name block on, peer block on)


def apply_mask(X: np.ndarray, nb0: int, E: int, pb0: int, mask: tuple) -> np.ndarray:
    X = X.copy()
    if not mask[0]:
        X[:, nb0:nb0 + E] = 0.0
    if not mask[1]:
        X[:, pb0:] = 0.0
    return X


def decode_top1(probs, meta, n_traj, tm, single_mode: bool):
    """Who&When has exactly one faulty agent per record (benchmark structure, not a label): keep only the agent with the highest
    faulty probability per record; its modes are those with p >= tm (at least the best one), or only the best mode if single_mode."""
    from ..schema import ERROR_MODES
    best = [(-1.0, None)] * n_traj
    for (ti, a), p in zip(meta, probs):
        if p[-1] > best[ti][0]:
            best[ti] = (p[-1], (a, p[:-1]))
    pred = [set() for _ in range(n_traj)]
    for ti, (_, ap) in enumerate(best):
        if ap is None:
            continue
        a, pm = ap
        ms = [int(np.argmax(pm))] if single_mode else (list(np.where(pm >= tm)[0]) or [int(np.argmax(pm))])
        pred[ti] = {(a, ERROR_MODES[m]) for m in ms}
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="wb")
    ap.add_argument("--tta", action="store_true", help="transductive self-training on unlabeled Who&When (separate 'tta_*' rows; uses NO Who&When label)")
    ap.add_argument("--top1", action="store_true", help="also report a one-faulty-agent-per-record decode on Who&When (benchmark-structure prior)")
    args = ap.parse_args()
    encs = args.encoders.split(",")
    sp = load_grouped_splits()
    test, ww = load_split("test"), load_split("who_when")
    trajs = {"train": sp["train"], "val": sp["val"], "test": test, "who_when": ww}
    vocab = agent_vocab(sp["train"])
    probs = {}     # (seed, mask, split) -> list over encoders
    meta, names_cache, src_prior = None, None, None
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
        pb0 = base["train"].shape[1]
        X = {k: np.hstack([base[k], peer_block(trajs[k], store), relational_features(trajs[k], store)]) for k in trajs}
        if src_prior is None:
            ytr = feats["train"][1]
            src_prior = ytr.sum(0) / ytr.sum()
        Xtr, Ytr = X["train"], feats["train"][1]
        copies = [apply_mask(Xtr, nb0, E, pb0, m) for m in MASKS.values()]
        Xaug, Yaug = np.vstack(copies), np.vstack([Ytr] * len(copies))
        for s in range(args.seeds):
            # validation for early stopping: full-block validation set stacked with name-off and peer-off versions
            Xv = np.vstack([apply_mask(X["val"], nb0, E, pb0, m) for m in MASKS.values()])
            Yv = np.vstack([feats["val"][1]] * len(MASKS))
            model, _, _, _ = train_one(Xaug, Yaug, Xv, Yv, s)
            for mname, m in MASKS.items():
                for k in ("val", "test", "who_when"):
                    xm = apply_mask(X[k], nb0, E, pb0, m)
                    with torch.no_grad():
                        p = torch.sigmoid(model(torch.tensor((xm - model.mu) / model.sd, dtype=torch.float32))).numpy()
                    probs.setdefault((s, mname, k), []).append(p)
            if args.tta:
                # self-training: confident own predictions on unlabeled Who&When become extra training rows (agent with highest p, its modes >= 0.4)
                xw = X["who_when"]
                with torch.no_grad():
                    pw = torch.sigmoid(model(torch.tensor((xw - model.mu) / model.sd, dtype=torch.float32))).numpy()
                best = {}
                for ri, ((ti, a), pr) in enumerate(zip(meta["who_when"], pw)):
                    if ti not in best or pr[-1] > pw[best[ti]][-1]:
                        best[ti] = ri
                Yp = np.zeros((len(xw), pw.shape[1] - 1), dtype=np.float32)
                keep = []
                for ti, ri in best.items():
                    if pw[ri][-1] >= 0.5:
                        ms = np.where(pw[ri][:-1] >= 0.4)[0]
                        if len(ms) == 0:
                            ms = [int(np.argmax(pw[ri][:-1]))]
                        Yp[ri, ms] = 1.0
                        keep += [rj for rj, (tj, _) in enumerate(meta["who_when"]) if tj == ti]
                keep = sorted(set(keep))
                if keep:
                    Xp = np.vstack([apply_mask(xw[keep], nb0, E, pb0, m) for m in MASKS.values()])
                    Yq = np.vstack([Yp[keep]] * len(MASKS))
                    model2, _, _, _ = train_one(np.vstack([Xaug, Xp]), np.vstack([Yaug, Yq]), Xv, Yv, s)
                else:
                    model2 = model
                for mname, m in MASKS.items():
                    for k in ("val", "test", "who_when"):
                        xm = apply_mask(X[k], nb0, E, pb0, m)
                        with torch.no_grad():
                            p = torch.sigmoid(model2(torch.tensor((xm - model2.mu) / model2.sd, dtype=torch.float32))).numpy()
                        probs.setdefault((s, "tta_" + mname, k), []).append(p)
        print(f"[{enc}] trained", flush=True)
    summary = {}
    for mname in list(MASKS) + (['tta_' + m for m in MASKS] if args.tta else []):
        for shift in (False, True):
            rows = []
            for s in range(args.seeds):
                ens = {k: np.mean(probs[(s, mname, k)], axis=0) for k in ("val", "test", "who_when")}
                _, ta, tm = tune2("prod", ens["val"], meta["val"], sp["val"])
                ww_p = ens["who_when"]
                if shift:
                    pri = estimate_prior(ww_p, meta["who_when"], len(ww), src_prior)
                    w = pri / src_prior
                    w = w / w.mean()
                    ww_p = ww_p.copy()
                    ww_p[:, :-1] = np.clip(ww_p[:, :-1] * w, 0, 1)
                r = {"test": evaluate([t.labels for t in test], decode2(ens["test"], meta["test"], len(test), "prod", ta, tm)),
                     "who_when": evaluate([t.labels for t in ww], decode2(ww_p, meta["who_when"], len(ww), "prod", ta, tm))}
                if args.top1:
                    r["top1"] = evaluate([t.labels for t in ww], decode_top1(ww_p, meta["who_when"], len(ww), tm, False))
                    r["top1m"] = evaluate([t.labels for t in ww], decode_top1(ww_p, meta["who_when"], len(ww), tm, True))
                rows.append(r)
                log_run(phase="10", model=f"ww_block_{mname}{'_shift' if shift else ''}", seed=s, split=f"who_when_grouped_{args.tag}", n_params=0,
                        config={"mask": mname, "shift": shift, "encoders": encs, "t_a": ta, "t_m": float(tm)},
                        metrics={"who_when": {l: r["who_when"][l] for l in LV}, "test": {l: r["test"][l] for l in LV}},
                        notes="ww_block: block-dropout model; thresholds from AEGIS val under the same mask; no Who&When label used")
            summary[(mname, shift)] = rows
    print(f"\n=== {args.tag}: block-dropout model, ensemble of {len(encs)} encoders, mean+-std over {args.seeds} seeds (micro F1 percent; macro after slash) ===")
    for (mname, shift), rows in summary.items():
        for split in ("test", "who_when"):
            if split == "test" and shift:
                continue
            print(f"{mname:11s} {'+shift' if shift else '      '} {split:9s} " + " | ".join(
                f"{l[:4]} {np.mean([100*r[split][l]['micro_f1'] for r in rows]):.2f}+-{np.std([100*r[split][l]['micro_f1'] for r in rows]):.2f}/{np.mean([100*r[split][l]['macro_f1'] for r in rows]):.2f}" for l in LV))
        if args.top1:
            for k in ("top1", "top1m"):
                if k in rows[0]:
                    print(f"{mname:11s} {'+shift' if shift else '      '} {k:9s} " + " | ".join(
                        f"{l[:4]} {np.mean([100*r[k][l]['micro_f1'] for r in rows]):.2f}+-{np.std([100*r[k][l]['micro_f1'] for r in rows]):.2f}/{np.mean([100*r[k][l]['macro_f1'] for r in rows]):.2f}" for l in LV))


if __name__ == "__main__":
    main()
