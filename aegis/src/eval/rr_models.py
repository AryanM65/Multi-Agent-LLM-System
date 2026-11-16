"""Block-dropout model with an OPTIONAL retrieval-feature block, and the per-agent probability dump that the shortlist / judge pipeline needs.

This is ww_block.py (same features, same four block masks, same training and decoding) plus two things:
  --retr 1 : appends a retrieval block (rr_common.retrieval_block: what do the nearest AEGIS-train agents say, per error mode) to the peer group.
             Under block-dropout the peer group (peer-contrast + relational cosines + retrieval block) is switched off as one unit, so the mask
             definitions are unchanged. Train agents retrieve with the SAME-QUERY rows masked out (no self / sibling leakage). val, test and
             Who&When retrieve from AEGIS train only (Who&When is never in the index).
  dump     : the ensemble probabilities (mean over encoders, one array per head seed, mask and split) go to aegis/results/rr/dump_<tag>.npz.
             Gate 0 (rr_gate0.py), the candidate builder (rr_candidates.py) and the fusion step (rr_fuse.py) read this file.
Run --retr 0 and --retr 1 with the SAME --encoders/--seeds to compare "model alone" with "model + retriever features" (no LLM involved).

Protocol: thresholds tuned on AEGIS grouped validation under the same mask; Who&When labels are never used for training/tuning; every row is logged
(metrics_history.jsonl, model name rr_<tag>_<mask>[_shift]) and printed. Label-shift rows are transductive.

Run: python -m aegis.src.eval.rr_models --encoders es_s0,es_s1,... --seeds 3 --retr 1 --tag rr_retr
     (smoke test: --encoders es_s0 --seeds 1 --retr 1 --tag rr_smoke --limit 400)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ..baselines.flat_mlp import make_targets, train_one
from ..baselines.flat_probe import relational_features
from ..embed import get_embeddings
from ..features import EmbStore, agent_vocab, build
from ..loaders import load_grouped_splits, load_paper_splits, load_split
from ..metrics import evaluate
from ..utils import log_run
from .improve_cpu import decode2, peer_block, tune2
from .rr_common import LV, RR_DIR, apply_shift, norm_pairs, retrieval_block, save_dump
from .whowhen_study import mask_cols
from .ww_block import MASKS, apply_mask
from .ww_focus import enc_dir


def qid_array(trajs: list, meta: list, qmap: dict) -> np.ndarray:
    return np.asarray([qmap.setdefault(trajs[ti].query, len(qmap)) for ti, _ in meta], dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--retr", type=int, default=0, help="1 = add the retrieval-feature block to the peer group")
    ap.add_argument("--retr-k", type=int, default=25)
    ap.add_argument("--retr-r", type=int, default=3)
    ap.add_argument("--tag", default="rr")
    ap.add_argument("--protocol", choices=["grouped", "paper"], default="grouped",
                    help="paper = train on the RELEASED train split (7,146), early-stop/tune on the released val (1,787); query leakage into val/test is possible, so AEGIS numbers are inflated and only a labelled extra row")
    ap.add_argument("--limit", type=int, default=0, help="smoke test only: use the first N train / val / test / Who&When trajectories")
    args = ap.parse_args()
    encs = args.encoders.split(",")
    sp = load_paper_splits() if args.protocol == "paper" else load_grouped_splits()
    test, ww = load_split("test"), load_split("who_when")
    trajs = {"train": sp["train"], "val": sp["val"], "test": test, "who_when": ww}
    if args.limit:
        trajs = {k: v[: args.limit] for k, v in trajs.items()}
        sp = {k: v[: args.limit] for k, v in sp.items()}
        test, ww = trajs["test"], trajs["who_when"]
    vocab = agent_vocab(sp["train"])
    probs: dict = {}        # (seed, mask, split) -> list over encoders
    meta, names_cache, src_prior = None, None, None
    t_start = time.time()
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
        if args.retr:
            qmap: dict = {}
            qid = {k: qid_array(trajs[k], meta[k], qmap) for k in trajs}
            Fi, Ti = feats["train"][0][:, :E], make_targets(feats["train"][1])
            for k in trajs:
                Fq = feats[k][0][:, :E]
                R = retrieval_block(Fq, Fi, Ti, qid_q=qid["train"] if k == "train" else None, qid_i=qid["train"] if k == "train" else None,
                                    k=args.retr_k, r=args.retr_r)
                X[k] = np.hstack([X[k], R])
        if src_prior is None:
            ytr = feats["train"][1]
            src_prior = ytr.sum(0) / ytr.sum()
        Xtr, Ytr = X["train"], feats["train"][1]
        copies = [apply_mask(Xtr, nb0, E, pb0, m) for m in MASKS.values()]
        Xaug, Yaug = np.vstack(copies), np.vstack([Ytr] * len(copies))
        Xv = np.vstack([apply_mask(X["val"], nb0, E, pb0, m) for m in MASKS.values()])
        Yv = np.vstack([feats["val"][1]] * len(MASKS))
        for s in range(args.seeds):
            model, _, _, _ = train_one(Xaug, Yaug, Xv, Yv, s)
            for mname, m in MASKS.items():
                for k in ("val", "test", "who_when"):
                    xm = apply_mask(X[k], nb0, E, pb0, m)
                    with torch.no_grad():
                        p = torch.sigmoid(model(torch.tensor((xm - model.mu) / model.sd, dtype=torch.float32))).numpy()
                    probs.setdefault((s, mname, k), []).append(p)
        print(f"[{enc}] trained ({(time.time() - t_start) / 60:.1f} min, feature dim {Xtr.shape[1]}, retr={args.retr})", flush=True)

    ens = {key: np.mean(v, axis=0) for key, v in probs.items()}
    RR_DIR.mkdir(parents=True, exist_ok=True)
    save_dump(RR_DIR / f"dump_{args.tag}.npz", ens, {k: meta[k] for k in ("val", "test", "who_when")},
              {k: [t.id for t in trajs[k]] for k in ("val", "test", "who_when")}, src_prior,
              {"tag": args.tag, "encoders": encs, "seeds": args.seeds, "retr": args.retr, "retr_k": args.retr_k, "retr_r": args.retr_r,
               "masks": list(MASKS), "limit": args.limit, "protocol": args.protocol, "note": "ensemble over encoders per head seed; raw probabilities (no label shift)"})
    print(f"saved {RR_DIR / f'dump_{args.tag}.npz'}", flush=True)

    summary = {}
    for mname in MASKS:
        for shift in (False, True):
            rows = []
            for s in range(args.seeds):
                _, ta, tm = tune2("prod", ens[(s, mname, "val")], meta["val"], sp["val"])
                ww_p = apply_shift(ens[(s, mname, "who_when")], meta["who_when"], len(ww), src_prior) if shift else ens[(s, mname, "who_when")]
                pw = decode2(ww_p, meta["who_when"], len(ww), "prod", ta, tm)
                r = {"test": evaluate([t.labels for t in test], decode2(ens[(s, mname, "test")], meta["test"], len(test), "prod", ta, tm)),
                     "who_when": evaluate([t.labels for t in ww], pw),
                     "who_when_norm": evaluate([norm_pairs(t.labels) for t in ww], [norm_pairs(x) for x in pw])}   # separate row: agent names normalised
                rows.append(r)
                if args.limit:      # smoke tests never touch the ledger
                    continue
                log_run(phase="11", model=f"rr_{args.tag}_{mname}{'_shift' if shift else ''}", seed=s, split=f"who_when_{args.protocol}_{args.tag}", n_params=0,
                        config={"protocol": args.protocol, "mask": mname, "shift": shift, "retr": args.retr, "encoders": encs, "t_a": ta, "t_m": float(tm)},
                        metrics={"who_when": {l: r["who_when"][l] for l in LV}, "who_when_norm": {l: r["who_when_norm"][l] for l in LV}, "test": {l: r["test"][l] for l in LV}},
                        notes="rr_models: block-dropout model" + (" + retrieval block" if args.retr else "") + "; thresholds from AEGIS val under the same mask; no Who&When label used")
            summary[(mname, shift)] = rows
    print(f"\n=== {args.tag}: retr={args.retr}, ensemble of {len(encs)} encoders, mean+-std over {args.seeds} head seeds (micro F1 percent; macro after slash) ===")
    for (mname, shift), rows in summary.items():
        for split in ("test", "who_when", "who_when_norm"):
            if split == "test" and shift:
                continue
            print(f"{mname:11s} {'+shift' if shift else '      '} {split:9s} " + " | ".join(
                f"{l[:4]} {np.mean([100 * r[split][l]['micro_f1'] for r in rows]):.2f}+-{np.std([100 * r[split][l]['micro_f1'] for r in rows]):.2f}/"
                f"{np.mean([100 * r[split][l]['macro_f1'] for r in rows]):.2f}" for l in LV))


if __name__ == "__main__":
    main()
