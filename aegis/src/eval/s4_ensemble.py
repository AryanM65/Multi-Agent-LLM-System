"""S4: block-dropout ensemble over the 14 encoders fine-tuned under the PUBLISHED protocol (released train/val minus test duplicates).

Same method as ww_block.py (the draft's), with three changes: (1) train/val = released split with every record removed whose conversation is a
verbatim copy of a record in either 600-record test file (the exact ids the encoders were trained on); (2) scored on the authors' repo file
(primary, decision D11), the HF test file and Who&When; (3) every row carries a record-level bootstrap 95% interval.
Protocol discipline: thresholds are tuned on the released validation split (same block mask as the evaluated split); no Who&When label is used;
label-shift (EM prior, Saerens et al.) reads UNLABELLED Who&When inputs only and is reported as a separate, transductive row.
Run (repo root): python -m aegis.src.eval.s4_ensemble --encoders 0,1,...,13 --seeds 3 --tag s4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ...scripts.compare_test_files import convo_hash
from ..baselines.flat_mlp import train_one
from ..baselines.flat_probe import relational_features
from ..embed import get_embeddings
from ..features import EmbStore, agent_vocab, build
from ..loaders import load_paper_splits, load_split
from ..metrics import evaluate
from ..utils import log_run
from .block_ablation import bootstrap, counts
from .improve_cpu import decode2, peer_block, tune2
from .whowhen_study import mask_cols
from .ww_block import MASKS, apply_mask
from .ww_focus import LV, estimate_prior

OUT = Path(__file__).resolve().parents[2] / "results"
TESTS = {"repo_test": "test_repo", "hf_test": "test", "who_when": "who_when"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", required=True, help="comma-separated S4 seeds, e.g. 0,1,2")
    ap.add_argument("--seeds", type=int, default=3, help="head seeds per encoder")
    ap.add_argument("--tag", default="s4")
    args = ap.parse_args()
    encs = [int(e) for e in args.encoders.split(",")]

    paper = load_paper_splits()
    tests = {k: load_split(v) for k, v in TESTS.items()}
    banned = {convo_hash(t) for k in ("repo_test", "hf_test") for t in tests[k]}
    sp = {k: [t for t in paper[k] if convo_hash(t) not in banned] for k in ("train", "val")}
    ids = json.load(open(OUT / "s4_ids_check.json")) if (OUT / "s4_ids_check.json").exists() else None
    print(f"train {len(sp['train'])} val {len(sp['val'])} (released {len(paper['train'])}/{len(paper['val'])})", flush=True)
    trajs = {"train": sp["train"], "val": sp["val"], **tests}
    vocab = agent_vocab(sp["train"])
    probs: dict = {}          # (head_seed, mask, split) -> list over encoders
    meta, names_cache, src_prior = None, None, None
    t_start = time.time()
    for enc in encs:
        store = EmbStore(str(OUT / "kaggle_logs" / f"s4_s{enc}" / "ft_shards"))
        feats = {k: build(ts, store, vocab) for k, ts in trajs.items()}
        meta = {k: v[2] for k, v in feats.items()}
        E = store.dim
        if names_cache is None:
            names = sorted({a for v in feats.values() for (_, a) in v[2]})
            names_cache = dict(zip(names, get_embeddings([n.replace("_", " ").replace("(", " ").replace(")", " ") for n in names], verbose=False)))
        base = {k: np.concatenate([mask_cols(v[0], E, len(vocab)), np.stack([names_cache[a] for (_, a) in v[2]])], axis=1) for k, v in feats.items()}
        nb0, pb0 = base["train"].shape[1] - E, base["train"].shape[1]
        X = {k: np.hstack([base[k], peer_block(trajs[k], store), relational_features(trajs[k], store)]) for k in trajs}
        if src_prior is None:
            ytr = feats["train"][1]
            src_prior = ytr.sum(0) / ytr.sum()
        Xtr, Ytr = X["train"], feats["train"][1]
        Xaug = np.vstack([apply_mask(Xtr, nb0, E, pb0, m) for m in MASKS.values()])
        Yaug = np.vstack([Ytr] * len(MASKS))
        Xv = np.vstack([apply_mask(X["val"], nb0, E, pb0, m) for m in MASKS.values()])
        Yv = np.vstack([feats["val"][1]] * len(MASKS))
        for s in range(args.seeds):
            model, _, epochs, _ = train_one(Xaug, Yaug, Xv, Yv, s)
            for mname, m in MASKS.items():
                for k in ("val", "repo_test", "hf_test", "who_when"):
                    xm = apply_mask(X[k], nb0, E, pb0, m)
                    with torch.no_grad():
                        p = torch.sigmoid(model(torch.tensor((xm - model.mu) / model.sd, dtype=torch.float32))).numpy()
                    probs.setdefault((s, mname, k), []).append(p)
        print(f"[encoder s4_s{enc}] trained {args.seeds} heads ({time.time() - t_start:.0f}s elapsed)", flush=True)

    rows, preds_out = {}, {}
    for mname in MASKS:
        for shift in (False, True):
            per = {k: [] for k in ("repo_test", "hf_test", "who_when")}
            for s in range(args.seeds):
                ens = {k: np.mean(probs[(s, mname, k)], axis=0) for k in ("val", "repo_test", "hf_test", "who_when")}
                _, ta, tm = tune2("prod", ens["val"], meta["val"], sp["val"])
                for k in per:
                    if shift and k != "who_when":
                        continue                      # label shift is applied to the natural target only
                    p = ens[k]
                    if shift:
                        pri = estimate_prior(p, meta[k], len(tests[k]), src_prior)
                        w = pri / src_prior
                        w = w / w.mean()
                        p = p.copy()
                        p[:, :-1] = np.clip(p[:, :-1] * w, 0, 1)
                    pred = decode2(p, meta[k], len(tests[k]), "prod", ta, tm)
                    per[k].append(counts([t.labels for t in tests[k]], pred))
                    res = evaluate([t.labels for t in tests[k]], pred)
                    log_run(phase="S4", model=f"s4_ens{len(encs)}_{mname}{'_shift' if shift else ''}", seed=s, split=f"paperdedup_{k}",
                            config={"mask": mname, "shift": shift, "encoders": encs, "t_a": ta, "t_m": float(tm)},
                            metrics={k: {l: res[l] for l in LV}},
                            notes="S4: encoders fine-tuned on released train minus test duplicates; thresholds from released val; no Who&When label used")
                    if s == 0:
                        preds_out[f"{mname}|{'shift' if shift else 'noshift'}|{k}"] = {"ids": [t.id for t in tests[k]], "pred": [sorted(map(list, q)) for q in pred]}
            for k, v in per.items():
                if v:
                    rows[f"{mname}|{'shift' if shift else 'noshift'}|{k}"] = bootstrap(np.stack(v))
    (OUT / f"{args.tag}_ensemble.json").write_text(json.dumps({"encoders": encs, "head_seeds": args.seeds, "rows": rows}, indent=1))
    (OUT / "preds" / f"{args.tag}_ensemble_seed0_preds.json").write_text(json.dumps(preds_out))
    print(f"\n=== {args.tag}: {len(encs)} encoders x {args.seeds} head seeds; micro-F1 percent [95% record bootstrap] ===")
    for key, r in rows.items():
        print(f"{key:34s} " + " | ".join(f"{lv} {r[lv]['micro']:.2f} [{r[lv]['lo']:.1f},{r[lv]['hi']:.1f}]" for lv in LV))


if __name__ == "__main__":
    main()
