"""Leave-one-framework-out validation of the block-dropout masks (CPU, saved vectors; no Who&When label used).
For each AEGIS framework f: train the block-dropout model on grouped train rows of the OTHER frameworks, tune thresholds on grouped val rows of the
other frameworks (same mask), score grouped train+val+test rows of framework f. Tells which mask (names+peer / names / peer / neither) transfers best
to an UNSEEN framework -- a principled, Who&When-free way to pick the Who&When operating point.
Run: python -m aegis.src.eval.ww_lofo --encoders es_s0,es_s1 --tag lofo1
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from ..baselines.flat_mlp import train_one
from ..baselines.flat_probe import relational_features
from ..embed import get_embeddings
from ..features import EmbStore, agent_vocab, build
from ..loaders import load_grouped_splits, load_split
from ..metrics import evaluate
from .improve_cpu import decode2, peer_block, tune2
from .whowhen_study import mask_cols
from .ww_block import MASKS, apply_mask
from .ww_focus import LV, enc_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", required=True)
    ap.add_argument("--tag", default="lofo")
    args = ap.parse_args()
    sp = load_grouped_splits()
    test = load_split("test")
    vocab = agent_vocab(sp["train"])
    pool = {"train": sp["train"], "val": sp["val"], "test": test}
    fws = sorted({t.framework for t in sp["train"]})
    print("frameworks:", {f: sum(t.framework == f for t in sp["train"]) for f in fws}, flush=True)
    res = defaultdict(list)    # (fw, mask) -> list of per-encoder probs info
    allp = {}
    for enc in args.encoders.split(","):
        store = EmbStore(enc_dir(enc))
        feats = {k: build(ts, store, vocab) for k, ts in pool.items()}
        E = store.dim
        names = sorted({a for v in feats.values() for (_, a) in v[2]})
        nc = dict(zip(names, get_embeddings([n.replace("_", " ").replace("(", " ").replace(")", " ") for n in names], verbose=False)))
        base = {k: np.concatenate([mask_cols(v[0], E, len(vocab)), np.stack([nc[a] for (_, a) in v[2]])], axis=1) for k, v in feats.items()}
        nb0, pb0 = base["train"].shape[1] - E, base["train"].shape[1]
        X = {k: np.hstack([base[k], peer_block(pool[k], store), relational_features(pool[k], store)]) for k in pool}
        rowfw = {k: np.array([pool[k][ti].framework for (ti, _) in feats[k][2]]) for k in pool}
        for f in fws:
            tr, va = rowfw["train"] != f, rowfw["val"] != f
            Xtr, Ytr = X["train"][tr], feats["train"][1][tr]
            Xaug = np.vstack([apply_mask(Xtr, nb0, E, pb0, m) for m in MASKS.values()])
            Yaug = np.vstack([Ytr] * len(MASKS))
            Xv = np.vstack([apply_mask(X["val"][va], nb0, E, pb0, m) for m in MASKS.values()])
            Yv = np.vstack([feats["val"][1][va]] * len(MASKS))
            model, _, _, _ = train_one(Xaug, Yaug, Xv, Yv, 0)
            for mname, m in MASKS.items():
                for k in ("val", "test"):
                    xm = apply_mask(X[k], nb0, E, pb0, m)
                    with torch.no_grad():
                        p = torch.sigmoid(model(torch.tensor((xm - model.mu) / model.sd, dtype=torch.float32))).numpy()
                    allp.setdefault((f, mname, k), []).append(p)
        print(f"[{enc}] done", flush=True)
        meta = {k: v[2] for k, v in feats.items()}
        rf = rowfw
    print(f"\n=== {args.tag}: leave-one-framework-out, encoders {args.encoders} (held-out framework rows from val+test; micro F1 percent) ===")
    tot = defaultdict(list)
    for f in fws:
        for mname in MASKS:
            # thresholds from val of the OTHER frameworks under the same mask
            pv = np.mean(allp[(f, mname, "val")], axis=0)
            keep_v = rf["val"] != f
            vt = [t for t in sp["val"] if t.framework != f]
            idx_map = {}
            sub_meta, sub_p = [], []
            remap = {}
            for (ti, a), p, fw in zip(meta["val"], pv, rf["val"]):
                if fw != f:
                    remap.setdefault(ti, len(remap))
                    sub_meta.append((remap[ti], a)); sub_p.append(p)
            _, ta, tm = tune2("prod", np.array(sub_p), sub_meta, vt)
            # held-out: val+test rows of framework f
            out = []
            for k, ts in (("val", sp["val"]), ("test", test)):
                pk = np.mean(allp[(f, mname, k)], axis=0)
                rm, sm, spp, hts = {}, [], [], []
                for (ti, a), p, fw in zip(meta[k], pk, rf[k]):
                    if fw == f:
                        if ti not in rm:
                            rm[ti] = len(rm); hts.append(ts[ti])
                        sm.append((rm[ti], a)); spp.append(p)
                if hts:
                    out.append(evaluate([t.labels for t in hts], decode2(np.array(spp), sm, len(hts), "prod", ta, tm)))
            if not out:
                continue
            n = [len(o.get("pair", {}) ) for o in out]
            r = {l: np.mean([100 * o[l]["micro_f1"] for o in out]) for l in LV}
            tot[mname].append(r)
            print(f"{f:14s} {mname:11s} " + " | ".join(f"{l[:4]} {r[l]:.2f}" for l in LV), flush=True)
    print("\n--- mean over held-out frameworks ---")
    for mname, rs in tot.items():
        print(f"{mname:11s} " + " | ".join(f"{l[:4]} {np.mean([r[l] for r in rs]):.2f}" for l in LV))


if __name__ == "__main__":
    main()
