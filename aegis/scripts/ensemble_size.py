"""Ensemble-size sensitivity and cost measurements for the S4 deduplicated-protocol ensemble (CPU only, uses the saved fine-tuned vectors).

Repeats the per-encoder head training of aegis/src/eval/s4_ensemble.py (same data, masks, seeds) but keeps each encoder's probabilities, so that
the ensemble can be rebuilt from the first k encoders (k = 1, 2, 4, 7, 14) and a single encoder (mean and SD over the 14). Thresholds are tuned on
the deduplicated released validation split for each configuration, as in the main run. Also records wall-clock cost per stage on this machine.
Output: aegis/results/ensemble_size.json   Run (repo root): btp-pipeline/btp-env/bin/python -m aegis.scripts.ensemble_size
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from aegis.scripts.compare_test_files import convo_hash
from aegis.src.baselines.flat_mlp import train_one
from aegis.src.baselines.flat_probe import relational_features
from aegis.src.embed import get_embeddings
from aegis.src.eval.block_ablation import counts, micro
from aegis.src.eval.improve_cpu import decode2, peer_block, tune2
from aegis.src.eval.rr_common import apply_shift
from aegis.src.eval.whowhen_study import mask_cols
from aegis.src.eval.ww_block import MASKS, apply_mask
from aegis.src.features import EmbStore, agent_vocab, build
from aegis.src.loaders import load_paper_splits, load_split

OUT = Path(__file__).resolve().parents[1] / "results"
TESTS = {"repo_test": "test_repo", "who_when": "who_when"}
LV = ("pair", "agent", "error")
ENCS, SEEDS, KS = list(range(14)), 3, (1, 2, 4, 7, 14)

paper = load_paper_splits()
tests = {k: load_split(v) for k, v in TESTS.items()}
tests_all = {**tests, "hf_test": load_split("test")}
banned = {convo_hash(t) for k in ("repo_test", "hf_test") for t in tests_all[k]}
sp = {k: [t for t in paper[k] if convo_hash(t) not in banned] for k in ("train", "val")}
trajs = {"train": sp["train"], "val": sp["val"], **tests}
vocab = agent_vocab(sp["train"])
probs, meta, names_cache, src_prior = {}, None, None, None
cost = {"feature_build_s": [], "head_train_s_per_3_heads": [], "inference_s": []}
for enc in ENCS:
    t0 = time.perf_counter()
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
        ytr = feats["train"][1]; src_prior = ytr.sum(0) / ytr.sum()
    Xtr, Ytr = X["train"], feats["train"][1]
    Xaug = np.vstack([apply_mask(Xtr, nb0, E, pb0, m) for m in MASKS.values()]); Yaug = np.vstack([Ytr] * len(MASKS))
    Xv = np.vstack([apply_mask(X["val"], nb0, E, pb0, m) for m in MASKS.values()]); Yv = np.vstack([feats["val"][1]] * len(MASKS))
    cost["feature_build_s"].append(time.perf_counter() - t0)
    t_head = t_inf = 0.0
    for s in range(SEEDS):
        t1 = time.perf_counter()
        model, _, _, _ = train_one(Xaug, Yaug, Xv, Yv, s)
        t_head += time.perf_counter() - t1
        t2 = time.perf_counter()
        for mname, m in MASKS.items():
            for k in ("val", "repo_test", "who_when"):
                xm = apply_mask(X[k], nb0, E, pb0, m)
                with torch.no_grad():
                    p = torch.sigmoid(model(torch.tensor((xm - model.mu) / model.sd, dtype=torch.float32))).numpy()
                probs.setdefault((s, mname, k), []).append(p)
        t_inf += time.perf_counter() - t2
    cost["head_train_s_per_3_heads"].append(t_head); cost["inference_s"].append(t_inf)
    print(f"encoder {enc} done", flush=True)
n_params = sum(p.numel() for p in model.parameters())


def evaluate_subset(encs, mask, split, shift):
    per = []
    for s in range(SEEDS):
        pv = np.mean([probs[(s, mask, "val")][e] for e in encs], axis=0)
        _, ta, tm = tune2("prod", pv, meta["val"], sp["val"])
        p = np.mean([probs[(s, mask, split)][e] for e in encs], axis=0)
        if shift:
            p = apply_shift(p, meta[split], len(tests[split]), src_prior)
        pred = decode2(p, meta[split], len(tests[split]), "prod", ta, tm)
        per.append(100 * micro(counts([t.labels for t in tests[split]], pred).sum(0)))
    return np.mean(per, axis=0)


CONFIGS = (("names+peer", "repo_test", False), ("names+peer", "who_when", False), ("names", "who_when", False), ("names", "who_when", True))
res = {"ks": list(KS), "n_head_seeds": SEEDS, "first_k": {}, "single_encoder": {}}
for mask, split, shift in CONFIGS:
    key = f"{mask}|{'shift' if shift else 'noshift'}|{split}"
    res["first_k"][key] = {str(k): {lv: round(float(v), 2) for lv, v in zip(LV, evaluate_subset(list(range(k)), mask, split, shift))} for k in KS}
    singles = np.stack([evaluate_subset([e], mask, split, shift) for e in ENCS])        # (14, 3)
    res["single_encoder"][key] = {lv: {"mean": round(float(singles[:, i].mean()), 2), "sd": round(float(singles[:, i].std(ddof=1)), 2),
                                       "min": round(float(singles[:, i].min()), 2), "max": round(float(singles[:, i].max()), 2)} for i, lv in enumerate(LV)}
    print(key, res["first_k"][key]["14"], flush=True)
res["cost"] = {"machine": f"{platform.machine()} {platform.processor() or ''} CPU, torch {torch.__version__}", "head_params": int(n_params),
               "mean_feature_build_s_per_encoder": round(float(np.mean(cost["feature_build_s"])), 1),
               "mean_head_train_s_per_encoder_3_heads_4_masks": round(float(np.mean(cost["head_train_s_per_3_heads"])), 1),
               "mean_inference_s_per_encoder_all_splits": round(float(np.mean(cost["inference_s"])), 2),
               "n_inference_rows": {k: int(len(meta[k])) for k in ("val", "repo_test", "who_when")}}
(OUT / "ensemble_size.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
