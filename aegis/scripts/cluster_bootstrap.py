"""Query-cluster bootstrap for the 14-encoder ensemble (all blocks, no shift), from the saved probability dump (CPU only).

Conversations built from one user query are correlated, so resampling single records can give intervals that are too narrow. Here whole query
groups are resampled (10,000 draws, pooled over 3 head seeds). Compared with the record-level interval used in the main tables.
Output: aegis/results/cluster_bootstrap.json    Run (repo root): btp-pipeline/btp-env/bin/python -m aegis.scripts.cluster_bootstrap
"""
import json
from pathlib import Path

import numpy as np

from aegis.scripts.compare_test_files import convo_hash
from aegis.src.eval.block_ablation import counts, micro
from aegis.src.eval.improve_cpu import decode2, tune2
from aegis.src.loaders import load_paper_splits, load_split

ROOT = Path(__file__).resolve().parents[2]
D = np.load(ROOT / "aegis/results/rr/dump_s4_dedup.npz", allow_pickle=True)
SETS = {"repo_test": "test_repo", "hf_test": "test", "who_when": "who_when"}
LV = ("pair", "agent", "error")
tests = {k: load_split(v) for k, v in SETS.items()}
paper = load_paper_splits()
banned = {convo_hash(t) for k in ("repo_test", "hf_test") for t in tests[k]}
val = [t for t in paper["val"] if convo_hash(t) not in banned]
meta = {k: list(zip(D[f"ti|{k}"].tolist(), D[f"ag|{k}"].tolist())) for k in ("val", *SETS)}
rng = np.random.default_rng(0)
out = {"method": "resample query groups, 10,000 draws, 3 head seeds pooled; ensemble all blocks, no shift; percent", "rows": {}}
for split, ts in tests.items():
    cs = []
    for s in range(3):
        _, ta, tm = tune2("prod", D[f"p|{s}|names+peer|val"], meta["val"], val)
        pred = decode2(D[f"p|{s}|names+peer|{split}"], meta[split], len(ts), "prod", ta, tm)
        cs.append(counts([t.labels for t in ts], pred))
    c = np.stack(cs)                                   # (3, n, 3, 3)
    groups = {}
    for i, t in enumerate(ts):
        groups.setdefault(t.query, []).append(i)
    G = [np.array(v) for v in groups.values()]
    n = len(ts)
    point = 100 * micro(c.sum(1)).mean(0)
    rec = np.array([100 * micro(c[:, rng.integers(0, n, n)].sum(1)).mean(0) for _ in range(2000)])
    clu = []
    for _ in range(10000):
        pick = rng.integers(0, len(G), len(G))
        ix = np.concatenate([G[j] for j in pick])
        clu.append(100 * micro(c[:, ix].sum(1)).mean(0))
    clu = np.array(clu)
    lo_r, hi_r = np.percentile(rec, [2.5, 97.5], axis=0)
    lo_c, hi_c = np.percentile(clu, [2.5, 97.5], axis=0)
    out["rows"][split] = {"records": n, "query_groups": len(G), **{lv: {"micro": round(float(point[i]), 2), "record_lo": round(float(lo_r[i]), 1), "record_hi": round(float(hi_r[i]), 1),
                                                                      "cluster_lo": round(float(lo_c[i]), 1), "cluster_hi": round(float(hi_c[i]), 1)} for i, lv in enumerate(LV)}}
    print(split, n, len(G), {lv: (out["rows"][split][lv]["micro"], out["rows"][split][lv]["record_lo"], out["rows"][split][lv]["record_hi"],
                                  out["rows"][split][lv]["cluster_lo"], out["rows"][split][lv]["cluster_hi"]) for lv in LV}, flush=True)
(ROOT / "aegis/results/cluster_bootstrap.json").write_text(json.dumps(out, indent=1))
