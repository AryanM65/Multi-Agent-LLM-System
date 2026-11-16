"""Paired record-level bootstrap of the label-shift gain on Who&When, for three protocols, from saved probability dumps (CPU only, no training).

Protocols: base  = earlier query-disjoint study (aegis/results/rr/dump_rr_base.npz, 14 old encoders, heads on 3,865 records)
           paper = old encoders, heads on the full released split with duplicates (dump_rr_paper.npz)
           s4    = deduplicated released protocol, 14 newly fine-tuned encoders (dump_s4_dedup.npz; saved from a CPU re-run of
                   aegis/src/eval/s4_ensemble.py that reproduced aegis/results/s4_ensemble.json byte for byte)
Gain = pair micro-F1 with label shift minus without, same 184 records, 3 head seeds pooled, 10,000 record resamples (one shared set of
resamples for every row, so differences between protocols are paired too), 95% percentile interval.
Output: aegis/results/shift_paired.json.   Run: btp-pipeline/btp-env/bin/python -m aegis.scripts.shift_paired   (repo root)"""
import json
from pathlib import Path
import numpy as np
from aegis.scripts.compare_test_files import convo_hash
from aegis.src.eval.block_ablation import counts, micro
from aegis.src.eval.improve_cpu import decode2, tune2
from aegis.src.eval.rr_common import apply_shift
from aegis.src.loaders import load_grouped_splits, load_paper_splits, load_split

ROOT = Path(__file__).resolve().parents[2]
RR = ROOT / "aegis/results/rr"
MASKS = ("names+peer", "names", "peer", "neither")
ww = load_split("who_when"); truth = [t.labels for t in ww]; n = len(ww)
paper = load_paper_splits()
banned = {convo_hash(t) for f in ("test_repo", "test") for t in load_split(f)}
VAL = {"base": load_grouped_splits()["val"], "paper": paper["val"], "s4": [t for t in paper["val"] if convo_hash(t) not in banned]}
D = {k: np.load(RR / f, allow_pickle=True) for k, f in (("base", "dump_rr_base.npz"), ("paper", "dump_rr_paper.npz"), ("s4", "dump_s4_dedup.npz"))}
for k, d in D.items():
    assert list(d["ids|who_when"]) == [t.id for t in ww], k
    assert list(d["ids|val"]) == [t.id for t in VAL[k]], k


def meta(d, k):
    return list(zip(d[f"ti|{k}"].tolist(), d[f"ag|{k}"].tolist()))


def cnt(proto, mask, shift):
    d = D[proto]; out = []
    for s in range(3):
        _, ta, tm = tune2("prod", d[f"p|{s}|{mask}|val"], meta(d, "val"), VAL[proto])
        pw = d[f"p|{s}|{mask}|who_when"]
        if shift:
            pw = apply_shift(pw, meta(d, "who_when"), n, d["src_prior"])
        out.append(counts(truth, decode2(pw, meta(d, "who_when"), n, "prod", ta, tm)))
    return np.stack(out)


C = {(p, m, sh): cnt(p, m, sh) for p in D for m in MASKS for sh in (False, True)}
rng = np.random.default_rng(0); IX = [rng.integers(0, n, n) for _ in range(10000)]; FULL = np.arange(n)
pm = lambda c, ix: micro(c[:, ix].sum(1))[:, 0].mean() * 100
gain = lambda p, m, ix: pm(C[(p, m, True)], ix) - pm(C[(p, m, False)], ix)


def ci(f):
    g = np.array([f(ix) for ix in IX]); lo, hi = np.percentile(g, [2.5, 97.5]); pt = f(FULL)
    return {"gain": round(float(pt), 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2), "excludes_zero": bool(lo > 0 or hi < 0)}


out = {"method": "paired record bootstrap, 10,000 resamples, 3 head seeds pooled, pair micro-F1, shift minus no shift", "gain": {}, "s4_minus": {}}
for p in D:
    for m in MASKS:
        out["gain"][f"{p}|{m}"] = ci(lambda ix, p=p, m=m: gain(p, m, ix))
for p in ("base", "paper"):
    for m in ("names+peer", "names"):
        out["s4_minus"][f"{p}|{m}"] = ci(lambda ix, p=p, m=m: gain("s4", m, ix) - gain(p, m, ix))
(ROOT / "aegis/results/shift_paired.json").write_text(json.dumps(out, indent=1))
for k, v in out["gain"].items(): print(k, v)
for k, v in out["s4_minus"].items(): print("s4 minus", k, v)
