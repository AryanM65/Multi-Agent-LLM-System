"""Robustness statistics for the S4 deduplicated-protocol ensemble, from the saved probability dump (CPU only, no training).

Uses aegis/results/rr/dump_s4_dedup.npz: ensemble-averaged probabilities (14 encoders) for 3 head seeds x 4 block masks x
{val, repo_test, hf_test, who_when}. Thresholds are re-tuned on the deduplicated released validation split exactly as in s4_ensemble.py.

Outputs aegis/results/robustness_stats.json with
  seed_stats   : per (mask, shift, split): mean and SD over the 3 head seeds of pair/agent/error micro-F1 (percent)
  paired       : paired record-level tests between configurations (pair micro-F1 difference, 10,000-resample bootstrap interval and a
                 two-sided Wilcoxon signed-rank test on per-record pair F1 averaged over head seeds)
  sensitivity  : pair micro-F1 when the tuned thresholds (t_a, t_m) are multiplied by a common factor
Mask keys follow the dump: names+peer = both blocks on, names = name on / peer off, peer = name off / peer on, neither = both off.
Run (repo root): btp-pipeline/btp-env/bin/python -m aegis.scripts.robustness_stats
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from aegis.scripts.compare_test_files import convo_hash
from aegis.src.eval.block_ablation import counts, micro
from aegis.src.eval.improve_cpu import decode2, tune2
from aegis.src.eval.rr_common import apply_shift
from aegis.src.loaders import load_paper_splits, load_split

ROOT = Path(__file__).resolve().parents[2]
D = np.load(ROOT / "aegis/results/rr/dump_s4_dedup.npz", allow_pickle=True)
MASKS = ("names+peer", "names", "peer", "neither")
SETS = {"repo_test": "test_repo", "hf_test": "test", "who_when": "who_when"}
LV = ("pair", "agent", "error")
tests = {k: load_split(v) for k, v in SETS.items()}
truth = {k: [t.labels for t in ts] for k, ts in tests.items()}
paper = load_paper_splits()
banned = {convo_hash(t) for k in ("repo_test", "hf_test") for t in tests[k]}
val = [t for t in paper["val"] if convo_hash(t) not in banned]
for k in tests:
    assert list(D[f"ids|{k}"]) == [t.id for t in tests[k]], k
assert list(D["ids|val"]) == [t.id for t in val]
meta = {k: list(zip(D[f"ti|{k}"].tolist(), D[f"ag|{k}"].tolist())) for k in ("val", *SETS)}
SRC = D["src_prior"]
N_BOOT = 10000


def tuned(s, mask):
    _, ta, tm = tune2("prod", D[f"p|{s}|{mask}|val"], meta["val"], val)
    return ta, tm


def pred_counts(s, mask, split, shift, scale=1.0):
    ta, tm = tuned(s, mask)
    p = D[f"p|{s}|{mask}|{split}"]
    if shift:
        p = apply_shift(p, meta[split], len(tests[split]), SRC)
    pred = decode2(p, meta[split], len(tests[split]), "prod", min(ta * scale, 1.0), min(tm * scale, 1.0))
    return counts(truth[split], pred)


TUNED = {(s, m): tuned(s, m) for s in range(3) for m in MASKS}
CACHE = {}


def C(mask, split, shift):  # (seeds, records, 3, 3)
    key = (mask, split, shift)
    if key not in CACHE:
        CACHE[key] = np.stack([pred_counts(s, mask, split, shift) for s in range(3)])
    return CACHE[key]


out = {"method": "3 head seeds over the 14-encoder ensemble; thresholds tuned on deduplicated released validation; percent", "seed_stats": {}, "paired": {}, "sensitivity": {}}

# ---- A. per-head-seed mean and SD ------------------------------------------------------------------------------------------------
for mask in MASKS:
    for split in SETS:
        for shift in ((False, True) if split == "who_when" else (False,)):
            c = C(mask, split, shift)
            per_seed = 100 * micro(c.sum(1))                      # (3 seeds, 3 levels)
            out["seed_stats"][f"{mask}|{'shift' if shift else 'noshift'}|{split}"] = {
                lv: {"mean": round(float(per_seed[:, i].mean()), 2), "sd": round(float(per_seed[:, i].std(ddof=1)), 2),
                     "per_seed": [round(float(x), 2) for x in per_seed[:, i]]} for i, lv in enumerate(LV)}

# ---- B. paired tests -------------------------------------------------------------------------------------------------------------
rng = np.random.default_rng(0)


def paired(a, b, split, label):
    """a, b: (seeds, records, 3, 3) counts on the same records. Difference in pair micro-F1 (a - b)."""
    n = a.shape[1]
    pm = lambda c, ix: micro(c[:, ix].sum(1))[:, 0].mean() * 100
    full = np.arange(n)
    point = pm(a, full) - pm(b, full)
    ix_all = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    d = np.array([pm(a, ix) - pm(b, ix) for ix in ix_all])
    lo, hi = np.percentile(d, [2.5, 97.5])
    fa, fb = (micro(x)[..., 0].mean(0) for x in (a, b))        # per-record pair F1, averaged over head seeds
    diff = fa - fb
    try:
        p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
    except ValueError:
        p = 1.0
    out["paired"][label] = {"split": split, "diff": round(float(point), 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2),
                            "bootstrap_excludes_zero": bool(lo > 0 or hi < 0), "wilcoxon_p": round(p, 4), "n_records": int(n),
                            "n_changed": int((diff != 0).sum())}


for split in ("repo_test", "who_when"):
    for m in ("names", "peer", "neither"):
        paired(C("names+peer", split, False), C(m, split, False), split, f"{split}: all-blocks minus {m} (no shift)")
for m in MASKS:
    paired(C(m, "who_when", True), C(m, "who_when", False), "who_when", f"who_when: shift minus no shift, mask {m}")
paired(C("names", "who_when", True), C("names+peer", "who_when", False), "who_when", "who_when: names+shift minus all-blocks no-shift")
paired(C("names", "who_when", True), C("neither", "who_when", False), "who_when", "who_when: names+shift minus neither no-shift")

# ---- C. threshold sensitivity ----------------------------------------------------------------------------------------------------
SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)
for mask, split, shift in (("names+peer", "repo_test", False), ("names+peer", "hf_test", False), ("names+peer", "who_when", False),
                           ("names", "who_when", False), ("names", "who_when", True)):
    row = {}
    for sc in SCALES:
        c = np.stack([pred_counts(s, mask, split, shift, sc) for s in range(3)])
        v = 100 * micro(c.sum(1))
        row[str(sc)] = {lv: round(float(v[:, i].mean()), 2) for i, lv in enumerate(LV)}
    out["sensitivity"][f"{mask}|{'shift' if shift else 'noshift'}|{split}"] = row

(ROOT / "aegis/results/robustness_stats.json").write_text(json.dumps(out, indent=1))
for k, v in out["seed_stats"].items():
    print(f"{k:34s} " + " | ".join(f"{lv} {v[lv]['mean']:.2f}±{v[lv]['sd']:.2f}" for lv in LV))
for k, v in out["paired"].items():
    print(f"{k:62s} {v['diff']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] wilcoxon p={v['wilcoxon_p']}")
for k, v in out["sensitivity"].items():
    print(k, {s: r["pair"] for s, r in v.items()})
