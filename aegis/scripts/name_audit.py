"""Agent-name consistency audit of the test files, and its effect on scores (CPU only, saved probability dump).

Finding checked here: in the authors' file (aegis.jsonl) many DyLAN gold agent labels are lowercase ('assistant1') while the speakers in the
conversation are 'Assistant1', so an exact-match scorer (the authors' evaluate.py, ours) can never credit them. Training data has no such mismatch.
Outputs aegis/results/name_audit.json:
  mismatch  : per file and framework, gold agent labels that do not equal any speaker name, and how many match after lower-casing / dropping '-' and '_'
  ids       : number of records and number of distinct record ids per file
  scores    : exact-match vs normalised-name micro-F1 of the 14-encoder ensemble (all blocks, no shift), mean and SD over 3 head seeds, with a
              2,000-resample record-level bootstrap interval for the pooled-seed point estimate
Run (repo root): btp-pipeline/btp-env/bin/python -m aegis.scripts.name_audit
"""
import json
import re
from pathlib import Path

import numpy as np

from aegis.scripts.compare_test_files import convo_hash
from aegis.src.eval.block_ablation import counts, micro
from aegis.src.eval.improve_cpu import decode2, tune2
from aegis.src.loaders import load_paper_splits, load_split

ROOT = Path(__file__).resolve().parents[2]
D = np.load(ROOT / "aegis/results/rr/dump_s4_dedup.npz", allow_pickle=True)
SETS = {"repo_test": "test_repo", "hf_test": "test"}
LV = ("pair", "agent", "error")
norm = lambda a: re.sub(r"[-_\s]", "", a.lower())
tests = {k: load_split(v) for k, v in SETS.items()}
out = {"mismatch": {}, "ids": {}, "scores": {}}

for split, ts in tests.items():
    out["ids"][split] = {"records": len(ts), "distinct_ids": len({t.id for t in ts})}
    fw = {}
    for t in ts:
        spk = set(t.agents); spk_n = {norm(a) for a in spk}
        for a, _ in t.labels:
            r = fw.setdefault(t.framework, {"gold_agent_labels": 0, "not_exact_speaker": 0, "fixed_by_normalising": 0})
            r["gold_agent_labels"] += 1
            if a not in spk:
                r["not_exact_speaker"] += 1
                r["fixed_by_normalising"] += norm(a) in spk_n
    out["mismatch"][split] = fw
tr = load_paper_splits()["train"]
bad = sum(a not in set(t.agents) for t in tr for a, _ in t.labels)
out["mismatch"]["train"] = {"gold_agent_labels": sum(len(t.label_agents) for t in tr), "not_exact_speaker": int(bad)}

paper = load_paper_splits()
banned = {convo_hash(t) for k in tests for t in tests[k]}
val = [t for t in paper["val"] if convo_hash(t) not in banned]
meta = {k: list(zip(D[f"ti|{k}"].tolist(), D[f"ag|{k}"].tolist())) for k in ("val", *SETS)}
rng = np.random.default_rng(0)


def mapped(ps, f):
    return [{(f(a), m) for a, m in s} for s in ps]


for split, ts in tests.items():
    truth = [t.labels for t in ts]
    res = {"exact": [], "normalised": []}
    cs = {"exact": [], "normalised": []}
    for s in range(3):
        _, ta, tm = tune2("prod", D[f"p|{s}|names+peer|val"], meta["val"], val)
        pred = decode2(D[f"p|{s}|names+peer|{split}"], meta[split], len(ts), "prod", ta, tm)
        # normalised scoring maps BOTH gold and predicted names through norm(); predictions use speaker names so only gold changes in effect
        for nm, f in (("exact", lambda a: a), ("normalised", norm)):
            c = counts(mapped(truth, f), mapped(pred, f))
            cs[nm].append(c); res[nm].append(100 * micro(c.sum(0)))
    out["scores"][split] = {}
    for nm in ("exact", "normalised"):
        arr = np.array(res[nm]); stack = np.stack(cs[nm]); n = stack.shape[1]
        boots = np.array([micro(stack[:, ix].sum(1)).mean(0) * 100 for ix in (rng.integers(0, n, n) for _ in range(2000))])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        out["scores"][split][nm] = {lv: {"mean": round(float(arr[:, i].mean()), 2), "sd": round(float(arr[:, i].std(ddof=1)), 2),
                                         "lo": round(float(lo[i]), 1), "hi": round(float(hi[i]), 1)} for i, lv in enumerate(LV)}
    fwi = {}
    for fwn in sorted({t.framework for t in ts}):
        idx = [i for i, t in enumerate(ts) if t.framework == fwn]
        fwi[fwn] = {}
        for nm in ("exact", "normalised"):
            c = np.concatenate([cs[nm][s][idx][None] for s in range(3)])
            v = 100 * micro(c.sum(1)).mean(0)
            fwi[fwn][nm] = {lv: round(float(v[i]), 2) for i, lv in enumerate(LV)}
        fwi[fwn]["n"] = len(idx)
    out["scores"][split]["by_framework"] = fwi

(ROOT / "aegis/results/name_audit.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
