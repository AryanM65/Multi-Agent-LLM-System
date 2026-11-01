"""Frequency baseline (Phase 4): per framework, always predict the k most common (agent, mode) pairs in train.

k is chosen on validation only. A pair a trajectory cannot contain (agent absent from its turns) is skipped.
Run: python -m aegis.src.baselines.frequency
"""
from __future__ import annotations

from collections import Counter

from ..loaders import load_grouped_splits, load_paper_splits
from ..metrics import evaluate
from ..utils import log_run, set_seed


def fit(train: list) -> dict:
    per_fw: dict = {}
    for t in train:
        per_fw.setdefault(t.framework, Counter()).update(t.labels)
    return {fw: [p for p, _ in c.most_common()] for fw, c in per_fw.items()}


def predict(trajs: list, ranking: dict, k: int) -> list:
    out = []
    for t in trajs:
        present = set(t.agents)
        ranked = [p for p in ranking.get(t.framework, []) if p[0] in present]
        out.append(set(ranked[:k]))
    return out


def run(name: str, splits: dict) -> dict:
    ranking = fit(splits["train"])
    truth = [t.labels for t in splits["val"]]
    best = None
    for k in range(1, 16):
        res = evaluate(truth, predict(splits["val"], ranking, k))
        score = sum(res[l]["micro_f1"] for l in ("pair", "agent", "error")) / 3
        if best is None or score > best[0]:
            best = (score, k, res)
    _, k, res = best
    print(f"[{name}] k={k} val pair {100*res['pair']['micro_f1']:.2f}/{100*res['pair']['macro_f1']:.2f} "
          f"agent {100*res['agent']['micro_f1']:.2f}/{100*res['agent']['macro_f1']:.2f} "
          f"error {100*res['error']['micro_f1']:.2f}/{100*res['error']['macro_f1']:.2f}")
    log_run(phase="4", model="frequency", config={"split": name, "k": k}, seed=0, split=f"val_{name}",
            metrics={"val": {l: res[l] for l in ("pair", "agent", "error")}}, notes="k chosen on val")
    return res


if __name__ == "__main__":
    set_seed(0)
    run("paper", load_paper_splits())
    run("grouped", load_grouped_splits())
