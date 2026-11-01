"""Random predictor, used only as a harness sanity check against the paper's Random row.

The paper's exact random definition is not in the released code (verified: no match under
third_party/AEGIS/evaluation). Empirically, drawing agents from the GLOBAL pool of 31 label names
(not the trajectory's own agents) with 1 pair (70%) or 2 pairs (30%) lands near the paper's row.
Those two knobs were fitted to the paper's numbers, so this is a harness check, not a baseline.

Run: python -m aegis.src.baselines.random_baseline
"""
from __future__ import annotations

import random

import numpy as np

from ..loaders import load_split
from ..metrics import evaluate
from ..schema import ERROR_MODES
from ..utils import log_run, set_seed

PAPER_RANDOM_AB = {"pair": (0.33, 0.21), "agent": (4.54, 3.56), "error": (11.23, 11.15)}


def predict(trajs: list, pool: list, seed: int, p_one: float = 0.7) -> list:
    rng = random.Random(seed)
    out = []
    for _ in trajs:
        k = 1 if rng.random() < p_one else 2
        out.append({(rng.choice(pool), rng.choice(ERROR_MODES)) for _ in range(k)})
    return out


if __name__ == "__main__":
    train, test = load_split("train"), load_split("test")
    pool = sorted({a for t in train for a, _ in t.labels})
    seeds = list(range(10))
    runs = []
    for s in seeds:
        set_seed(s)
        runs.append(evaluate([t.labels for t in test], predict(test, pool, s)))
    mean = {lvl: {k: float(np.mean([r[lvl][k] for r in runs])) for k in runs[0][lvl]} for lvl in ("pair", "agent", "error")}
    for lvl, (pm, pM) in PAPER_RANDOM_AB.items():
        print(f"{lvl:5s} ours {100*mean[lvl]['micro_f1']:.2f}/{100*mean[lvl]['macro_f1']:.2f}   paper {pm}/{pM}")
    log_run(phase="2", model="random_global_pool", config={"pool": "global31", "p_one": 0.7, "seeds": seeds},
            seed=-1, split="test", metrics={"aegis_bench": mean}, notes="harness sanity check vs paper Random row; test used only for reproduction, no tuning on a model")
