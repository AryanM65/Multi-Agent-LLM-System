"""Flat MLP with structure-aware decoding (Phase 4b), validation only.

Same model and probabilities as flat_mlp.py. Only the decoder changes, using label structure found in
docs/error_analysis.md (all statistics come from the TRAIN split):
  1. Roles that were never faulty in train (fault rate 0, present in >= 20 trajectories) are never predicted.
  2. The number of faulty agents k is taken from train: the most common faulty-agent count for
     (framework, number of agents present), falling back to the framework's most common count.
  3. The k agents with the highest p_agent are predicted faulty.
  4. Each predicted agent gets its argmax mode; extra modes with p_mode >= t_m are added (t_m tuned on val;
     1.01 = no extra modes).

Run: python -m aegis.src.baselines.flat_topk [--seeds 3]
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict

import numpy as np

from ..features import EmbStore, agent_vocab, build
from ..loaders import load_grouped_splits, load_paper_splits
from ..metrics import evaluate
from ..schema import ERROR_MODES
from .flat_mlp import decode, train_one, tune
from ..utils import log_run

EXTRA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def structure_from_train(train: list, min_present: int = 20) -> dict:
    present, faulty = Counter(), Counter()
    k_by_n, k_by_fw = defaultdict(Counter), defaultdict(Counter)
    for t in train:
        agents = {u.agent for u in t.turns}
        bad = {a for a, _ in t.labels}
        for a in agents:
            present[(t.framework, a)] += 1
            faulty[(t.framework, a)] += a in bad
        k_by_n[(t.framework, len(agents))][len(bad)] += 1
        k_by_fw[t.framework][len(bad)] += 1
    never = {key for key, n in present.items() if n >= min_present and faulty[key] == 0}
    k_n = {key: c.most_common(1)[0][0] for key, c in k_by_n.items()}
    k_fw = {fw: c.most_common(1)[0][0] for fw, c in k_by_fw.items()}
    return {"never": never, "k_n": k_n, "k_fw": k_fw}


def decode_topk(probs, meta, trajs, struct, t_extra: float) -> list:
    n = len(trajs)
    rows = [[] for _ in range(n)]
    for r, (ti, a) in enumerate(meta):
        rows[ti].append((a, r))
    pred = [set() for _ in range(n)]
    for ti, t in enumerate(trajs):
        cand = [(a, r) for a, r in rows[ti] if (t.framework, a) not in struct["never"]] or rows[ti]
        k = struct["k_n"].get((t.framework, len(rows[ti])), struct["k_fw"].get(t.framework, 2))
        cand.sort(key=lambda ar: -probs[ar[1]][-1])
        for a, r in cand[:max(k, 1)]:
            pm = probs[r][:-1]
            pred[ti].add((a, ERROR_MODES[int(np.argmax(pm))]))
            for mi in np.where(pm >= t_extra)[0]:
                pred[ti].add((a, ERROR_MODES[int(mi)]))
    return pred


def tune_topk(probs, meta, trajs, struct):
    truth = [t.labels for t in trajs]
    best = None
    for te in EXTRA_GRID:
        res = evaluate(truth, decode_topk(probs, meta, trajs, struct, te))
        score = sum(res[l]["micro_f1"] for l in ("pair", "agent", "error")) / 3
        if best is None or score > best[0]:
            best = (score, te, res)
    return best


def fmt(res) -> str:
    return (f"pair {100*res['pair']['micro_f1']:.2f}/{100*res['pair']['macro_f1']:.2f} "
            f"agent {100*res['agent']['micro_f1']:.2f}/{100*res['agent']['macro_f1']:.2f} "
            f"error {100*res['error']['micro_f1']:.2f}/{100*res['error']['macro_f1']:.2f}")


def run(name: str, splits: dict, store: EmbStore, seeds: list) -> None:
    vocab = agent_vocab(splits["train"])
    Xtr, Ytr, _ = build(splits["train"], store, vocab)
    Xva, Yva, meta_va = build(splits["val"], store, vocab)
    struct = structure_from_train(splits["train"])
    print(f"[{name}] never-faulty roles: {sorted(struct['never'])}", flush=True)
    base, top = [], []
    for s in seeds:
        t0 = time.time()
        _, probs, epochs, n_params = train_one(Xtr, Ytr, Xva, Yva, s)
        _, ta, tm, res_thr = tune(probs, meta_va, splits["val"])
        _, te, res_top = tune_topk(probs, meta_va, splits["val"], struct)
        base.append(res_thr)
        top.append(res_top)
        print(f"[{name}] seed {s} threshold : {fmt(res_thr)}", flush=True)
        print(f"[{name}] seed {s} top-k     : {fmt(res_top)}  (t_extra {te})", flush=True)
        log_run(phase="4b", model="flat_mlp_topk", seed=s, split=f"val_{name}", n_params=n_params,
                train_seconds=time.time() - t0,
                config={"split": name, "decoder": "topk_agents_per_framework", "t_extra": te, "epochs_run": epochs},
                metrics={"val": {l: res_top[l] for l in ("pair", "agent", "error")}},
                notes="same MLP as flat_mlp; structure-aware decoder; val only; no test eval")
    for label, rs in (("threshold", base), ("top-k", top)):
        for lvl in ("pair", "agent", "error"):
            for k in ("micro_f1", "macro_f1"):
                v = [100 * r[lvl][k] for r in rs]
                print(f"[{name}] {label:9s} {lvl:5s} {k}: {np.mean(v):.2f} +- {np.std(v):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    store = EmbStore()
    for name, splits in (("paper", load_paper_splits()), ("grouped", load_grouped_splits())):
        run(name, splits, store, list(range(args.seeds)))
