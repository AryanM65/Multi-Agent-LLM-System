"""Pair / agent / error micro+macro F1, matching the authors' evaluation/evaluate.py exactly.

Faithfulness notes (verified against third_party/AEGIS/evaluation/evaluate.py):
- Macro-F1 averages over the UNION of classes seen in either true or predicted sets, over the
  whole evaluated set. Agent-level classes are raw agent names (e.g. 'Node0', 'Solver'),
  pair-level classes are (agent, mode) tuples, error-level classes are mode codes.
- Samples whose prediction is unparseable (non-list) are skipped by the authors; here the caller
  passes an empty set instead (PLAN.md 3a: count unparseable as empty) and reports the rate.
"""
from __future__ import annotations

from typing import Iterable


def prf(true_sets: list, pred_sets: list) -> dict:
    classes: set = set()
    for s in true_sets:
        classes |= s
    for s in pred_sets:
        classes |= s
    stats = {c: [0, 0, 0] for c in classes}   # tp, fp, fn
    mtp = mfp = mfn = 0
    for t, p in zip(true_sets, pred_sets):
        tp, fp, fn = t & p, p - t, t - p
        mtp, mfp, mfn = mtp + len(tp), mfp + len(fp), mfn + len(fn)
        for c in tp:
            stats[c][0] += 1
        for c in fp:
            stats[c][1] += 1
        for c in fn:
            stats[c][2] += 1

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    mp, mr, mf = f1(mtp, mfp, mfn)
    per = [f1(*s) for s in stats.values()]
    n = len(per) or 1
    return {
        "micro_p": mp, "micro_r": mr, "micro_f1": mf,
        "macro_p": sum(x[0] for x in per) / n,
        "macro_r": sum(x[1] for x in per) / n,
        "macro_f1": sum(x[2] for x in per) / n,
    }


def evaluate(true_pairs: Iterable, pred_pairs: Iterable) -> dict:
    """Each element: a set of (agent_name, error_type) tuples for one trajectory."""
    true_pairs, pred_pairs = [set(s) for s in true_pairs], [set(s) for s in pred_pairs]
    proj_a = lambda ss: [{a for a, _ in s} for s in ss]
    proj_e = lambda ss: [{e for _, e in s} for s in ss]
    return {
        "pair": prf(true_pairs, pred_pairs),
        "agent": prf(proj_a(true_pairs), proj_a(pred_pairs)),
        "error": prf(proj_e(true_pairs), proj_e(pred_pairs)),
        "exact_match": sum(t == p for t, p in zip(true_pairs, pred_pairs)) / max(len(true_pairs), 1),
    }


def avg12(ab: dict, ww: dict) -> float:
    """Paper Table 1 Avg.: mean of 12 numbers (3 levels x micro/macro x 2 benchmarks), in percent."""
    vals = [m[lvl][k] for m in (ab, ww) for lvl in ("pair", "agent", "error") for k in ("micro_f1", "macro_f1")]
    return 100 * sum(vals) / len(vals)
