"""Gate 0 for the rerank idea (no LLM, CPU only): can ANY re-ranker help? Reads the probability dump written by rr_models.py.

For every (agent, mode) pair of every trajectory the score is p_agent * p_mode (the same score the production decoder uses; Who&When gets the
transductive label-shift correction, AEGIS splits do not). Pairs are ranked per trajectory and:
  recall@k   = (# true pairs inside the top-k) / (# true pairs)            micro over all true pairs; unreachable pairs (true agent that never speaks) count as misses
  hit@k      = share of trajectories with at least one true pair in the top-k
  agent-recall@k = share of true agents that appear in at least one top-k pair
  oracle F1@k = pair/agent/error micro F1 if a PERFECT judge kept only the true pairs inside the top-k (a trajectory with none falls back to its top-1
                pair, like the production decoder). This is a loose ceiling, NOT a stop rule: it assumes a perfect judge.
  model F1   = the production two-threshold decoder tuned on AEGIS grouped val (for reference)
Gate 0 decision (heuristic bar of ours, see rerank_rag_context.md section 5): PASS if recall@5 >= 1.5 x recall@1 on AEGIS VALIDATION.
The choice of k must come from AEGIS validation only; the Who&When columns are diagnostics and must not be used to pick k.

Run: python -m aegis.src.eval.rr_gate0 --dump aegis/results/rr/dump_rr_base.npz --tag rr_base
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ..loaders import load_grouped_splits, load_split
from ..metrics import evaluate
from ..schema import ERROR_MODES
from .improve_cpu import decode2, tune2
from .rr_common import LV, RR_DIR, all_candidates, load_dump, micro, norm_pairs, pairs_of, split_probs

KS = (1, 2, 3, 5, 8, 12)


def diagnostics(truth: list, cands: list, ks=KS) -> dict:
    n_true = sum(len(t) for t in truth)
    out = {"n_traj": len(truth), "n_true_pairs": n_true, "mean_candidates": float(np.mean([len(c) for c in cands]))}
    reachable = sum(len({(a, m) for (a, m) in t} & {(c[1], ERROR_MODES[c[2]]) for c in cc}) for t, cc in zip(truth, cands))
    out["reachable_share"] = reachable / max(n_true, 1)
    for k in ks:
        tops = [pairs_of(c[:k]) for c in cands]
        out[f"recall@{k}"] = sum(len(t & p) for t, p in zip(truth, tops)) / max(n_true, 1)
        out[f"hit@{k}"] = float(np.mean([len(t & p) > 0 for t, p in zip(truth, tops)]))
        n_ag = sum(len({a for a, _ in t}) for t in truth)
        out[f"agent_recall@{k}"] = sum(len({a for a, _ in t} & {a for a, _ in p}) for t, p in zip(truth, tops)) / max(n_ag, 1)
        pred = [(t & p) or pairs_of(c[:1]) for t, p, c in zip(truth, tops, cands)]
        out[f"oracle_f1@{k}"] = dict(zip(LV, micro(evaluate(truth, pred))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mask", default="names+peer")
    args = ap.parse_args()
    d = load_dump(args.dump)
    sp = load_grouped_splits()
    trajs = {"val": sp["val"], "test": load_split("test"), "who_when": load_split("who_when")}
    for k, ts in trajs.items():
        ids = d["ids"][k]
        trajs[k] = ts[: len(ids)]
        assert [t.id for t in trajs[k]] == ids, f"dump ids do not match the {k} split (was the dump made with --limit?)"
    report = {"dump": args.dump, "info": d["info"], "mask": args.mask}
    probs = {k: split_probs(d, k, args.mask) for k in trajs}
    _, ta, tm = tune2("prod", probs["val"], d["meta"]["val"], trajs["val"])
    report["thresholds"] = {"t_a": ta, "t_m": float(tm)}
    for k, ts in trajs.items():
        truth = [t.labels for t in ts]
        cands = all_candidates(probs[k], d["meta"][k], len(ts))
        rep = diagnostics(truth, cands)
        pred = decode2(probs[k], d["meta"][k], len(ts), "prod", ta, tm)
        rep["model_f1"] = dict(zip(LV, micro(evaluate(truth, pred))))
        if k == "who_when":     # separate disclosed row: agent names normalised on predictions and labels (see rr_common.norm_agent)
            rep["model_f1_name_normalised"] = dict(zip(LV, micro(evaluate([norm_pairs(t) for t in truth], [norm_pairs(x) for x in pred]))))
            cn = [{(c[1], ERROR_MODES[c[2]]) for c in cc} for cc in cands]
            rep["reachable_share_name_normalised"] = sum(len(norm_pairs(t) & norm_pairs(c)) for t, c in zip(truth, cn)) / max(rep["n_true_pairs"], 1)
        report[k] = rep
    v = report["val"]
    ratio = v["recall@5"] / max(v["recall@1"], 1e-9)
    report["gate0"] = {"rule": "recall@5 >= 1.5 x recall@1 on AEGIS validation", "ratio": ratio, "pass": bool(ratio >= 1.5)}
    RR_DIR.mkdir(parents=True, exist_ok=True)
    out = RR_DIR / f"gate0_{args.tag}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"=== Gate 0 ({args.tag}, mask {args.mask}) ===")
    for k in ("val", "test", "who_when"):
        r = report[k]
        print(f"[{k}] trajectories {r['n_traj']}, true pairs {r['n_true_pairs']}, reachable {100 * r['reachable_share']:.1f}%, mean candidates {r['mean_candidates']:.1f}")
        print("  recall@k      " + "  ".join(f"@{x} {100 * r[f'recall@{x}']:.1f}" for x in KS))
        print("  hit@k         " + "  ".join(f"@{x} {100 * r[f'hit@{x}']:.1f}" for x in KS))
        print("  agent-recall@k " + "  ".join(f"@{x} {100 * r[f'agent_recall@{x}']:.1f}" for x in KS))
        print("  oracle F1 pair@k " + "  ".join(f"@{x} {r[f'oracle_f1@{x}']['pair']:.1f}" for x in KS) + "   (ceiling, perfect judge)")
        print("  model F1 (pair/agent/error) " + " / ".join(f"{x:.2f}" for x in r["model_f1"].values()))
        if "model_f1_name_normalised" in r:
            print("  [separate row, name-normalised] model F1 (pair/agent/error) " + " / ".join(f"{x:.2f}" for x in r["model_f1_name_normalised"].values())
                  + f"; reachable share {100 * r['reachable_share_name_normalised']:.1f}%")
    g = report["gate0"]
    print(f"GATE 0 {'PASS' if g['pass'] else 'FAIL'}: recall@5 / recall@1 on AEGIS val = {g['ratio']:.2f} (needs >= 1.5). Saved {out}")


if __name__ == "__main__":
    main()
