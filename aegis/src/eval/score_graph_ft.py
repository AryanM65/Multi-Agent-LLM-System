"""CPU scorer for preds.json exported by aegis/scripts/kaggle_finetune_graph.py (end-to-end MiniLM + turn-graph fine-tune).

preds.json = {split: [{"id", "agents": [names in first-appearance order], "probs": [[14 mode probs, 1 agent prob], ...]}]}
for splits es / val / test / who_when. The thresholds (t_a, t_m) are tuned on VAL ONLY (flat_mlp.tune); test and Who&When are
scored once. Prints micro/macro pair/agent/error, avg12 (paper Table 1 definition), floors, and a per-framework test breakdown.

Run: python -m aegis.src.eval.score_graph_ft --preds <preds.json> --tag ftg_l2
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from ..baselines.flat_mlp import decode, tune
from ..loaders import load_grouped_splits, load_split
from ..metrics import evaluate
from ..schema import ERROR_MODES
from ..utils import log_run

LV = ("pair", "agent", "error")


def line(res) -> str:
    return " ".join(f"{l} {100*res[l]['micro_f1']:.2f}/{100*res[l]['macro_f1']:.2f}" for l in LV)


def avg12(rt, rw) -> float:
    return float(np.mean([100 * r[l][k] for r in (rt, rw) for l in LV for k in ("micro_f1", "macro_f1")]))


def floor(trajs):
    return evaluate([t.labels for t in trajs], [{(a, m) for a in set(t.agents) for m in ERROR_MODES} for t in trajs])


def arrays(pred_rows: list, trajs: list):
    """probs (rows x 15) and meta [(traj_index, agent)] aligned with `trajs` by id; agents must match the trajectory's agents."""
    by_id = {r["id"]: r for r in pred_rows}
    probs, meta = [], []
    for ti, t in enumerate(trajs):
        r = by_id[t.id]
        assert list(r["agents"]) == list(t.agents), f"agent order mismatch in {t.id}"
        for a, p in zip(r["agents"], r["probs"]):
            probs.append(p)
            meta.append((ti, a))
    return np.asarray(probs, dtype=np.float32), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--tag", required=True, help="ledger split-name suffix, e.g. ftg_l2")
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()
    preds = json.load(open(args.preds, encoding="utf-8"))
    val = load_grouped_splits()["val"]
    test, ww = load_split("test"), load_split("who_when")
    if len(preds["val"]) < len(val):   # smoke runs export only a slice
        keep = {r["id"] for r in preds["val"]}
        val = [t for t in val if t.id in keep]
    for name, ts in (("test", test), ("who_when", ww)):
        if len(preds[name]) < len(ts):
            keep = {r["id"] for r in preds[name]}
            if name == "test":
                test = [t for t in ts if t.id in keep]
            else:
                ww = [t for t in ts if t.id in keep]
    pv, mv = arrays(preds["val"], val)
    _, ta, tm, rv = tune(pv, mv, val)
    print(f"[{args.tag}] thresholds tuned on val only: t_a {ta} t_m {tm}  val {line(rv)}", flush=True)
    out = {}
    for name, ts in (("test", test), ("who_when", ww)):
        p, m = arrays(preds[name], ts)
        out[name] = evaluate([t.labels for t in ts], decode(p, m, len(ts), ta, tm))
        print(f"[{args.tag}] {name:8s} {line(out[name])}", flush=True)
        print(f"[{args.tag}] floor {name:8s} {line(floor(ts))}", flush=True)
    print(f"[{args.tag}] avg12 {avg12(out['test'], out['who_when']):.2f}   (paper Table 1 Avg. definition; DCL 12.61, Aegis-SFT 26.51)", flush=True)

    # per-framework test breakdown (same thresholds)
    p, m = arrays(preds["test"], test)
    pred_sets = decode(p, m, len(test), ta, tm)
    groups = defaultdict(list)
    for i, t in enumerate(test):
        groups[t.framework].append(i)
    for fw, idx in sorted(groups.items()):
        r = evaluate([test[i].labels for i in idx], [pred_sets[i] for i in idx])
        print(f"[{args.tag}] test framework {fw:13s} n={len(idx):3d} {line(r)}", flush=True)

    if not args.no_ledger:
        for tag, res in out.items():
            log_run(phase="8", model="graph_finetune_e2e", seed=0, split=f"{tag}_{args.tag}", n_params=0,
                    config={"preds": args.preds, "t_a": ta, "t_m": tm},
                    metrics={tag: {l: res[l] for l in LV}},
                    notes="end-to-end MiniLM + turn-graph fine-tune; thresholds tuned on grouped val only; test/Who&When scored once")


if __name__ == "__main__":
    main()
