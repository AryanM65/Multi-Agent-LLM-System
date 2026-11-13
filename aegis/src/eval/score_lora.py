"""Score the LoRA reader's generated predictions (preds_<split>.json from kaggle_lora_reader.py) with the same metrics as everything else.

Run: python -m aegis.src.eval.score_lora --dir aegis/results/kaggle_logs/lora --tag lora1
Also reports the 'present agents x all modes' floor on the same records for reference. Who&When labels are only used to SCORE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..loaders import load_grouped_splits, load_split
from ..metrics import evaluate
from ..utils import log_run

LV = ("pair", "agent", "error")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    sp = load_grouped_splits()
    sets = {"test": load_split("test"), "who_when": load_split("who_when"), "val": sp["val"]}
    for split, trajs in sets.items():
        f = Path(args.dir) / f"preds_{split}.json"
        if not f.exists():
            continue
        preds = json.loads(f.read_text())
        sub = [t for t in trajs if t.id in preds]
        pred = [{(a, m) for a, m in preds[t.id]["pairs"]} for t in sub]
        res = evaluate([t.labels for t in sub], pred)
        n_empty = sum(1 for p in pred if not p)
        print(f"{split}: n {len(sub)} empty predictions {n_empty} | " + " | ".join(f"{l} {100*res[l]['micro_f1']:.2f}/{100*res[l]['macro_f1']:.2f}" for l in LV))
        log_run(phase="11", model="lora_reader", seed=0, split=f"{split}_grouped_{args.tag}", n_params=0, config={"dir": args.dir, "n": len(sub), "empty": n_empty},
                metrics={split: {l: res[l] for l in LV}}, notes="LoRA small-LM reader, trained on AEGIS grouped train only; Who&When zero-shot")


if __name__ == "__main__":
    main()
