"""Score zero-shot LLM outputs (Phase 3a) with the authors' metric definition.

Usage:
  python -m aegis.src.eval.score_zeroshot <outputs_dir> [--log]

<outputs_dir> holds zeroshot_test.jsonl and zeroshot_who_when.jsonl written by
scripts/kaggle_zeroshot_qwen.py. Parsing mirrors the authors' evaluate.py: take the
faulty_agents list, optionally nested under final_answer / Final Answer. Unparseable output
counts as an empty prediction (the authors skip it instead); the rate is reported.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ..loaders import load_split
from ..metrics import avg12, evaluate
from ..schema import normalize_mode
from ..utils import log_run

PAPER_QWEN7B = {
    "aegis_bench": {"pair": (5.02, 2.52), "agent": (27.55, 14.49), "error": (14.96, 11.36)},
    "who_when": {"pair": (2.31, 1.14), "agent": (40.92, 23.50), "error": (3.64, 1.77)},
}


def _candidates(text: str):
    """Yield JSON values found in text: the whole string, then each top-level {...} span."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        yield json.loads(text)
        return
    except Exception:
        pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    yield json.loads(text[start: i + 1])
                except Exception:
                    pass


def parse_prediction(raw: str):
    """Return (set of (agent, mode) pairs, parsed_ok)."""
    for obj in _candidates(raw):
        if not isinstance(obj, dict):
            continue
        for key in ("final_answer", "Final Answer"):
            if isinstance(obj.get(key), dict):
                obj = obj[key]
                break
        items = obj.get("faulty_agents")
        if isinstance(items, list):
            pairs = set()
            for d in items:
                if isinstance(d, dict) and d.get("agent_name") is not None and d.get("error_type") is not None:
                    try:
                        pairs.add((str(d["agent_name"]).strip(), normalize_mode(d["error_type"])))
                    except ValueError:
                        pairs.add((str(d["agent_name"]).strip(), str(d["error_type"]).strip()))
            return pairs, True
    return set(), False


def score_split(split: str, path: Path) -> dict:
    gold = {t.id: t for t in load_split(split)}
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    true_sets, pred_sets, bad, trunc, not_stop = [], [], 0, 0, 0
    for r in rows:
        pairs, ok = parse_prediction(r["raw_output"])
        bad += not ok
        trunc += bool(r.get("truncated"))
        not_stop += r.get("finish_reason") != "stop"
        true_sets.append(set(gold[r["id"]].labels))
        pred_sets.append(pairs)
    res = evaluate(true_sets, pred_sets)
    res["n"] = len(rows)
    res["unparseable"] = bad
    res["truncated"] = trunc
    res["finish_not_stop"] = not_stop
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs_dir")
    ap.add_argument("--log", action="store_true", help="append to results/metrics_history.jsonl")
    ap.add_argument("--model", default="qwen2.5-7b-instruct-awq_zeroshot")
    args = ap.parse_args()
    d = Path(args.outputs_dir)
    results = {}
    for key, split, fn in (("aegis_bench", "test", "zeroshot_test.jsonl"), ("who_when", "who_when", "zeroshot_who_when.jsonl")):
        if (d / fn).exists():
            results[key] = score_split(split, d / fn)
    for key, r in results.items():
        print(f"== {key}: n={r['n']} unparseable={r['unparseable']} truncated={r['truncated']} finish!=stop={r['finish_not_stop']}")
        for lvl in ("pair", "agent", "error"):
            pm, pM = PAPER_QWEN7B[key][lvl]
            print(f"  {lvl:5s} micro/macro F1 ours {100*r[lvl]['micro_f1']:.2f}/{100*r[lvl]['macro_f1']:.2f}   paper {pm}/{pM}")
    if "aegis_bench" in results and "who_when" in results:
        print(f"  avg12 ours {avg12(results['aegis_bench'], results['who_when']):.2f}   paper 12.43")
    if args.log and results:
        log_run(phase="3a", model=args.model, config={"outputs_dir": d.name}, seed=0, split="test+who_when",
                metrics={k: v for k, v in results.items()}, notes="zero-shot reproduction, no tuning")


if __name__ == "__main__":
    main()
