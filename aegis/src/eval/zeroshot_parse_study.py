"""Re-score saved zero-shot outputs under the authors' own parse/scoring rules (no GPU).

Question: does our agent micro-F1 of 49.2 (paper: 27.55) move when we copy the authors' evaluation code exactly?
Variants: ours; authors' regex (last {"faulty_agents": [...]} match, failure -> empty); authors' evaluate.calculate_metrics
imported from the repo; unparseable skipped instead of empty; raw (un-normalised) strings. Prints a table, writes nothing.

Usage: python -m aegis.src.eval.zeroshot_parse_study aegis/results/kaggle_logs/full
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

from ..loaders import load_split
from ..metrics import evaluate
from .score_zeroshot import parse_prediction

AUTH = Path(__file__).resolve().parents[2] / "third_party" / "AEGIS" / "evaluation" / "evaluate.py"
PATTERN = re.compile(r'{\s*"faulty_agents"\s*:\s*\[.*?\]\s*}', re.DOTALL)


def authors_parse(raw: str):
    """local_model._parse_response: strip code fences, take the LAST faulty_agents object; else json.loads(whole)."""
    clean = raw.strip()
    if "```json" in clean:
        m = re.search(r"```json\s*({.*?})\s*```", clean, re.DOTALL)
        if m:
            clean = m.group(1)
    elif "```" in clean:
        clean = clean.replace("```json", "").replace("```", "").strip()
    try:
        ms = PATTERN.findall(clean)
        obj = json.loads(ms[-1]) if ms else json.loads(clean)
        items = obj.get("faulty_agents", [])
    except Exception:
        return [], False
    return (items if isinstance(items, list) else []), True


def load_authors_metrics():
    spec = importlib.util.spec_from_file_location("authors_eval", AUTH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.calculate_metrics


def to_sets(items):
    pairs = {(d.get("agent_name"), d.get("error_type")) for d in items if isinstance(d, dict)}
    return pairs, {a for a, _ in pairs}, {e for _, e in pairs}


def main() -> None:
    d = Path(sys.argv[1])
    gold = {t.id: t for t in load_split("test")}
    rows = [json.loads(line) for line in (d / "zeroshot_test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    calc = load_authors_metrics()
    variants = {}

    # 1. ours (score_zeroshot.parse_prediction, normalised modes, unparseable -> empty)
    t, p = [], []
    for r in rows:
        t.append(set(gold[r["id"]].labels))
        p.append(parse_prediction(r["raw_output"])[0])
    res = evaluate(t, p)
    variants["ours (normalised, unparseable=empty)"] = tuple(100 * res[l]["micro_f1"] for l in ("pair", "agent", "error")) + tuple(100 * res[l]["macro_f1"] for l in ("pair", "agent", "error"))

    # 2. authors' parse + authors' calculate_metrics (raw strings), unparseable -> empty (their behaviour)
    # 3. same but unparseable skipped
    for label, skip in (("authors parse + authors metrics, unparseable=empty", False), ("authors parse + authors metrics, unparseable=skipped", True)):
        tg, pg, ta, pa, te, pe = [], [], [], [], [], []
        for r in rows:
            items, ok = authors_parse(r["raw_output"])
            if skip and not ok:
                continue
            g = gold[r["id"]].labels
            gp, ga, ge = set(g), {a for a, _ in g}, {e for _, e in g}
            pp, pa_, pe_ = to_sets(items)
            tg.append(gp); pg.append(pp); ta.append(ga); pa.append(pa_); te.append(ge); pe.append(pe_)
        out = [calc(x, y) for x, y in ((tg, pg), (ta, pa), (te, pe))]
        variants[label] = tuple(100 * m["f1_micro"] for m in out) + tuple(100 * m["f1_macro"] for m in out)

    # 4. diagnostics on predicted agent names
    names_in = names_tot = n_empty = 0
    for r in rows:
        items, ok = authors_parse(r["raw_output"])
        n_empty += not items
        pred_agents = {x.get("agent_name") for x in items if isinstance(x, dict)}
        names_tot += len(pred_agents)
        names_in += len(pred_agents & set(gold[r["id"]].agents))

    print(f"{'variant':58s} pair(mi/ma)      agent(mi/ma)     error(mi/ma)")
    for k, v in variants.items():
        print(f"{k:58s} {v[0]:5.2f}/{v[3]:5.2f}   {v[1]:5.2f}/{v[4]:5.2f}   {v[2]:5.2f}/{v[5]:5.2f}")
    print(f"paper Qwen2.5-7B-Instruct                                  5.02/ 2.52   27.55/14.49   14.96/11.36")
    print(f"predicted agent names present in the trajectory: {names_in}/{names_tot} ({100*names_in/names_tot:.1f}%); empty predictions: {n_empty}/{len(rows)}")


if __name__ == "__main__":
    main()
