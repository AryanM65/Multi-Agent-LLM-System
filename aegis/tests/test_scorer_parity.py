"""Parity test: our aegis/src/metrics.py vs the authors' evaluation/evaluate.py on identical inputs.

The authors' file is third-party code in aegis/third_party/AEGIS (git-ignored; clone https://github.com/kfq20/Aegis).
It imports pandas, so the test is skipped if either is missing.
Run with an interpreter that has pandas, for example: btp-pipeline/btp-env/bin/python -m pytest aegis/tests/test_scorer_parity.py
"""
import importlib.util
import json
import random
from pathlib import Path

import pytest

from aegis.src.metrics import evaluate

pytest.importorskip("pandas")
AUTHORS = Path(__file__).resolve().parents[1] / "third_party" / "AEGIS" / "evaluation" / "evaluate.py"
pytestmark = pytest.mark.skipif(not AUTHORS.exists(), reason="authors' repo not cloned")

MODES = [f"FM-{g}.{i}" for g, n in ((1, 5), (2, 6), (3, 3)) for i in range(1, n + 1)]  # 14 MAST modes
AGENTS = ["Node0", "Node1", "Solver", "Critic", "Planner", "Coder", "Judge", "Orchestrator"]
KEYS = (("pair", "Global Level (Exact Pair Matching)"), ("agent", "Agent Level"), ("error", "Error Type Level"))
TOL = 1e-12


def load_authors():
    spec = importlib.util.spec_from_file_location("authors_evaluate", AUTHORS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def random_sets(rng, n):
    def one():
        k = rng.choice([0, 0, 1, 1, 2, 3])  # include empty sets
        return {(rng.choice(AGENTS), rng.choice(MODES)) for _ in range(k)}
    return [one() for _ in range(n)], [one() for _ in range(n)]


def compare(ours, theirs):
    for lvl, key in KEYS:
        for k_ours, k_theirs in (("micro_p", "precision_micro"), ("micro_r", "recall_micro"), ("micro_f1", "f1_micro"),
                                 ("macro_p", "precision_macro"), ("macro_r", "recall_macro"), ("macro_f1", "f1_macro")):
            assert ours[lvl][k_ours] == pytest.approx(theirs[key][k_theirs], abs=TOL), (lvl, k_ours)


def project(sets, which):
    if which == "pair":
        return [set(s) for s in sets]
    idx = 0 if which == "agent" else 1
    return [{x[idx] for x in s} for s in sets]


def test_calculate_metrics_fuzz():
    """200 random datasets, all six numbers at all three levels, against the authors' calculate_metrics."""
    ev = load_authors()
    rng = random.Random(0)
    for _ in range(200):
        t, p = random_sets(rng, rng.randint(1, 40))
        ours = evaluate(t, p)
        theirs = {key: ev.calculate_metrics(project(t, lvl), project(p, lvl)) for lvl, key in KEYS}
        compare(ours, theirs)


def test_end_to_end_jsonl(tmp_path, monkeypatch):
    """Write predictions in the authors' JSONL format, run their evaluate_results, compare to ours."""
    ev = load_authors()
    rng = random.Random(1)
    t, p = random_sets(rng, 120)
    path = tmp_path / "preds.jsonl"
    with open(path, "w") as f:
        for ts, ps in zip(t, p):
            rec = {
                "response_text": json.dumps({"final_answer": {"faulty_agents": [
                    {"agent_name": a, "error_type": m} for a, m in sorted(ps)]}}),
                "ground_truth": {"faulty_agents": [{"agent_name": a, "error_type": m} for a, m in sorted(ts)]},
            }
            f.write(json.dumps(rec) + "\n")

    captured = {}
    orig = ev.pd.DataFrame.from_dict

    def spy(d, orient="columns", **kw):
        captured.update(d)
        return orig(d, orient=orient, **kw)

    monkeypatch.setattr(ev.pd.DataFrame, "from_dict", staticmethod(spy))
    ev.evaluate_results(str(path))
    # records whose predicted list is empty fall through to model_detection in the authors' parser and count as
    # an empty prediction too, so all 120 records are scored by both sides
    compare(evaluate(t, p), captured)
    assert captured["Overall"]["Exact Match Ratio"] == pytest.approx(evaluate(t, p)["exact_match"], abs=TOL)


def test_unparseable_rule_is_the_only_documented_difference(tmp_path):
    """The authors SKIP records whose prediction is not a list; our harness scores them as empty (aegis/src/metrics.py).
    This test pins the difference so it can never be hidden: for the paper comparison with published rows we must use
    the authors' rule (drop them), which is the caller's job. Our own models always emit parseable output."""
    ev = load_authors()
    path = tmp_path / "preds.jsonl"
    gt = {"faulty_agents": [{"agent_name": "A", "error_type": "FM-1.1"}]}
    rows = [
        {"response_text": json.dumps({"final_answer": {"faulty_agents": [{"agent_name": "A", "error_type": "FM-1.1"}]}}),
         "ground_truth": gt},
        {"model_detection": 7, "ground_truth": gt},  # non-list prediction -> skipped by the authors
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    captured = {}
    orig = ev.pd.DataFrame.from_dict
    ev.pd.DataFrame.from_dict = staticmethod(lambda d, orient="columns", **kw: (captured.update(d), orig(d, orient=orient, **kw))[1])
    try:
        ev.evaluate_results(str(path))
    finally:
        ev.pd.DataFrame.from_dict = orig
    assert captured["Overall"]["Skipped Samples (OOM failures)"] == 1
    assert captured["Global Level (Exact Pair Matching)"]["f1_micro"] == 1.0  # skipped record not counted
    ours_all = evaluate([{("A", "FM-1.1")}, {("A", "FM-1.1")}], [{("A", "FM-1.1")}, set()])
    assert ours_all["pair"]["micro_f1"] < 1.0  # scoring it as empty lowers the score: this is the difference
