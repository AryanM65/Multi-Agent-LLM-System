"""test_uncertainty.py — unit tests for uncertainty metrics.

Run: python -m tests.test_uncertainty  (from btp-pipeline/)
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.uncertainty import jaccard_uncertainty, per_item_inclusion_frequency


def test_jaccard_uncertainty_worked_example():
    """Master plan's worked example: {a,b},{a,b,c},{a,d},{a,b,d} -> jaccard ~0.49."""
    item_sets = [["a", "b"], ["a", "b", "c"], ["a", "d"], ["a", "b", "d"]]
    result = jaccard_uncertainty(item_sets)
    assert abs(result - 0.49) < 0.02, f"expected ~0.49, got {result}"


def test_jaccard_uncertainty_identical_sets():
    item_sets = [["a", "b"], ["a", "b"], ["a", "b"]]
    assert jaccard_uncertainty(item_sets) == 0.0


def test_jaccard_uncertainty_disjoint_sets():
    item_sets = [["a"], ["b"]]
    assert jaccard_uncertainty(item_sets) == 1.0


def test_jaccard_uncertainty_single_sample():
    assert jaccard_uncertainty([["a", "b"]]) == 0.0


def test_per_item_inclusion_frequency():
    item_sets = [["a", "b"], ["a", "b", "c"], ["a", "d"], ["a", "b", "d"]]
    freq = per_item_inclusion_frequency(item_sets)
    assert freq["a"] == 1.0
    assert freq["b"] == 0.75
    assert freq["c"] == 0.25
    assert freq["d"] == 0.5


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
