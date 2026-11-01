import pytest

from aegis.src.metrics import avg12, evaluate, prf


def P(*x):
    return set(x)


def test_perfect():
    t = [P(("A", "FM-1.1"), ("B", "FM-2.2")), P(("A", "FM-3.1"))]
    r = evaluate(t, t)
    for lvl in ("pair", "agent", "error"):
        assert r[lvl]["micro_f1"] == 1.0 and r[lvl]["macro_f1"] == 1.0
    assert r["exact_match"] == 1.0


def test_empty_pred():
    r = evaluate([P(("A", "FM-1.1"))], [set()])
    assert r["pair"]["micro_f1"] == 0.0 and r["agent"]["macro_f1"] == 0.0


def test_partial_hand_computed():
    t = [P(("A", "FM-1.1"), ("B", "FM-2.2"))]
    p = [P(("A", "FM-1.1"), ("C", "FM-2.2"))]
    r = evaluate(t, p)
    assert r["pair"]["micro_p"] == 0.5 and r["pair"]["micro_r"] == 0.5
    assert r["agent"]["macro_f1"] == pytest.approx(1 / 3)   # classes A,B,C -> 1,0,0
    assert r["agent"]["micro_f1"] == pytest.approx(0.5)
    assert r["error"]["micro_f1"] == 1.0                    # modes right, agent ignored


def test_macro_over_union_classes():
    r = prf([{"x"}, {"y"}], [{"x"}, {"z"}])
    assert r["macro_f1"] == pytest.approx(1 / 3)


def test_avg12():
    perfect = evaluate([P(("A", "FM-1.1"))], [P(("A", "FM-1.1"))])
    assert avg12(perfect, perfect) == pytest.approx(100.0)
