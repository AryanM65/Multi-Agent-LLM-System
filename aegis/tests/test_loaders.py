import pytest

from aegis.src.loaders import PATHS, assert_no_query_overlap, load_grouped_splits, load_split, parse_record
from aegis.src.schema import normalize_mode

needs_data = pytest.mark.skipif(not PATHS["test"].exists(), reason="dataset not downloaded")


def test_normalize_mode():
    assert normalize_mode("EM-1.1") == "FM-1.1"
    assert normalize_mode(" fm_3.3 ") == "FM-3.3"
    with pytest.raises(ValueError):
        normalize_mode("FM-9.9")


def test_parse_record_drops_leaky_fields():
    rec = {
        "id": "x", "metadata": {"framework": "macnet", "benchmark": "GSM8K", "num_injected_agents": 2},
        "input": {"query": "q", "conversation_history": [
            {"step": 1, "agent_name": " Node0 ", "content": None, "phase": "reasoning", "is_injected": True}]},
        "output": {"faulty_agents": [{"agent_name": "Node0", "error_type": "EM-2.1", "injection_strategy": "s"}]},
        "ground_truth": {"injected_agents": [1]},
    }
    t = parse_record(rec, "train")
    assert t.turns[0].agent == "Node0" and t.turns[0].text == ""
    assert t.labels == {("Node0", "FM-2.1")}
    assert not hasattr(t.turns[0], "is_injected")


@needs_data
def test_counts_and_grouped_split():
    assert len(load_split("test")) == 600
    g = load_grouped_splits()
    assert_no_query_overlap(g)
    assert len(g["train"]) > 0 and len(g["val"]) > 0
