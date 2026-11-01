from aegis.src.eval.score_zeroshot import parse_prediction


def test_plain_json():
    p, ok = parse_prediction('{"faulty_agents": [{"agent_name": "A", "error_type": "FM-1.1"}]}')
    assert ok and p == {("A", "FM-1.1")}


def test_fenced_and_prefixed():
    raw = 'Analysis...\n```json\n{"faulty_agents": [{"agent_name": " B ", "error_type": "EM-2.3"}]}\n```'
    p, ok = parse_prediction(raw)
    assert ok and p == {("B", "FM-2.3")}


def test_nested_final_answer():
    p, ok = parse_prediction('{"Final Answer": {"faulty_agents": [{"agent_name": "C", "error_type": "FM-3.1"}]}}')
    assert ok and p == {("C", "FM-3.1")}


def test_empty_list_is_valid_and_garbage_is_not():
    assert parse_prediction('{"faulty_agents": []}') == (set(), True)
    assert parse_prediction("I think agent A is wrong") == (set(), False)
