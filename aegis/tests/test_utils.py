import json

import numpy as np

from aegis.src.utils import log_run, set_seed


def test_set_seed_reproducible():
    set_seed(3)
    a = np.random.rand(3)
    set_seed(3)
    b = np.random.rand(3)
    assert (a == b).all()


def test_log_run_appends(tmp_path):
    p = tmp_path / "h.jsonl"
    log_run(phase="0", model="t", config={"a": 1}, seed=1, split="val", metrics={}, path=p)
    log_run(phase="0", model="t", config={"a": 1}, seed=2, split="val", metrics={}, path=p)
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["config_hash"] == json.loads(lines[1])["config_hash"]
