"""Load Aegis-Bench / Who&When into Trajectory objects and build splits.

Splits
- "paper":   as released on HF (train 7146 / val 1787 / test 600). Test queries overlap train
             (see docs/dataset_audit.md), so scores are optimistic.
- "grouped": same test set, but every train/val record sharing a query with a test record is
             dropped, and train/val is re-split by query (80/20, seeded). Report both.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .schema import Trajectory, Turn, normalize_mode

AEGIS_ROOT = Path(__file__).resolve().parents[1]
RAW = AEGIS_ROOT / "data" / "raw"
WHOWHEN_PATH = AEGIS_ROOT / "third_party" / "AEGIS" / "evaluation" / "data" / "whowhen.jsonl"

PATHS = {
    "train": RAW / "Aegis-Training" / "train.jsonl",
    "val": RAW / "Aegis-Training" / "val.jsonl",
    "test": RAW / "Aegis-Bench" / "test.jsonl",
    "who_when": WHOWHEN_PATH,
}


def parse_record(rec: dict, split: str) -> Trajectory:
    """Build a Trajectory. Only whitelisted fields are read (no is_injected / ground_truth)."""
    turns = [
        Turn(step=int(t["step"]), agent=str(t["agent_name"]).strip(), text=t.get("content") or "")
        for t in rec["input"]["conversation_history"]
    ]
    labels = frozenset(
        (str(a["agent_name"]).strip(), normalize_mode(a["error_type"]))
        for a in rec["output"]["faulty_agents"]
    )
    md = rec["metadata"]
    return Trajectory(
        id=rec["id"], split=split, framework=md["framework"], benchmark=md["benchmark"],
        query=rec["input"]["query"], turns=turns, labels=labels,
    )


def load_split(split: str) -> list:
    out = []
    with PATHS[split].open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(parse_record(json.loads(line), split))
    return out


def load_paper_splits() -> dict:
    return {s: load_split(s) for s in ("train", "val", "test")}


def load_grouped_splits(seed: int = 0, val_frac: float = 0.2) -> dict:
    """Leakage-safe splits: no query shared between train, val and test."""
    paper = load_paper_splits()
    test = paper["test"]
    test_queries = {t.query for t in test}
    pool = [t for t in paper["train"] + paper["val"] if t.query not in test_queries]
    queries = sorted({t.query for t in pool})
    rng = random.Random(seed)
    rng.shuffle(queries)
    val_q = set(queries[: int(len(queries) * val_frac)])
    return {
        "train": [t for t in pool if t.query not in val_q],
        "val": [t for t in pool if t.query in val_q],
        "test": test,
    }


def assert_no_query_overlap(splits: dict) -> None:
    names = list(splits)
    sets = {n: {t.query for t in splits[n]} for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ov = sets[a] & sets[b]
            assert not ov, f"{len(ov)} queries shared between {a} and {b}"


if __name__ == "__main__":
    p = load_paper_splits()
    print({k: len(v) for k, v in p.items()})
    g = load_grouped_splits()
    print("grouped:", {k: len(v) for k, v in g.items()})
    assert_no_query_overlap(g)
    print("grouped split: zero query overlap OK")
    print("who_when:", len(load_split("who_when")))
