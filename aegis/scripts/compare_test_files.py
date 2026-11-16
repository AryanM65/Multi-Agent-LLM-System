"""S2 part A: compare the HF test file (Aegis-Bench/test.jsonl) with the authors' repo file (evaluation/data/aegis.jsonl).

Question: which 600-record file could the paper's test numbers come from, and does it matter for our row?
Reports: record counts, id/query/conversation overlap between the two files, overlap of each file with HF train/val,
framework and mode mix, and label-set sizes. Run: python -m aegis.scripts.compare_test_files  (writes the JSON log)
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from ..src.loaders import PATHS, load_split

OUT = Path(__file__).resolve().parents[1] / "results" / "test_file_comparison.json"


def convo_hash(t) -> str:
    return hashlib.sha1("\n".join(f"{x.agent}\t{x.text}" for x in t.turns).encode()).hexdigest()


def describe(name: str, trajs: list) -> dict:
    fw = Counter(t.framework for t in trajs)
    modes = Counter(m for t in trajs for _, m in t.labels)
    return {
        "name": name, "n": len(trajs), "unique_queries": len({t.query for t in trajs}),
        "frameworks": dict(sorted(fw.items())), "mode_counts": dict(sorted(modes.items())),
        "mean_labels_per_record": sum(len(t.labels) for t in trajs) / len(trajs),
        "mean_turns": sum(len(t.turns) for t in trajs) / len(trajs),
    }


def overlap(a: list, b: list) -> dict:
    ids_a, ids_b = {t.id for t in a}, {t.id for t in b}
    q_a, q_b = {t.query for t in a}, {t.query for t in b}
    c_a, c_b = {convo_hash(t) for t in a}, {convo_hash(t) for t in b}
    return {"ids": len(ids_a & ids_b), "queries": len(q_a & q_b), "conversations": len(c_a & c_b),
            "records_of_A_with_query_in_B": sum(t.query in q_b for t in a)}


def main() -> None:
    hf, repo = load_split("test"), load_split("test_repo")
    train, val = load_split("train"), load_split("val")
    res = {
        "hf_test": describe("HF Aegis-Bench/test.jsonl", hf),
        "repo_test": describe("repo evaluation/data/aegis.jsonl", repo),
        "hf_vs_repo": overlap(hf, repo),
        "hf_test_vs_train": overlap(hf, train), "hf_test_vs_val": overlap(hf, val),
        "repo_test_vs_train": overlap(repo, train), "repo_test_vs_val": overlap(repo, val),
        "repo_file_sha1": hashlib.sha1(PATHS["test_repo"].read_bytes()).hexdigest(),
        "hf_file_sha1": hashlib.sha1(PATHS["test"].read_bytes()).hexdigest(),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
