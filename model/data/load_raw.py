"""Load dataset/trials.jsonl + dataset/topology_pool.json and sanity-check them.

Repo-root-relative paths are used throughout this package (run scripts from
the project root: `D:\\aryan\\Projects\\Multi agent LLM System`).
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOPOLOGY_PATH = REPO_ROOT / "dataset" / "topology_pool.json"
DEFAULT_TRIALS_PATH = REPO_ROOT / "dataset" / "trials.jsonl"


def load_topologies(path=DEFAULT_TOPOLOGY_PATH):
    topos = json.load(open(path, encoding="utf-8"))
    return {t["topology_id"]: t for t in topos}


def load_trials(path=DEFAULT_TRIALS_PATH):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def validate(trials, topologies):
    """Re-runs the checks done by hand on 2026-09-26. Raises AssertionError on
    any regression (e.g. if the dataset file gets regenerated/edited later)."""
    errors = []

    for i, t in enumerate(trials):
        tid = t.get("topology_id")
        if tid not in topologies:
            errors.append(f"trial {i}: unknown topology_id {tid!r}")
            continue

        expected_nodes = set(topologies[tid]["nodes"].keys())
        have_uncertainty = set(t.get("uncertainties", {}).keys())
        have_samples = set(t.get("samples", {}).keys())

        if expected_nodes - have_uncertainty:
            errors.append(f"trial {i} ({tid}): missing uncertainties for {expected_nodes - have_uncertainty}")
        if expected_nodes - have_samples:
            errors.append(f"trial {i} ({tid}): missing samples for {expected_nodes - have_samples}")

        for node, samples in t.get("samples", {}).items():
            if len(samples) < 1:
                errors.append(f"trial {i} ({tid}): node {node} has empty samples list")

        fault_config = t.get("fault_config") or {}
        target = fault_config.get("target_node")
        if target is not None and target not in expected_nodes:
            errors.append(f"trial {i} ({tid}): fault target {target!r} not in topology's nodes")

    if errors:
        raise AssertionError(f"{len(errors)} validation error(s), first 10:\n" + "\n".join(errors[:10]))

    return True


def summary(trials, topologies):
    from collections import Counter

    split_counts = Counter(topologies[t["topology_id"]]["split"] for t in trials)
    label_counts = Counter(t["true_label"] for t in trials)
    topo_counts = Counter(t["topology_id"] for t in trials)

    return {
        "total_trials": len(trials),
        "total_topologies_referenced": len(topo_counts),
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "trials_per_topology": dict(topo_counts),
    }


if __name__ == "__main__":
    topologies = load_topologies()
    trials = load_trials()
    validate(trials, topologies)
    print(f"Loaded and validated {len(trials)} trials across {len(topologies)} topologies. All checks passed.")
    for k, v in summary(trials, topologies).items():
        print(f"  {k}: {v}")
