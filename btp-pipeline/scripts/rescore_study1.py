"""
rescore_study1.py — Retroactive re-scoring of existing Study 1 logs.

Phase 2 Definition of Done requirement:
  Apply the new semantic_uncertainty / jaccard_uncertainty functions to the
  raw samples already stored in logs/trials.jsonl, and produce a before/after
  comparison table (lexical vs semantic per node, per fault type).

Since the old Reasoner samples were generated without the FINAL ANSWER: marker,
this script uses retroactive_extract_conclusion() (heuristic, not marker-based)
to extract a conclusion sentence from each old Reasoner sample.

Output:
  results/study1_rescore.jsonl   — per-trial rescored records
  results/study1_rescore_table.csv — aggregated before/after comparison table

Usage:
    python scripts/rescore_study1.py
    python scripts/rescore_study1.py --log-path logs/trials.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import LOG_DIR, RESULTS_DIR
from src.uncertainty import (
    retroactive_extract_conclusion,
    semantic_uncertainty,
    jaccard_uncertainty,
    parse_retrieved_items,
    per_item_inclusion_frequency,
)


# ---------------------------------------------------------------------------
# Rescore a single trial record
# ---------------------------------------------------------------------------

def rescore_record(record: Dict) -> Dict:
    """Compute new uncertainty metrics from saved samples in a trial record.

    Returns a new dict with additional keys:
      semantic_uncertainty_{node}   — for each node that has samples
      jaccard_uncertainty_retriever — if retriever returned multi-item outputs
      item_frequencies_retriever    — per-item inclusion rates
    """
    samples_by_node: Dict[str, List[str]] = record.get("samples", {})
    rescored: Dict = {
        "question": record.get("question", ""),
        "true_label": record.get("true_label", ""),
        "topology_id": record.get("topology_id", ""),
        "lexical_uncertainties": record.get("uncertainties", {}),
        "semantic_uncertainties": {},
        "jaccard_uncertainty_retriever": None,
        "item_frequencies_retriever": None,
    }

    for node_name, samples in samples_by_node.items():
        if not samples:
            continue

        if node_name == "reasoner":
            # Use retroactive heuristic extraction (no FINAL ANSWER: marker in old data).
            conclusions = [retroactive_extract_conclusion(s) for s in samples]
            rescored["semantic_uncertainties"][node_name] = semantic_uncertainty(conclusions)
            rescored[f"conclusions_{node_name}"] = conclusions

        elif node_name == "writer":
            rescored["semantic_uncertainties"][node_name] = semantic_uncertainty(samples)

        elif node_name == "retriever":
            rescored["semantic_uncertainties"][node_name] = semantic_uncertainty(samples)
            item_sets = [parse_retrieved_items(s) for s in samples]
            if any(len(s) > 1 for s in item_sets):
                rescored["jaccard_uncertainty_retriever"] = jaccard_uncertainty(item_sets)
                rescored["item_frequencies_retriever"] = per_item_inclusion_frequency(item_sets)

    return rescored


# ---------------------------------------------------------------------------
# Aggregate comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(rescored_records: List[Dict]) -> "pd.DataFrame":
    """Build a per-fault-type, per-node comparison table of lexical vs semantic."""
    import pandas as pd

    rows = []
    for rec in rescored_records:
        ft = rec.get("true_label", "unknown")
        for node in ["retriever", "reasoner", "writer"]:
            lex = rec.get("lexical_uncertainties", {}).get(node)
            sem = rec.get("semantic_uncertainties", {}).get(node)
            rows.append({
                "fault_type": ft,
                "node": node,
                "lexical_uncertainty": lex,
                "semantic_uncertainty": sem,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    table = (
        df.groupby(["fault_type", "node"])
        .agg(
            n_trials=("lexical_uncertainty", "count"),
            lexical_mean=("lexical_uncertainty", "mean"),
            semantic_mean=("semantic_uncertainty", "mean"),
        )
        .round(4)
    )
    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retroactively rescore Study 1 logs with new uncertainty metrics."
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=f"{LOG_DIR}/trials.jsonl",
        help="Path to the Study 1 trials JSONL log.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=RESULTS_DIR,
        help="Directory to write rescored output files.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        print(f"[rescore] Log file not found: {args.log_path}")
        sys.exit(1)

    print(f"[rescore] Loading records from '{args.log_path}'...")
    records = []
    with open(args.log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[rescore] Loaded {len(records)} records.")

    print("[rescore] Rescoring records (loading sentence-transformers if needed)...")
    rescored = [rescore_record(r) for r in records]

    os.makedirs(args.output_dir, exist_ok=True)

    # Write per-trial rescored JSONL
    rescore_path = os.path.join(args.output_dir, "study1_rescore.jsonl")
    with open(rescore_path, "w") as f:
        for r in rescored:
            f.write(json.dumps(r) + "\n")
    print(f"[rescore] Per-trial rescored records written to '{rescore_path}'.")

    # Write comparison table
    try:
        import pandas as pd
        table = build_comparison_table(rescored)
        table_path = os.path.join(args.output_dir, "study1_rescore_table.csv")
        table.to_csv(table_path)
        print(f"[rescore] Comparison table written to '{table_path}'.")
        print("\n=== Before/After Comparison (mean per fault_type × node) ===")
        print(table.to_string())
        print()
    except ImportError:
        print("[rescore] pandas not available — skipping comparison table.")


if __name__ == "__main__":
    main()
