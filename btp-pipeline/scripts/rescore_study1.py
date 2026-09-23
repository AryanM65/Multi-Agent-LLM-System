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
  results/study1_rescore.jsonl      — per-trial rescored records
  results/study1_rescore_table.csv  — aggregated before/after comparison table

Usage:
    python scripts/rescore_study1.py
    python scripts/rescore_study1.py --log-path logs/trials.jsonl
    python scripts/rescore_study1.py --auto   # called by run_study1.py --auto-rescore
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure") and sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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

def build_comparison_table(rescored_records: List[Dict]) -> List[Dict]:
    """Build a per-fault-type, per-node comparison table of lexical vs semantic."""
    from collections import defaultdict

    groups = defaultdict(lambda: {"lex_vals": [], "sem_vals": []})
    for rec in rescored_records:
        ft = rec.get("true_label", "unknown")
        for node in ["retriever", "reasoner", "writer"]:
            lex = rec.get("lexical_uncertainties", {}).get(node)
            sem = rec.get("semantic_uncertainties", {}).get(node)
            if lex is not None and lex == lex:
                groups[(ft, node)]["lex_vals"].append(float(lex))
            if sem is not None and sem == sem:
                groups[(ft, node)]["sem_vals"].append(float(sem))

    table_rows = []
    for (ft, node), data in sorted(groups.items()):
        lex_v = data["lex_vals"]
        sem_v = data["sem_vals"]
        n = max(len(lex_v), len(sem_v))
        lex_mean = sum(lex_v) / len(lex_v) if lex_v else float("nan")
        sem_mean = sum(sem_v) / len(sem_v) if sem_v else float("nan")
        table_rows.append({
            "fault_type": ft,
            "node": node,
            "n_trials": n,
            "lexical_mean": round(lex_mean, 4) if lex_mean == lex_mean else None,
            "semantic_mean": round(sem_mean, 4) if sem_mean == sem_mean else None,
        })
    return table_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_rich_table(table_rows: List[Dict]) -> None:
    """Print a colour-highlighted before/after comparison to the terminal."""
    BOLD  = "\033[1m"
    GREEN = "\033[92m"
    CYAN  = "\033[96m"
    RESET = "\033[0m"
    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  Before / After: Lexical vs Semantic Uncertainty (mean per group){RESET}")
    print(f"{BOLD}{'='*72}{RESET}")
    print(f"  {'Fault Type':<15} {'Node':<14} {'n':>4}  "
          f"{CYAN}{'Lex (old)':>10}{RESET}  {GREEN}{'Sem (new)':>10}{RESET}  {'Δ':>8}")
    print(f"  {'-'*15} {'-'*14} {'-'*4}  {'-'*10}  {'-'*10}  {'-'*8}")
    for row in table_rows:
        ft  = row["fault_type"]
        node= row["node"]
        n   = row.get("n_trials", 0)
        lex = row.get("lexical_mean")
        sem = row.get("semantic_mean")
        delta_str = f"{(sem - lex):+.4f}" if (lex is not None and sem is not None) else "   N/A  "
        lex_str   = f"{lex:>10.4f}" if lex is not None else "   N/A  "
        sem_str   = f"{sem:>10.4f}" if sem is not None else "   N/A  "
        print(f"  {ft:<15} {node:<14} {n:>4}  "
              f"{CYAN}{lex_str}{RESET}  {GREEN}{sem_str}{RESET}  {delta_str:>8}")
    print(f"{BOLD}{'='*72}{RESET}\n")


def main(log_path: str = None, output_dir: str = None) -> None:
    """Entry point — also callable programmatically from run_study1.py."""
    import csv
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
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Non-interactive mode (called from run_study1.py --auto-rescore).",
    )
    args = parser.parse_args()
    # Programmatic overrides (when called from run_study1.py)
    if log_path:    args.log_path    = log_path
    if output_dir:  args.output_dir  = output_dir

    if not os.path.exists(args.log_path):
        print(f"[rescore] Log file not found: {args.log_path}")
        sys.exit(1)

    print(f"[rescore] Loading records from '{args.log_path}'...")
    records = []
    with open(args.log_path, encoding="utf-8") as f:
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
    with open(rescore_path, "w", encoding="utf-8") as f:
        for r in rescored:
            f.write(json.dumps(r) + "\n")
    print(f"[rescore] Per-trial rescored records written to '{rescore_path}'.")

    # Write comparison table
    table_rows = build_comparison_table(rescored)
    table_path = os.path.join(args.output_dir, "study1_rescore_table.csv")
    if table_rows:
        with open(table_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["fault_type", "node", "n_trials", "lexical_mean", "semantic_mean"])
            writer.writeheader()
            writer.writerows(table_rows)
        print(f"[rescore] Comparison table written to '{table_path}'.")
        _print_rich_table(table_rows)


if __name__ == "__main__":
    main()
