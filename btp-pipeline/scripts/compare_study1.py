"""
compare_study1.py -- Before vs After comparison for Study 1.

Compares the OLD Study 1 results (original lexical-only pipeline, 45 trials)
against the NEW Study 1 results (Ollama + semantic uncertainty + per-node
thresholds).

Generates:
  results/study1_comparison.csv  -- per-fault-type metrics table
  results/study1_comparison.md   -- human-readable Markdown report

Usage:
    python scripts/compare_study1.py \\
        --old-log logs/trials.jsonl \\
        --new-log logs/trials_new.jsonl \\
        --output-dir results/

Run after both studies are complete.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


FAULT_LABELS = ["noise", "contamination", "ceiling"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_trials(path: str) -> List[Dict]:
    """Load JSONL trial records, skipping skip-records and clean controls."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("_is_skip"):
                continue
            records.append(r)
    return records


def accuracy_metrics(records: List[Dict]) -> Dict:
    """Compute per-fault-type and overall accuracy for a set of records."""
    fault_records = [r for r in records if r.get("true_label") in FAULT_LABELS]
    total = len(fault_records)
    diagnosed = [r for r in fault_records
                 if r.get("diagnosed_label") not in (None, "no_fault_detected")]
    correct = [r for r in diagnosed
               if r.get("diagnosed_label") == r.get("true_label")]

    per_type = {}
    for ft in FAULT_LABELS:
        ft_recs = [r for r in fault_records if r.get("true_label") == ft]
        ft_diag = [r for r in ft_recs
                   if r.get("diagnosed_label") not in (None, "no_fault_detected")]
        ft_correct = [r for r in ft_diag
                      if r.get("diagnosed_label") == r.get("true_label")]
        detection_rate = len(ft_diag) / len(ft_recs) if ft_recs else 0.0
        accuracy = len(ft_correct) / len(ft_diag) if ft_diag else 0.0
        per_type[ft] = {
            "total": len(ft_recs),
            "detected": len(ft_diag),
            "correct": len(ft_correct),
            "detection_rate": detection_rate,
            "accuracy": accuracy,
        }

    overall_detection = len(diagnosed) / total if total else 0.0
    overall_accuracy = len(correct) / len(diagnosed) if diagnosed else 0.0
    return {
        "total_fault_trials": total,
        "total_detected": len(diagnosed),
        "total_correct": len(correct),
        "overall_detection_rate": overall_detection,
        "overall_accuracy": overall_accuracy,
        "per_type": per_type,
    }


def uncertainty_stats(records: List[Dict]) -> Dict:
    """Compute mean uncertainty per node across fault types.

    Pulls 'uncertainties' (lexical) and 'semantic_uncertainties' (semantic)
    from records. Falls back gracefully if either field is absent.
    """
    node_lex: Dict[str, List[float]] = defaultdict(list)
    node_sem: Dict[str, List[float]] = defaultdict(list)

    for r in records:
        for node, val in (r.get("uncertainties") or {}).items():
            if val is not None:
                node_lex[node].append(float(val))
        for node, val in (r.get("semantic_uncertainties") or {}).items():
            if val is not None:
                node_sem[node].append(float(val))

    result = {}
    all_nodes = sorted(set(list(node_lex) + list(node_sem)))
    for node in all_nodes:
        result[node] = {
            "lexical_mean": round(sum(node_lex[node]) / len(node_lex[node]), 4)
            if node_lex[node] else None,
            "semantic_mean": round(sum(node_sem[node]) / len(node_sem[node]), 4)
            if node_sem[node] else None,
        }
    return result


def confusion_matrix(records: List[Dict]) -> Dict:
    """Return a 3x3 dict of {true: {diagnosed: count}}."""
    cm = {r: {c: 0 for c in FAULT_LABELS} for r in FAULT_LABELS}
    for rec in records:
        tl = rec.get("true_label")
        dl = rec.get("diagnosed_label")
        if tl in cm and dl in cm.get(tl, {}):
            cm[tl][dl] += 1
    return cm


def print_cm(cm: Dict, title: str) -> str:
    """Render a confusion matrix as a Markdown table string."""
    lines = [f"### {title}", ""]
    col_w = 16
    header = f"| {'true \\ diagnosed':>20} |" + "".join(f" {c:>{col_w}} |" for c in FAULT_LABELS)
    sep = f"|{'-'*22}|" + "".join(f"{'-'*(col_w+2)}|" for _ in FAULT_LABELS)
    lines += [header, sep]
    for r in FAULT_LABELS:
        row = f"| {r:>20} |" + "".join(f" {cm[r][c]:>{col_w}d} |" for c in FAULT_LABELS)
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare(old_path: str, new_path: str, output_dir: str) -> None:
    print(f"[compare] Loading OLD log: {old_path}")
    old_recs = load_trials(old_path)
    print(f"[compare] Loading NEW log: {new_path}")
    new_recs = load_trials(new_path)

    old_fault = [r for r in old_recs if r.get("true_label") in FAULT_LABELS]
    new_fault = [r for r in new_recs if r.get("true_label") in FAULT_LABELS]

    old_m = accuracy_metrics(old_recs)
    new_m = accuracy_metrics(new_recs)
    old_u = uncertainty_stats(old_fault)
    new_u = uncertainty_stats(new_fault)
    old_cm = confusion_matrix([r for r in old_fault
                                if r.get("diagnosed_label") not in (None, "no_fault_detected")])
    new_cm = confusion_matrix([r for r in new_fault
                                if r.get("diagnosed_label") not in (None, "no_fault_detected")])

    os.makedirs(output_dir, exist_ok=True)

    # --- CSV summary ---
    csv_path = os.path.join(output_dir, "study1_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "old", "new", "delta"])

        def row(name, old_v, new_v):
            delta = round(new_v - old_v, 4) if isinstance(old_v, float) else ""
            w.writerow([name, round(old_v, 4) if isinstance(old_v, float) else old_v,
                        round(new_v, 4) if isinstance(new_v, float) else new_v, delta])

        row("total_fault_trials", old_m["total_fault_trials"], new_m["total_fault_trials"])
        row("overall_detection_rate", old_m["overall_detection_rate"], new_m["overall_detection_rate"])
        row("overall_accuracy", old_m["overall_accuracy"], new_m["overall_accuracy"])
        for ft in FAULT_LABELS:
            row(f"{ft}_detection_rate",
                old_m["per_type"][ft]["detection_rate"],
                new_m["per_type"][ft]["detection_rate"])
            row(f"{ft}_accuracy",
                old_m["per_type"][ft]["accuracy"],
                new_m["per_type"][ft]["accuracy"])

    print(f"[compare] CSV saved: {csv_path}")

    # --- Markdown report ---
    md_lines = [
        "# Study 1: Before vs After Comparison Report",
        "",
        f"- **Old log**: `{old_path}` ({len(old_fault)} fault trials)",
        f"- **New log**: `{new_path}` ({len(new_fault)} fault trials)",
        "",
        "---",
        "",
        "## 1. Overall Diagnostic Performance",
        "",
        f"| Metric | OLD (lexical) | NEW (semantic) | Delta |",
        f"|--------|--------------|----------------|-------|",
        f"| Fault trials | {old_m['total_fault_trials']} | {new_m['total_fault_trials']} | - |",
        f"| Detection rate | {old_m['overall_detection_rate']:.1%} | {new_m['overall_detection_rate']:.1%} | {new_m['overall_detection_rate']-old_m['overall_detection_rate']:+.1%} |",
        f"| Accuracy (of detected) | {old_m['overall_accuracy']:.1%} | {new_m['overall_accuracy']:.1%} | {new_m['overall_accuracy']-old_m['overall_accuracy']:+.1%} |",
        "",
        "## 2. Per Fault Type",
        "",
        "| Fault Type | OLD detect% | NEW detect% | OLD acc% | NEW acc% |",
        "|------------|-------------|-------------|----------|----------|",
    ]
    for ft in FAULT_LABELS:
        op = old_m["per_type"][ft]
        np_ = new_m["per_type"][ft]
        md_lines.append(
            f"| {ft} | {op['detection_rate']:.1%} | {np_['detection_rate']:.1%} "
            f"| {op['accuracy']:.1%} | {np_['accuracy']:.1%} |"
        )

    md_lines += [
        "",
        "## 3. Uncertainty Statistics (fault trials only)",
        "",
        "| Node | OLD lex mean | NEW lex mean | OLD sem mean | NEW sem mean |",
        "|------|-------------|-------------|-------------|-------------|",
    ]
    all_nodes = sorted(set(list(old_u) + list(new_u)))
    for node in all_nodes:
        ou = old_u.get(node, {})
        nu = new_u.get(node, {})
        md_lines.append(
            f"| {node} | {ou.get('lexical_mean', 'N/A')} | {nu.get('lexical_mean', 'N/A')}"
            f" | {ou.get('semantic_mean', 'N/A')} | {nu.get('semantic_mean', 'N/A')} |"
        )

    md_lines += [
        "",
        "## 4. Confusion Matrices",
        "",
        print_cm(old_cm, "OLD (lexical pipeline)"),
        "",
        print_cm(new_cm, "NEW (semantic + per-node thresholds)"),
        "",
        "---",
        "",
        "> Generated by `scripts/compare_study1.py`",
    ]

    md_path = os.path.join(output_dir, "study1_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[compare] Markdown report saved: {md_path}")

    # Print summary to terminal
    print()
    print("=" * 60)
    print("STUDY 1 BEFORE vs AFTER SUMMARY")
    print("=" * 60)
    print(f"{'':25s} {'OLD':>12s} {'NEW':>12s} {'DELTA':>10s}")
    print("-" * 60)
    print(f"{'Overall detection rate':25s} "
          f"{old_m['overall_detection_rate']:>12.1%} "
          f"{new_m['overall_detection_rate']:>12.1%} "
          f"{new_m['overall_detection_rate']-old_m['overall_detection_rate']:>+10.1%}")
    print(f"{'Overall accuracy':25s} "
          f"{old_m['overall_accuracy']:>12.1%} "
          f"{new_m['overall_accuracy']:>12.1%} "
          f"{new_m['overall_accuracy']-old_m['overall_accuracy']:>+10.1%}")
    for ft in FAULT_LABELS:
        op = old_m["per_type"][ft]
        np_ = new_m["per_type"][ft]
        print(f"  {ft+' detect%':23s} "
              f"{op['detection_rate']:>12.1%} "
              f"{np_['detection_rate']:>12.1%} "
              f"{np_['detection_rate']-op['detection_rate']:>+10.1%}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compare Study 1 OLD vs NEW results.")
    parser.add_argument("--old-log", default="logs/trials.jsonl")
    parser.add_argument("--new-log", default="logs/trials_new.jsonl")
    parser.add_argument("--output-dir", default="results/")
    args = parser.parse_args()
    compare(args.old_log, args.new_log, args.output_dir)


if __name__ == "__main__":
    main()
