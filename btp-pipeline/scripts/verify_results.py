"""verify_results.py — recompute detection/accuracy numbers straight from a raw
trials JSONL log. Never trust a written results summary; always recompute.

This project's history includes multiple prose "results summaries" that did not
match their own underlying logs when checked. This script is the antidote: point
it at a trials log (and optionally its matching skip log) and it prints the true
numbers, computed fresh, every time.

Usage:
    python scripts/verify_results.py logs/trials_new.jsonl
    python scripts/verify_results.py logs/trials_new.jsonl --skip-log logs/trials_new_skip.jsonl
    python scripts/verify_results.py logs/pilot_k5_fix.jsonl --exclude-fallback
        # re-run excluding any trial where a node's used_thinking_fallback flag
        # fired for any sample, to check whether accuracy recovers once
        # fallback-degraded (truncated-reasoning) samples are removed.
"""

import argparse
import json
from collections import Counter, defaultdict


def _load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _trial_used_fallback(r: dict) -> bool:
    """True if any node in this trial had any sample flagged used_thinking_fallback."""
    fallback_map = r.get("used_thinking_fallback") or {}
    return any(any(flags) for flags in fallback_map.values())


def _fallback_sample_lengths(r: dict) -> list:
    """Word counts of every sample flagged used_thinking_fallback in this trial.

    No separate length field is logged — a fallback sample's text is already
    in `samples` at the same index as its True flag in `used_thinking_fallback`,
    so quality can be judged post-hoc from data already collected instead of
    adding a redundant schema field. A binary flag alone can't distinguish a
    3-character fragment ("Who") from a longer, plausibly-usable sentence;
    this gives that distinction for free.
    """
    lengths = []
    fallback_map = r.get("used_thinking_fallback") or {}
    samples_map = r.get("samples") or {}
    for node, flags in fallback_map.items():
        node_samples = samples_map.get(node, [])
        for i, flagged in enumerate(flags):
            if flagged and i < len(node_samples):
                lengths.append(len(node_samples[i].split()))
    return lengths


def verify_results(log_path: str, skip_log_path: str = None, exclude_fallback: bool = False):
    all_records = _load_jsonl(log_path)
    trial_records = [r for r in all_records if not r.get("_is_skip")]

    fallback_count = sum(1 for r in trial_records if _trial_used_fallback(r))
    if trial_records:
        print(f"Trials using thinking-fallback (any node/sample): {fallback_count}/{len(trial_records)} "
              f"({100 * fallback_count / len(trial_records):.1f}%)")

        all_fallback_lengths = [ln for r in trial_records for ln in _fallback_sample_lengths(r)]
        if all_fallback_lengths:
            all_fallback_lengths.sort()
            n = len(all_fallback_lengths)
            median = all_fallback_lengths[n // 2]
            very_short = sum(1 for ln in all_fallback_lengths if ln <= 3)
            print(f"  Fallback sample word counts: min={all_fallback_lengths[0]} "
                  f"median={median} max={all_fallback_lengths[-1]}  "
                  f"({very_short}/{n} are <=3 words — likely unusable fragments, not just 'binary flag' noise)")

    if exclude_fallback:
        removed = fallback_count
        trial_records = [r for r in trial_records if not _trial_used_fallback(r)]
        print(f"--exclude-fallback: removed {removed} fallback-contaminated trials, "
              f"{len(trial_records)} remain.\n")

    control = [r for r in trial_records if r.get("is_control")]
    faults = [r for r in trial_records if not r.get("is_control")]

    print("=" * 60)
    print(f"RAW LOG: {log_path}" + (" (fallback-excluded)" if exclude_fallback else ""))
    print("=" * 60)
    print(f"Total records in log:      {len(all_records)}")
    print(f"  Control (clean) trials:  {len(control)}")
    print(f"  Fault trials:            {len(faults)}")

    if skip_log_path:
        skips = _load_jsonl(skip_log_path)
        reasons = Counter(r.get("reason", "unknown") for r in skips)
        print(f"  Skipped trials (separate log): {len(skips)}")
        for reason, n in sorted(reasons.items()):
            print(f"    {reason}: {n}")
        total_attempted = len(trial_records) + len(skips)
        print(f"Total attempted (trials + skips): {total_attempted}")

    if not faults:
        print("\nNo fault trials found — nothing further to compute.")
        return

    diagnosed = [r for r in faults if r.get("diagnosed_label") not in (None, "no_fault_detected")]
    print()
    print(f"Fault trials:              {len(faults)}")
    print(f"Diagnosed (non-null):      {len(diagnosed)} "
          f"({100 * len(diagnosed) / len(faults):.1f}%)")

    correct = sum(1 for r in diagnosed if r["diagnosed_label"] == r["true_label"])
    if diagnosed:
        print(f"Correct among diagnosed:   {correct}/{len(diagnosed)} "
              f"({100 * correct / len(diagnosed):.1f}%)")
    else:
        print("Correct among diagnosed:   n/a (nothing diagnosed)")

    print()
    print("Per-fault-type breakdown (of fault trials only):")
    by_fault_type = defaultdict(lambda: {"total": 0, "detected": 0, "correct": 0})
    for r in faults:
        ft = r["true_label"]
        by_fault_type[ft]["total"] += 1
        if r.get("diagnosed_label") not in (None, "no_fault_detected"):
            by_fault_type[ft]["detected"] += 1
            if r["diagnosed_label"] == ft:
                by_fault_type[ft]["correct"] += 1
    for ft, stats in sorted(by_fault_type.items()):
        det_pct = 100 * stats["detected"] / stats["total"] if stats["total"] else 0.0
        print(f"  {ft:15s} {stats['detected']}/{stats['total']} detected ({det_pct:.1f}%), "
              f"{stats['correct']}/{stats['total']} correct")

    print()
    print("Confusion matrix (true_label, diagnosed_label) -> count:")
    matrix = Counter((r["true_label"], r.get("diagnosed_label") or "no_fault_detected") for r in faults)
    for k, v in sorted(matrix.items()):
        print(f"  {k}: {v}")

    print()
    print("Sanity checks:")
    empty_retriever_samples = sum(
        1 for r in trial_records
        for s in r.get("samples", {}).get("retriever", [])
        if s == ""
    )
    print(f"  Empty retriever samples across all trials: {empty_retriever_samples}"
          + ("  <-- non-zero: uncertainty metrics for affected trials may be corrupted"
             if empty_retriever_samples else ""))

    if len(diagnosed) < 20:
        print(f"  NOTE: only {len(diagnosed)} diagnosed trials — treat percentages above as a "
              f"small pilot, not a conclusive result (see plan Section 1.3).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", help="Path to the trials JSONL log to verify.")
    parser.add_argument("--skip-log", dest="skip_log_path", default=None,
                         help="Optional path to the matching skipped-trials JSONL log.")
    parser.add_argument("--exclude-fallback", action="store_true",
                         help="Exclude trials where any node used the thinking-field "
                              "fallback for any sample (see src/nodes.py:_ollama_generate).")
    args = parser.parse_args()
    verify_results(args.log_path, args.skip_log_path, args.exclude_fallback)


if __name__ == "__main__":
    main()
