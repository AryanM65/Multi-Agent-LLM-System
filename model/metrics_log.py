"""Append-only run history: every training run's config + resulting metrics
gets appended as one line to model/metrics_history.jsonl.

This is what lets us answer "did that last tweak actually help?" without
relying on memory of terminal scrollback -- every run is on record, in order,
so results across experiments (different hyperparams, dataset versions,
architecture changes) can be compared side by side later.
"""
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = REPO_ROOT / "model" / "metrics_history.jsonl"


def log_run(config: dict, metrics: dict, notes: str = "", path=DEFAULT_LOG_PATH):
    """
    config: hyperparameters / run setup, e.g.
        {"hidden_dim": 32, "num_layers": 2, "dropout": 0.2, "lr": 1e-3,
         "epochs_requested": 150, "patience": 30, "seed": 0}
    metrics: per-split results, e.g.
        {"train": {"top1_accuracy": 0.30, "top2_accuracy": 0.48, "n": 251},
         "val":   {"top1_accuracy": 0.27, "top2_accuracy": 0.41, "n": 78},
         "ood":   {"top1_accuracy": 0.22, "top2_accuracy": 0.35, "n": 164},
         "baseline_ood_top1_accuracy": 0.159,
         "best_epoch": 12, "epochs_run": 43}
    """
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "metrics": metrics,
        "notes": notes,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_history(path=DEFAULT_LOG_PATH):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def print_history(path=DEFAULT_LOG_PATH):
    history = load_history(path)
    if not history:
        print("No runs logged yet.")
        return

    print(f"{'#':<3} {'timestamp':<20} {'train_top1':>10} {'val_top1':>9} {'ood_top1':>9} {'ood_top2':>9} {'baseline_ood':>13} {'best_ep':>8}")
    for i, r in enumerate(history):
        m = r["metrics"]
        print(f"{i:<3} {r['timestamp']:<20} "
              f"{m.get('train', {}).get('top1_accuracy', float('nan')):>10.3f} "
              f"{m.get('val', {}).get('top1_accuracy', float('nan')):>9.3f} "
              f"{m.get('ood', {}).get('top1_accuracy', float('nan')):>9.3f} "
              f"{m.get('ood', {}).get('top2_accuracy', float('nan')):>9.3f} "
              f"{m.get('baseline_ood_top1_accuracy', float('nan')):>13.3f} "
              f"{m.get('best_epoch', -1):>8}")


if __name__ == "__main__":
    print_history()
