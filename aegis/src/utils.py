"""Shared helpers: seeding and the metrics_history.jsonl ledger writer."""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

AEGIS_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = AEGIS_ROOT / "results" / "metrics_history.jsonl"


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def config_hash(config: dict) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:10]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=AEGIS_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def log_run(
    *,
    phase: str,
    model: str,
    config: dict,
    seed: int,
    split: str,
    metrics: dict,
    n_params: int = 0,
    train_seconds: float = 0.0,
    notes: str = "",
    path: Path | None = None,
) -> dict:
    """Append one run record (schema: PLAN.md section 9) to metrics_history.jsonl."""
    path = Path(path) if path else HISTORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    rec = {
        "run_id": f"{ts}_{model}_seed{seed}",
        "git_commit": git_commit(),
        "phase": phase,
        "model": model,
        "config": config,
        "config_hash": config_hash(config),
        "seed": seed,
        "split": split,
        "metrics": metrics,
        "n_params": n_params,
        "train_seconds": train_seconds,
        "notes": notes,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec
