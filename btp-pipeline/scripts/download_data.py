"""
download_data.py — Cache HotpotQA distractor split locally.

Idempotent: if the dataset already exists on disk, skips the download.
Run once before any pipeline scripts.

Usage:
    python scripts/download_data.py
"""

import os
import sys

# Allow importing from src/ when running as a script from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import DATA_DIR


def main() -> None:
    if os.path.exists(DATA_DIR) and any(os.scandir(DATA_DIR)):
        print(f"[download_data] Dataset already cached at '{DATA_DIR}'. Skipping download.")
        return

    print("[download_data] Downloading HotpotQA distractor split (validation)...")
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    os.makedirs(DATA_DIR, exist_ok=True)
    ds.save_to_disk(DATA_DIR)
    print(f"[download_data] Saved {len(ds)} examples to '{DATA_DIR}'.")


if __name__ == "__main__":
    main()
