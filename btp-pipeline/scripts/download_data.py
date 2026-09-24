"""
download_data.py -- Cache HotpotQA distractor split locally.

Idempotent: if the dataset already exists on disk, skips the download.
Run once before any pipeline scripts.

Usage:
    python scripts/download_data.py
"""

import json
import os
import sys
import urllib.request

# Allow importing from src/ when running as a script from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import DATA_DIR


# Official HotpotQA distractor validation set (hosted on HuggingFace as a file)
HF_HOTPOT_URL = (
    "https://huggingface.co/datasets/hotpot_qa/resolve/main/data/"
    "hotpot_dev_distractor_v1.json.gz"
)
# Direct mirror (no authentication needed)
DIRECT_URL = "https://huggingface.co/datasets/hotpot_qa/resolve/refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet"

# Fallback: use the raw GitHub release file
GITHUB_URL = "https://raw.githubusercontent.com/hotpotQA/hotpot_qa/master/data/hotpot_dev_distractor_v1.json"


def main() -> None:
    if os.path.exists(DATA_DIR) and any(os.scandir(DATA_DIR)):
        print(f"[download_data] Dataset already cached at '{DATA_DIR}'. Skipping download.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    # Try Parquet via HuggingFace (fastest, no script required)
    print("[download_data] Attempting download from HuggingFace Parquet endpoint...")
    try:
        _download_parquet()
        return
    except Exception as e:
        print(f"[download_data] Parquet download failed: {e}")

    print("[download_data] Attempting standard HuggingFace datasets API...")
    try:
        _download_via_hf_api()
        return
    except Exception as e:
        print(f"[download_data] HF API download failed: {e}")

    print("[download_data] All download methods failed.")
    print("Please manually download hotpot_dev_distractor_v1.json and run:")
    print("  python scripts/convert_hotpot_json.py <path/to/json>")
    sys.exit(1)


def _download_parquet() -> None:
    """Download the pre-converted Parquet file from HuggingFace."""
    import pandas as pd
    from datasets import Dataset

    parquet_url = "https://huggingface.co/datasets/hotpot_qa/resolve/refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet"
    parquet_path = os.path.join(DATA_DIR, "validation.parquet")

    print(f"[download_data] Downloading from {parquet_url}")
    req = urllib.request.Request(parquet_url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=120) as r, open(parquet_path, "wb") as f:
        f.write(r.read())
    print(f"[download_data] Downloaded parquet to {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"[download_data] Loaded {len(df)} rows from parquet")

    # Convert to HuggingFace Dataset and save to disk
    ds = Dataset.from_pandas(df)
    ds.save_to_disk(DATA_DIR)
    print(f"[download_data] Saved {len(ds)} examples to '{DATA_DIR}'.")


def _download_via_hf_api() -> None:
    """Try loading via the HuggingFace datasets library."""
    from datasets import load_dataset

    # Try without trust_remote_code (newer API)
    try:
        ds = load_dataset("hotpot_qa", "distractor", split="validation")
    except Exception:
        # Try explicit split with newer API style
        ds = load_dataset("BeIR/hotpotqa", split="validation")

    os.makedirs(DATA_DIR, exist_ok=True)
    ds.save_to_disk(DATA_DIR)
    print(f"[download_data] Saved {len(ds)} examples to '{DATA_DIR}'.")


if __name__ == "__main__":
    main()
