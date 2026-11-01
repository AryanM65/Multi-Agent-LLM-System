"""Embed every turn text of Aegis-Bench / Who&When with all-MiniLM-L6-v2 on a Kaggle GPU.

Standalone (no repo imports). Output shards are compatible with aegis/src/embed.py's cache:
  minilm_shard_<n>.npz with arrays keys (sha1 hex of utf-8 text) and vecs (float32, 384-d).
Download them into aegis/data/cache/ and aegis.src.embed.get_embeddings() will find them.

Inputs: train.jsonl, val.jsonl, test.jsonl, whowhen.jsonl anywhere under --input-dir.
Texts are exactly the turn contents (None -> ""), max_seq_length 256, no normalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SHARD_SIZE = 20000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="/kaggle/input")
    ap.add_argument("--output-dir", default="/kaggle/working")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts: dict = {}
    for name in ("test.jsonl", "whowhen.jsonl", "val.jsonl", "train.jsonl"):
        path = sorted(root.rglob(name))[0]
        n_turns = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                for t in json.loads(line)["input"]["conversation_history"]:
                    txt = t.get("content") or ""
                    texts.setdefault(hashlib.sha1(txt.encode("utf-8")).hexdigest(), txt)
                    n_turns += 1
        print(f"{name}: {n_turns} turns, {len(texts)} unique so far", flush=True)

    keys = list(texts)
    if args.limit:
        keys = keys[: args.limit]
    keys.sort(key=lambda k: len(texts[k]))   # similar lengths per batch, little padding

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cuda")
    model.max_seq_length = 256
    t0 = time.time()
    for s in range(0, len(keys), SHARD_SIZE):
        part = keys[s: s + SHARD_SIZE]
        vecs = model.encode([texts[k] for k in part], batch_size=256, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
        np.savez(out / f"minilm_shard_{s // SHARD_SIZE:04d}.npz", keys=np.array(part), vecs=vecs)
        print(f"{min(s + SHARD_SIZE, len(keys))}/{len(keys)}  {time.time() - t0:.0f}s", flush=True)
    (out / "embed_info.json").write_text(json.dumps({"unique_texts": len(keys), "model": MODEL_NAME,
                                                      "seconds": round(time.time() - t0)}), encoding="utf-8")


if __name__ == "__main__":
    main()
