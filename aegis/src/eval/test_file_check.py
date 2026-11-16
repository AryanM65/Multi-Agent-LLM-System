"""S2 part B: score ONE trained model on both 600-record test files (and Who&When), paper protocol.

Why: the paper does not say which test file its published numbers used (HF Aegis-Bench/test.jsonl or the authors' repo
file evaluation/data/aegis.jsonl). Training and threshold tuning are identical for both rows (released train, released val),
so any difference between the rows is caused by the test file alone.

Model: flat MLP on frozen MiniLM vectors (the cheap model whose pipeline is fully local). Per seed it also saves per-record
predictions to aegis/results/preds/ so bootstrap intervals can be computed later without retraining.
Run: python -m aegis.src.eval.test_file_check [--seeds 5]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..baselines import frequency
from ..baselines.flat_mlp import decode, train_one, tune
from ..features import EmbStore, agent_vocab, build
from ..loaders import load_paper_splits, load_split
from ..metrics import evaluate
from ..schema import ERROR_MODES
from ..utils import log_run

LV = ("pair", "agent", "error")
PREDS = Path(__file__).resolve().parents[2] / "results" / "preds"
SETS = {"hf_test": "test", "repo_test": "test_repo", "who_when": "who_when"}


def line(res) -> str:
    return " ".join(f"{l} {100 * res[l]['micro_f1']:.2f}/{100 * res[l]['macro_f1']:.2f}" for l in LV)


def floor(trajs):
    return evaluate([t.labels for t in trajs], [{(a, m) for a in set(t.agents) for m in ERROR_MODES} for t in trajs])


def main(n_seeds: int, dedup: bool) -> None:
    PREDS.mkdir(parents=True, exist_ok=True)
    store = EmbStore()
    sp = load_paper_splits()
    sets = {k: load_split(v) for k, v in SETS.items()}
    tag = "_dedup" if dedup else ""
    if dedup:
        # drop released train/val records whose conversation text is a verbatim copy of a record in either 600-record test file
        from ...scripts.compare_test_files import convo_hash
        banned = {convo_hash(t) for k in ("hf_test", "repo_test") for t in sets[k]}
        before = {k: len(v) for k, v in sp.items()}
        sp = {**sp, **{k: [t for t in sp[k] if convo_hash(t) not in banned] for k in ("train", "val")}}
        print(f"dedup: train {before['train']} -> {len(sp['train'])}, val {before['val']} -> {len(sp['val'])}", flush=True)
    vocab = agent_vocab(sp["train"])
    Xtr, Ytr, _ = build(sp["train"], store, vocab)
    Xva, Yva, mva = build(sp["val"], store, vocab)
    feats = {k: build(ts, store, vocab) for k, ts in sets.items()}

    print("rows: model | file | pair micro/macro | agent | error", flush=True)
    for k, ts in sets.items():
        print(f"  floor     {k:9s} {line(floor(ts))}", flush=True)
    ranking = frequency.fit(sp["train"])
    truth_val = [t.labels for t in sp["val"]]
    kbest = max((sum(evaluate(truth_val, frequency.predict(sp["val"], ranking, k))[l]["micro_f1"] for l in LV), k) for k in range(1, 16))[1]
    for k, ts in sets.items():
        print(f"  freq k={kbest:<2d} {k:9s} {line(evaluate([t.labels for t in ts], frequency.predict(ts, ranking, kbest)))}", flush=True)

    runs = []
    for s in range(n_seeds):
        t0 = time.time()
        model, probs, epochs, n_params = train_one(Xtr, Ytr, Xva, Yva, s)
        _, ta, tm, _ = tune(probs, mva, sp["val"])          # thresholds chosen on the released val only
        out = {}
        for k, ts in sets.items():
            X, _, meta = feats[k]
            with torch.no_grad():
                p = torch.sigmoid(model(torch.tensor((X - model.mu) / model.sd, dtype=torch.float32))).numpy()
            pred = decode(p, meta, len(ts), ta, tm)
            out[k] = evaluate([t.labels for t in ts], pred)
            (PREDS / f"s2_flat_paper{tag}_seed{s}_{k}.json").write_text(json.dumps(
                {"ids": [t.id for t in ts], "pred": [sorted(map(list, q)) for q in pred]}))
            log_run(phase="S2", model="flat_mlp", seed=s, split=f"paper{tag}_{k}", n_params=n_params,
                    config={"protocol": "paper" + tag, "t_a": ta, "t_m": tm, "epochs_run": epochs, "file": k},
                    metrics={k: {l: out[k][l] for l in LV}}, train_seconds=time.time() - t0,
                    notes="S2: same model scored on both 600-record test files; embeddings recomputed locally 2026-11-16 (MPS)")
        runs.append(out)
        print(f"  flat_mlp seed {s}: " + " || ".join(f"{k} {line(out[k])}" for k in sets) + f"  ({time.time() - t0:.0f}s)", flush=True)

    print("\nmean +- std over seeds (micro / macro):", flush=True)
    for k in sets:
        parts = []
        for l in LV:
            for m in ("micro_f1", "macro_f1"):
                v = [100 * r[k][l][m] for r in runs]
                parts.append(f"{l[:4]} {m[:5]} {np.mean(v):.2f}+-{np.std(v):.2f}")
        print(f"  flat_mlp {k:9s} " + " | ".join(parts), flush=True)
    # paired per-seed difference between the two test files (same models, different records)
    for l in LV:
        d = [100 * (r["hf_test"][l]["micro_f1"] - r["repo_test"][l]["micro_f1"]) for r in runs]
        print(f"  {l} micro  hf_test - repo_test per seed: " + ", ".join(f"{x:+.2f}" for x in d) + f"  mean {np.mean(d):+.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dedup", action="store_true", help="remove train/val records whose conversation also occurs in a test file")
    a = ap.parse_args()
    main(a.seeds, a.dedup)
