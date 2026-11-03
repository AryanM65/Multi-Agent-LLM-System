"""Final scoring on the held-out test set and Who&When (Phase 6). Run ONCE per configuration.

Protocol (test-set discipline): every model is trained on the split's train set with early stopping on its val set; decision
thresholds (or k for the frequency baseline) are chosen on val only; the frozen model is then scored on the 600-record test
set and on Who&When. Nothing here is tuned on test. Two training protocols are reported:
  paper   - released train/val (comparable to published rows, query leakage present)
  grouped - query-disjoint train/val (generalisation claim)
Models: trivial floor (every present agent x all 14 modes), frequency, flat MLP, pooled-projection (0 layers), turn GNN (2 layers).

Run: python -m aegis.src.eval.final_test [--seeds 5]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from ..baselines import frequency
from ..baselines.flat_mlp import decode, train_one, tune
from ..embed import get_embeddings
from ..features import EmbStore, agent_vocab, build
from ..gnn.turn_gnn import train_gnn
from ..graph import make_items
from ..loaders import load_grouped_splits, load_paper_splits, load_split
from ..metrics import evaluate
from ..schema import ERROR_MODES
from ..utils import log_run

LV = ("pair", "agent", "error")
DRY = False   # --dry-run: score slices of VAL instead of test/Who&When and do not write the ledger


def line(res) -> str:
    return " ".join(f"{l} {100*res[l]['micro_f1']:.2f}/{100*res[l]['macro_f1']:.2f}" for l in LV)


def record(name, model, seed, res_test, res_ww, extra, n_params=0):
    if DRY:
        return
    for tag, res in (("test", res_test), ("who_when", res_ww)):
        log_run(phase="6", model=model, seed=seed, split=f"{tag}_{name}", n_params=n_params,
                config=extra, metrics={tag: {l: res[l] for l in LV}},
                notes="final scoring; config and thresholds chosen on val only")


def summarize(label, runs):
    for tag in ("test", "who_when"):
        parts = []
        for l in LV:
            for k in ("micro_f1", "macro_f1"):
                v = [100 * r[tag][l][k] for r in runs]
                parts.append(f"{l[:4]} {k[:5]} {np.mean(v):.2f}+-{np.std(v):.2f}")
        print(f"  {label:18s} {tag:8s} " + " | ".join(parts), flush=True)


def floor(trajs):
    return evaluate([t.labels for t in trajs], [{(a, m) for a in set(t.agents) for m in ERROR_MODES} for t in trajs])


def run_split(name: str, splits: dict, ww: list, test: list, store: EmbStore, seeds: list) -> None:
    print(f"\n===== train protocol: {name} =====", flush=True)
    vocab = agent_vocab(splits["train"])
    qs = sorted({t.query for ts in (splits["train"], splits["val"], test, ww) for t in ts})
    qvec = dict(zip(qs, get_embeddings(qs, verbose=False)))
    feats = {k: build(ts, store, vocab) for k, ts in (("train", splits["train"]), ("val", splits["val"]), ("test", test), ("who_when", ww))}
    trajs = {"train": splits["train"], "val": splits["val"], "test": test, "who_when": ww}

    # trivial floor and frequency (deterministic)
    rt, rw = floor(test), floor(ww)
    print("  floor              test     ", line(rt), "\n  floor              who_when ", line(rw))
    record(name, "floor_all_agents_all_modes", 0, rt, rw, {"split": name})
    ranking = frequency.fit(splits["train"])
    truth_val = [t.labels for t in splits["val"]]
    best = max(((sum(evaluate(truth_val, frequency.predict(splits["val"], ranking, k))[l]["micro_f1"] for l in LV), k) for k in range(1, 16)))
    k = best[1]
    rt = evaluate([t.labels for t in test], frequency.predict(test, ranking, k))
    rw = evaluate([t.labels for t in ww], frequency.predict(ww, ranking, k))
    print(f"  frequency (k={k})   test     ", line(rt), "\n                     who_when ", line(rw))
    record(name, "frequency", 0, rt, rw, {"split": name, "k": k})

    # flat MLP
    Xtr, Ytr, _ = feats["train"]
    Xva, Yva, mva = feats["val"]
    runs = []
    for s in seeds:
        t0 = time.time()
        model, probs, epochs, n_params = train_one(Xtr, Ytr, Xva, Yva, s)
        _, ta, tm, _ = tune(probs, mva, splits["val"])
        out = {}
        for tag in ("test", "who_when"):
            X, _, meta = feats[tag]
            with torch.no_grad():
                p = torch.sigmoid(model(torch.tensor((X - model.mu) / model.sd, dtype=torch.float32))).numpy()
            out[tag] = evaluate([t.labels for t in trajs[tag]], decode(p, meta, len(trajs[tag]), ta, tm))
        runs.append(out)
        record(name, "flat_mlp", s, out["test"], out["who_when"], {"split": name, "t_a": ta, "t_m": tm, "epochs_run": epochs}, n_params)
        print(f"  flat_mlp seed {s} test {line(out['test'])}  ({time.time()-t0:.0f}s)", flush=True)
    summarize("flat_mlp", runs)

    # GNN: 0 layers (control) and 2 layers
    items = {k_: make_items(trajs[k_], store, qvec, *feats[k_]) for k_ in feats}
    for layers in (0, 2):
        runs = []
        for s in seeds:
            t0 = time.time()
            probs, epochs, n_params, extra = train_gnn(items["train"], items["val"], s, layers,
                                                       extra_items={"test": items["test"], "who_when": items["who_when"]})
            _, ta, tm, _ = tune(probs, feats["val"][2], splits["val"])
            out = {tag: evaluate([t.labels for t in trajs[tag]], decode(extra[tag], feats[tag][2], len(trajs[tag]), ta, tm))
                   for tag in ("test", "who_when")}
            runs.append(out)
            record(name, f"turn_gnn_L{layers}", s, out["test"], out["who_when"], {"split": name, "layers": layers, "t_a": ta, "t_m": tm, "epochs_run": epochs}, n_params)
            print(f"  turn_gnn_L{layers} seed {s} test {line(out['test'])}  ({time.time()-t0:.0f}s)", flush=True)
        summarize(f"turn_gnn_L{layers}", runs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="code-path check on slices of val; test/Who&When are not read")
    args = ap.parse_args()
    DRY = args.dry_run
    store = EmbStore()
    for name, loader in (("grouped", load_grouped_splits), ("paper", load_paper_splits)):
        sp = loader()
        if DRY:
            test, ww = sp["val"][:100], sp["val"][100:200]
        else:
            test, ww = load_split("test"), load_split("who_when")
        run_split(name, sp, ww, test, store, list(range(1 if DRY else args.seeds)))
