"""Reproduction rows with record-level intervals: DCL-style (frozen encoder) under the deduplicated protocol, and the saved zero-shot Qwen2.5-7B outputs.

* DCL-style: trained on the released train split minus test duplicates, early stopping and one decision threshold on the deduplicated validation
  split, then scored once per seed on the authors' file, the Hugging Face file and Who&When (5 seeds). It is NOT the published DCL (frozen encoder,
  different optimiser schedule; see aegis/docs/reproduction_log.md, section 3b).
* Qwen2.5-7B zero-shot: the saved Kaggle outputs (aegis/results/kaggle_logs/full; AWQ 4-bit, greedy, 512 new tokens, long prompts truncated) scored
  on the Hugging Face file and Who&When. It was NOT run on the authors' file. Unparseable outputs count as empty predictions.
Output: aegis/results/baselines_dedup.json (micro-F1 with 2,000-resample record bootstrap, macro-F1 point estimate, percent)
Run (repo root): btp-pipeline/btp-env/bin/python -m aegis.scripts.baselines_dedup
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from aegis.scripts.compare_test_files import convo_hash
from aegis.src.baselines.dcl_style import (GRID, Batcher, DCLStyle, decode, init_prototypes, losses, pair_probs)
from aegis.src.eval.block_ablation import bootstrap, counts
from aegis.src.eval.score_zeroshot import parse_prediction
from aegis.src.features import EmbStore, agent_vocab
from aegis.src.loaders import load_paper_splits, load_split
from aegis.src.metrics import evaluate
from aegis.src.utils import set_seed

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "aegis/results"
LV = ("pair", "agent", "error")
SEEDS, EPOCHS, PATIENCE = 5, 20, 5
SETS = {"repo_test": "test_repo", "hf_test": "test", "who_when": "who_when"}


def summarise(per_seed_counts, macros):
    r = bootstrap(np.stack(per_seed_counts))
    for lv in LV:
        r[lv]["macro"] = 100 * float(np.mean([m[lv] for m in macros]))
    return r


out = {"seeds": SEEDS, "rows": {}, "notes": {}}

# ---------------- Qwen2.5-7B zero-shot (saved outputs) ----------------
for key, split, fname in (("hf_test", "test", "zeroshot_test.jsonl"), ("who_when", "who_when", "zeroshot_who_when.jsonl")):
    gold = {t.id: t for t in load_split(split)}
    rows = [json.loads(line) for line in (RES / "kaggle_logs/full" / fname).read_text(encoding="utf-8").splitlines() if line.strip()]
    truth = [set(gold[r["id"]].labels) for r in rows]
    parsed = [parse_prediction(r["raw_output"]) for r in rows]
    pred = [p for p, _ in parsed]
    ev = evaluate(truth, pred)
    r = bootstrap(counts(truth, pred)[None])
    for lv in LV:
        r[lv]["macro"] = 100 * ev[lv]["macro_f1"]
    out["rows"][f"qwen_zeroshot|{key}"] = r
    out["notes"][f"qwen_zeroshot|{key}"] = {"n": len(rows), "unparseable": int(sum(not ok for _, ok in parsed)),
                                              "truncated": int(sum(bool(x.get("truncated")) for x in rows))}
    print("qwen", key, {lv: round(r[lv]["micro"], 2) for lv in LV}, flush=True)

# ---------------- DCL-style under the deduplicated protocol ----------------
paper = load_paper_splits()
tests = {k: load_split(v) for k, v in SETS.items()}
banned = {convo_hash(t) for k in ("repo_test", "hf_test") for t in tests[k]}
sp = {k: [t for t in paper[k] if convo_hash(t) not in banned] for k in ("train", "val")}
store = EmbStore()
vocab = agent_vocab(sp["train"])
modes0, names0 = init_prototypes(vocab)
tr, va = Batcher(sp["train"], store, vocab), Batcher(sp["val"], store, vocab)
vtruth = [t.labels for t in sp["val"]]
vorder = list(range(len(sp["val"])))
ev_b = {k: (Batcher(ts, store, vocab), [t.labels for t in ts]) for k, ts in tests.items()}
print(f"DCL-style: train {len(tr.items)} val {len(va.items)}", flush=True)
per = {k: [] for k in tests}
macros = {k: [] for k in tests}
cost = []
for seed in range(SEEDS):
    set_seed(seed)
    t0 = time.time()
    model = DCLStyle(modes0, names0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    steps_total = EPOCHS * ((len(tr.items) + 63) // 64)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps_total)) * 0.5 * (1 + np.cos(np.pi * s / steps_total)))
    best, state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(tr.items))
        for i in range(0, len(perm), 64):
            b = tr.batch(perm[i: i + 64])
            opt.zero_grad()
            loss, _ = losses(model(b), b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for i in range(0, len(vorder), 128):
                b = va.batch(vorder[i: i + 128])
                vl += losses(model(b), b)[0].item() * len(vorder[i: i + 128])
        vl /= len(vorder)
        if vl < best - 1e-4:
            best, state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state)
    vprobs = pair_probs(model, va, vorder)
    thr = max(GRID, key=lambda t: sum(evaluate(vtruth, decode(vprobs, va, vorder, t))[l]["micro_f1"] for l in LV))
    for k, (eb, etruth) in ev_b.items():
        eorder = list(range(len(eb.items)))
        pred = decode(pair_probs(model, eb, eorder), eb, eorder, thr)
        per[k].append(counts(etruth, pred))
        ev = evaluate(etruth, pred)
        macros[k].append({lv: ev[lv]["macro_f1"] for lv in LV})
    cost.append(time.time() - t0)
    print(f"seed {seed} thr {thr} epochs {ep + 1} ({cost[-1]:.0f}s)", flush=True)
for k in tests:
    out["rows"][f"dcl_style|{k}"] = summarise(per[k], macros[k])
    print("dcl", k, {lv: round(out["rows"][f"dcl_style|{k}"][lv]["micro"], 2) for lv in LV}, flush=True)
out["notes"]["dcl_style"] = {"train_records": len(tr.items), "val_records": len(va.items), "mean_train_seconds_per_seed": round(float(np.mean(cost)), 1),
                              "protocol": "released train/val minus test duplicates; frozen MiniLM; thr tuned on val"}
(RES / "baselines_dedup.json").write_text(json.dumps(out, indent=1))
