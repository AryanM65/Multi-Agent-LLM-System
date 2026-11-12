"""Fine-tune all-MiniLM-L6-v2 end-to-end on the GROUPED split (Kaggle GPU), then export fine-tuned embeddings.

Standalone (no repo imports). Inputs under --input-dir (any depth):
  train.jsonl, val.jsonl, test.jsonl, whowhen.jsonl   (dataset aegis-raw-data)
  grouped_ids.json {"train":[ids],"val":[ids]}         (dataset aegis-grouped-ids; produced by loaders.load_grouped_splits)

Model: trainable MiniLM (mean pooling, max_len 256) -> per-agent mean of its turn vectors -> concat metadata
(framework one-hot, agent-name one-hot with UNK, 4 count scalars; same as aegis/src/features.py) -> MLP head
(256-256, dropout .3) -> 14 mode logits + 1 agent logit, BCE. Same head/targets as flat_mlp.py, only the encoder is trainable.
Early stopping on grouped-val BCE. Test and Who&When are NEVER used for training or selection: they are only
embedded at the end so the CPU pipeline (flat MLP / GNN) can be re-run on the fine-tuned vectors.

Outputs (/kaggle/working): ft_shards/minilm_shard_<n>.npz (keys = sha1 of utf-8 text, vecs float32 384-d, same format as
aegis/src/embed.py cache), ft_log.json (per-epoch losses), run_info.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODES = [f"FM-1.{i}" for i in range(1, 6)] + [f"FM-2.{i}" for i in range(1, 7)] + [f"FM-3.{i}" for i in range(1, 4)]
MIDX = {m: i for i, m in enumerate(MODES)}
FWS = ["agentverse", "dylan", "llm_debate", "macnet", "magentic_one", "smoagents", "who_when"]
MAXLEN, MAX_TURNS = 256, 30


def sha(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def norm_mode(c: str) -> str:
    c = str(c).strip().upper().replace("EM-", "FM-").replace("FM_", "FM-")
    return c


def find(root: Path, name: str) -> Path:
    return sorted(root.rglob(name))[0]


def read(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def to_traj(rec: dict) -> dict:
    by: dict = {}
    for t in rec["input"]["conversation_history"]:
        by.setdefault(str(t["agent_name"]).strip(), []).append(t.get("content") or "")
    labels = {}
    for a in rec.get("output", {}).get("faulty_agents", []):
        labels.setdefault(str(a["agent_name"]).strip(), set()).add(norm_mode(a["error_type"]))
    return {"id": rec["id"], "fw": rec["metadata"]["framework"], "by": by, "labels": labels,
            "n_total": len(rec["input"]["conversation_history"]), "query": rec["input"]["query"]}


def agent_rows(tr: dict, vocab: dict):
    """One row per agent: (turn texts, meta vector, 14-d target)."""
    order = list(tr["by"])
    rows = []
    for rank, a in enumerate(order):
        texts = tr["by"][a]
        meta = np.zeros(len(FWS) + len(vocab) + 1 + 4, dtype=np.float32)
        meta[FWS.index(tr["fw"])] = 1.0
        meta[len(FWS) + vocab.get(a, len(vocab))] = 1.0
        o = len(FWS) + len(vocab) + 1
        meta[o] = math.log1p(len(texts))
        meta[o + 1] = len(texts) / max(tr["n_total"], 1)
        meta[o + 2] = math.log1p(tr["n_total"])
        meta[o + 3] = rank / max(len(order) - 1, 1)
        y = np.zeros(len(MODES), dtype=np.float32)
        for m in tr["labels"].get(a, ()):
            y[MIDX[m]] = 1.0
        rows.append((texts, meta, y))
    return rows


class Net(nn.Module):
    def __init__(self, d_meta: int):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.enc = AutoModel.from_pretrained(MODEL)
        self.head = nn.Sequential(nn.Linear(384 + d_meta, 256), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(MODES) + 1))

    def embed(self, texts: list, dev) -> torch.Tensor:
        b = self.tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
        h = self.enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def forward(self, rows: list, mu, sd, dev):
        flat, owner = [], []
        for i, (texts, _, _) in enumerate(rows):
            if len(texts) > MAX_TURNS:
                idx = np.linspace(0, len(texts) - 1, MAX_TURNS).round().astype(int)
                texts = [texts[j] for j in idx]
            flat += texts
            owner += [i] * len(texts)
        with torch.autocast("cuda", dtype=torch.float16):
            v = self.embed(flat, dev)
        v = v.float()
        owner_t = torch.tensor(owner, device=dev)
        agg = torch.zeros(len(rows), 384, device=dev).index_add_(0, owner_t, v)
        cnt = torch.zeros(len(rows), device=dev).index_add_(0, owner_t, torch.ones(len(owner), device=dev))
        meta = torch.tensor((np.stack([r[1] for r in rows]) - mu) / sd, dtype=torch.float32, device=dev)
        return self.head(torch.cat([agg / cnt.unsqueeze(1), meta], dim=1))


def targets(rows, dev):
    y = torch.tensor(np.stack([r[2] for r in rows]), device=dev)
    return torch.cat([y, (y.sum(1, keepdim=True) > 0).float()], dim=1)


def batches(trajs, vocab, bs, shuffle, rng):
    idx = list(range(len(trajs)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        rows = []
        for j in idx[i: i + bs]:
            rows += agent_rows(trajs[j], vocab)
        yield rows


@torch.no_grad()
def val_loss(net, trajs, vocab, mu, sd, dev, bs):
    net.eval()
    lossf = nn.BCEWithLogitsLoss(reduction="sum")
    tot, n = 0.0, 0
    for rows in batches(trajs, vocab, bs, False, None):
        y = targets(rows, dev)
        tot += lossf(net(rows, mu, sd, dev), y).item()
        n += y.numel()
    return tot / n


@torch.no_grad()
def export(net, texts: dict, out: Path, dev, shard=20000):
    net.eval()
    keys = sorted(texts, key=lambda k: len(texts[k]))
    out.mkdir(parents=True, exist_ok=True)
    for s in range(0, len(keys), shard):
        part, vecs = keys[s: s + shard], []
        for i in range(0, len(part), 256):
            with torch.autocast("cuda", dtype=torch.float16):
                v = net.embed([texts[k] for k in part[i: i + 256]], dev)
            vecs.append(v.float().cpu().numpy())
        np.savez(out / f"minilm_shard_{s // shard:03d}.npz", keys=np.array(part), vecs=np.concatenate(vecs).astype(np.float32))
        print(f"exported {min(s + shard, len(keys))}/{len(keys)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="/kaggle/input")
    ap.add_argument("--output-dir", default="/kaggle/working")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8, help="trajectories per batch")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use this many train/val trajectories")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    root, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    ids = json.loads(find(root, "grouped_ids.json").read_text())
    want_tr, want_va = set(ids["train"]), set(ids["val"])
    allrecs = {}
    for name in ("train.jsonl", "val.jsonl"):
        for rec in read(find(root, name)):
            if rec["id"] in want_tr or rec["id"] in want_va:
                allrecs[rec["id"]] = to_traj(rec)
    tr = [allrecs[i] for i in ids["train"] if i in allrecs]
    va = [allrecs[i] for i in ids["val"] if i in allrecs]
    if args.limit:
        tr, va = tr[: args.limit], va[: max(args.limit // 4, 8)]
    print(f"grouped train {len(tr)} val {len(va)}", flush=True)

    counts: dict = {}
    for t in tr:
        for a in t["by"]:
            counts[a] = counts.get(a, 0) + 1
    vocab = {a: i for i, (a, _) in enumerate(sorted(counts.items(), key=lambda kv: -kv[1]))}
    meta_all = np.stack([r[1] for t in tr for r in agent_rows(t, vocab)])
    mu, sd = meta_all.mean(0), meta_all.std(0)
    sd = np.where(sd < 1e-3, 1.0, sd)

    net = Net(meta_all.shape[1]).to(dev)
    opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": 2e-5},
                             {"params": net.head.parameters(), "lr": 1e-3}], weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(args.seed)
    best, best_state, bad, log = 1e9, None, 0, []
    for ep in range(args.epochs):
        net.train()
        t0, tot, nb = time.time(), 0.0, 0
        for rows in batches(tr, vocab, args.bs, True, rng):
            opt.zero_grad()
            loss = lossf(net(rows, mu, sd, dev), targets(rows, dev))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item(); nb += 1
            if nb % 100 == 0:
                print(f"  ep {ep} batch {nb} train loss {tot / nb:.4f} ({time.time() - t0:.0f}s)", flush=True)
        vl = val_loss(net, va, vocab, mu, sd, dev, args.bs)
        log.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1), "val_loss": vl, "seconds": time.time() - t0})
        print(f"epoch {ep + 1}: train {tot / max(nb, 1):.4f} val {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
        (out / "ft_log.json").write_text(json.dumps(log, indent=1))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(best_state)

    texts: dict = {}
    for name in ("test.jsonl", "whowhen.jsonl", "val.jsonl", "train.jsonl"):
        for rec in read(find(root, name)):
            for t in rec["input"]["conversation_history"]:
                c = t.get("content") or ""
                texts.setdefault(sha(c), c)
            q = rec["input"]["query"]
            texts.setdefault(sha(q), q)
    if args.limit:
        texts = dict(list(texts.items())[:2000])
    print(f"embedding {len(texts)} unique texts with the fine-tuned encoder", flush=True)
    export(net, texts, out / "ft_shards", dev)
    (out / "run_info.json").write_text(json.dumps({"best_val_loss": best, "epochs_run": len(log), "n_texts": len(texts),
                                                    "train": len(tr), "val": len(va), "args": vars(args)}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
