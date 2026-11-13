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

MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # overridden by --model
MODES = [f"FM-1.{i}" for i in range(1, 6)] + [f"FM-2.{i}" for i in range(1, 7)] + [f"FM-3.{i}" for i in range(1, 4)]
MIDX = {m: i for i, m in enumerate(MODES)}
FWS = ["agentverse", "dylan", "llm_debate", "macnet", "magentic_one", "smoagents", "who_when"]
MAXLEN, MAX_TURNS = 256, 30
NO_NAMES, NO_SCALARS = False, False   # --no-names: zero framework+agent-name columns of the head input; --no-scalars: zero the 4 count scalars (targeted augmentation for Who&When transfer)
SEQ = False   # --seq: small transformer over an agent's turn vectors (order-aware pooling)
ADV_FW, SUPCON = 0.0, 0.0   # --adv-fw LAMBDA: gradient-reversal framework classifier on the pooled agent vector; --supcon W: supervised contrastive loss by error-mode overlap
ATTN = False   # --attn: gated attention pooling over an agent's turns instead of the plain mean
CHUNK_CHARS, MAX_CHUNKS = 0, 3   # set by --chunk-chars / --max-chunks (0 = off)


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
        if not NO_NAMES:
            meta[FWS.index(tr["fw"])] = 1.0
            meta[len(FWS) + vocab.get(a, len(vocab))] = 1.0
        o = len(FWS) + len(vocab) + 1
        if not NO_SCALARS:
            meta[o] = math.log1p(len(texts))
            meta[o + 1] = len(texts) / max(tr["n_total"], 1)
            meta[o + 2] = math.log1p(tr["n_total"])
            meta[o + 3] = rank / max(len(order) - 1, 1)
        y = np.zeros(len(MODES), dtype=np.float32)
        for m in tr["labels"].get(a, ()):
            y[MIDX[m]] = 1.0
        rows.append((texts, meta, y))
    return rows


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Net(nn.Module):
    def __init__(self, d_meta: int, model: str = MODEL):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)
        self.enc = AutoModel.from_pretrained(model)
        self.dim = self.enc.config.hidden_size
        if ATTN:
            self.att_v, self.att_u, self.att_w = nn.Linear(self.dim, 128), nn.Linear(self.dim, 128), nn.Linear(128, 1)
        if SEQ:
            layer = nn.TransformerEncoderLayer(self.dim, 4, 2 * self.dim, 0.1, batch_first=True)
            self.seq = nn.TransformerEncoder(layer, 2)
            self.pos = nn.Embedding(MAX_TURNS + 1, self.dim)
        if ADV_FW:
            self.fw_head = nn.Sequential(nn.Linear(self.dim, 128), nn.ReLU(), nn.Linear(128, len(FWS)))
        self.last_pooled = None
        self.head = nn.Sequential(nn.Linear(self.dim + d_meta, 256), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, len(MODES) + 1))

    def embed(self, texts: list, dev) -> torch.Tensor:
        owner = None
        if CHUNK_CHARS:
            # long messages: split into pieces of CHUNK_CHARS characters (at most MAX_CHUNKS), embed each, average per message
            parts, owner_l = [], []
            for i, t in enumerate(texts):
                pieces = [t[j: j + CHUNK_CHARS] for j in range(0, max(len(t), 1), CHUNK_CHARS)][:MAX_CHUNKS] or [""]
                parts += pieces
                owner_l += [i] * len(pieces)
            owner = torch.tensor(owner_l, device=dev)
            texts = parts
        b = self.tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
        h = self.enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        if owner is None:
            return v
        n = int(owner.max()) + 1
        agg = torch.zeros(n, v.shape[1], device=dev, dtype=v.dtype).index_add_(0, owner, v)
        cnt = torch.zeros(n, device=dev, dtype=v.dtype).index_add_(0, owner, torch.ones(len(owner), device=dev, dtype=v.dtype))
        return agg / cnt.unsqueeze(1)

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
        if SEQ:
            B, Tm = len(rows), max(1, max(owner.count(i) for i in range(len(rows))))
            pad = torch.zeros(B, Tm, self.dim, device=dev)
            msk = torch.ones(B, Tm, dtype=torch.bool, device=dev)
            cur = [0] * B
            for j, o in enumerate(owner):
                pad[o, cur[o]] = v[j] + self.pos(torch.tensor(cur[o], device=dev))
                msk[o, cur[o]] = False
                cur[o] += 1
            h = self.seq(pad, src_key_padding_mask=msk)
            keep = (~msk).unsqueeze(-1).float()
            pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        elif ATTN:
            sc = self.att_w(torch.tanh(self.att_v(v)) * torch.sigmoid(self.att_u(v))).squeeze(-1)
            e = torch.exp(sc - sc.max())
            den = torch.zeros(len(rows), device=dev).index_add_(0, owner_t, e).clamp(min=1e-9)
            pooled = torch.zeros(len(rows), self.dim, device=dev).index_add_(0, owner_t, e.unsqueeze(1) * v) / den.unsqueeze(1)
        else:
            agg = torch.zeros(len(rows), self.dim, device=dev).index_add_(0, owner_t, v)
            cnt = torch.zeros(len(rows), device=dev).index_add_(0, owner_t, torch.ones(len(owner), device=dev))
            pooled = agg / cnt.unsqueeze(1)
        self.last_pooled = pooled
        meta = torch.tensor((np.stack([r[1] for r in rows]) - mu) / sd, dtype=torch.float32, device=dev)
        return self.head(torch.cat([pooled, meta], dim=1))


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
        for i in range(0, len(part), 64 if CHUNK_CHARS else 256):
            with torch.autocast("cuda", dtype=torch.float16):
                v = net.embed([texts[k] for k in part[i: i + (64 if CHUNK_CHARS else 256)]], dev)
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
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (needed for base-size encoders on a 15 GB T4)")
    ap.add_argument("--export-dir", default="ft_shards")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--es-holdout", type=float, default=0.0,
                    help="fraction of TRAIN (grouped by query) held out for encoder early stopping; grouped val is then never "
                         "seen by the encoder, so downstream validation numbers are clean (0 = old behaviour: early-stop on val)")
    ap.add_argument("--chunk-chars", type=int, default=0, help="split messages longer than this many characters into pieces, embed each piece and average (0 = off)")
    ap.add_argument("--max-chunks", type=int, default=3, help="at most this many pieces per message when --chunk-chars is set")
    ap.add_argument("--no-names", action="store_true", help="zero framework and agent-name columns in the head input during fine-tuning")
    ap.add_argument("--no-scalars", action="store_true", help="zero the 4 count scalars in the head input during fine-tuning")
    ap.add_argument("--seq", action="store_true", help="small transformer over an agent's turn vectors (order-aware pooling)")
    ap.add_argument("--adv-fw", type=float, default=0.0, help="weight of the gradient-reversal framework classifier (0 = off)")
    ap.add_argument("--supcon", type=float, default=0.0, help="weight of the supervised contrastive loss by error-mode overlap (0 = off)")
    ap.add_argument("--attn", action="store_true", help="gated attention pooling over turns (end-to-end)")
    ap.add_argument("--max-len", type=int, default=256, help="tokens kept per message (default 256; larger = reads longer messages, slower)")
    args = ap.parse_args()
    global MAXLEN, CHUNK_CHARS, MAX_CHUNKS, ATTN, NO_NAMES, NO_SCALARS, ADV_FW, SUPCON, SEQ
    SEQ = args.seq
    ADV_FW, SUPCON = args.adv_fw, args.supcon
    ATTN = args.attn
    NO_NAMES, NO_SCALARS = args.no_names, args.no_scalars
    MAXLEN = args.max_len
    CHUNK_CHARS, MAX_CHUNKS = args.chunk_chars, args.max_chunks
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
    es = va
    if args.es_holdout > 0:
        queries = sorted({t["query"] for t in tr})
        random.Random(1234).shuffle(queries)
        held = set(queries[: max(1, int(len(queries) * args.es_holdout))])
        es = [t for t in tr if t["query"] in held]
        tr = [t for t in tr if t["query"] not in held]
        print(f"early-stopping slice: {len(es)} train trajectories ({len(held)} queries) held out of training", flush=True)
    print(f"grouped train {len(tr)} early-stop {len(es)} val(untouched if es-holdout) {len(va)}", flush=True)

    counts: dict = {}
    for t in tr:
        for a in t["by"]:
            counts[a] = counts.get(a, 0) + 1
    vocab = {a: i for i, (a, _) in enumerate(sorted(counts.items(), key=lambda kv: -kv[1]))}
    meta_all = np.stack([r[1] for t in tr for r in agent_rows(t, vocab)])
    mu, sd = meta_all.mean(0), meta_all.std(0)
    sd = np.where(sd < 1e-3, 1.0, sd)   # zeroed columns (--no-names / --no-scalars) must not divide by 0
    sd = np.where(sd < 1e-3, 1.0, sd)

    net = Net(meta_all.shape[1], args.model).to(dev)
    if args.grad_ckpt:
        net.enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": 2e-5},
                             {"params": [q for n_, q in net.named_parameters() if not n_.startswith("enc.")], "lr": 1e-3}], weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(args.seed)
    best, best_state, bad, log = 1e9, None, 0, []
    for ep in range(args.epochs):
        net.train()
        t0, tot, nb = time.time(), 0.0, 0
        for rows in batches(tr, vocab, args.bs, True, rng):
            opt.zero_grad()
            tg = targets(rows, dev)
            loss = lossf(net(rows, mu, sd, dev), tg)
            if ADV_FW:
                fwy = torch.tensor([int(np.argmax(r[1][:len(FWS)])) for r in rows], device=dev)
                loss = loss + ADV_FW * nn.functional.cross_entropy(net.fw_head(_GRL.apply(net.last_pooled, 1.0)), fwy)
            if SUPCON and len(rows) > 2:
                z = nn.functional.normalize(net.last_pooled, dim=1)
                lab = tg[:, :-1] > 0.5
                clean = ~lab.any(1)
                pos = ((lab.float() @ lab.float().T) > 0) | (clean.unsqueeze(1) & clean.unsqueeze(0))
                eye = torch.eye(len(rows), dtype=torch.bool, device=dev)
                pos = pos & ~eye
                sim = (z @ z.T) / 0.1
                sim = sim.masked_fill(eye, -1e9)
                logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
                has = pos.any(1)
                if has.any():
                    loss = loss + SUPCON * (-(logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item(); nb += 1
            if nb % 100 == 0:
                print(f"  ep {ep} batch {nb} train loss {tot / nb:.4f} ({time.time() - t0:.0f}s)", flush=True)
        vl = val_loss(net, es, vocab, mu, sd, dev, args.bs)
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
    export(net, texts, out / args.export_dir, dev)
    (out / "run_info.json").write_text(json.dumps({"best_val_loss": best, "epochs_run": len(log), "n_texts": len(texts),
                                                    "train": len(tr), "val": len(va), "es": len(es), "args": vars(args)}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
