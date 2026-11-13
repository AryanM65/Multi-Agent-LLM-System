"""End-to-end fine-tune of MiniLM + turn-graph GNN on the GROUPED split (Kaggle GPU).

Why: until now the graph model ran on top of FROZEN or separately fine-tuned vectors. This job trains the encoder AND the graph model
together (loss flows through message passing into the encoder), which is the fair test of whether the graph helps.

Standalone (no repo imports). Inputs under --input-dir (any depth):
  train.jsonl, val.jsonl, test.jsonl, whowhen.jsonl   (dataset aegis-raw-data)
  grouped_ids.json {"train":[ids],"val":[ids]}         (dataset aegis-grouped-ids)

Model (same as aegis/src/gnn/turn_gnn.py, but with a trainable encoder):
  trainable MiniLM (mean pooling, max_len 256) -> node vectors for each turn + one query node
  -> Linear(d_enc,128)+ReLU+Dropout -> L relational mean-aggregation layers (4 relations: seq, same-agent, query->turn, turn->query;
     residual + LayerNorm) -> mean-pool the turns of each agent -> concat metadata -> MLP head 256-256 -> 14 mode logits + 1 agent logit.
  --layers 0 gives the 0-layer control (projection + pooling, no message passing).
Early stopping on --es-holdout (query-grouped slice of grouped TRAIN) so grouped val is never seen by the encoder.
Test and Who&When are NEVER used for training or selection; their predictions are only exported at the end.

Outputs (/kaggle/working): preds.json  {split: [{"id", "agents":[...], "probs":[[15 floats]...]}]} for es/val/test/who_when,
ft_log.json (per-epoch losses), run_info.json. Score preds.json offline on CPU (thresholds tuned on val only).
--name-free zeroes the framework and agent-name one-hot metadata columns (Who&When has free-form names).
"""
import argparse
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
MAXLEN, MAX_NODES, N_REL = 256, 40, 4


def norm_mode(c: str) -> str:
    return str(c).strip().upper().replace("EM-", "FM-").replace("FM_", "FM-")


def find(root: Path, name: str) -> Path:
    return sorted(root.rglob(name))[0]


def read(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def to_traj(rec: dict) -> dict:
    turns = [(str(t["agent_name"]).strip(), t.get("content") or "") for t in rec["input"]["conversation_history"]]
    labels: dict = {}
    for a in rec.get("output", {}).get("faulty_agents", []):
        labels.setdefault(str(a["agent_name"]).strip(), set()).add(norm_mode(a["error_type"]))
    return {"id": rec["id"], "fw": rec["metadata"]["framework"], "turns": turns, "labels": labels, "query": rec["input"]["query"]}


def edges_for(agent_of_turn: list) -> np.ndarray:
    """Same edges as aegis/src/graph.py: (src, dst, rel); the query node id is n (placed last)."""
    n = len(agent_of_turn)
    e = []
    for i in range(n - 1):
        e += [(i, i + 1, 0), (i + 1, i, 0)]
    last: dict = {}
    for i, a in enumerate(agent_of_turn):
        if a in last:
            j = last[a]
            e += [(j, i, 1), (i, j, 1)]
        last[a] = i
    for i in range(n):
        e += [(n, i, 2), (i, n, 3)]
    return np.asarray(e, dtype=np.int64).reshape(-1, 3)


def keep_indices(turns: list) -> list:
    """At most MAX_NODES turns: evenly spaced, plus the first turn of every agent so no agent loses all of its turns."""
    n = len(turns)
    if n <= MAX_NODES:
        return list(range(n))
    idx = set(np.linspace(0, n - 1, MAX_NODES).round().astype(int).tolist())
    seen = set()
    for i, (a, _) in enumerate(turns):
        if a not in seen:
            seen.add(a)
            idx.add(i)
    return sorted(idx)


def meta_rows(tr: dict, order: list, vocab: dict, name_free: bool):
    n_total = len(tr["turns"])
    cnt = {a: 0 for a in order}
    for a, _ in tr["turns"]:
        cnt[a] += 1
    rows = []
    for rank, a in enumerate(order):
        meta = np.zeros(len(FWS) + len(vocab) + 1 + 4, dtype=np.float32)
        if not name_free:
            meta[FWS.index(tr["fw"])] = 1.0
            meta[len(FWS) + vocab.get(a, len(vocab))] = 1.0
        o = len(FWS) + len(vocab) + 1
        meta[o] = math.log1p(cnt[a])
        meta[o + 1] = cnt[a] / max(n_total, 1)
        meta[o + 2] = math.log1p(n_total)
        meta[o + 3] = rank / max(len(order) - 1, 1)
        y = np.zeros(len(MODES), dtype=np.float32)
        for m in tr["labels"].get(a, ()):
            y[MIDX[m]] = 1.0
        rows.append((meta, y))
    return rows


class RelLayer(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.self_lin = nn.Linear(d, d)
        self.rel_lin = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(N_REL)])
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, src, dst, rel):
        out = self.self_lin(h)
        for r, lin in enumerate(self.rel_lin):
            m = rel == r
            if not m.any():
                continue
            s, d_ = src[m], dst[m]
            agg = torch.zeros_like(h).index_add_(0, d_, lin(h)[s])
            deg = torch.zeros(h.shape[0], device=h.device).index_add_(0, d_, torch.ones(len(d_), device=h.device))
            out = out + agg / deg.clamp(min=1).unsqueeze(1)
        return self.norm(h + self.drop(torch.relu(out)))


class Net(nn.Module):
    def __init__(self, d_meta: int, model: str, layers: int, d: int = 128, dropout: float = 0.3):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)
        self.enc = AutoModel.from_pretrained(model)
        self.dim = self.enc.config.hidden_size
        self.inp = nn.Sequential(nn.Linear(self.dim, d), nn.ReLU(), nn.Dropout(dropout))
        self.layers = nn.ModuleList([RelLayer(d, dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(d + d_meta, 256), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, len(MODES) + 1))

    def embed(self, texts: list, dev) -> torch.Tensor:
        b = self.tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
        h = self.enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def forward(self, batch: dict, mu, sd, dev):
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
            v = self.embed(batch["texts"], dev)
        h = self.inp(v.float())
        src, dst, rel, pool = (torch.as_tensor(batch[k], device=dev) for k in ("src", "dst", "rel", "pool"))
        for layer in self.layers:
            h = layer(h, src, dst, rel)
        keep = pool >= 0
        n_rows = batch["n_rows"]
        agg = torch.zeros(n_rows, h.shape[1], device=dev).index_add_(0, pool[keep], h[keep])
        cnt = torch.zeros(n_rows, device=dev).index_add_(0, pool[keep], torch.ones(int(keep.sum()), device=dev))
        meta = torch.tensor((batch["meta"] - mu) / sd, dtype=torch.float32, device=dev)
        return self.head(torch.cat([agg / cnt.clamp(min=1).unsqueeze(1), meta], dim=1))


def make_batch(trajs: list, vocab: dict, name_free: bool) -> dict:
    texts, src, dst, rel, pool, metas, ys, agents = [], [], [], [], [], [], [], []
    off = row = 0
    for tr in trajs:
        order: dict = {}
        for a, _ in tr["turns"]:
            order.setdefault(a, len(order))
        mr = meta_rows(tr, list(order), vocab, name_free)
        idx = keep_indices(tr["turns"])
        sub = [tr["turns"][i] for i in idx]
        aot = [order[a] for a, _ in sub]
        ed = edges_for(aot)
        n = len(sub)
        texts += [t for _, t in sub] + [tr["query"]]
        src.append(ed[:, 0] + off); dst.append(ed[:, 1] + off); rel.append(ed[:, 2])
        pool.append(np.asarray([a + row for a in aot] + [-1]))
        metas += [m for m, _ in mr]
        ys += [y for _, y in mr]
        agents.append(list(order))
        off += n + 1
        row += len(order)
    return {"texts": texts, "src": np.concatenate(src), "dst": np.concatenate(dst), "rel": np.concatenate(rel),
            "pool": np.concatenate(pool), "meta": np.stack(metas), "y": np.stack(ys), "n_rows": row, "agents": agents}


def targets(y: np.ndarray, dev):
    y = torch.tensor(y, device=dev)
    return torch.cat([y, (y.sum(1, keepdim=True) > 0).float()], dim=1)


@torch.no_grad()
def predict(net, trajs, vocab, mu, sd, dev, bs, name_free):
    net.eval()
    out, loss_sum, n = [], 0.0, 0
    lossf = nn.BCEWithLogitsLoss(reduction="sum")
    for i in range(0, len(trajs), bs):
        chunk = trajs[i: i + bs]
        b = make_batch(chunk, vocab, name_free)
        lg = net(b, mu, sd, dev)
        y = targets(b["y"], dev)
        loss_sum += lossf(lg, y).item()
        n += y.numel()
        pr = torch.sigmoid(lg).cpu().numpy()
        k = 0
        for tr, ags in zip(chunk, b["agents"]):
            out.append({"id": tr["id"], "agents": ags, "probs": [[round(float(x), 5) for x in r] for r in pr[k: k + len(ags)]]})
            k += len(ags)
    return out, loss_sum / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="/kaggle/input")
    ap.add_argument("--output-dir", default="/kaggle/working")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8, help="trajectories per batch")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use this many train trajectories")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layers", type=int, default=2, help="0 = control without message passing")
    ap.add_argument("--es-holdout", type=float, default=0.15)
    ap.add_argument("--name-free", action="store_true", help="zero the framework and agent-name one-hot metadata columns")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    root, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(f"grouped train {len(tr)} early-stop {len(es)} val {len(va)} layers {args.layers} name_free {args.name_free}", flush=True)

    counts: dict = {}
    for t in tr:
        for a in {a for a, _ in t["turns"]}:
            counts[a] = counts.get(a, 0) + 1
    vocab = {a: i for i, (a, _) in enumerate(sorted(counts.items(), key=lambda kv: -kv[1]))}
    meta_all = np.concatenate([np.stack([m for m, _ in meta_rows(t, list(dict.fromkeys(a for a, _ in t["turns"])), vocab, args.name_free)]) for t in tr])
    mu, sd = meta_all.mean(0), meta_all.std(0)
    sd = np.where(sd < 1e-3, 1.0, sd)

    net = Net(meta_all.shape[1], args.model, args.layers).to(dev)
    if args.grad_ckpt:
        net.enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    enc_params = list(net.enc.parameters())
    other = [p for n_, p in net.named_parameters() if not n_.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": 2e-5}, {"params": other, "lr": 1e-3}], weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))
    rng = random.Random(args.seed)
    best, best_state, bad, log = 1e9, None, 0, []
    for ep in range(args.epochs):
        net.train()
        t0, tot, nb = time.time(), 0.0, 0
        idx = list(range(len(tr)))
        rng.shuffle(idx)
        for i in range(0, len(idx), args.bs):
            b = make_batch([tr[j] for j in idx[i: i + args.bs]], vocab, args.name_free)
            opt.zero_grad()
            loss = lossf(net(b, mu, sd, dev), targets(b["y"], dev))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item(); nb += 1
            if nb % 100 == 0:
                print(f"  ep {ep} batch {nb} train loss {tot / nb:.4f} ({time.time() - t0:.0f}s)", flush=True)
        _, vl = predict(net, es, vocab, mu, sd, dev, args.bs, args.name_free)
        log.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1), "es_loss": vl, "seconds": time.time() - t0})
        print(f"epoch {ep + 1}: train {tot / max(nb, 1):.4f} es {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
        (out / "ft_log.json").write_text(json.dumps(log, indent=1))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(best_state)

    preds = {}
    for name, trajs in (("es", es), ("val", va)):
        preds[name], _ = predict(net, trajs, vocab, mu, sd, dev, args.bs, args.name_free)
    for name, fname in (("test", "test.jsonl"), ("who_when", "whowhen.jsonl")):
        trajs = [to_traj(r) for r in read(find(root, fname))]
        if args.limit:
            trajs = trajs[: max(args.limit // 4, 8)]
        preds[name], _ = predict(net, trajs, vocab, mu, sd, dev, args.bs, args.name_free)
        print(f"predicted {name}: {len(trajs)} trajectories", flush=True)
    (out / "preds.json").write_text(json.dumps(preds))
    (out / "run_info.json").write_text(json.dumps({"best_es_loss": best, "epochs_run": len(log), "train": len(tr), "es": len(es), "val": len(va),
                                                    "vocab": list(vocab), "args": vars(args)}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
