"""DCL-style baseline (Phase 3b): a re-implementation of the published DCL design from its description
(paper Appendix C / Table 7). NOT the authors' code (none was released) and NOT end-to-end: the MiniLM encoder is FROZEN
and read from the embedding cache, so this is "DCL-style (frozen encoder)". Differences from the paper are listed below.

Design (as described): a trajectory is a bag of turns. A projection maps each turn embedding to d=128. Two prototype banks,
one over agent names and one over the 14 error modes (initialised from text embeddings), score turns by cosine similarity.
For each agent, multiple-instance top-K (K=3) pooling over that agent's turns gives agent, mode and pair logits.
No graph, no edges.
Losses: asymmetric BCE (agent, trajectory-level error, pair), supervised contrastive on agent evidence (tau 0.07, positives
share a mode), hierarchical consistency (pair prob <= min(agent prob, error prob), squared hinge).
Loss weights: cls 1.0, con 0.4, hier 0.6 (paper Table 7).

Deviations from the paper: frozen encoder (paper fine-tunes it), lr 1e-3 and 20 epochs on a tiny head instead of lr 1e-4 and
2 x 500 steps, agent prototypes keyed by agent NAME (paper: text definitions of roles), turn speaker is used to group turns
into agents, batch 64 instead of 128, decision rule from D10 (single pair threshold tuned on val, at least one pair).

Run: python -m aegis.src.baselines.dcl_style [--seeds 3]
Validation only; the test set is not touched.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..embed import CACHE_DIR
from ..features import EmbStore, agent_vocab
from ..loaders import AEGIS_ROOT, load_grouped_splits, load_paper_splits, load_split
from ..metrics import evaluate
from ..schema import ERROR_MODES, MODE_INDEX
from ..utils import log_run, set_seed

K_TOP = 3
GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------- prototype initialisation
def mode_definitions() -> list:
    text = (AEGIS_ROOT / "third_party" / "AEGIS" / "evaluation" / "prompt.txt").read_text(encoding="utf-8")
    defs = {}
    for m in re.finditer(r"- (FM-\d\.\d): \*\*(.+?)\*\* - (.+)", text):
        defs[m.group(1)] = f"{m.group(2)}. {m.group(3)}"
    return [defs[c] for c in ERROR_MODES]


def init_prototypes(vocab: list) -> tuple:
    """Embed the 14 mode definitions and the agent names once (tiny: < 100 short strings), cached to disk."""
    path = CACHE_DIR / f"proto_init_{hashlib.sha1(chr(10).join(vocab).encode()).hexdigest()[:10]}.npz"
    if path.exists():
        z = np.load(path)
        return z["modes"], z["names"]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    modes = model.encode(mode_definitions(), convert_to_numpy=True).astype(np.float32)
    names = model.encode([re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", v) for v in vocab] + ["unknown"],
                         convert_to_numpy=True).astype(np.float32)
    np.savez(path, modes=modes, names=names)
    return modes, names


# ---------------------------------------------------------------- data
class Batcher:
    def __init__(self, trajs: list, store: EmbStore, vocab: list):
        vidx = {a: i for i, a in enumerate(vocab)}
        self.items = []
        for t in trajs:
            agents = t.agents
            aidx = {a: i for i, a in enumerate(agents)}
            E = np.stack([store.lookup(u.text) for u in t.turns])
            name_ids = np.array([vidx.get(u.agent, len(vocab)) for u in t.turns])
            agent_ids = np.array([aidx[u.agent] for u in t.turns])
            Y = np.zeros((len(agents), len(ERROR_MODES)), dtype=np.float32)
            for a, m in t.labels:
                if a in aidx:
                    Y[aidx[a], MODE_INDEX[m]] = 1.0
            self.items.append((E, name_ids, agent_ids, Y, agents))

    def batch(self, idx) -> dict:
        its = [self.items[i] for i in idx]
        B, Tm, Am = len(its), max(len(i[0]) for i in its), max(len(i[4]) for i in its)
        E = np.zeros((B, Tm, 384), np.float32)
        nid = np.zeros((B, Tm), np.int64)
        aid = np.zeros((B, Tm), np.int64)
        tmask = np.zeros((B, Tm), bool)
        Y = np.zeros((B, Am, len(ERROR_MODES)), np.float32)
        amask = np.zeros((B, Am), bool)
        for b, (e, n, a, y, ag) in enumerate(its):
            E[b, : len(e)], nid[b, : len(e)], aid[b, : len(e)], tmask[b, : len(e)] = e, n, a, True
            Y[b, : len(ag)], amask[b, : len(ag)] = y, True
        return {k: torch.tensor(v) for k, v in dict(E=E, nid=nid, aid=aid, tmask=tmask, Y=Y, amask=amask).items()}


# ---------------------------------------------------------------- model
def topk_mean(scores: torch.Tensor, member: torch.Tensor, k: int) -> torch.Tensor:
    """scores (B,A,T,C), member (B,A,T): mean of the top-k member scores per (B,A,C)."""
    s = scores.masked_fill(~member.unsqueeze(-1), float("-inf"))
    vals = s.topk(min(k, s.shape[2]), dim=2).values
    valid = torch.isfinite(vals)
    return vals.masked_fill(~valid, 0.0).sum(2) / valid.sum(2).clamp(min=1)


class DCLStyle(nn.Module):
    def __init__(self, modes0: np.ndarray, names0: np.ndarray, d: int = 128, rank: int = 64):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(384, d), nn.GELU(), nn.LayerNorm(d))
        self.to_proto = nn.Linear(384, d, bias=False)
        self.register_buffer("modes0", torch.tensor(modes0))
        self.register_buffer("names0", torch.tensor(names0))
        self.mode_delta = nn.Parameter(torch.zeros(modes0.shape[0], d))
        self.name_delta = nn.Parameter(torch.zeros(names0.shape[0], d))
        self.log_scale = nn.Parameter(torch.tensor(2.3))        # learnable temperature, init scale 10
        self.bias_a = nn.Parameter(torch.tensor(-1.0))
        self.bias_m = nn.Parameter(torch.full((modes0.shape[0],), -2.0))
        self.U = nn.Linear(d, rank, bias=False)
        self.V = nn.Linear(d, rank, bias=False)

    def forward(self, b: dict) -> dict:
        E, nid, aid, tmask = b["E"], b["nid"], b["aid"], b["tmask"]
        B, T, _ = E.shape
        A = b["amask"].shape[1]
        H = F.normalize(self.proj(E), dim=-1)                                       # B,T,d
        Q = F.normalize(self.to_proto(self.modes0) + self.mode_delta, dim=-1)       # 14,d
        P = F.normalize(self.to_proto(self.names0) + self.name_delta, dim=-1)       # V+1,d
        scale = self.log_scale.exp()
        sm = scale * H @ Q.T + self.bias_m                                          # B,T,14
        sa = scale * (H * P[nid]).sum(-1, keepdim=True) + self.bias_a               # B,T,1
        member = (aid.unsqueeze(1) == torch.arange(A).view(1, A, 1)) & tmask.unsqueeze(1)   # B,A,T
        za = topk_mean(sa.unsqueeze(1).expand(B, A, T, 1), member, K_TOP).squeeze(-1)       # B,A
        zam = topk_mean(sm.unsqueeze(1).expand(B, A, T, sm.shape[-1]), member, K_TOP)       # B,A,14
        ze = topk_mean(sm.unsqueeze(1), tmask.unsqueeze(1), K_TOP).squeeze(1)               # B,14
        # evidence vector per agent: mean of its top-K turns by agent score (for contrastive loss and the bilinear term)
        masked = sa.squeeze(-1).unsqueeze(1).expand(B, A, T).masked_fill(~member, float("-inf"))
        top = masked.topk(min(K_TOP, T), dim=2)
        gath = torch.gather(H.unsqueeze(1).expand(B, A, T, H.shape[-1]), 2,
                            top.indices.unsqueeze(-1).expand(-1, -1, -1, H.shape[-1]))
        valid = torch.isfinite(top.values).unsqueeze(-1)
        ev = (gath * valid).sum(2) / valid.sum(2).clamp(min=1)                              # B,A,d
        bil = (self.U(ev).unsqueeze(2) * self.V(Q).view(1, 1, Q.shape[0], -1)).sum(-1) / 8.0   # B,A,14
        return {"za": za, "zam": zam + bil, "ze": ze, "ev": ev}


def asl(logits, target, mask, gamma_neg: float = 2.0, clip: float = 0.05):
    p = torch.sigmoid(logits)
    pos = target * torch.log(p.clamp(min=1e-8))
    pn = (p - clip).clamp(min=0)
    neg = (1 - target) * (pn ** gamma_neg) * torch.log((1 - pn).clamp(min=1e-8))
    loss = -(pos + neg)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def supcon(ev, Y, amask, tau: float = 0.07):
    faulty = amask & (Y.sum(-1) > 0)
    z = F.normalize(ev[faulty], dim=-1)
    y = Y[faulty]
    n = z.shape[0]
    if n < 2:
        return ev.sum() * 0.0
    sim = z @ z.T / tau
    eye = torch.eye(n, dtype=torch.bool)
    sim = sim.masked_fill(eye, float("-inf"))
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = ((y @ y.T) > 0) & ~eye
    cnt = pos.sum(1)
    ok = cnt > 0
    if ok.sum() == 0:
        return ev.sum() * 0.0
    return (-(logp.masked_fill(~pos, 0.0).sum(1)[ok] / cnt[ok])).mean()


def losses(out: dict, b: dict, w=(1.0, 0.4, 0.6)) -> tuple:
    Y, amask = b["Y"], b["amask"]
    ya = (Y.sum(-1) > 0).float()
    ye = (Y.sum(1) > 0).float()
    m_pair = amask.unsqueeze(-1).float().expand_as(Y)
    cls = asl(out["zam"], Y, m_pair) + asl(out["za"], ya, amask.float()) + asl(out["ze"], ye, torch.ones_like(ye))
    con = supcon(out["ev"], Y, amask)
    pa, pm, pe = torch.sigmoid(out["za"]).unsqueeze(-1), torch.sigmoid(out["zam"]), torch.sigmoid(out["ze"]).unsqueeze(1)
    hier = (F.relu(pm - torch.minimum(pa, pe)) ** 2 * m_pair).sum() / m_pair.sum().clamp(min=1)
    return w[0] * cls + w[1] * con + w[2] * hier, (cls.item(), con.item(), hier.item())


# ---------------------------------------------------------------- eval
def pair_probs(model, batcher, order, bs: int = 128) -> list:
    model.eval()
    res = []
    with torch.no_grad():
        for i in range(0, len(order), bs):
            idx = order[i: i + bs]
            b = batcher.batch(idx)
            out = model(b)
            p = torch.sigmoid(out["zam"]).numpy()
            for j, ti in enumerate(idx):
                res.append(p[j, : len(batcher.items[ti][4])])
    return res


def decode(probs: list, batcher, order, thr: float) -> list:
    preds = []
    for p, ti in zip(probs, order):
        agents = batcher.items[ti][4]
        sel = {(agents[a], ERROR_MODES[m]) for a, m in zip(*np.where(p >= thr))}
        if not sel:
            a, m = np.unravel_index(np.argmax(p), p.shape)
            sel = {(agents[a], ERROR_MODES[m])}
        preds.append(sel)
    return preds


def run(name: str, splits: dict, store: EmbStore, seeds: list, epochs: int = 20, patience: int = 5,
        eval_sets: dict | None = None) -> None:
    vocab = agent_vocab(splits["train"])
    modes0, names0 = init_prototypes(vocab)
    tr, va = Batcher(splits["train"], store, vocab), Batcher(splits["val"], store, vocab)
    truth = [t.labels for t in splits["val"]]
    vorder = list(range(len(splits["val"])))
    extra = {k: (Batcher(v, store, vocab), [t.labels for t in v]) for k, v in (eval_sets or {}).items()}
    print(f"[{name}] train {len(tr.items)} val {len(va.items)} vocab {len(vocab)}", flush=True)
    allres = []
    for seed in seeds:
        set_seed(seed)
        t0 = time.time()
        model = DCLStyle(modes0, names0)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        steps_total = epochs * ((len(tr.items) + 63) // 64)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / (0.1 * steps_total)) * 0.5 * (1 + np.cos(np.pi * s / steps_total)))
        best, state, bad = 1e9, None, 0
        for ep in range(epochs):
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
                if bad >= patience:
                    break
        model.load_state_dict(state)
        probs = pair_probs(model, va, vorder)
        bestres = None
        for thr in GRID:
            r = evaluate(truth, decode(probs, va, vorder, thr))
            sc = sum(r[l]["micro_f1"] for l in ("pair", "agent", "error")) / 3
            if bestres is None or sc > bestres[0]:
                bestres = (sc, thr, r)
        _, thr, r = bestres
        allres.append(r)
        final = {}
        for en, (eb, etruth) in extra.items():   # one-shot evaluation with the config frozen on val
            eorder = list(range(len(eb.items)))
            final[en] = evaluate(etruth, decode(pair_probs(model, eb, eorder), eb, eorder, thr))
            e = final[en]
            print(f"[{name}] seed {seed} {en}: pair {100*e['pair']['micro_f1']:.2f}/{100*e['pair']['macro_f1']:.2f} "
                  f"agent {100*e['agent']['micro_f1']:.2f}/{100*e['agent']['macro_f1']:.2f} "
                  f"error {100*e['error']['micro_f1']:.2f}/{100*e['error']['macro_f1']:.2f}", flush=True)
        print(f"[{name}] seed {seed}: epochs {ep + 1} thr {thr}  pair {100*r['pair']['micro_f1']:.2f}/{100*r['pair']['macro_f1']:.2f} "
              f"agent {100*r['agent']['micro_f1']:.2f}/{100*r['agent']['macro_f1']:.2f} "
              f"error {100*r['error']['micro_f1']:.2f}/{100*r['error']['macro_f1']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        log_run(phase="3b", model="dcl_style_frozen", seed=seed, split=f"val_{name}", n_params=n_params,
                train_seconds=time.time() - t0, config={"split": name, "d": 128, "K": K_TOP, "lr": 1e-3, "epochs_run": ep + 1, "thr": thr,
                                                         "loss_w": [1.0, 0.4, 0.6]},
                metrics={"val": {l: r[l] for l in ("pair", "agent", "error")},
                         **{en: {l: e[l] for l in ("pair", "agent", "error")} for en, e in final.items()}},
                notes="DCL-style, frozen MiniLM from cache; threshold tuned on val" + ("; test/who_when evaluated once (published-row reproduction)" if final else "; no test eval"))
    for lvl in ("pair", "agent", "error"):
        for k in ("micro_f1", "macro_f1"):
            v = [100 * r[lvl][k] for r in allres]
            print(f"[{name}] {lvl:5s} {k}: {np.mean(v):.2f} +- {np.std(v):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--splits", default="paper,grouped")
    ap.add_argument("--eval-test", action="store_true", help="also score test and Who&When once (published-row reproduction)")
    args = ap.parse_args()
    store = EmbStore()
    sp = {"paper": load_paper_splits, "grouped": load_grouped_splits}
    for nm in args.splits.split(","):
        splits = sp[nm]()
        ev = {"test": splits["test"], "who_when": load_split("who_when")} if args.eval_test else None
        run(nm, splits, store, list(range(args.seeds)), eval_sets=ev)
