"""Shared helpers for the "model shortlist + retrieval + LLM judge" idea (Track A; see local-notes/rerank_rag_context.md).

Track A rule: Who&When is NEVER used for training, tuning, thresholds, retrieval index or prompt examples. Everything here that touches Who&When
(shortlists, label-shift) uses unlabeled inputs and the model's own probabilities only; Who&When labels appear ONLY in the scoring functions
(recall@k diagnostics, final scoring) and are never read by any tuning code.

Contents
  retrieval_block   - leave-same-query-out nearest-neighbour feature block (57 numbers per agent row) for the retriever-only model
  apply_shift       - Saerens-style label-shift correction of Who&When mode probabilities (identical to ww_block.py)
  load_dump / dump  - read/write the per-agent probability dump (npz) produced by rr_models.py
  shortlist         - top-k (agent, mode) candidates per trajectory from a probability matrix
  pairs_from_mask   - helper to turn per-record candidate selections into prediction sets
  bootstrap_ci      - percentile bootstrap over records for pair/agent/error micro F1 (and paired differences)
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from ..metrics import evaluate
from ..schema import ERROR_MODES
from .ww_focus import estimate_prior

NM = len(ERROR_MODES)
LV = ("pair", "agent", "error")
RR_DIR = Path(os.environ.get("RR_DIR") or Path(__file__).resolve().parents[2] / "results" / "rr")   # RR_DIR env var: redirect for smoke tests
RETR_DIM = NM + 1 + 3 * NM        # 15 neighbour-label averages + 3 numbers per mode


# ----------------------------------------------------------------------------------------------------------------------------------
# Who&When agent-name normalisation: a SEPARATE, disclosed scoring row (never replaces the raw-name row that is comparable with the paper).
# One fixed rule, applied to predictions AND labels together: strip a trailing parenthetical ("Orchestrator (thought)" -> "Orchestrator"), lower-case.
# It was chosen after noticing that 20/184 Who&When true pairs name an agent absent under that exact name in the turns (post-hoc data-handling choice).
# ----------------------------------------------------------------------------------------------------------------------------------
def norm_agent(a: str) -> str:
    return re.sub(r"\s*\(.*\)\s*$", "", a).strip().lower()


def norm_pairs(pairs) -> set:
    return {(norm_agent(a), m) for a, m in pairs}


# ----------------------------------------------------------------------------------------------------------------------------------
# retrieval block (retriever-only variant, section 4.5 of rerank_rag_context.md)
# ----------------------------------------------------------------------------------------------------------------------------------
def _unit_t(a):
    import torch

    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    return t / t.norm(dim=1, keepdim=True).clamp(min=1e-9)


def retrieval_block(Fq: np.ndarray, Fi: np.ndarray, Ti: np.ndarray, qid_q=None, qid_i=None, k: int = 25, r: int = 3,
                    topn: int = 50, chunk: int = 1000, temp: float = 10.0) -> np.ndarray:
    """What do the most similar AEGIS-TRAIN agents say? Returns len(Fq) x RETR_DIM features.

    Fq: query agent vectors (rows x E); Fi: index agent vectors (AEGIS train rows x E); Ti: index targets (14 modes + agent flag, 0/1).
    qid_q / qid_i: integer query ids. When both are given, index rows with the SAME query as the query row are masked out (used for train rows,
    so a train agent never retrieves its own conversation or a sibling conversation of the same task). val/test/Who&When rows pass None
    (their queries are disjoint from train by construction of the grouped split; Who&When is never in the index).

    Features per row:
      [0:15]  similarity-weighted label average over the k nearest index rows (14 modes + faulty flag), weights softmax(temp * cosine)
      then for every mode m (3 numbers):
        nearest-positive cosine, mean cosine of the r nearest positives (index agents with mode m),
        contrast = (mean r nearest positives) - (mean of the r nearest NEGATIVES within the topn neighbours; negatives = agents without mode m).
    """
    import torch

    Fi_t, Ti_t = _unit_t(Fi), torch.from_numpy(np.ascontiguousarray(Ti, dtype=np.float32))
    Yi = Ti_t[:, :NM] > 0.5
    pos_idx = [torch.nonzero(Yi[:, m]).squeeze(1) for m in range(NM)]
    qi = torch.from_numpy(np.asarray(qid_i)) if qid_i is not None else None
    out = []
    for s in range(0, len(Fq), chunk):
        q = _unit_t(Fq[s:s + chunk])
        S = q @ Fi_t.T                                                    # c x N cosine
        if qid_q is not None and qi is not None:
            same = torch.from_numpy(np.asarray(qid_q[s:s + chunk]))[:, None] == qi[None, :]
            S = S.masked_fill(same, -1.0)
        vals, idx = S.topk(min(topn, S.shape[1]), dim=1)                   # c x topn
        w = torch.softmax(vals[:, :k] * temp, dim=1)
        lab = (w.unsqueeze(2) * Ti_t[idx[:, :k]]).sum(1)                   # c x 15
        Yn = Yi[idx]                                                       # c x topn x 14
        feats = [lab]
        for m in range(NM):
            P = pos_idx[m]
            if len(P) > 0:
                tv = S[:, P].topk(min(r, len(P)), dim=1).values
                pos1, pos_mean = tv[:, 0], tv.mean(1)
            else:
                pos1 = pos_mean = torch.full((S.shape[0],), -1.0)
            nv = torch.where(~Yn[:, :, m], vals, torch.full_like(vals, -2.0))
            nt = nv.topk(min(r, nv.shape[1]), dim=1).values
            nt = torch.where(nt < -1.5, vals[:, -1:].expand_as(nt), nt)     # fewer than r negatives in topn: use the topn-th similarity
            feats += [pos1.unsqueeze(1), pos_mean.unsqueeze(1), (pos_mean - nt.mean(1)).unsqueeze(1)]
        out.append(torch.cat(feats, dim=1).numpy())
    return np.concatenate(out).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------------------------------
# label shift (same arithmetic as ww_block.py: prior EM on unlabeled predictions, weights normalised to mean 1, clipped)
# ----------------------------------------------------------------------------------------------------------------------------------
def apply_shift(p: np.ndarray, meta: list, n_traj: int, src_prior: np.ndarray) -> np.ndarray:
    pri = estimate_prior(p, meta, n_traj, src_prior)
    w = pri / src_prior
    w = w / w.mean()
    q = p.copy()
    q[:, :-1] = np.clip(q[:, :-1] * w, 0, 1)
    return q


# ----------------------------------------------------------------------------------------------------------------------------------
# probability dump (npz) written by rr_models.py
# ----------------------------------------------------------------------------------------------------------------------------------
def save_dump(path: Path, probs: dict, meta: dict, ids: dict, src_prior: np.ndarray, info: dict) -> None:
    """probs: (seed, mask, split) -> rows x 15 ensemble probabilities. meta[split]: list of (traj_index, agent). ids[split]: trajectory ids."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {f"p|{s}|{m}|{k}": v.astype(np.float32) for (s, m, k), v in probs.items()}
    for k, mt in meta.items():
        arrs[f"ti|{k}"] = np.asarray([ti for ti, _ in mt], dtype=np.int64)
        arrs[f"ag|{k}"] = np.asarray([a for _, a in mt], dtype=str)
        arrs[f"ids|{k}"] = np.asarray(ids[k], dtype=str)
    arrs["src_prior"] = np.asarray(src_prior, dtype=np.float64)
    arrs["info"] = np.asarray(json.dumps(info))
    np.savez_compressed(path, **arrs)


def load_dump(path) -> dict:
    """Returns {'p': {(seed, mask, split): array}, 'meta': {split: [(ti, agent)]}, 'ids': {...}, 'src_prior': arr, 'info': dict}."""
    import json

    z = np.load(path, allow_pickle=False)
    d = {"p": {}, "meta": {}, "ids": {}, "src_prior": z["src_prior"], "info": json.loads(str(z["info"]))}
    for key in z.files:
        if key.startswith("p|"):
            _, s, m, k = key.split("|")
            d["p"][(int(s), m, k)] = z[key]
        elif key.startswith("ti|"):
            k = key.split("|")[1]
            d["meta"][k] = list(zip(z[key].tolist(), z[f"ag|{k}"].tolist()))
        elif key.startswith("ids|"):
            d["ids"][key.split("|")[1]] = z[key].tolist()
    return d


def mean_probs(dump: dict, split: str, mask: str = "names+peer") -> np.ndarray:
    """Average over the head seeds stored in the dump (each seed is already an ensemble over encoders)."""
    seeds = sorted({s for (s, m, k) in dump["p"] if m == mask and k == split})
    return np.mean([dump["p"][(s, mask, split)] for s in seeds], axis=0)


def split_probs(dump: dict, split: str, mask: str = "names+peer", shift: bool | None = None) -> np.ndarray:
    """Model probabilities for a split. Who&When gets label-shift by default (transductive, no labels); AEGIS splits never do."""
    p = mean_probs(dump, split, mask)
    if shift is None:
        shift = split == "who_when"
    if shift:
        n_traj = max(ti for ti, _ in dump["meta"][split]) + 1
        p = apply_shift(p, dump["meta"][split], n_traj, dump["src_prior"])
    return p


# ----------------------------------------------------------------------------------------------------------------------------------
# candidates
# ----------------------------------------------------------------------------------------------------------------------------------
def all_candidates(probs: np.ndarray, meta: list, n_traj: int) -> list:
    """Per trajectory: list of (score, agent, mode_index, p_agent, p_mode) over every (agent, mode) pair, sorted by score = p_agent * p_mode."""
    cand = [[] for _ in range(n_traj)]
    for (ti, a), p in zip(meta, probs):
        pa, pm = float(p[-1]), p[:-1]
        for mi in range(NM):
            cand[ti].append((pa * float(pm[mi]), a, mi, pa, float(pm[mi])))
    for ti in range(n_traj):
        cand[ti].sort(key=lambda c: -c[0])
    return cand


def shortlist(probs: np.ndarray, meta: list, n_traj: int, k: int) -> list:
    """Top-k candidates per trajectory (k <= 0 keeps all pairs)."""
    cand = all_candidates(probs, meta, n_traj)
    return cand if k <= 0 else [c[:k] for c in cand]


def pairs_of(cands: list) -> set:
    return {(c[1], ERROR_MODES[c[2]]) for c in cands}


# ----------------------------------------------------------------------------------------------------------------------------------
# scoring helpers
# ----------------------------------------------------------------------------------------------------------------------------------
def micro(res: dict) -> tuple:
    return tuple(100 * res[l]["micro_f1"] for l in LV)


def bootstrap_ci(truth: list, preds: dict, n: int = 1000, seed: int = 0, ref: str | None = None) -> dict:
    """Percentile bootstrap over RECORDS. truth: list of label sets; preds: {row name: list of predicted pair sets (same order)}.
    Returns {name: {level: (mean, lo, hi)}} for pair/agent/error micro F1 (percent); if ref is given also {name+'-'+ref: paired difference CI}."""
    rng = np.random.default_rng(seed)
    N = len(truth)
    samples = {name: [] for name in preds}
    for _ in range(n):
        idx = rng.integers(0, N, N)
        t = [truth[i] for i in idx]
        for name, pr in preds.items():
            samples[name].append(micro(evaluate(t, [pr[i] for i in idx])))
    out = {}
    for name, arr in samples.items():
        a = np.asarray(arr)
        out[name] = {l: (float(a[:, j].mean()), float(np.percentile(a[:, j], 2.5)), float(np.percentile(a[:, j], 97.5))) for j, l in enumerate(LV)}
    if ref is not None and ref in samples:
        r = np.asarray(samples[ref])
        for name, arr in samples.items():
            if name == ref:
                continue
            d = np.asarray(arr) - r
            out[f"{name}-minus-{ref}"] = {l: (float(d[:, j].mean()), float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5)))
                                          for j, l in enumerate(LV)}
    return out
