"""Build the shortlist + labelled retrieval examples that the LLM judge (aegis/scripts/kaggle_judge.py) reads. CPU only, no LLM, no Who&When label.

For every trajectory of the requested splits it keeps the top-k (agent, mode) candidates by p_agent * p_mode (Who&When probabilities get the same
transductive label-shift as everywhere else) and, for each candidate, attaches two example sets:
  examples_rag  : MODE-FIRST contrastive retrieval from AEGIS GROUPED TRAIN only. For candidate (agent a, mode m): embed a (fine-tuned MiniLM mean turn vector),
                  fetch the r nearest index agents that HAVE mode m (positives) and the r nearest that do NOT (negatives); the second of each kind is taken
                  from a different framework than the first when one exists among the nearest 40. Each example carries its own ground-truth labels
                  (labels of OTHER AEGIS train conversations, never of the conversation being judged). If the best positive cosine is below --min-sim the
                  examples are dropped for that candidate (weak-match fallback = plain judge).
  examples_rand : control with the same format and the same number/kind of examples (r positives, r negatives for mode m) but chosen at random from the index.
Leakage rules (asserted): index = AEGIS grouped train agent rows only; no example comes from the same query as the judged record; Who&When is never indexed.
The released data has only an injection STRATEGY tag per faulty agent (no free-text injection plan), so examples show the agent's labelled modes, not a rationale.

Output: aegis/results/rr/cand_<tag>_<split>.jsonl, one line per trajectory:
  {id, split, framework, cands: [{agent, mode, rank, p_agent, p_mode, score, examples_rag: [...], examples_rand: [...]}]}
  example = {label (1 if this example agent has the candidate mode, else 0), framework, agent, modes (all labelled modes of that agent), snippet, sim}

Run: python -m aegis.src.eval.rr_candidates --dump aegis/results/rr/dump_rr_base.npz --tag rr_base --k 5 --retr-enc es_s0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random

import numpy as np

from ..baselines.flat_mlp import make_targets
from ..features import EmbStore, agent_vocab, build
from ..loaders import load_grouped_splits, load_split
from ..schema import ERROR_MODES
from .rr_common import NM, RR_DIR, load_dump, shortlist, split_probs
from .ww_focus import enc_dir

PER_TURN_CHARS = 260
MAX_TURNS = 4


def agent_snippet(traj, agent: str) -> str:
    """About 200-300 tokens of the agent's own turns: the first two and the last two (each cut to PER_TURN_CHARS)."""
    ts = [u.text.strip().replace("\n", " ") for u in traj.turns if u.agent == agent]
    if len(ts) > MAX_TURNS:
        ts = ts[: MAX_TURNS // 2] + ts[-(MAX_TURNS // 2):]
    return " | ".join(t[:PER_TURN_CHARS] for t in ts)


def agent_modes(traj, agent: str) -> list:
    return sorted(m for a, m in traj.labels if a == agent)


def seed_of(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def pick(rows: np.ndarray, sims: np.ndarray, fw: np.ndarray, r: int) -> list:
    """rows: candidate index rows (positives or negatives), sims: their cosine to the query. Nearest first; the 2nd pick prefers another framework."""
    if len(rows) == 0:
        return []
    top = rows[np.argsort(-sims[rows])[:40]]
    chosen = [int(top[0])]
    for j in top[1:]:
        if len(chosen) >= r:
            break
        if fw[j] != fw[chosen[0]] or len(top) <= r:
            chosen.append(int(j))
    for j in top[1:]:
        if len(chosen) >= r:
            break
        if int(j) not in chosen:
            chosen.append(int(j))
    return chosen[:r]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=5, help="shortlist size per trajectory (0 = all agent-mode pairs)")
    ap.add_argument("--r", type=int, default=2, help="positive and negative examples per candidate")
    ap.add_argument("--min-sim", type=float, default=0.0, help="drop the RAG examples of a candidate if its best positive cosine is below this")
    ap.add_argument("--retr-enc", default="es_s0", help="encoder whose fine-tuned agent vectors are the retrieval space")
    ap.add_argument("--splits", default="val,test,who_when")
    ap.add_argument("--mask", default="names+peer")
    ap.add_argument("--max-records", type=int, default=0, help="debug: first N trajectories per split")
    ap.add_argument("--subsample", type=int, default=0, help="judge only a seeded random subset of N trajectories of val and test (the judge is expensive; the "
                    "released order is grouped by framework, so a plain prefix would be single-framework). Who&When always keeps all records.")
    ap.add_argument("--subsample-seed", type=int, default=0)
    args = ap.parse_args()
    d = load_dump(args.dump)
    sp = load_grouped_splits()
    trajs = {"train": sp["train"], "val": sp["val"], "test": load_split("test"), "who_when": load_split("who_when")}
    store = EmbStore(enc_dir(args.retr_enc))
    vocab = agent_vocab(sp["train"])
    Xtr, Ytr, mtr = build(trajs["train"], store, vocab)
    E = store.dim
    Fi = Xtr[:, :E] / np.clip(np.linalg.norm(Xtr[:, :E], axis=1, keepdims=True), 1e-9, None)
    Ti = make_targets(Ytr)
    fw_i = np.asarray([trajs["train"][ti].framework for ti, _ in mtr])
    qry_i = np.asarray([trajs["train"][ti].query for ti, _ in mtr])
    pos_rows = [np.nonzero(Ti[:, m] > 0.5)[0] for m in range(NM)]
    neg_rows = [np.nonzero(Ti[:, m] <= 0.5)[0] for m in range(NM)]
    RR_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in args.splits.split(","):
        ts = trajs[split]
        ids = d["ids"][split]
        ts = ts[: len(ids)]
        assert [t.id for t in ts] == ids, f"dump ids do not match the {split} split"
        n = args.max_records or len(ts)
        ts = ts[:n]
        sel = list(range(len(ts)))
        if args.subsample and split != "who_when" and args.subsample < len(ts):
            sel = sorted(random.Random(args.subsample_seed).sample(sel, args.subsample))
        probs = split_probs(d, split, args.mask)
        meta = d["meta"][split]
        sl = shortlist(probs, meta, len(d["ids"][split]), args.k)
        Xs, _, ms = build(ts, store, vocab)
        row_of = {m: i for i, m in enumerate(ms)}
        sim_cache: dict = {}
        n_drop = n_cand = 0
        with (RR_DIR / f"cand_{args.tag}_{split}.jsonl").open("w", encoding="utf-8") as f:
            for ti in sel:
                t = ts[ti]
                out = {"id": t.id, "split": split, "framework": t.framework, "cands": []}
                for rank, (score, agent, mi, pa, pm) in enumerate(sl[ti]):
                    c = {"agent": agent, "mode": ERROR_MODES[mi], "rank": rank + 1, "p_agent": pa, "p_mode": pm, "score": score}
                    ri = row_of[(ti, agent)]
                    if ri not in sim_cache:
                        q = Xs[ri, :E]
                        sim_cache[ri] = Fi @ (q / max(float(np.linalg.norm(q)), 1e-9))
                    sims = sim_cache[ri]
                    rng = random.Random(seed_of(t.id, agent, mi))

                    def build_ex(rows_idx):
                        exs = []
                        for j in rows_idx:
                            ti_j, a_j = mtr[j]
                            tj = trajs["train"][ti_j]
                            assert split == "who_when" or tj.query != t.query, "retrieval leaked a same-query example"
                            exs.append({"label": int(Ti[j, mi] > 0.5), "framework": tj.framework, "agent": a_j, "modes": agent_modes(tj, a_j),
                                        "snippet": agent_snippet(tj, a_j), "sim": float(sims[j])})
                        rng.shuffle(exs)
                        return exs

                    P = pick(pos_rows[mi], sims, fw_i, args.r)
                    N = pick(neg_rows[mi], sims, fw_i, args.r)
                    n_cand += 1
                    if P and float(sims[P[0]]) >= args.min_sim:
                        c["examples_rag"] = build_ex(P + N)
                    else:
                        c["examples_rag"] = []
                        n_drop += 1
                    rp = rng.sample(list(pos_rows[mi]), min(args.r, len(pos_rows[mi])))
                    rn = rng.sample(list(neg_rows[mi]), min(args.r, len(neg_rows[mi])))
                    c["examples_rand"] = build_ex(rp + rn)
                    out["cands"].append(c)
                f.write(json.dumps(out) + "\n")
        stats[split] = {"records": len(sel), "candidates": n_cand, "rag_examples_dropped_weak_match": n_drop}
        print(f"[{split}] {len(sel)} records, {n_cand} candidates, RAG examples dropped for weak match: {n_drop}", flush=True)
    (RR_DIR / f"cand_{args.tag}_info.json").write_text(json.dumps({"args": vars(args), "stats": stats, "dump_info": d["info"]}, indent=1))


if __name__ == "__main__":
    main()
