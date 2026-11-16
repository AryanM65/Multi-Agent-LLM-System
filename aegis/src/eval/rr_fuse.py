"""Fuse the trained model with the frozen-LLM judge scores, tune ONLY on AEGIS validation, pre-commit, then score once. CPU only.

Inputs
  aegis/results/rr/dump_<dump tag>.npz           model probabilities (rr_models.py)
  aegis/results/rr/cand_<tag>_<split>.jsonl      shortlist + examples (rr_candidates.py)
  aegis/results/rr/judge_<variant>_<split>*.jsonl  judge log-odds J from kaggle_judge.py (variants plain, rag, rand; several chunk files are concatenated)

Fused score of candidate (agent, mode):   s = wm * logit(p_agent * p_mode)  +  sum_v  w_v * z_v      (wm = 1 unless stated)
  z_v = the judge log-odds J_v z-scored over all candidates of the SAME split (unlabeled statistic; transductive in the same weak sense as label-shift)
Decoding: a candidate is predicted if s >= thr (one global threshold). A trajectory with no candidate above thr predicts its single best candidate
(every record has a fault). Candidates outside the shortlist are never predicted.

Rows (all reported, none hidden):
  model_alone_prod   the production two-threshold decoder on ALL pairs (reference; thresholds tuned on val)
  shortlist_model    s = logit(p) on the shortlist only (what the shortlist alone gives)
  plain / rag / rand  model + that judge variant (rand = random-examples control for the retrieval claim)
  plain+rag          model + both judges
  llm_only_plain / llm_only_rag   rank by the judge alone inside the shortlist (wm = 0)
Weights w_v come from a small grid and are chosen by 2-fold CROSS-FITTING on AEGIS validation (halves split by QUERY: weights chosen on the score of
a half using the threshold tuned on the other half), then the final threshold is tuned on the whole validation set.

Phases
  tune  : AEGIS validation only. Writes aegis/results/rr/precommit_<tag>.json (weights, thresholds, row definitions, headline row chosen by a fixed rule,
          timestamp, file hashes). Prints the approximate leave-one-framework-out check (Gate 1/2): the tuning is repeated with one validation framework
          held out; NOTE the encoders are not retrained for that, so it is weaker than a true leave-one-framework-out.
  final : refuses to run without the precommit file and refuses to overwrite an existing final_<tag>.json (use --force-rescore only on purpose).
          Scores AEGIS test and Who&When ONCE with the precommitted numbers, with bootstrap 95% intervals and paired differences vs model_alone_prod.
Who&When is never used to choose weights, thresholds, k or the headline row.

Run:  python -m aegis.src.eval.rr_fuse --phase tune  --dump-tag rr_base --cand-tag rr_base --variants plain,rag,rand --tag rrf1
      python -m aegis.src.eval.rr_fuse --phase final --dump-tag rr_base --cand-tag rr_base --variants plain,rag,rand --tag rrf1
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from datetime import datetime

import numpy as np

from ..loaders import load_grouped_splits, load_split
from ..metrics import evaluate
from ..schema import ERROR_MODES
from ..utils import log_run
from .improve_cpu import decode2, tune2
from .rr_common import LV, RR_DIR, bootstrap_ci, load_dump, micro, norm_pairs, split_probs

GRID_W = (0.0, 0.25, 0.5, 1.0, 2.0)
ROWS_ORDER = ["shortlist_model", "plain", "rag", "rand", "plain+rag", "llm_only_plain", "llm_only_rag"]
HEADLINE_POOL = ["shortlist_model", "plain", "rag", "plain+rag"]      # fixed rule: best cross-fit validation objective among these


def sha1(path) -> str:
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:12]


def obj(truth, pred) -> float:
    return float(np.mean(micro(evaluate(truth, pred))))


# ----------------------------------------------------------------------------------------------------------------------------------
# data assembly
# ----------------------------------------------------------------------------------------------------------------------------------
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def load_judge(variant: str, split: str) -> dict:
    out: dict = {}
    for p in sorted(RR_DIR.glob(f"judge_{variant}_{split}*.jsonl")):
        for r in read_jsonl(p):
            out[r["id"]] = {(s["agent"], s["mode"]): s["J"] for s in r["scores"]}
    return out


class Data:
    """Records present in the candidate file AND in every requested judge variant, in candidate-file order."""

    def __init__(self, split: str, cand_tag: str, variants: list, labels: dict, queries: dict, frameworks: dict):
        cands = read_jsonl(RR_DIR / f"cand_{cand_tag}_{split}.jsonl")
        judges = {v: load_judge(v, split) for v in variants}
        self.split, self.variants, self.ids, self.recs, self.dropped = split, variants, [], [], 0
        for c in cands:
            if any(c["id"] not in judges[v] for v in variants):
                self.dropped += 1
                continue
            ag = [x["agent"] for x in c["cands"]]
            md = [x["mode"] for x in c["cands"]]
            sc = np.clip(np.asarray([x["score"] for x in c["cands"]], dtype=np.float64), 1e-4, 1 - 1e-4)
            J = {v: np.asarray([judges[v][c["id"]][(a, m)] for a, m in zip(ag, md)], dtype=np.float64) for v in variants}
            self.ids.append(c["id"])
            self.recs.append({"agents": ag, "modes": md, "lp": np.log(sc / (1 - sc)), "J": J})
        self.truth = [labels[i] for i in self.ids]
        self.query = [queries[i] for i in self.ids]
        self.fw = [frameworks[i] for i in self.ids]
        # z-score each judge variant over all candidates of this split (no labels)
        for v in variants:
            allj = np.concatenate([r["J"][v] for r in self.recs]) if self.recs else np.zeros(1)
            mu, sd = allj.mean(), max(allj.std(), 1e-6)
            for r in self.recs:
                r.setdefault("z", {})[v] = (r["J"][v] - mu) / sd

    def fused(self, wm: float, w: dict) -> list:
        return [wm * r["lp"] + sum(w.get(v, 0.0) * r["z"][v] for v in self.variants) for r in self.recs]

    def decode(self, fused: list, thr: float, idx=None) -> list:
        idx = range(len(self.recs)) if idx is None else idx
        out = []
        for i in idx:
            r, s = self.recs[i], fused[i]
            keep = np.nonzero(s >= thr)[0]
            if len(keep) == 0:
                keep = [int(np.argmax(s))]
            out.append({(r["agents"][j], r["modes"][j]) for j in keep})
        return out


def best_thr(data: Data, fused: list, idx: list) -> tuple:
    allv = np.concatenate([fused[i] for i in idx])
    grid = np.unique(np.percentile(allv, np.linspace(40, 99.5, 70)))
    truth = [data.truth[i] for i in idx]
    best = (-1.0, None)
    for t in grid:
        s = obj(truth, data.decode(fused, float(t), idx))
        if s > best[0]:
            best = (s, float(t))
    return best[1], best[0]


def row_specs(variants: list) -> dict:
    """row name -> (wm, list of judge variants that get a weight grid, fixed weight for llm-only rows)."""
    spec = {"shortlist_model": (1.0, [], None)}
    for v in ("plain", "rag", "rand"):
        if v in variants:
            spec[v] = (1.0, [v], None)
    if "plain" in variants and "rag" in variants:
        spec["plain+rag"] = (1.0, ["plain", "rag"], None)
    for v in ("plain", "rag"):
        if v in variants:
            spec[f"llm_only_{v}"] = (0.0, [v], 1.0)
    return spec


def weight_combos(vs: list, fixed):
    if fixed is not None:
        return [{vs[0]: fixed}]
    if not vs:
        return [{}]
    return [dict(zip(vs, c)) for c in itertools.product(GRID_W, repeat=len(vs))]


def tune_row(data: Data, wm: float, vs: list, fixed, folds: list) -> dict:
    """Cross-fit the weights on AEGIS validation (halves given by `folds`), then tune the final threshold on all of it."""
    allidx = list(range(len(data.recs)))
    best = None
    for w in weight_combos(vs, fixed):
        f = data.fused(wm, w)
        cf = []
        for a, b in ((0, 1), (1, 0)):
            t, _ = best_thr(data, f, folds[a])
            cf.append(obj([data.truth[i] for i in folds[b]], data.decode(f, t, folds[b])))
        score = float(np.mean(cf))
        key = (score, -sum(w.values()))
        if best is None or key > best[0]:
            best = (key, w, score)
    _, w, score = best
    f = data.fused(wm, w)
    thr, in_sample = best_thr(data, f, allidx)
    return {"wm": wm, "weights": w, "thr": thr, "val_crossfit": score, "val_in_sample": in_sample}


def make_folds(data: Data) -> list:
    h = [int(hashlib.sha1(q.encode()).hexdigest(), 16) % 2 for q in data.query]
    return [[i for i, x in enumerate(h) if x == 0], [i for i, x in enumerate(h) if x == 1]]


# ----------------------------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["tune", "final"], required=True)
    ap.add_argument("--dump-tag", required=True)
    ap.add_argument("--cand-tag", required=True)
    ap.add_argument("--variants", default="plain,rag,rand")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mask", default="names+peer")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--final-splits", default="test,who_when", help="splits scored by --phase final (a run that judged only who_when needs who_when)")
    ap.add_argument("--force-rescore", action="store_true")
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]
    dump_path = RR_DIR / f"dump_{args.dump_tag}.npz"
    d = load_dump(dump_path)
    sp = load_grouped_splits()
    trajs = {"val": sp["val"], "test": load_split("test"), "who_when": load_split("who_when")}
    labels = {t.id: t.labels for ts in trajs.values() for t in ts}
    queries = {t.id: t.query for ts in trajs.values() for t in ts}
    frameworks = {t.id: t.framework for ts in trajs.values() for t in ts}
    spec = row_specs(variants)
    pre_path = RR_DIR / f"precommit_{args.tag}.json"
    final_path = RR_DIR / f"final_{args.tag}.json"

    probs = {k: split_probs(d, k, args.mask) for k in trajs}
    ids_all = {k: d["ids"][k] for k in trajs}
    _, ta, tm = tune2("prod", probs["val"], d["meta"]["val"], trajs["val"][: len(ids_all["val"])])

    def model_alone(split: str, ids: list) -> list:
        ts = trajs[split][: len(ids_all[split])]
        pred = decode2(probs[split], d["meta"][split], len(ts), "prod", ta, tm)
        pos = {t.id: i for i, t in enumerate(ts)}
        return [pred[pos[i]] for i in ids]

    # ------------------------------------------------------------------ tune
    if args.phase == "tune":
        val = Data("val", args.cand_tag, variants, labels, queries, frameworks)
        print(f"AEGIS val: {len(val.recs)} records with candidates and all judge variants ({val.dropped} dropped for missing judge scores)")
        assert len(val.recs) >= 50, "too few validation records have judge scores to tune on"
        folds = make_folds(val)
        rows = {name: tune_row(val, wm, vs, fixed, folds) for name, (wm, vs, fixed) in spec.items()}
        rows["model_alone_prod"] = {"t_a": ta, "t_m": float(tm)}
        truth_v = val.truth
        ma_v = model_alone("val", val.ids)
        print(f"\n=== tune on AEGIS val ({args.tag}); objective = mean of pair/agent/error micro F1 (percent) ===")
        print(f"- model_alone_prod (all pairs, tuned in sample): {obj(truth_v, ma_v):.2f}")
        for name in ROWS_ORDER:
            if name in rows:
                r = rows[name]
                f = val.fused(r["wm"], r["weights"])
                pr = val.decode(f, r["thr"])
                print(f"- {name}: weights {r['weights']} thr {r['thr']:.3f}; cross-fit {r['val_crossfit']:.2f}, in-sample {r['val_in_sample']:.2f} "
                      f"(pair/agent/error {' / '.join(f'{x:.2f}' for x in micro(evaluate(truth_v, pr)))})")
        pool = [n for n in HEADLINE_POOL if n in rows]
        headline = max(pool, key=lambda n: (rows[n]["val_crossfit"], -len(rows[n]["weights"])))
        print(f"- headline row (fixed rule: best cross-fit validation objective among {pool}): {headline}")

        # approximate leave-one-framework-out (Gate 1/2): tune on the other frameworks (in-sample thr), score the held-out one
        print("\n=== approximate leave-one-framework-out inside AEGIS validation (encoder NOT retrained) ===")
        lofo: dict = {n: [] for n in rows if n != "model_alone_prod"}
        for fw in sorted(set(val.fw)):
            te = [i for i, x in enumerate(val.fw) if x == fw]
            tr = [i for i, x in enumerate(val.fw) if x != fw]
            if len(te) < 30 or len(tr) < 50:
                continue
            for name, (wm, vs, fixed) in spec.items():
                best = None
                for w in weight_combos(vs, fixed):
                    f = val.fused(wm, w)
                    t, s = best_thr(val, f, tr)
                    if best is None or (s, -sum(w.values())) > best[0]:
                        best = ((s, -sum(w.values())), w, t)
                _, w, t = best
                f = val.fused(wm, w)
                lofo[name].append((fw, len(te), obj([val.truth[i] for i in te], val.decode(f, t, te)), micro(evaluate([val.truth[i] for i in te], val.decode(f, t, te)))[0]))
        for name, res in lofo.items():
            if res:
                print(f"- {name}: held-out frameworks " + ", ".join(f"{fw}({n}) obj {o:.1f}/pair {p:.1f}" for fw, n, o, p in res) +
                      f"; mean obj {np.mean([o for _, _, o, _ in res]):.2f}, mean pair {np.mean([p for _, _, _, p in res]):.2f}")
        def m(n):
            return np.mean([o for _, _, o, _ in lofo[n]]) if lofo.get(n) else float("nan")
        if "plain" in lofo:
            print(f"- Gate 1 (plain judge fused beats shortlist_model on held-out frameworks): {'PASS' if m('plain') > m('shortlist_model') else 'FAIL'} "
                  f"({m('plain'):.2f} vs {m('shortlist_model'):.2f})")
        if "rag" in lofo and "rand" in lofo and "plain" in lofo:
            print(f"- Gate 2 (rag beats the random-examples control and plain): {'PASS' if m('rag') > max(m('rand'), m('plain')) else 'FAIL'} "
                  f"(rag {m('rag'):.2f}, rand {m('rand'):.2f}, plain {m('plain'):.2f})")

        pre = {"timestamp": datetime.now().isoformat(timespec="seconds"), "tag": args.tag, "dump": str(dump_path), "dump_sha1": sha1(dump_path),
               "cand_tag": args.cand_tag, "cand_files": {s: sha1(RR_DIR / f"cand_{args.cand_tag}_{s}.jsonl") for s in ("val", "test", "who_when")
                                                           if (RR_DIR / f"cand_{args.cand_tag}_{s}.jsonl").exists()},
               "variants": variants, "mask": args.mask, "grid": list(GRID_W), "rows": rows, "headline_row": headline,
               "candidate_info": json.loads((RR_DIR / f"cand_{args.cand_tag}_info.json").read_text()) if (RR_DIR / f"cand_{args.cand_tag}_info.json").exists() else None,
               "judge_run_info": [json.loads(p.read_text()) for p in sorted(RR_DIR.glob("run_info_*.json"))],
               "lofo_approx": {n: [(fw, n_, o) for fw, n_, o, _ in r] for n, r in lofo.items()},
               "rule": "chosen on AEGIS validation only; Who&When not read in this phase"}
        pre_path.write_text(json.dumps(pre, indent=1, default=float))
        print(f"\nwrote {pre_path}. Next: python -m aegis.src.eval.rr_fuse --phase final --dump-tag {args.dump_tag} --cand-tag {args.cand_tag} "
              f"--variants {args.variants} --tag {args.tag}")
        return

    # ------------------------------------------------------------------ final
    if not pre_path.exists():
        raise SystemExit(f"refusing to score Who&When: no precommit file {pre_path}. Run --phase tune first.")
    if final_path.exists() and not args.force_rescore:
        raise SystemExit(f"{final_path} exists: Who&When/test were already scored once for this tag. Use a new --tag or --force-rescore on purpose.")
    pre = json.loads(pre_path.read_text())
    assert pre["dump_sha1"] == sha1(dump_path), "the probability dump changed after the precommit"
    assert pre["variants"] == variants, "variants differ from the precommit"
    report = {"tag": args.tag, "precommit": str(pre_path), "precommit_time": pre["timestamp"], "headline_row": pre["headline_row"], "splits": {}}
    for split in args.final_splits.split(","):
        data = Data(split, args.cand_tag, variants, labels, queries, frameworks)
        preds = {"model_alone_prod": model_alone(split, data.ids)}
        for name, r in pre["rows"].items():
            if name == "model_alone_prod":
                continue
            preds[name] = data.decode(data.fused(r["wm"], r["weights"]), r["thr"])
        ci = bootstrap_ci(data.truth, preds, n=args.boot, ref="model_alone_prod")
        report["splits"][split] = {"records": len(data.recs), "dropped": data.dropped, "point": {n: dict(zip(LV, micro(evaluate(data.truth, p)))) for n, p in preds.items()}, "ci": ci}
        print(f"\n=== {split}: {len(data.recs)} records ({data.dropped} dropped for missing judge scores); point estimate [95% bootstrap CI], micro F1 pair/agent/error ===")
        for name in ["model_alone_prod"] + [n for n in ROWS_ORDER if n in preds]:
            pt = report["splits"][split]["point"][name]
            print(f"- {name}{' (HEADLINE)' if name == pre['headline_row'] else ''}: " + "; ".join(
                f"{l} {pt[l]:.2f} [{ci[name][l][1]:.2f}, {ci[name][l][2]:.2f}]" for l in LV))
        for key, v in ci.items():
            if key.endswith("minus-model_alone_prod"):
                print(f"  - paired difference {key}: " + "; ".join(f"{l} {v[l][0]:+.2f} [{v[l][1]:+.2f}, {v[l][2]:+.2f}]" for l in LV))
        if split == "who_when":    # separate, disclosed row: agent names normalised on predictions AND labels (rr_common.norm_agent); raw-name rows above stay the headline
            tn = [norm_pairs(t) for t in data.truth]
            pn = {n: [norm_pairs(x) for x in p] for n, p in preds.items()}
            cin = bootstrap_ci(tn, pn, n=args.boot, ref="model_alone_prod")
            report["splits"]["who_when_name_normalised"] = {"point": {n: dict(zip(LV, micro(evaluate(tn, p)))) for n, p in pn.items()}, "ci": cin}
            print(chr(10) + "=== who_when, NAME-NORMALISED scoring (separate row; post-hoc data-handling choice, not comparable with the paper's raw-name numbers) ===")
            for name in ["model_alone_prod"] + [n for n in ROWS_ORDER if n in pn]:
                pt = report["splits"]["who_when_name_normalised"]["point"][name]
                print(f"- {name}: " + "; ".join(f"{l} {pt[l]:.2f} [{cin[name][l][1]:.2f}, {cin[name][l][2]:.2f}]" for l in LV))
        for name, p in preds.items():
            if os.environ.get("RR_DIR"):      # redirected output dir = smoke test: never write to the ledger
                continue
            log_run(phase="12", model=f"rr_fuse_{args.tag}_{name}", seed=0, split=f"{split}_grouped_{args.tag}", n_params=0,
                    config={"row": name, "precommit": str(pre_path), "dump": str(dump_path)}, metrics={split: {l: evaluate(data.truth, p)[l] for l in LV}},
                    notes="rr_fuse final: precommitted weights/thresholds chosen on AEGIS val only; scored once")
    final_path.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {final_path}")


if __name__ == "__main__":
    main()
