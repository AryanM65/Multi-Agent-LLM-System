"""Per-benchmark and per-framework micro-F1 of the S4 ensemble (head seed 0, all blocks) on the authors' test file.
Inputs : aegis/results/preds/s4_ensemble_seed0_preds.json, aegis/third_party/AEGIS/evaluation/data/aegis.jsonl
Outputs: aegis/results/breakdown_by_group.json, paper/table_breakdown.tex
Run    : python3 -m aegis.scripts.breakdown_by_group   (from the repo root; needs numpy)"""
import ast, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from aegis.src.eval.block_ablation import counts, micro

ROOT = Path(__file__).resolve().parents[2]
P = json.load(open(ROOT / "aegis/results/preds/s4_ensemble_seed0_preds.json"))["names+peer|noshift|repo_test"]
pred = {i: {tuple(x) for x in p} for i, p in zip(P["ids"], P["pred"])}
recs = {}
for line in open(ROOT / "aegis/third_party/AEGIS/evaluation/data/aegis.jsonl"):
    r = json.loads(line)
    md = r["metadata"] if isinstance(r["metadata"], dict) else ast.literal_eval(r["metadata"])
    out = r["output"] if isinstance(r["output"], dict) else ast.literal_eval(r["output"])
    recs[r["id"]] = (md, {(f["agent_name"], f["error_type"]) for f in out["faulty_agents"]})
res = {}
for key in ("benchmark", "framework"):
    groups = defaultdict(list)
    for i in P["ids"]:
        groups[recs[i][0][key]].append(i)
    for g, ids in sorted(groups.items()):
        c = counts([recs[i][1] for i in ids], [pred[i] for i in ids]).sum(0)
        m = micro(c) * 100
        res[f"{key}|{g}"] = {"n": len(ids), "pair": float(m[0]), "agent": float(m[1]), "error": float(m[2])}
json.dump(res, open(ROOT / "aegis/results/breakdown_by_group.json", "w"), indent=1)
t = "\\begin{table}[t]\n\\caption{Where the 14-encoder ensemble does well: micro-F1 (\\%) on the authors' test file by benchmark and by framework (head seed 0, all blocks; one prediction per record, no intervals, so small groups are noisy).}\n\\label{tab:breakdown}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\nGroup & Value & $n$ & Pair & Agent & Error \\\\\n\\midrule\n"
for k, v in res.items():
    g, name = k.split("|")
    t += f"{g} & {name.replace('_', ' ')} & {v['n']} & {v['pair']:.1f} & {v['agent']:.1f} & {v['error']:.1f} \\\\\n"
t += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
(ROOT / "paper/table_breakdown.tex").write_text(t)
print(t)

# ---- per failure mode (error level): precision / recall / F1 of the ensemble on the authors' file ----
from collections import Counter
tp, fp, fn = Counter(), Counter(), Counter()
for i in P["ids"]:
    g, p = recs[i][1], pred[i]
    for a, m in p:
        (tp if (a, m) in g else fp)[m] += 1
    for a, m in g:
        if (a, m) not in p:
            fn[m] += 1
modes = sorted(set(tp) | set(fp) | set(fn), key=lambda m: tuple(int(x) for x in m[3:].split(".")))
pm = {}
for m in modes:
    pr = tp[m] / max(tp[m] + fp[m], 1); rc = tp[m] / max(tp[m] + fn[m], 1)
    pm[m] = {"gold": tp[m] + fn[m], "pred": tp[m] + fp[m], "precision": 100 * pr, "recall": 100 * rc, "f1": 100 * 2 * pr * rc / max(pr + rc, 1e-9)}
json.dump(pm, open(ROOT / "aegis/results/per_mode_pair.json", "w"), indent=1)
t = "\\begin{table}[t]\n\\caption{Per failure mode, pair level: gold and predicted counts, precision, recall and F1 (\\%) of the ensemble on the authors' file (head seed 0, all blocks). A pair counts as correct only if agent and mode both match.}\n\\label{tab:permode}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\nMode & Gold & Pred. & Prec. & Rec. & F1 \\\\\n\\midrule\n"
for m, v in pm.items():
    t += f"{m} & {v['gold']} & {v['pred']} & {v['precision']:.1f} & {v['recall']:.1f} & {v['f1']:.1f} \\\\\n"
t += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
(ROOT / "paper/table_permode.tex").write_text(t)
print(t)
