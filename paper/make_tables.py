"""Generate paper/tables_generated.tex and paper/numbers.tex from committed evidence only.

Inputs : aegis/results/published_rows.json      (AEGIS paper Table 1, copied as reported)
         aegis/results/block_ablation.json      (floors, frequency, shortcut-only, content-only, full; released and duplicate-free protocols)
         aegis/results/s4_ensemble.json         (14-encoder block-dropout ensemble, published protocol, three test sets)
         aegis/results/test_file_comparison.json(duplicate / overlap counts)
Run    : python3 paper/make_tables.py   (from the repo root; standard library only)
Every number in the paper's abstract and tables comes from this script, so it cannot drift from the evidence files.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "aegis" / "results"
OUT = Path(__file__).resolve().parent
pub = json.load(open(R / "published_rows.json"))["rows"]
ab = json.load(open(R / "block_ablation.json"))["rows"]
ens = json.load(open(R / "s4_ensemble.json"))["rows"]
cmp_ = json.load(open(R / "test_file_comparison.json"))
LV = ("pair", "agent", "error")


def cell(r, lv, ci=True):
    s = f"{r[lv]['micro']:.2f}"
    return s + (f" {{\\scriptsize[{r[lv]['lo']:.1f}, {r[lv]['hi']:.1f}]}}" if ci else "")


def row(label, src, r):
    return f"{label} & {src} & " + " & ".join(cell(r, lv) for lv in LV) + " \\\\\n"


def pubrow(label, key, bench="ab"):
    p = pub[key]
    return f"{label} & rep. & " + " & ".join(f"{p[f'{bench}_{lv}_micro']:.2f}" for lv in LV) + " \\\\\n"


T = []
# ---- Table: AEGIS-Bench, authors' file (primary) ----
t = "\\begin{table}[t]\n\\caption{AEGIS-Bench, authors' test file (primary). Micro-F1 (\\%); ``rep.'' = copied from Table~1 of~\\cite{aegis}; ``ours'' = run by us with 95\\% record-level bootstrap intervals. Our training data have the test duplicates removed (Section~\\ref{sec:protocol}).}\n\\label{tab:aegis}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}llrrr@{}}\n\\toprule\nModel & Src. & Pair & Agent & Error \\\\\n\\midrule\n"
for lab, key in (("Random", "Random"), ("Qwen2.5-7B (zero-shot)", "Qwen2.5-7B-Instruct"), ("GPT-4.1 (zero-shot)", "GPT-4.1"), ("o3 (zero-shot)", "o3"),
                 ("DCL", "DCL (Ours)"), ("Aegis-SFT", "Qwen2.5-14B-Instruct + SFT (Aegis-SFT)"), ("Aegis-GRPO", "Qwen2.5-14B-Instruct + GRPO (Aegis-GRPO)")):
    t += pubrow(lab, key)
t += "\\midrule\n"
t += row("Trivial floor", "ours", ab["floor|repo_test"])
t += row("Frequency baseline", "ours", ab["frequency|dedup|repo_test"])
t += row("Shortcut-only (no text)", "ours", ab["shortcut_only|dedup|repo_test"])
t += row("Content-only", "ours", ab["content_only|dedup|repo_test"])
t += row("Full, frozen encoder, 1 model", "ours", ab["full|dedup|repo_test"])
t += row("\\textbf{Fine-tuned, 14 encoders}", "ours", ens["names+peer|noshift|repo_test"])
t += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
T.append(t)

# ---- Table: AEGIS-Bench, HF file (secondary) ----
t = "\\begin{table}[t]\n\\caption{AEGIS-Bench, Hugging Face test file (secondary), same models as Table~\\ref{tab:aegis}. Micro-F1 (\\%) with 95\\% intervals.}\n\\label{tab:aegishf}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nModel & Pair & Agent & Error \\\\\n\\midrule\n"
for lab, key in (("Trivial floor", "floor|hf_test"), ("Frequency baseline", "frequency|dedup|hf_test"), ("Shortcut-only (no text)", "shortcut_only|dedup|hf_test"),
                 ("Content-only", "content_only|dedup|hf_test"), ("Full, frozen encoder, 1 model", "full|dedup|hf_test")):
    t += f"{lab} & " + " & ".join(cell(ab[key], lv) for lv in LV) + " \\\\\n"
t += "\\textbf{Fine-tuned, 14 encoders} & " + " & ".join(cell(ens["names+peer|noshift|hf_test"], lv) for lv in LV) + " \\\\\n"
t += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
T.append(t)

# ---- Table: effect of duplicates ----
t = "\\begin{table}[t]\n\\caption{Effect of verbatim duplicates on one model (frozen MiniLM, flat MLP, 5 seeds). Pair micro-F1 (\\%) with 95\\% intervals. ``Released'' trains on the released train/validation data; ``deduplicated'' removes every record whose conversation also occurs in a test file.}\n\\label{tab:leak}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nTest file & Released & Deduplicated & Drop \\\\\n\\midrule\n"
for lab, f in (("Authors' file", "repo_test"), ("Hugging Face file", "hf_test")):
    a, b = ab[f"full|paper|{f}"], ab[f"full|dedup|{f}"]
    t += f"{lab} & {cell(a, 'pair')} & {cell(b, 'pair')} & {a['pair']['micro'] - b['pair']['micro']:.2f} \\\\\n"
t += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
T.append(t)

# ---- Table: shortcut decomposition ----
t = "\\begin{table}[t]\n\\caption{What the score is made of (frozen MiniLM, flat MLP, 5 seeds, deduplicated training data). Pair micro-F1 (\\%) with 95\\% intervals; shortcut-only sees agent name, framework and turn counts but never the text.}\n\\label{tab:shortcut}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nInput & Authors' file & HF file & Who\\&When \\\\\n\\midrule\n"
for lab, v in (("Trivial floor", None), ("Shortcut-only", "shortcut_only"), ("Content-only", "content_only"), ("Full", "full")):
    if v is None:
        cells = [cell(ab["floor|repo_test"], "pair"), cell(ab["floor|hf_test"], "pair"), cell(ab["floor|who_when"], "pair")]
    else:
        cells = [cell(ab[f"{v}|dedup|{f}"], "pair") for f in ("repo_test", "hf_test", "who_when")]
    t += f"{lab} & " + " & ".join(cells) + " \\\\\n"
t += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
T.append(t)

# ---- Table: Who&When ----
t = "\\begin{table}[t]\n\\caption{Who\\&When (184 natural failures, out-of-domain; no Who\\&When label is used for training or tuning). Micro-F1 (\\%); intervals are 95\\% record-level bootstrap. Label shift reads unlabelled Who\\&When inputs (transductive). $^\\dagger$ mask chosen after seeing Who\\&When results (post hoc).}\n\\label{tab:ww}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}llrrr@{}}\n\\toprule\nModel & Src. & Pair & Agent & Error \\\\\n\\midrule\n"
for lab, key in (("Random", "Random"), ("DCL", "DCL (Ours)"), ("Aegis-SFT", "Qwen2.5-14B-Instruct + SFT (Aegis-SFT)"), ("GPT-4.1 (zero-shot)", "GPT-4.1"), ("o3 (zero-shot)", "o3")):
    t += pubrow(lab, key, "ww")
t += "\\midrule\n"
t += row("Trivial floor", "ours", ab["floor|who_when"])
t += row("Shortcut-only (1 model)", "ours", ab["shortcut_only|dedup|who_when"])
t += row("Full, frozen encoder (1 model)", "ours", ab["full|dedup|who_when"])
t += row("14 encoders, all blocks", "ours", ens["names+peer|noshift|who_when"])
t += row("\\quad + label shift", "ours", ens["names+peer|shift|who_when"])
t += row("14 encoders, peer block hidden$^\\dagger$", "ours", ens["names|noshift|who_when"])
t += row("\\quad + label shift$^\\dagger$", "ours", ens["names|shift|who_when"])
t += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
T.append(t)

# ---- Table: mask x shift grid ----
t = "\\begin{table}[t]\n\\caption{Block masks at test time (14-encoder ensemble, 3 head seeds), pair / agent / error micro-F1 (\\%). AEGIS: authors' file. Intervals for every cell are in \\texttt{aegis/results/s4\\_ensemble.json}.}\n\\label{tab:grid}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nMask (name, peer) & AEGIS & Who\\&When & + label shift \\\\\n\\midrule\n"


def trip(r):
    return " / ".join(f"{r[lv]['micro']:.1f}" for lv in LV)


for lab, m in (("on, on (all blocks)", "names+peer"), ("on, off", "names"), ("off, on", "peer"), ("off, off", "neither")):
    t += f"{lab} & {trip(ens[m + '|noshift|repo_test'])} & {trip(ens[m + '|noshift|who_when'])} & {trip(ens[m + '|shift|who_when'])} \\\\\n"
t += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
T.append(t)

T = [t.replace("\\begin{table}[t]", "\\begin{table*}[t]").replace("\\end{table}", "\\end{table*}") if any(k in t for k in ("tab:aegis}", "tab:ww}", "tab:shortcut}")) else t for t in T]
(OUT / "tables_generated.tex").write_text("% generated by paper/make_tables.py; do not edit by hand\n" + "\n".join(T))

# ---- numbers.tex: macros used in the abstract and text ----
N = {}


def put(name, r, lv, lo_hi=True):
    N[name] = f"{r[lv]['micro']:.2f}"
    if lo_hi:
        N[name + "Lo"] = f"{r[lv]['lo']:.1f}"
        N[name + "Hi"] = f"{r[lv]['hi']:.1f}"


put("ensAEpair", ens["names+peer|noshift|repo_test"], "pair"); put("ensAEagent", ens["names+peer|noshift|repo_test"], "agent", False)
put("ensAEerror", ens["names+peer|noshift|repo_test"], "error", False); put("ensHFpair", ens["names+peer|noshift|hf_test"], "pair")
put("shortAEpair", ab["shortcut_only|dedup|repo_test"], "pair"); put("shortHFrel", ab["shortcut_only|paper|hf_test"], "pair", False)
put("contentAEpair", ab["content_only|dedup|repo_test"], "pair", False); put("fullAEdedup", ab["full|dedup|repo_test"], "pair"); put("fullAErel", ab["full|paper|repo_test"], "pair")
put("fullHFdedup", ab["full|dedup|hf_test"], "pair"); put("fullHFrel", ab["full|paper|hf_test"], "pair")
N["dropAE"] = f"{ab['full|paper|repo_test']['pair']['micro'] - ab['full|dedup|repo_test']['pair']['micro']:.1f}"
N["dropHF"] = f"{ab['full|paper|hf_test']['pair']['micro'] - ab['full|dedup|hf_test']['pair']['micro']:.1f}"
put("floorAE", ab["floor|repo_test"], "pair", False); put("floorHF", ab["floor|hf_test"], "pair", False); put("floorWW", ab["floor|who_when"], "pair", False)
put("wwAllNo", ens["names+peer|noshift|who_when"], "pair"); put("wwAllShift", ens["names+peer|shift|who_when"], "pair")
put("wwNameNo", ens["names|noshift|who_when"], "pair"); put("wwNameShift", ens["names|shift|who_when"], "pair")
put("wwNameShiftErr", ens["names|shift|who_when"], "error", False); put("wwAllAgent", ens["names+peer|noshift|who_when"], "agent", False)
put("wwShortPair", ab["shortcut_only|dedup|who_when"], "pair"); put("wwFullPair", ab["full|dedup|who_when"], "pair")
N["pubSFT"] = f"{pub['Qwen2.5-14B-Instruct + SFT (Aegis-SFT)']['ab_pair_micro']:.2f}"
N["pubSFTagent"] = f"{pub['Qwen2.5-14B-Instruct + SFT (Aegis-SFT)']['ab_agent_micro']:.2f}"
N["pubSFTerror"] = f"{pub['Qwen2.5-14B-Instruct + SFT (Aegis-SFT)']['ab_error_micro']:.2f}"
N["pubDCL"] = f"{pub['DCL (Ours)']['ab_pair_micro']:.2f}"
N["pubOthree"] = f"{pub['o3']['ab_pair_micro']:.2f}"
N["pubOthreeWW"] = f"{pub['o3']['ww_pair_micro']:.2f}"
N["pubOthreeWWagent"] = f"{pub['o3']['ww_agent_micro']:.2f}"
N["pubOthreeWWerr"] = f"{pub['o3']['ww_error_micro']:.2f}"
h, v, hv = cmp_["hf_test_vs_train"], cmp_["hf_test_vs_val"], cmp_["hf_vs_repo"]
N["dupTrain"] = str(h["conversations"]); N["dupVal"] = str(v["conversations"]); N["qTrain"] = str(h["records_of_A_with_query_in_B"])
N["sharedConv"] = str(hv["conversations"]); N["repoQTrain"] = str(cmp_["repo_test_vs_train"]["records_of_A_with_query_in_B"]); N["repoDupTrain"] = str(cmp_["repo_test_vs_train"]["conversations"])
N["dupTrainPct"] = f"{100 * h['conversations'] / 600:.1f}"
(OUT / "numbers.tex").write_text("% generated by paper/make_tables.py\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in N.items()))
print(len(N), "macros;", len(T), "tables")
