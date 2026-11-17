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
# (grid table superseded by table_gridci.tex; not appended)

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

# ======== additions: full grid with intervals, figures ========
G = "\\begin{table}[t]\n\\caption{Test-time block masks, 14-encoder ensemble (3 head seeds): pair micro-F1 (\\%) with 95\\% record-level bootstrap intervals. Label shift is applied to Who\\&When only.}\n\\label{tab:gridci}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nName / peer block & Authors' file & HF file & Who\\&When & + shift \\\\\n\\midrule\n"
for lab, m in (("on / on", "names+peer"), ("on / off$^\\dagger$", "names"), ("off / on", "peer"), ("off / off", "neither")):
    c = [cell(ens[f"{m}|noshift|{f}"], "pair") for f in ("repo_test", "hf_test", "who_when")] + [cell(ens[f"{m}|shift|who_when"], "pair")]
    G += f"{lab} & " + " & ".join(c) + " \\\\\n"
G += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
G = G.replace("\\begin{table}[t]", "\\begin{table*}[t]").replace("\\end{table}", "\\end{table*}")
(OUT / "table_gridci.tex").write_text(G)

# Figure 1: Who&When pair micro-F1 by mask, with / without shift (error bars = interval)
def coords(shift):
    pts = []
    for i, m in enumerate(("names+peer", "names", "peer", "neither")):
        r = ens[f"{m}|{'shift' if shift else 'noshift'}|who_when"]["pair"]
        pts.append(f"({i + 1},{r['micro']:.2f}) +- (0,{r['micro'] - r['lo']:.2f})")
    return " ".join(pts)

fig = f"""\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[ybar,width=\\columnwidth,height=4.3cm,bar width=7pt,ymin=0,ymax=14,ylabel={{Pair micro-F1 (\\%)}},
 symbolic x coords={{1,2,3,4}},xtick=data,xticklabels={{on/on,on/off,off/on,off/off}},xlabel={{name / peer block at test time}},
 legend style={{at={{(0.5,1.02)}},anchor=south,legend columns=3,font=\\footnotesize}},tick label style={{font=\\footnotesize}},
 error bars/y dir=both,error bars/y explicit]
\\addplot+[fill=gray!55,draw=black] coordinates {{{coords(False)}}};
\\addplot+[fill=blue!55,draw=black] coordinates {{{coords(True)}}};
\\draw[dashed] (axis cs:1,{ab['floor|who_when']['pair']['micro']:.2f}) -- (axis cs:4,{ab['floor|who_when']['pair']['micro']:.2f});
\\legend{{no shift,label shift,trivial floor}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Who\\&When pair micro-F1 of the 14-encoder ensemble by test-time block mask, with and without label shift (bars: 95\\% intervals; dashed: trivial floor). Shift helped clearly only with the name block on and the peer block hidden (Table~\\ref{{tab:shiftgain}}).}}
\\label{{fig:ww}}
\\end{{figure}}
"""
(OUT / "fig_ww.tex").write_text(fig)

# Figure 2: authors' file pair micro-F1, published rows vs ours
rows = [("Random", pub["Random"]["ab_pair_micro"]), ("floor", ab["floor|repo_test"]["pair"]["micro"]), ("o3", pub["o3"]["ab_pair_micro"]), ("DCL", pub["DCL (Ours)"]["ab_pair_micro"]),
        ("short.", ab["shortcut_only|dedup|repo_test"]["pair"]["micro"]), ("SFT", pub["Qwen2.5-14B-Instruct + SFT (Aegis-SFT)"]["ab_pair_micro"]),
        ("frozen", ab["full|dedup|repo_test"]["pair"]["micro"]), ("14 enc.", ens["names+peer|noshift|repo_test"]["pair"]["micro"])]
labels = ",".join(r[0] for r in rows)
pts = " ".join(f"({i + 1},{r[1]:.2f})" for i, r in enumerate(rows))
fig2 = f"""\\begin{{figure}}[t]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[ybar,width=\\columnwidth,height=4.1cm,bar width=9pt,ymin=0,ymax=42,ylabel={{Pair micro-F1 (\\%)}},symbolic x coords={{1,2,3,4,5,6,7,8}},
 xtick=data,xticklabels={{{labels}}},tick label style={{font=\\scriptsize}},nodes near coords,nodes near coords style={{font=\\tiny}},x tick label style={{rotate=30,anchor=east}}]
\\addplot+[fill=gray!55,draw=black] coordinates {{{pts}}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{AEGIS-Bench, authors' file: published rows (Random, o3, DCL, Aegis-SFT), our trivial floor, shortcut-only (no text), frozen-encoder model and the 14-encoder ensemble.}}
\\label{{fig:aegis}}
\\end{{figure}}
"""
(OUT / "fig_aegis.tex").write_text(fig2)
print("figures written")

# ======== all published rows, micro/macro, both benchmarks ========
def esc(s):
    return s.replace("&", "\\&").replace("+", "{+}")

PT = "\\begin{table*}[t]\n\\caption{All 25 published rows copied as reported from Table~1 of~\\cite{aegis}: micro / macro F1 (\\%) at pair, agent and error level, on AEGIS-Bench (AB) and Who\\&When (WW). Not recomputed by us.}\n\\label{tab:pubfull}\n\\centering\\tiny\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{@{}l" + "r" * 12 + "@{}}\n\\toprule\n & \\multicolumn{6}{c}{AEGIS-Bench} & \\multicolumn{6}{c}{Who\\&When} \\\\\n\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}\nModel & Pair mi & Pair ma & Agent mi & Agent ma & Err mi & Err ma & Pair mi & Pair ma & Agent mi & Agent ma & Err mi & Err ma \\\\\n\\midrule\n"
for name, v in pub.items():
    vals = [v[f"{b}_{lv}_{m}"] for b in ("ab", "ww") for lv in LV for m in ("micro", "macro")]
    PT += esc(name) + " & " + " & ".join(f"{x:.2f}" for x in vals) + " \\\\\n"
PT += "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
(OUT / "table_pubfull.tex").write_text(PT)
print("pubfull written")

# ======== label-shift gain by protocol (paired intervals) and macros ========
SP = json.load(open(R / "shift_paired.json"))
def g(p, m, key="gain"):
    v = SP[key][f"{p}|{m}"]
    return f"{v['gain']:+.2f} {{\\scriptsize[{v['lo']:+.2f}, {v['hi']:+.2f}]}}"
T3 = "\\begin{table*}[t]\n\\caption{Effect of label shift on Who\\&When pair micro-F1 (shift minus no shift, percentage points) by protocol and block mask, with paired 95\\% record-level bootstrap intervals (10,000 resamples of the 184 records, 3 head seeds pooled). Bold: interval excludes zero. ``Earlier'': 14 encoders and heads trained on the 3,865 query-disjoint records. ``Full released'': the same old encoders with heads trained on the full released split (7,146 records, duplicates included). ``Dedup.'': the 14 newly fine-tuned encoders of the deduplicated protocol. All shift rows are transductive.}\n\\label{tab:shiftgain}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nProtocol & on / on & on / off & off / on & off / off \\\\\n\\midrule\n"
for lab, p in (("Earlier (query-disjoint)", "base"), ("Full released (duplicates)", "paper"), ("Dedup. (this paper)", "s4")):
    cells = []
    for m in MASKS_ORDER if False else ("names+peer", "names", "peer", "neither"):
        c = g(p, m); cells.append(f"\\textbf{{{c}}}" if SP["gain"][f"{p}|{m}"]["excludes_zero"] else c)
    T3 += f"{lab} & " + " & ".join(cells) + " \\\\\n"
T3 += "\\midrule\nDedup. minus earlier & " + g("base", "names+peer", "s4_minus").replace("{\\scriptsize", "{\\scriptsize") + " & " + g("base", "names", "s4_minus") + " & -- & -- \\\\\n"
T3 += "Dedup. minus full released & " + g("paper", "names+peer", "s4_minus") + " & " + g("paper", "names", "s4_minus") + " & -- & -- \\\\\n"
T3 += "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
(OUT / "table_shiftgain.tex").write_text(T3)
M = {}
for p, tag in (("base", "Old"), ("paper", "Full"), ("s4", "Dd")):
    for m, mt in (("names+peer", "OO"), ("names", "OF"), ("peer", "FO"), ("neither", "FF")):
        v = SP["gain"][f"{p}|{m}"]
        M[f"sg{tag}{mt}"] = f"{v['gain']:+.2f}"; M[f"sg{tag}{mt}Lo"] = f"{v['lo']:+.2f}"; M[f"sg{tag}{mt}Hi"] = f"{v['hi']:+.2f}"
for p, tag in (("base", "Old"), ("paper", "Full")):
    for m, mt in (("names+peer", "OO"), ("names", "OF")):
        v = SP["s4_minus"][f"{p}|{m}"]
        M[f"sd{tag}{mt}"] = f"{v['gain']:+.2f}"; M[f"sd{tag}{mt}Lo"] = f"{v['lo']:+.2f}"; M[f"sd{tag}{mt}Hi"] = f"{v['hi']:+.2f}"
with open(OUT / "numbers.tex", "a") as fh:
    fh.write("% paired label-shift gains, from aegis/results/shift_paired.json\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
print("shiftgain table +", len(M), "macros")

# ======== robustness: head-seed SD, paired tests, sensitivity, name audit, cost (aegis/results/{robustness_stats,name_audit,ensemble_size}.json) ========
RB = json.load(open(R / "robustness_stats.json"))
NA = json.load(open(R / "name_audit.json"))
ES = json.load(open(R / "ensemble_size.json")) if (R / "ensemble_size.json").exists() else None
MASK_LAB = {"names+peer": "on / on", "names": "on / off", "peer": "off / on", "neither": "off / off"}


def ms(v, lv):
    return f"{v[lv]['mean']:.2f} $\\pm$ {v[lv]['sd']:.2f}"


def pfmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


S = "\\begin{table}[t]\n\\caption{Head-seed variability: mean $\\pm$ SD over the three head seeds of the 14-encoder ensemble (micro-F1, \\%). Block masks are name / peer (on or off). Shift rows are transductive and use Who\\&When inputs only.}\n\\label{tab:seedsd}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}llrrr@{}}\n\\toprule\nTest set, mask & Shift & Pair & Agent & Error \\\\\n\\midrule\n"
for lab, key in (("Authors' file, on / on", "names+peer|noshift|repo_test"), ("HF file, on / on", "names+peer|noshift|hf_test"),
                 ("Who\\&When, on / on", "names+peer|noshift|who_when"), ("Who\\&When, on / off", "names|noshift|who_when"),
                 ("Who\\&When, off / on", "peer|noshift|who_when"), ("Who\\&When, off / off", "neither|noshift|who_when"),
                 ("Who\\&When, on / on", "names+peer|shift|who_when"), ("Who\\&When, on / off", "names|shift|who_when")):
    v = RB["seed_stats"][key]
    S += f"{lab} & {'yes' if '|shift|' in key else 'no'} & " + " & ".join(ms(v, lv) for lv in LV) + " \\\\\n"
S += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
(OUT / "table_seedsd.tex").write_text(S)

PR = RB["paired"]
P_ROWS = (("Authors' file: on/on minus off/off", "repo_test: all-blocks minus neither (no shift)"),
          ("Authors' file: on/on minus off/on", "repo_test: all-blocks minus names (no shift)"),
          ("Authors' file: on/on minus on/off", "repo_test: all-blocks minus peer (no shift)"),
          ("Who\\&W.: on/on minus off/off, no shift", "who_when: all-blocks minus neither (no shift)"),
          ("Who\\&W.: shift minus no shift, on/on", "who_when: shift minus no shift, mask names+peer"),
          ("Who\\&W.: shift minus no shift, on/off", "who_when: shift minus no shift, mask names"),
          ("Who\\&W.: shift minus no shift, off/on", "who_when: shift minus no shift, mask peer"),
          ("Who\\&W.: shift minus no shift, off/off", "who_when: shift minus no shift, mask neither"),
          ("Who\\&W.: on/off + shift minus on/on no shift", "who_when: names+shift minus all-blocks no-shift"),
          ("Who\\&W.: on/off + shift minus off/off no shift", "who_when: names+shift minus neither no-shift"))
S = "\\begin{table}[t]\n\\caption{Paired tests on the same records (pair micro-F1 difference, percentage points; 10,000-resample record-level bootstrap interval; two-sided Wilcoxon signed-rank test on per-record pair F1 averaged over head seeds). Authors' file: 600 records; Who\\&When: 184. Ten tests are shown, so a Bonferroni level is 0.005.}\n\\label{tab:paired}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lrr@{}}\n\\toprule\nComparison & Difference [95\\% interval] & Wilcoxon $p$ \\\\\n\\midrule\n"
for lab, key in P_ROWS:
    v = PR[key]
    c = f"{v['diff']:+.2f} {{\\scriptsize[{v['lo']:+.2f}, {v['hi']:+.2f}]}}"
    S += f"{lab} & {'\\textbf{' + c + '}' if v['bootstrap_excludes_zero'] else c} & {pfmt(v['wilcoxon_p'])} \\\\\n"
S += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
(OUT / "table_paired.tex").write_text(S)

SN = RB["sensitivity"]
S = "\\begin{table}[t]\n\\caption{Sensitivity of pair micro-F1 (\\%) to the decision thresholds. Both tuned thresholds $(t_a, t_m)$ are multiplied by the same factor (1.0 is the tuned setting). Mean over three head seeds.}\n\\label{tab:sens}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nSetting & 0.5 & 0.75 & 1.0 & 1.25 & 1.5 \\\\\n\\midrule\n"
for lab, key in (("Authors' file, on / on", "names+peer|noshift|repo_test"), ("HF file, on / on", "names+peer|noshift|hf_test"),
                 ("Who\\&W., on / on", "names+peer|noshift|who_when"), ("Who\\&W., on / off", "names|noshift|who_when"), ("Who\\&W., on / off + shift", "names|shift|who_when")):
    S += f"{lab} & " + " & ".join(f"{SN[key][s]['pair']:.2f}" for s in ("0.5", "0.75", "1.0", "1.25", "1.5")) + " \\\\\n"
S += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
(OUT / "table_sens.tex").write_text(S)

SC = NA["scores"]["repo_test"]
S = "\\begin{table}[t]\n\\caption{Agent names in the authors' test file. Gold agent labels that equal no speaker name in their own conversation, by framework (none in the 16,586 training labels or the Hugging Face file), and the ensemble's micro-F1 (\\%) under exact and normalised (case, hyphen and underscore ignored) name matching. Mean over three head seeds.}\n\\label{tab:names}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n & Unmatched labels & Agent, exact & Agent, normalised & Pair, normalised \\\\\n\\midrule\n"
for fw, lab in (("dylan", "DyLAN (87 records)"), ("macnet", "MacNet (100 records)")):
    mm = NA["mismatch"]["repo_test"][fw]
    bf = SC["by_framework"][fw]
    S += f"{lab} & {mm['not_exact_speaker']} of {mm['gold_agent_labels']} & {bf['exact']['agent']:.1f} & {bf['normalised']['agent']:.1f} & {bf['normalised']['pair']:.1f} \\\\\n"
S += f"All 600 records & {sum(v['not_exact_speaker'] for v in NA['mismatch']['repo_test'].values())} of {sum(v['gold_agent_labels'] for v in NA['mismatch']['repo_test'].values())} & {SC['exact']['agent']['mean']:.1f} & {SC['normalised']['agent']['mean']:.1f} & {SC['normalised']['pair']['mean']:.1f} \\\\\n"
S += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
(OUT / "table_names.tex").write_text(S)

M2 = {"nmDyMis": NA["mismatch"]["repo_test"]["dylan"]["not_exact_speaker"], "nmDyTot": NA["mismatch"]["repo_test"]["dylan"]["gold_agent_labels"],
      "nmMacMis": NA["mismatch"]["repo_test"]["macnet"]["not_exact_speaker"],
      "nmAgentExact": f"{SC['exact']['agent']['mean']:.2f}", "nmAgentNorm": f"{SC['normalised']['agent']['mean']:.2f}",
      "nmAgentNormLo": f"{SC['normalised']['agent']['lo']:.1f}", "nmAgentNormHi": f"{SC['normalised']['agent']['hi']:.1f}",
      "nmPairNorm": f"{SC['normalised']['pair']['mean']:.2f}", "nmPairNormLo": f"{SC['normalised']['pair']['lo']:.1f}", "nmPairNormHi": f"{SC['normalised']['pair']['hi']:.1f}",
      "nmDyAgentExact": f"{SC['by_framework']['dylan']['exact']['agent']:.1f}", "nmDyAgentNorm": f"{SC['by_framework']['dylan']['normalised']['agent']:.1f}",
      "nmDistinctIds": NA["ids"]["repo_test"]["distinct_ids"],
      "sdAEpair": f"{RB['seed_stats']['names+peer|noshift|repo_test']['pair']['sd']:.2f}",
      "sdWWallPair": f"{RB['seed_stats']['names+peer|noshift|who_when']['pair']['sd']:.2f}",
      "sdWWnameShiftPair": f"{RB['seed_stats']['names|shift|who_when']['pair']['sd']:.2f}",
      "ptAEallNeither": f"{PR['repo_test: all-blocks minus neither (no shift)']['diff']:+.2f}",
      "ptAEallNeitherLo": f"{PR['repo_test: all-blocks minus neither (no shift)']['lo']:+.2f}",
      "ptAEallNeitherHi": f"{PR['repo_test: all-blocks minus neither (no shift)']['hi']:+.2f}",
      "ptAEallNames": f"{PR['repo_test: all-blocks minus names (no shift)']['diff']:+.2f}",
      "ptWWnameShiftP": pfmt(PR["who_when: shift minus no shift, mask names"]["wilcoxon_p"]).replace("$<$", "$<$"),
      "ptWWnameShiftVsAll": f"{PR['who_when: names+shift minus all-blocks no-shift']['diff']:+.2f}",
      "ptWWnameShiftVsAllLo": f"{PR['who_when: names+shift minus all-blocks no-shift']['lo']:+.2f}",
      "ptWWnameShiftVsAllHi": f"{PR['who_when: names+shift minus all-blocks no-shift']['hi']:+.2f}",
      "sensAEmin": f"{min(SN['names+peer|noshift|repo_test'][s]['pair'] for s in ('0.75', '1.0', '1.25', '1.5')):.2f}",
      "sensAEhalf": f"{SN['names+peer|noshift|repo_test']['0.5']['pair']:.2f}",
      "sensShiftMin": f"{min(v['pair'] for v in SN['names|shift|who_when'].values()):.2f}",
      "sensShiftMax": f"{max(v['pair'] for v in SN['names|shift|who_when'].values()):.2f}"}
if ES:
    FK, SE = ES["first_k"]["names+peer|noshift|repo_test"], ES["single_encoder"]["names+peer|noshift|repo_test"]
    FKW, SEW = ES["first_k"]["names|shift|who_when"], ES["single_encoder"]["names|shift|who_when"]
    S = "\\begin{table}[t]\n\\caption{Ensemble size. Pair micro-F1 (\\%), mean over three head seeds, when the ensemble is built from the first $k$ of the 14 fine-tuned encoders (``single'': mean $\\pm$ SD, over the 14 encoders used alone). Thresholds are re-tuned on validation for each $k$.}\n\\label{tab:ensize}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\nSetting & single & $k{=}1$ & 2 & 4 & 7 & 14 \\\\\n\\midrule\n"
    for lab, fk, se in (("Authors' file, on / on", FK, SE), ("Who\\&W., on / off + shift", FKW, SEW)):
        S += f"{lab} & {se['pair']['mean']:.2f} $\\pm$ {se['pair']['sd']:.2f} & " + " & ".join(f"{fk[str(k)]['pair']:.2f}" for k in (1, 2, 4, 7, 14)) + " \\\\\n"
    S += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    (OUT / "table_ensize.tex").write_text(S)
    cst = ES["cost"]
    M2.update({"ensSingleAE": f"{SE['pair']['mean']:.2f}", "ensSingleAEsd": f"{SE['pair']['sd']:.2f}", "ensKtwoAE": f"{FK['2']['pair']:.2f}", "ensKfourAE": f"{FK['4']['pair']:.2f}",
               "ensKsevenAE": f"{FK['7']['pair']:.2f}", "ensKfourteenAE": f"{FK['14']['pair']:.2f}",
               "ensSingleWWshift": f"{SEW['pair']['mean']:.2f}", "ensSingleWWshiftSd": f"{SEW['pair']['sd']:.2f}", "ensFourteenWWshift": f"{FKW['14']['pair']:.2f}",
               "costFeat": cst["mean_feature_build_s_per_encoder"], "costHead": cst["mean_head_train_s_per_encoder_3_heads_4_masks"],
               "costInf": cst["mean_inference_s_per_encoder_all_splits"], "costParams": f"{cst['head_params']:,}".replace(",", "{,}")})
with open(OUT / "numbers.tex", "a") as fh:
    fh.write("% robustness numbers, from robustness_stats.json, name_audit.json, ensemble_size.json\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M2.items()))
print("robustness tables +", len(M2), "macros")

# ======== reproduction check: our runs of lighter published baselines (aegis/results/baselines_dedup.json) ========
BD = json.load(open(R / "baselines_dedup.json"))["rows"]


def ci3(r):
    return " & ".join(f"{r[lv]['micro']:.2f} {{\\scriptsize[{r[lv]['lo']:.1f}, {r[lv]['hi']:.1f}]}}" for lv in LV)


def pub3(key, pre):
    return " & ".join(f"{pub[key][f'{pre}_{lv}_micro']:.2f}" for lv in LV)


S = "\\begin{table*}[t]\n\\caption{Reproduction check (micro-F1, \\%). Rows marked ``rep.'' are copied from Table~1 of~\\cite{aegis}; every row marked ``ours'' was run by us, scored with the authors' scorer on the same test records and with 95\\% record-level intervals. The last four rows (trivial floor, frequency, flat MLP, shortcut-only MLP) use the same deduplicated training data as the ensemble; the flat MLP and shortcut-only MLP use frozen embeddings and five head seeds. Qwen2.5-7B (zero-shot, AWQ 4-bit, greedy, long prompts truncated) was run on the Hugging Face file only. DCL-style is our frozen-encoder re-implementation of the published design, trained under the deduplicated protocol (5 seeds), and is not the published DCL. Frequency scores zero on Who\\&When because it predicts agent names seen in training, and Who\\&When agent names are free-form. The published rows were evaluated on a test file the paper does not name.}\n\\label{tab:repro}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n & & \\multicolumn{3}{c}{AEGIS-Bench test file} & \\multicolumn{3}{c}{Who\\&When} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\nModel & Test file & Pair & Agent & Error & Pair & Agent & Error \\\\\n\\midrule\n"
S += f"Qwen2.5-7B, rep. & unnamed & {pub3('Qwen2.5-7B-Instruct', 'ab')} & {pub3('Qwen2.5-7B-Instruct', 'ww')} \\\\\n"
S += f"Qwen2.5-7B, ours & HF & {ci3(BD['qwen_zeroshot|hf_test'])} & {ci3(BD['qwen_zeroshot|who_when'])} \\\\\n"
S += "\\midrule\n"
S += f"DCL, rep. & unnamed & {pub3('DCL (Ours)', 'ab')} & {pub3('DCL (Ours)', 'ww')} \\\\\n"
S += f"DCL-style, ours & authors' & {ci3(BD['dcl_style|repo_test'])} & {ci3(BD['dcl_style|who_when'])} \\\\\n"
S += f"DCL-style, ours & HF & {ci3(BD['dcl_style|hf_test'])} & -- \\\\\n"
S += "\\midrule\n"
for lab, k in (("Trivial floor, ours", "floor"), ("Frequency, ours", "frequency|dedup"), ("Flat MLP, frozen MiniLM, ours", "full|dedup"), ("Shortcut-only MLP, ours", "shortcut_only|dedup")):
    S += f"{lab} & authors' & {ci3(ab[k + '|repo_test'] if k != 'floor' else ab['floor|repo_test'])} & {ci3(ab[k + '|who_when'] if k != 'floor' else ab['floor|who_when'])} \\\\\n"
S += "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
(OUT / "table_repro.tex").write_text(S)
Q, QW, DA, DH, DW = BD["qwen_zeroshot|hf_test"], BD["qwen_zeroshot|who_when"], BD["dcl_style|repo_test"], BD["dcl_style|hf_test"], BD["dcl_style|who_when"]
M3 = {"rpQwPair": f"{Q['pair']['micro']:.2f}", "rpQwAgent": f"{Q['agent']['micro']:.2f}", "rpQwError": f"{Q['error']['micro']:.2f}",
      "rpQwWWpair": f"{QW['pair']['micro']:.2f}", "rpQwWWagent": f"{QW['agent']['micro']:.2f}", "rpQwWWerror": f"{QW['error']['micro']:.2f}",
      "pubQwPair": f"{pub['Qwen2.5-7B-Instruct']['ab_pair_micro']:.2f}", "pubQwAgent": f"{pub['Qwen2.5-7B-Instruct']['ab_agent_micro']:.2f}",
      "pubQwError": f"{pub['Qwen2.5-7B-Instruct']['ab_error_micro']:.2f}", "pubDCLagent": f"{pub['DCL (Ours)']['ab_agent_micro']:.2f}",
      "pubDCLerror": f"{pub['DCL (Ours)']['ab_error_micro']:.2f}", "pubDCLwwAgent": f"{pub['DCL (Ours)']['ww_agent_micro']:.2f}",
      "pubDCLwwError": f"{pub['DCL (Ours)']['ww_error_micro']:.2f}", "pubDCLwwPair": f"{pub['DCL (Ours)']['ww_pair_micro']:.2f}",
      "rpDclPair": f"{DA['pair']['micro']:.2f}", "rpDclAgent": f"{DA['agent']['micro']:.2f}", "rpDclError": f"{DA['error']['micro']:.2f}",
      "rpDclPairLo": f"{DA['pair']['lo']:.1f}", "rpDclPairHi": f"{DA['pair']['hi']:.1f}",
      "rpDclWWpair": f"{DW['pair']['micro']:.2f}", "rpDclWWagent": f"{DW['agent']['micro']:.2f}", "rpDclWWerror": f"{DW['error']['micro']:.2f}",
      "floorAEagent": f"{ab['floor|repo_test']['agent']['micro']:.2f}", "floorHFagent": f"{ab['floor|hf_test']['agent']['micro']:.2f}"}
with open(OUT / "numbers.tex", "a") as fh:
    fh.write("% reproduction check, from baselines_dedup.json\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M3.items()))
print("repro table +", len(M3), "macros")
