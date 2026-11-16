"""Build the paper-ready comparison tables (Markdown) from saved evidence only.

Inputs (all committed):  aegis/results/published_rows.json  (AEGIS paper Table 1, copied as reported)
                         aegis/results/block_ablation.json  (our runs: floors, frequency, full / shortcut-only / content-only,
                                                             record-level bootstrap 95% intervals, 5 seeds)
Output:                  aegis/results/comparison_tables.md
Conventions: micro-F1 in percent; "reported" = copied from the paper, "ours" = run by us under the stated protocol;
the authors' repo file is the primary AEGIS-Bench test set (decision D11), the HF file is secondary.
Run: python -I aegis/scripts/build_comparison_tables.py
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
pub = json.load(open(R / "published_rows.json"))["rows"]
ours = json.load(open(R / "block_ablation.json"))["rows"]
PUB_KEYS = [("Random", "Random"), ("DCL", "DCL (Ours)"), ("Qwen2.5-7B-Instruct (zero-shot)", "Qwen2.5-7B-Instruct"),
            ("Qwen2.5-14B-Instruct (zero-shot)", "Qwen2.5-14B-Instruct"), ("o3 (zero-shot)", "o3"),
            ("Aegis-SFT", "Qwen2.5-14B-Instruct + SFT (Aegis-SFT)"), ("Aegis-GRPO", "Qwen2.5-14B-Instruct + GRPO (Aegis-GRPO)")]


def cell(r, lv):
    return f"{r[lv]['micro']:.2f} [{r[lv]['lo']:.1f}, {r[lv]['hi']:.1f}]"


def table(bench: str, files: list) -> str:
    pre = "ab" if bench == "ab" else "ww"
    head = "| Row | Source | Pair | Agent | Error |\n|---|---|---|---|---|\n"
    out = head
    for label, key in PUB_KEYS:
        p = pub[key]
        out += f"| {label} | reported | {p[pre + '_pair_micro']:.2f} | {p[pre + '_agent_micro']:.2f} | {p[pre + '_error_micro']:.2f} |\n"
    for f in files:
        out += f"| *Trivial floor ({f})* | ours | " + " | ".join(cell(ours[f"floor|{f}"], lv) for lv in ("pair", "agent", "error")) + " |\n"
    for proto, ptxt in (("paper", "released train/val"), ("dedup", "released train/val, test-duplicates removed")):
        for var, vtxt in (("frequency", "frequency baseline"), ("shortcut_only", "shortcut-only (no text)"),
                          ("content_only", "content-only (no name/framework)"), ("full", "full model")):
            for f in files:
                key = f"{var}|{proto}|{f}"
                if key not in ours or (var == "frequency" and bench == "ww"):
                    continue   # frequency baseline predicts nothing on Who&When: its agent names never occur in training
                out += f"| {vtxt}, {ptxt}, {f} | ours | " + " | ".join(cell(ours[key], lv) for lv in ("pair", "agent", "error")) + " |\n"
    return out


def leakage_table() -> str:
    out = "| Test file | Full model, released train/val | Same, duplicates removed | Drop (pair) |\n|---|---|---|---|\n"
    for f in ("repo_test", "hf_test"):
        a, b = ours[f"full|paper|{f}"]["pair"]["micro"], ours[f"full|dedup|{f}"]["pair"]["micro"]
        out += f"| {f} | {cell(ours[f'full|paper|{f}'], 'pair')} | {cell(ours[f'full|dedup|{f}'], 'pair')} | {a - b:.2f} |\n"
    return out


def main() -> None:
    text = "# Comparison tables (generated; do not edit by hand)\n\n"
    text += ("Micro-F1 in percent. 'reported' rows are copied from the AEGIS paper's Table 1 (arXiv 2509.14295v6); 'ours' rows are run by us "
             "(frozen MiniLM + flat MLP, 5 seeds) with record-level bootstrap 95% intervals in brackets. The paper does not name its test file; the "
             "authors' repo file `evaluation/data/aegis.jsonl` matches its description (100 per benchmark) and is primary (decision D11).\n\n")
    text += "## AEGIS-Bench (pair / agent / error micro-F1)\n\n" + table("ab", ["repo_test", "hf_test"]) + "\n"
    text += "## Who&When (zero-shot: no Who&When label used for training or tuning)\n\n" + table("ww", ["who_when"]) + "\n"
    text += "## Effect of verbatim-duplicate leakage on the full model (pair micro-F1)\n\n" + leakage_table() + "\n"
    text += ("Notes: (1) reported agent-level numbers sit below our trivial floor; we report them as published and do not interpret them. "
             "(2) The shortcut-only model never reads the conversation. (3) Who&When rows here are the simple model, not the ensemble with block-dropout and "
             "label shift in the draft; those need the S4 runs.\n")
    (R / "comparison_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
