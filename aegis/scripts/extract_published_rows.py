"""Extract Table 1 (micro/macro F1, Aegis-Bench and Who&When) from the AEGIS paper (arXiv 2509.14295 v6, ICLR 2026).

Usage: python extract_published_rows.py <pdf_text_file> <out_json>
<pdf_text_file> is text extracted from the paper PDF (e.g. with pypdf). Arrows in the printed table are dropped.
Column order in the paper: AB pair mu/M, agent mu/M, error mu/M | WW pair mu/M, agent mu/M, error mu/M | Avg.
These rows are the ONLY reference numbers the paper may cite (guide's rule: only published results); they are copied as reported.
"""
import json
import re
import sys

COLS = ["ab_pair_micro", "ab_pair_macro", "ab_agent_micro", "ab_agent_macro", "ab_error_micro", "ab_error_macro",
        "ww_pair_micro", "ww_pair_macro", "ww_agent_micro", "ww_agent_macro", "ww_error_micro", "ww_error_macro", "avg"]
NUM = re.compile(r"-?\d+\.\d+")


def main(txt_path: str, out_path: str) -> None:
    lines = open(txt_path, encoding="utf-8", errors="replace").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Table 1: Main results"))
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("Table 2"))  # first table after Table 1
    rows = {}
    base = ""
    for l in lines[start:end]:
        clean = re.sub(r"[↑↓–]", "", l)
        ms = list(NUM.finditer(clean))
        if len(ms) < 13:
            continue
        ms = ms[-13:]                      # the 13 table numbers are the last 13 on the line (names may contain decimals)
        name = clean[: ms[0].start()].strip()
        if not name or not re.match(r"^[A-Za-z+]", name):
            continue
        if name.startswith("+"):
            name = f"{base} {name}"        # '+ SFT' / '+ GRPO' belong to the preceding base model
        elif "only-" not in name and not name.startswith("w/o"):
            base = name
        assert name not in rows, name
        rows[name] = dict(zip(COLS, (float(m.group()) for m in ms)))
    out = {"source": "Kong et al., AEGIS, ICLR 2026, arXiv 2509.14295v6, Table 1 (copied as reported)", "rows": rows}
    json.dump(out, open(out_path, "w"), indent=1)
    print(len(rows), "rows:", list(rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
