"""Regenerates the 'full values' appendix (every run, every seed, micro AND macro F1 at pair / agent / error level) inside
local-notes/approaches_vs_paper.md and local-notes/results.md, straight from the ledger aegis/results/metrics_history.jsonl.
Re-run any time:  python local-notes/gen_full_values.py      (local only; never commit)
Format: bullets only (user rule: no tables). Numbers are percent.
"""
import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "aegis/results/metrics_history.jsonl"
START, END = "<!-- FULL_VALUES_START -->", "<!-- FULL_VALUES_END -->"
LV = ("pair", "agent", "error")

PAPER = """### Paper rows, all 12 numbers (Table 1; micro / macro F1, percent)
- Random: AEGIS pair 0.33/0.21, agent 4.54/3.56, error 11.23/11.15; Who&When pair 0.11/0.05, agent 1.06/0.83, error 8.74/7.14; average 4.08
- DCL: AEGIS pair 8.33/5.30, agent 22.93/20.23, error 24.73/27.70; Who&When pair 1.60/0.77, agent 8.40/6.07, error 14.67/10.57; average 12.61
- Qwen2.5-7B zero-shot: AEGIS pair 5.02/2.52, agent 27.55/14.49, error 14.96/11.36; Who&When pair 2.31/1.14, agent 40.92/23.50, error 3.64/1.77; average 12.43
- Qwen2.5-14B zero-shot: AEGIS pair 5.47/2.20, agent 35.78/12.71, error 20.24/5.91; Who&When pair 0.00/0.00, agent 49.88/33.19, error 1.56/1.35; average 13.99
- Aegis-SFT (14B fine-tuned): AEGIS pair 16.62/9.99, agent 76.53/47.97, error 27.53/27.66; Who&When pair 4.03/2.08, agent 51.14/36.94, error 9.87/7.77; average 26.51
- Aegis-GRPO (14B): AEGIS pair 6.84/2.55, agent 49.74/18.38, error 21.19/16.10; Who&When pair 2.45/1.49, agent 54.43/40.88, error 4.15/2.67; average 18.41
- o3 zero-shot: AEGIS pair 7.86/2.27, agent 40.31/23.27, error 22.37/16.76; Who&When pair 7.41/3.98, agent 53.10/42.55, error 14.88/8.63; average 20.24
- Trivial floor (ours): AEGIS test pair 7.35/6.35, agent 68.58/61.98, error 27.14/26.99; Who&When pair 3.42/2.85, agent 38.32/38.41, error 13.33/12.34; average 25.59
"""

PHASE_TITLE = OrderedDict([
    ("3a", "Zero-shot Qwen2.5-7B reproduction (phase 3a)"),
    ("6r", "Frozen MiniLM models, final held-out scoring (phase 6r; split 'grouped' = our protocol, 'paper' = released leaky split)"),
    ("6ft", "Fine-tuned MiniLM (phase 6ft; split without a tag = first encoder stopped on validation; tags es_s0 / es_s1 = clean encoders)"),
    ("7", "Who&When study variants (phase 7; base / namefree / nameemb, decoders thr and top1; tags = encoder)"),
    ("8", "End-to-end graph fine-tune (phase 8; ftg_l2 = 2-layer graph, ftg_l0 = control, nf = name-free)"),
    ("9", "CPU improvement experiments (phase 9; variants base / peer / namedrop / peernd; decoders prod / pm / min / perclass; tag = encoder)"),
])


def fmt(m):
    parts = []
    for lv in LV:
        d = m.get(lv)
        parts.append(f"{lv} {100 * d['micro_f1']:.2f}/{100 * d['macro_f1']:.2f}" if d else f"{lv} n/a")
    return ", ".join(parts)


def build(which_phases):
    rows = [json.loads(l) for l in open(LEDGER, encoding="utf8")]
    groups = defaultdict(list)
    for r in rows:
        ph = str(r.get("phase"))
        if ph not in which_phases:
            continue
        for key, m in (r.get("metrics") or {}).items():
            if not isinstance(m, dict) or "pair" not in m and "agent" not in m:
                continue
            groups[(ph, r.get("split"), key, r.get("model"))].append((r.get("seed"), m))
    out = []
    for ph in which_phases:
        gs = [(k, v) for k, v in groups.items() if k[0] == ph]
        if not gs:
            continue
        out.append(f"### {PHASE_TITLE.get(ph, ph)}")
        for (p, split, key, model), items in sorted(gs, key=lambda kv: (str(kv[0][1]), str(kv[0][3]), str(kv[0][2]))):
            items.sort(key=lambda t: (t[0] is None, t[0]))
            out.append(f"- {model} | scored on {key} | split tag {split} | {len(items)} run(s); each value is micro/macro F1")
            for seed, m in items:
                out.append(f"  - seed {seed}: {fmt(m)}")
        out.append("")
    return "\n".join(out)


def inject(path: Path, text: str):
    s = path.read_text(encoding="utf8")
    block = f"{START}\n{text}\n{END}"
    if START in s:
        s = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _m: block, s, flags=re.S)
    else:
        s = s.rstrip() + "\n\n" + block + "\n"
    path.write_text(s, encoding="utf8")


def main():
    cmp_text = ("## Appendix: full values (every run and seed, micro AND macro F1, not only averages)\n"
                "Generated from the ledger `aegis/results/metrics_history.jsonl` by `local-notes/gen_full_values.py` (re-run to refresh). "
                "Each bullet is one model on one split; below it one line per seed with pair / agent / error as micro/macro percent.\n\n"
                + PAPER + "\n" + build(["3a", "6r", "6ft", "7", "8"]))
    res_text = ("## Appendix: full values of this push (every run and seed, micro AND macro F1, not only averages)\n"
                "Generated from the ledger by `local-notes/gen_full_values.py`. Phase 9 = improve_cpu runs; each line is one seed, pair / agent / error as micro/macro percent, "
                "on grouped validation, grouped test and Who&When.\n\n" + build(["9"]))
    inject(ROOT / "local-notes/approaches_vs_paper.md", cmp_text)
    inject(ROOT / "local-notes/results.md", res_text)
    print("done")


if __name__ == "__main__":
    main()
