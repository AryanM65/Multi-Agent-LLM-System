"""Local helper (no network, no token): build the Kaggle kernel directory for aegis/scripts/kaggle_judge.py and the small dataset that carries the candidate files.

  python aegis/scripts/rr_make_judge_kernel.py --user aryanmahajan665 --name aegis-rr-judge-a --out <build dir> \
      --cand-tag rrc --splits val,who_when --variant plain,rag,rand --model Qwen/Qwen2.5-7B-Instruct --four-bit --limit 300 --minutes 520

Creates
  <out>/<name>/<name>.py + kernel-metadata.json     the kernel (T4 GPU, internet on, datasets <user>/aegis-raw-data and <user>/aegis-rr-cands)
  <out>/aegis-rr-cands/cand_<tag>_<split>.jsonl + dataset-metadata.json   the private dataset holding the shortlist/example files
Then (by hand, token read per command from kaggle.txt as KAGGLE_API_TOKEN, never written anywhere):
  kaggle datasets create -p <out>/aegis-rr-cands          (first time; later: kaggle datasets version -p <out>/aegis-rr-cands -m "msg")
  kaggle kernels push -p <out>/<name>
Pull the output the moment the kernel is COMPLETE (kaggle kernels output <user>/<name> -p aegis/results/kaggle_logs/<short>) and copy the judge_*.jsonl files to aegis/results/rr/.
The candidate files contain AEGIS-train text snippets and model probabilities only; Who&When candidates contain the model's own shortlist (no label).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RR = HERE.parent / "results" / "rr"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--name", required=True, help="kernel slug, e.g. aegis-rr-judge-a")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cand-tag", required=True)
    ap.add_argument("--splits", default="val,who_when", help="candidate splits this kernel judges, in this order")
    ap.add_argument("--variant", default="plain,rag,rand")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=540.0, help="global time budget inside the kernel (Kaggle GPU sessions stop at about 9 h)")
    ap.add_argument("--char-budget", type=int, default=20000)
    ap.add_argument("--spawn", type=int, default=0, help="processes per kernel, one per GPU (2 on a 2xT4 session)")
    ap.add_argument("--shard-base", type=int, default=0, help="first shard id of this kernel")
    ap.add_argument("--shard-total", type=int, default=0, help="total shards over all kernels")
    args = ap.parse_args()
    out = Path(args.out)
    kdir, ddir = out / args.name, out / "aegis-rr-cands"
    kdir.mkdir(parents=True, exist_ok=True)
    ddir.mkdir(parents=True, exist_ok=True)
    cands = [f"cand_{args.cand_tag}_{s}.jsonl" for s in args.splits.split(",")]
    for c in cands:
        shutil.copy(RR / c, ddir / c)
    (ddir / "dataset-metadata.json").write_text(json.dumps({"title": "aegis-rr-cands", "id": f"{args.user}/aegis-rr-cands", "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    argv = ["--cands", ",".join(cands), "--variant", args.variant, "--model", args.model, "--minutes", str(args.minutes), "--char-budget", str(args.char_budget)]
    if args.four_bit:
        argv.append("--four-bit")
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.skip:
        argv += ["--skip", str(args.skip)]
    if args.shard_total:
        argv += ["--shard-total", str(args.shard_total), "--shard-base", str(args.shard_base)]
    if args.spawn:
        argv += ["--spawn", str(args.spawn)]
    src = (HERE / "kaggle_judge.py").read_text(encoding="utf-8").replace("from __future__ import annotations\n", "")
    head = ("import os, subprocess, sys\n"
            "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')\n"
            "if '--cands' not in sys.argv:      # the kernel itself; child processes it spawns are started WITH arguments and must not re-install or re-add them\n"
            "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'bitsandbytes>=0.43.1', 'accelerate'], check=False)\n"
            f"    sys.argv += {argv!r}\n")
    (kdir / f"{args.name}.py").write_text(head + src, encoding="utf-8")
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.user}/{args.name}", "title": args.name, "code_file": f"{args.name}.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"],
        "dataset_sources": [f"{args.user}/aegis-raw-data", f"{args.user}/aegis-rr-cands"], "kernel_sources": [], "competition_sources": [],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4"}, indent=1))
    print(f"kernel dir {kdir}\ndataset dir {ddir}\nargv {argv}")


if __name__ == "__main__":
    main()
