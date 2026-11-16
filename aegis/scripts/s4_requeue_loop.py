"""Every 5 minutes: for each launched S4 kernel still QUEUED, delete it and push it again (user instruction 2026-11-16), until none is queued.

Safety (CLAUDE.md: never delete a RUNNING kernel before its output is saved):
- status is re-read immediately before every delete and only a kernel that is QUEUED (no output exists) is deleted;
- COMPLETE and ERROR kernels are pulled into aegis/results/kaggle_logs/s4_s<seed>/ first and then left alone (never re-pushed automatically);
- RUNNING kernels are never touched.
Tokens come from env vars KAGGLE_TOKEN_<ACCOUNT> only. Run from the repo root:
  S4_SEEDS=0,1,3,4,6,7,8,9,10,11,12,13 python -m aegis.scripts.s4_requeue_loop [--interval 300] [--max-rounds 36]
"""
import argparse
import os
import re
import tempfile
import time
from datetime import datetime

from .make_s4_kernel import main as build_kernel
from .s4_status_pull import ASSIGNMENT, LOGS, kaggle


def status(acct: str, seed: int) -> str:
    out = kaggle(acct, "kernels", "status", f"{acct}/aegis-s4-ft-s{seed}")
    m = re.search(r"KernelWorkerStatus\.(\w+)", out)
    return m.group(1) if m else "UNKNOWN"


def pull(acct: str, seed: int) -> None:
    dest = LOGS / f"s4_s{seed}"
    if dest.exists() and any(dest.iterdir()):
        return
    dest.mkdir(parents=True, exist_ok=True)
    kaggle(acct, "kernels", "output", f"{acct}/aegis-s4-ft-s{seed}", "-p", str(dest))
    print(f"    pulled seed {seed} -> {dest.name} ({len(list(dest.iterdir()))} files)", flush=True)


def requeue(acct: str, seed: int) -> None:
    ref = f"{acct}/aegis-s4-ft-s{seed}"
    if status(acct, seed) != "QUEUED":          # re-check right before deleting
        print(f"    seed {seed}: no longer queued, not deleting", flush=True)
        return
    kaggle(acct, "kernels", "delete", ref, "-y")
    with tempfile.TemporaryDirectory() as d:
        build_kernel(d, seed, acct, "aegis-s4-data" if acct == "shubhamy90" else "aegis-raw-data")
        out = kaggle(acct, "kernels", "push", "-p", d)
    print(f"    seed {seed}: deleted and re-pushed: {out.strip().splitlines()[-1][:90] if out.strip() else '?'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--max-rounds", type=int, default=36)
    a = ap.parse_args()
    seeds = [int(s) for s in os.environ["S4_SEEDS"].split(",")]
    for rnd in range(1, a.max_rounds + 1):
        queued, line = [], []
        for seed in seeds:
            acct = ASSIGNMENT[seed]
            st = status(acct, seed)
            line.append(f"s{seed}:{st}")
            if st in ("COMPLETE", "ERROR"):
                pull(acct, seed)
            elif st == "QUEUED":
                queued.append(seed)
        print(f"[{datetime.now():%H:%M:%S}] round {rnd}: " + " ".join(line), flush=True)
        if not queued:
            print("no queued kernels left: done", flush=True)
            return
        for seed in queued:
            requeue(ASSIGNMENT[seed], seed)
        time.sleep(a.interval)
    print("max rounds reached", flush=True)


if __name__ == "__main__":
    main()
