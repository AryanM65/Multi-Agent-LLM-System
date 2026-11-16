"""Status of the S4 fine-tuning kernels, and durable pull of every finished one (CLAUDE.md: never leave a log only on Kaggle).

Tokens are read ONLY from environment variables KAGGLE_TOKEN_<ACCOUNT> (account name upper-cased); nothing is written to a file.
For each kernel in ASSIGNMENT: print its status; if COMPLETE or ERROR and not yet saved, run `kaggle kernels output` into
aegis/results/kaggle_logs/s4_s<seed>/ (a failed run is kept too: it is evidence).
Run: KAGGLE_TOKEN_SHUBHAMY90=... KAGGLE_TOKEN_AMAHAJ55=... python -I aegis/scripts/s4_status_pull.py
"""
import os
import re
import subprocess
from pathlib import Path

LOGS = Path(__file__).resolve().parents[1] / "results" / "kaggle_logs"
# seed -> account. 14 encoders (seeds 0-13), as in the draft's ensemble; split over five accounts (2026-11-16).
ASSIGNMENT = {0: "shubhamy90", 1: "shubhamy90", 2: "shubhamy90", 3: "aryanmahajan665", 4: "aryanmahajan665", 5: "aryanmahajan665",
              6: "aryanmahajan7", 7: "aryanmahajan7", 8: "aryanmahajan7", 9: "amahaj55", 10: "amahaj55", 11: "amahaj55",
              12: "aryanomgh65", 13: "aryanomgh65"}


def kaggle(account: str, *args: str) -> str:
    env = dict(os.environ, KAGGLE_API_TOKEN=os.environ[f"KAGGLE_TOKEN_{account.upper()}"])
    r = subprocess.run(["kaggle", *args], env=env, capture_output=True, text=True)
    return "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "outdated" not in l)


def main() -> None:
    for seed, acct in ASSIGNMENT.items():
        if f"KAGGLE_TOKEN_{acct.upper()}" not in os.environ:
            print(f"seed {seed:2d} {acct:16s} (no token in env)")
            continue
        ref = f"{acct}/aegis-s4-ft-s{seed}"
        out = kaggle(acct, "kernels", "status", ref)
        m = re.search(r"KernelWorkerStatus\.(\w+)", out)
        st = m.group(1) if m else out.strip()[:60]
        dest = LOGS / f"s4_s{seed}"
        saved = dest.exists() and any(dest.iterdir())
        note = ""
        if st in ("COMPLETE", "ERROR") and not saved:
            dest.mkdir(parents=True, exist_ok=True)
            pull = kaggle(acct, "kernels", "output", ref, "-p", str(dest))
            note = f"  -> pulled to {dest.name} ({len(list(dest.iterdir()))} files)"
        print(f"seed {seed:2d} {acct:16s} {st}{' (saved)' if saved else ''}{note}")


if __name__ == "__main__":
    main()
