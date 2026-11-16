"""Build a Kaggle kernel directory for one S4 fine-tuning run (published protocol, duplicates removed).

A kernel takes ONE script, so the arguments are prepended to a copy of kaggle_finetune_minilm.py.
Usage: python -m aegis.scripts.make_s4_kernel <out_dir> <seed> [user=shubhamy90]
Then: KAGGLE_API_TOKEN=... kaggle kernels push -p <out_dir>   (token only as an env var, never written to a file)
"""
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "kaggle_finetune_minilm.py"


def main(out: str, seed: int, user: str = "shubhamy90") -> None:
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    src = SRC.read_text()
    marker = 'if __name__ == "__main__":\n'
    assert src.count(marker) == 1
    # arguments are set inside the __main__ block: a line above the file's `from __future__` import would be a SyntaxError
    src = src.replace(marker, marker + f"    import sys\n    sys.argv = ['kaggle_finetune_minilm.py', '--es-holdout', '0.15', '--seed', '{seed}']\n")
    (d / "script.py").write_text(src)
    slug = f"aegis-s4-ft-s{seed}"
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}", "title": slug, "code_file": "script.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{user}/aegis-s4-data", f"{user}/aegis-s4-ids"], "competition_sources": [], "kernel_sources": []}, indent=1))
    print(d, slug)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), *(sys.argv[3:4]))
