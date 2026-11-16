"""Stage the Kaggle inputs for S4 (fine-tune under the PUBLISHED protocol, duplicates removed).

Writes into <staging>/data   : train.jsonl val.jsonl test.jsonl whowhen.jsonl aegis.jsonl (copied from the HF download and the authors' repo)
                  <staging>/ids    : grouped_ids.json {"train": [...], "val": [...]}  (the file name the fine-tune script expects; here it holds the
                                     RELEASED train/val ids minus every record whose conversation is a verbatim copy of a record in either 600-record test file)
Both folders are meant to be uploaded as private Kaggle datasets. The staging directory must be OUTSIDE the repo (about 250 MB).
Run: python -m aegis.scripts.make_s4_inputs <staging_dir>
"""
import json
import shutil
import sys
from pathlib import Path

from ..src.loaders import PATHS, load_paper_splits, load_split
from .compare_test_files import convo_hash


def main(staging: str) -> None:
    st = Path(staging)
    (st / "data").mkdir(parents=True, exist_ok=True)
    (st / "ids").mkdir(parents=True, exist_ok=True)
    for name, key in (("train.jsonl", "train"), ("val.jsonl", "val"), ("test.jsonl", "test"),
                      ("whowhen.jsonl", "who_when"), ("aegis.jsonl", "test_repo")):
        shutil.copyfile(PATHS[key], st / "data" / name)
    sp = load_paper_splits()
    banned = {convo_hash(t) for k in ("test", "test_repo") for t in load_split(k)}
    ids = {k: [t.id for t in sp[k] if convo_hash(t) not in banned] for k in ("train", "val")}
    (st / "ids" / "grouped_ids.json").write_text(json.dumps(ids))
    print({k: (len(sp[k]), len(v)) for k, v in ids.items()}, "(released, kept)")


if __name__ == "__main__":
    main(sys.argv[1])
