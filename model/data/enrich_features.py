"""Post-hoc enrichment pass: fills in `inference_gaps` (currently always null
in the raw dataset) and adds a new `item_frequencies` field, using the raw
per-node sample texts already stored in each trial (`samples`). No
regeneration needed -- see docs/model/futurework.md Section 2.

inference_gap[node] = 1 - cosine_similarity(embed(parent_output), embed(node_output))
    "how much did this node change the information it was given" -- uses the
    node's own first sample as its output, and its parent's(s') first sample
    (concatenated if multiple parents) as a stand-in for its input. Root
    nodes (no parents) use the trial's question as the input side instead.

item_frequencies[node] = mean per-item inclusion rate across the node's own
    k samples (only meaningful for Retriever-role, multi-item outputs;
    None for single-line outputs or non-Retriever roles).
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BTP_SRC = REPO_ROOT / "btp-pipeline"
if str(BTP_SRC) not in sys.path:
    sys.path.insert(0, str(BTP_SRC))

from src.uncertainty import get_embedder, parse_retrieved_items, per_item_inclusion_frequency  # noqa: E402


def compute_inference_gaps(trial: dict, topology: dict, embedder) -> dict:
    node_ids = list(topology["nodes"].keys())
    parents_of = {n: [] for n in node_ids}
    for src, dst in topology["edges"]:
        parents_of[dst].append(src)

    gaps = {}
    texts_a, texts_b, keys = [], [], []
    for node in node_ids:
        node_samples = trial["samples"].get(node)
        if not node_samples:
            gaps[node] = None
            continue
        output_text = node_samples[0]

        parents = parents_of.get(node, [])
        if parents:
            parent_texts = [trial["samples"][p][0] for p in parents if trial["samples"].get(p)]
            input_text = " ".join(parent_texts) if parent_texts else trial["question"]
        else:
            input_text = trial["question"]

        texts_a.append(input_text)
        texts_b.append(output_text)
        keys.append(node)

    if keys:
        embs_a = embedder.encode(texts_a, normalize_embeddings=True)
        embs_b = embedder.encode(texts_b, normalize_embeddings=True)
        for node, ea, eb in zip(keys, embs_a, embs_b):
            gap = float(1.0 - float((ea * eb).sum()))
            gaps[node] = round(gap, 4)

    return gaps


def compute_item_frequencies(trial: dict, topology: dict) -> dict:
    freqs = {}
    for node, meta in topology["nodes"].items():
        if meta["role"] != "retriever":
            continue
        node_samples = trial["samples"].get(node)
        if not node_samples:
            continue
        item_sets = [parse_retrieved_items(s) for s in node_samples]
        if all(len(s) <= 1 for s in item_sets):
            continue  # single-line outputs -- not a real multi-item set
        per_item = per_item_inclusion_frequency(item_sets)
        if per_item:
            freqs[node] = round(sum(per_item.values()) / len(per_item), 4)
    return freqs


def compute_node_embeddings(trial: dict, topology: dict, embedder) -> dict:
    """Mean-pooled sentence-embedding across a node's own k samples -- the
    real per-node signal a scalar uncertainty number throws away. See
    docs/model/futurework.md Section 2 / the embedding-features experiment.
    """
    embeddings = {}
    all_texts, spans = [], []  # spans: (node, start, end) into all_texts
    for node in topology["nodes"]:
        node_samples = trial["samples"].get(node)
        if not node_samples:
            continue
        start = len(all_texts)
        all_texts.extend(node_samples)
        spans.append((node, start, len(all_texts)))

    if not all_texts:
        return embeddings

    all_embs = embedder.encode(all_texts, normalize_embeddings=True)
    for node, start, end in spans:
        mean_vec = all_embs[start:end].mean(axis=0)
        embeddings[node] = [round(float(x), 5) for x in mean_vec]
    return embeddings


def enrich(trials: list, topologies: dict, with_embeddings: bool = True) -> list:
    embedder = get_embedder()
    for i, trial in enumerate(trials):
        topo = topologies[trial["topology_id"]]
        trial["inference_gaps"] = compute_inference_gaps(trial, topo, embedder)
        trial["item_frequencies"] = compute_item_frequencies(trial, topo)
        if with_embeddings:
            trial["node_embeddings"] = compute_node_embeddings(trial, topo, embedder)
        if (i + 1) % 100 == 0:
            print(f"  enriched {i + 1}/{len(trials)}", flush=True)
    return trials


if __name__ == "__main__":
    from model.data.load_raw import load_topologies, load_trials

    IN_PATH = REPO_ROOT / "dataset" / "trials.jsonl"
    OUT_PATH = REPO_ROOT / "dataset" / "trials.jsonl"
    BACKUP_PATH = REPO_ROOT / "dataset" / "trials_1762_preenrich.jsonl"

    topologies = load_topologies()
    trials = load_trials(path=IN_PATH)
    print(f"Loaded {len(trials)} trials. Backing up to {BACKUP_PATH} before enriching in place.")

    with open(IN_PATH, encoding="utf-8") as fsrc, open(BACKUP_PATH, "w", encoding="utf-8") as fdst:
        fdst.write(fsrc.read())

    trials = enrich(trials, topologies)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for t in trials:
            f.write(json.dumps(t) + "\n")

    print(f"Done. Wrote enriched dataset to {OUT_PATH}.")
