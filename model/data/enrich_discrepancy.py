"""Post-hoc per-node features computed from the stored k-sample outputs.

Motivation (see docs/model/model.md): every feature the model had was an
*uncertainty* measure, and uncertainty only moves for one of the three fault
types. Measured against clean controls on the 1785-record dataset:

    contamination   mean d(semantic_unc) = +0.235   (66.8% move up)  -> detectable
    noise           mean d(semantic_unc) = +0.062   (43.0% move up)  -> weak
    ceiling         mean d(semantic_unc) = -0.015   (24.2% move up)  -> INVERTED

Ceiling faults truncate a node (reasoner capped at 2 steps, writer at 5
words), and shorter output is *more* self-consistent -- so uncertainty falls.
No uncertainty-derived feature can localize them; single-feature oracle hit
rate on ceiling was 0.03-0.28 across every signal already present.

Length relative to the node's own clean baseline localizes them at 0.716
(random = 0.199). The discrepancy features cover the other two types:
contamination injects content with no upstream origin, noise drops it.

Adds per trial:
    node_novel_ratio      content emitted with no traceable upstream source
    node_dropped_ratio    upstream content this node failed to carry forward
    node_sibling_disagree divergence from nodes sharing a parent
    node_child_novel      mean novelty of this node's children (downstream symptom)
    node_len_mean         mean output length in words over the k samples
    node_len_std          length variability across the k samples
    node_len_z            length vs this (topology, node)'s clean-control mean
"""
from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict

STOP = set("""the a an and or of to in is are was were be been being for on at by with from as that this these those
it its they them their there here what which who whom whose when where why how not no nor but if then than
so such can could should would may might must will shall do does did done have has had having i you he she we
about into over under again further once all any both each few more most other some only own same too very
s t just now also please given return only based answer question following evidence text sentences paragraph""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A token must appear in at least this fraction of a node's k self-consistency
# samples to count as part of that node's profile -- filters per-sample
# sampling noise so the profile reflects what the node consistently says.
PROFILE_MIN_FRAC = 0.5


def _content_tokens(text) -> set:
    return {t for t in _TOKEN_RE.findall(str(text).lower())
            if t not in STOP and (len(t) >= 4 or t.isdigit())}


def _node_profile(samples) -> set:
    if not samples:
        return set()
    counts = Counter()
    for s in samples:
        counts.update(_content_tokens(s))
    need = max(1, int(len(samples) * PROFILE_MIN_FRAC))
    return {t for t, c in counts.items() if c >= need}


def _mean_len(samples):
    if not samples:
        return None
    return statistics.mean(len(str(s).split()) for s in samples)


def _std_len(samples):
    if not samples or len(samples) < 2:
        return 0.0
    return statistics.pstdev([len(str(s).split()) for s in samples])


def compute_length_baselines(trials: list) -> tuple[dict, dict]:
    """Mean/std output length per (topology_id, node_id), from CLEAN controls only.

    Clean controls carry no fault label, so using them as a reference point is
    calibration, not leakage. It does mean a deployed model needs clean
    baseline runs for its own topology -- that is a real operational
    requirement, not an artifact of the evaluation.
    """
    acc = defaultdict(list)
    for t in trials:
        if (t.get("fault_config") or {}).get("target_node"):
            continue
        for node_id, samples in (t.get("samples") or {}).items():
            v = _mean_len(samples)
            if v is not None:
                acc[(t["topology_id"], node_id)].append(v)
    mu = {k: statistics.mean(v) for k, v in acc.items() if v}
    # guard against a zero-variance baseline making z blow up
    sd = {k: (statistics.pstdev(v) or 1.0) for k, v in acc.items() if v}
    return mu, sd


def enrich_discrepancy(trials: list, topologies: dict) -> list:
    """Adds the per-node discrepancy + length features to each trial, in place."""
    basemu, basesd = compute_length_baselines(trials)

    for trial in trials:
        topo = topologies[trial["topology_id"]]
        nodes = list(topo["nodes"].keys())
        edges = topo.get("edges") or []
        parents = {n: [p for p, c in edges if c == n] for n in nodes}
        children = {n: [c for p, c in edges if p == n] for n in nodes}

        samples = trial.get("samples") or {}
        prof = {n: _node_profile(samples.get(n) or []) for n in nodes}
        q_tokens = _content_tokens(trial.get("question", ""))

        novel, dropped, sib = {}, {}, {}
        len_mean, len_std, len_z = {}, {}, {}

        for n in nodes:
            pn = prof[n]
            # legitimate content sources: this node's parents, plus the question
            src = set(q_tokens)
            for p in parents[n]:
                src |= prof[p]

            if pn:
                novel[n] = len(pn - src) / len(pn)
            if src:
                dropped[n] = len(src - pn) / len(src)

            sibs = {s for p in parents[n] for s in children[p] if s != n}
            if sibs and pn:
                sims = [len(pn & prof[s]) / max(1, len(pn | prof[s])) for s in sibs]
                if sims:
                    sib[n] = 1.0 - sum(sims) / len(sims)

            v = _mean_len(samples.get(n))
            if v is not None:
                len_mean[n] = v
                len_std[n] = _std_len(samples.get(n))
                key = (trial["topology_id"], n)
                if key in basemu:
                    # negative => unusually terse for this node: the ceiling signature
                    len_z[n] = (v - basemu[key]) / basesd[key]

        child_novel = {}
        for n in nodes:
            vals = [novel[c] for c in children[n] if c in novel]
            if vals:
                child_novel[n] = sum(vals) / len(vals)

        trial["node_novel_ratio"] = novel
        trial["node_dropped_ratio"] = dropped
        trial["node_sibling_disagree"] = sib
        trial["node_child_novel"] = child_novel
        trial["node_len_mean"] = len_mean
        trial["node_len_std"] = len_std
        trial["node_len_z"] = len_z

    return trials


if __name__ == "__main__":
    import argparse
    import json

    from model.data.load_raw import load_topologies, load_trials

    ap = argparse.ArgumentParser()
    ap.add_argument("in_path")
    ap.add_argument("out_path")
    args = ap.parse_args()

    topologies = load_topologies()
    trials = load_trials(path=args.in_path)
    enrich_discrepancy(trials, topologies)

    with open(args.out_path, "w", encoding="utf-8") as fh:
        for t in trials:
            fh.write(json.dumps(t) + "\n")

    covered = sum(1 for t in trials if t.get("node_len_z"))
    print(f"enriched {len(trials)} trials -> {args.out_path}")
    print(f"  node_len_z populated for {covered}/{len(trials)} trials")
