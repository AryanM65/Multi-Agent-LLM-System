"""Zero-training oracle probe: does a candidate feature actually localize the fault?

For each faulty trial we compute a per-node score, then check how often the
argmax within that graph is the true faulty node. Compared against the
per-graph random baseline (1/num_nodes). A feature only matters if it beats
random by a wide margin -- see the existing signals, which don't.
"""
import json
import re
from collections import Counter

STOP = set("""the a an and or of to in is are was were be been being for on at by with from as that this these those
it its it's they them their there here what which who whom whose when where why how not no nor but if then than
so such can could should would may might must will shall do does did done have has had having i you he she we
about into over under again further once all any both each few more most other some only own same too very
s t just now also please given return only based answer question following evidence text sentences paragraph""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+")


def content_tokens(text: str) -> set:
    toks = TOKEN_RE.findall(str(text).lower())
    return {t for t in toks if t not in STOP and (len(t) >= 4 or t.isdigit())}


def node_profile(samples: list, min_frac: float = 0.5) -> set:
    """Tokens appearing in at least `min_frac` of the k self-consistency samples.

    Thresholding filters out per-sample sampling noise, so the profile reflects
    what the node *consistently* says rather than one stochastic draw.
    """
    if not samples:
        return set()
    counts = Counter()
    for s in samples:
        counts.update(content_tokens(s))
    need = max(1, int(len(samples) * min_frac))
    return {t for t, c in counts.items() if c >= need}


def build_scores(trial: dict, topo: dict) -> dict:
    """Returns {feature_name: {node_id: score}} for one trial."""
    nodes = list(topo["nodes"].keys())
    edges = topo.get("edges") or []
    parents = {n: [p for p, c in edges if c == n] for n in nodes}
    children = {n: [c for p, c in edges if p == n] for n in nodes}

    samples = trial.get("samples") or {}
    prof = {n: node_profile(samples.get(n) or []) for n in nodes}
    q_tokens = content_tokens(trial.get("question", ""))

    novel, dropped, sib, child_novel = {}, {}, {}, {}
    for n in nodes:
        pn = prof[n]
        # Source of legitimate content: parents' output, or the question for roots.
        src = set()
        for p in parents[n]:
            src |= prof[p]
        if not parents[n]:
            src |= q_tokens
        src |= q_tokens  # question is always a legitimate source

        if pn:
            # Content this node emits that has no upstream origin -> injection.
            novel[n] = len(pn - src) / len(pn)
        if src:
            # Upstream content this node failed to carry forward -> dropping.
            dropped[n] = len(src - pn) / len(src)

        # Disagreement with siblings (nodes sharing a parent), same job, different answer.
        sibs = {s for p in parents[n] for s in children[p] if s != n}
        if sibs and pn:
            sims = []
            for s in sibs:
                ps = prof[s]
                if pn or ps:
                    sims.append(len(pn & ps) / max(1, len(pn | ps)))
            if sims:
                sib[n] = 1.0 - (sum(sims) / len(sims))

        # How much novelty this node's *children* show -- corruption is visible
        # downstream of the faulty node, not only at it.
        cn = [novel.get(c) for c in children[n]]
        cn = [v for v in cn if v is not None]
        if cn:
            child_novel[n] = sum(cn) / len(cn)

    return {
        "novel_ratio": novel,
        "dropped_ratio": dropped,
        "sibling_disagree": sib,
        "child_novel": child_novel,
    }


def main():
    topos = {t["topology_id"]: t for t in json.load(open("dataset/topology_pool.json", encoding="utf-8"))}
    rows = [json.loads(l) for l in open("dataset/trials_k10.jsonl", encoding="utf-8") if l.strip()]
    faulty = [r for r in rows if (r.get("fault_config") or {}).get("target_node")]

    # existing signals, for a like-for-like comparison
    existing = ["uncertainties", "semantic_uncertainties"]

    agg = {}
    by_type = {}
    rand_hits = 0.0
    for r in faulty:
        topo = topos[r["topology_id"]]
        tgt = r["fault_config"]["target_node"]
        ftype = r.get("true_label")
        rand_hits += 1.0 / len(topo["nodes"])

        scores = build_scores(r, topo)
        for sig in existing:
            d = r.get(sig) or {}
            scores[sig] = {n: v for n, v in d.items() if isinstance(v, (int, float))}

        for name, sc in scores.items():
            sc = {n: v for n, v in sc.items() if n in topo["nodes"]}
            if len(sc) < 2:
                continue
            hit = 1 if max(sc, key=sc.get) == tgt else 0
            a = agg.setdefault(name, [0, 0])
            a[0] += hit
            a[1] += 1
            b = by_type.setdefault(name, {}).setdefault(ftype, [0, 0])
            b[0] += hit
            b[1] += 1

    n = len(faulty)
    print(f"faulty trials: {n}   random baseline: {rand_hits / n:.3f}\n")
    print(f"{'feature':20s} {'argmax hit':>10s} {'cover':>8s}   per-fault-type")
    for name, (hit, tot) in sorted(agg.items(), key=lambda kv: -kv[1][0] / max(1, kv[1][1])):
        bt = " ".join(
            f"{t}={h/c:.2f}" for t, (h, c) in sorted(by_type[name].items()) if c
        )
        print(f"{name:20s} {hit/tot:>10.3f} {tot/n:>8.0%}   {bt}")


if __name__ == "__main__":
    main()
