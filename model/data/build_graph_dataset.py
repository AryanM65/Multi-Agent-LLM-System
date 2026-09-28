"""Turn each trial record + its topology's structure into one PyG graph.

Feature vector per node (8-dim), see docs/model/model-plan.md Step 2 for the full
explanation with a worked example:
    [lexical_unc, semantic_unc, jaccard_unc, has_semantic, has_jaccard,
     is_retriever, is_reasoner, is_writer]

Known, accepted data quirk (see docs/model/model.md Section 3, item 5):
`retriever_c` in the `star` and `triple_retriever_fanin` topologies is never
a true fault target. We don't special-case it here (the graph is still built
normally, features and all) -- it's handled at evaluation time instead, by
excluding it from the candidate/target-node set for those two topologies.
"""
import torch
from torch_geometric.data import Data

ROLE_LIST = ["retriever", "reasoner", "writer"]

# Per-node scalars added by model/data/enrich_discrepancy.py. Each contributes
# a (value, has_value) pair to the feature vector, same convention as sem/jac.
DISCREPANCY_FIELDS = [
    "node_novel_ratio",
    "node_dropped_ratio",
    "node_sibling_disagree",
    "node_child_novel",
    "node_len_mean",
    "node_len_std",
    "node_len_z",
]

# all-MiniLM-L6-v2 output dim -- see model/data/enrich_features.py's
# compute_node_embeddings(). Used to zero-pad when node_embeddings is
# missing (pre-enrichment data), so in_dim stays consistent either way.
EMBEDDING_DIM = 384

# Trials whose fault target is None (`is_control` / clean baseline) get this
# label. Handled as an extra "no fault" class -- see build_dataset()'s
# `num_classes` / class-index remapping below.
NO_FAULT_LABEL = "__no_fault__"


def node_feature_vector(trial: dict, node_id: str, role: str, use_embeddings: bool = False,
                        use_embedding_variance: bool = False) -> list:
    lex = trial["uncertainties"].get(node_id)
    lex = lex if lex is not None else 0.0

    sem = trial.get("semantic_uncertainties", {}).get(node_id)
    has_sem = 1.0 if sem is not None else 0.0
    sem = sem if sem is not None else 0.0

    jac = trial.get("jaccard_uncertainties", {}).get(node_id)
    has_jac = 1.0 if jac is not None else 0.0
    jac = jac if jac is not None else 0.0

    # Added via model/data/enrich_features.py post-hoc pass -- see
    # docs/model/futurework.md Section 2. Missing/None on un-enriched data,
    # handled the same "value + has_X flag" way as sem/jac above.
    inf_gap = trial.get("inference_gaps", {}).get(node_id)
    has_inf_gap = 1.0 if inf_gap is not None else 0.0
    inf_gap = inf_gap if inf_gap is not None else 0.0

    item_freq = trial.get("item_frequencies", {}).get(node_id)
    has_item_freq = 1.0 if item_freq is not None else 0.0
    item_freq = item_freq if item_freq is not None else 0.0

    role_onehot = [1.0 if role == r else 0.0 for r in ROLE_LIST]

    scalar_feats = [lex, sem, jac, has_sem, has_jac, inf_gap, has_inf_gap, item_freq, has_item_freq] + role_onehot

    # Discrepancy + length features from model/data/enrich_discrepancy.py.
    # Every pre-existing feature is an uncertainty measure, and uncertainty
    # only moves for contamination faults -- it is flat for noise and
    # *inverted* for ceiling (truncated output is more self-consistent, so
    # uncertainty falls). node_len_z is what actually localizes ceiling:
    # single-feature oracle hit rate 0.716 vs 0.199 random, where every
    # uncertainty-derived signal scored 0.03-0.28. See enrich_discrepancy.py.
    for field in DISCREPANCY_FIELDS:
        v = trial.get(field, {})
        v = v.get(node_id) if isinstance(v, dict) else None
        scalar_feats += [float(v) if v is not None else 0.0, 1.0 if v is not None else 0.0]

    if not use_embeddings:
        return scalar_feats

    emb = trial.get("node_embeddings", {}).get(node_id)
    emb = emb if emb is not None else [0.0] * EMBEDDING_DIM
    feats = scalar_feats + list(emb)

    if use_embedding_variance:
        std_emb = trial.get("node_embedding_std", {}).get(node_id)
        std_emb = std_emb if std_emb is not None else [0.0] * EMBEDDING_DIM
        feats = feats + list(std_emb)

    return feats


def trial_to_graph(trial: dict, topology: dict, bidirectional: bool = False, use_embeddings: bool = False,
                   use_embedding_variance: bool = False) -> Data:
    node_ids = list(topology["nodes"].keys())
    idx_of = {n: i for i, n in enumerate(node_ids)}

    x = [
        node_feature_vector(trial, n, topology["nodes"][n]["role"], use_embeddings=use_embeddings,
                            use_embedding_variance=use_embedding_variance)
        for n in node_ids
    ]

    edges = topology["edges"]
    if edges:
        src_list = [idx_of[src] for src, dst in edges]
        dst_list = [idx_of[dst] for src, dst in edges]
        if bidirectional:
            # Add reverse edges too: forward matches real pipeline execution
            # order, backward lets a fault symptom's message-passing trace
            # back toward its cause -- see docs/model/futurework.md Section 6.
            src_list, dst_list = src_list + dst_list, dst_list + src_list
        edge_index = [src_list, dst_list]
    else:
        edge_index = [[], []]

    fault_config = trial.get("fault_config") or {}
    # "" (not None) for clean/control trials -- PyG's Data silently drops
    # None-valued kwargs, which breaks the `g.target_node` lookup downstream.
    target_node = fault_config.get("target_node") or ""

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        num_nodes=len(node_ids),
        # raw string target ("" for clean trials), resolved to a class index
        # later in build_dataset()
        target_node=target_node,
        true_label=trial.get("true_label"),
        topology_id=trial["topology_id"],
        question=trial.get("question"),
        node_ids=node_ids,  # keep the id->index mapping around for eval/error-analysis
    )


def build_dataset(trials: list, topologies: dict, bidirectional: bool = False, use_embeddings: bool = False,
                  use_embedding_variance: bool = False) -> list:
    """Returns a list of PyG Data graphs, one per trial, with `y` set to a
    per-graph node-class-index (0..num_nodes-1), or the index for
    "no fault" if the trial is a clean control -- see NO_FAULT_LABEL.

    Each graph's node-index space is local to that graph's own topology
    (graphs are NOT batched into one shared label space here).
    """
    graphs = []
    for trial in trials:
        topo = topologies[trial["topology_id"]]
        g = trial_to_graph(trial, topo, bidirectional=bidirectional, use_embeddings=use_embeddings,
                          use_embedding_variance=use_embedding_variance)

        if not g.target_node:
            # "no fault" is its own class, appended after all real nodes
            y = len(g.node_ids)
        else:
            y = g.node_ids.index(g.target_node)

        g.y = torch.tensor([y], dtype=torch.long)
        graphs.append(g)

    return graphs


if __name__ == "__main__":
    from model.data.load_raw import load_topologies, load_trials, validate

    topologies = load_topologies()
    trials = load_trials()
    validate(trials, topologies)

    graphs = build_dataset(trials, topologies)
    print(f"Built {len(graphs)} graphs.")

    g0 = graphs[0]
    print(f"\nExample graph (topology={g0.topology_id}, true_label={g0.true_label}):")
    print(f"  nodes: {g0.node_ids}")
    print(f"  x shape: {tuple(g0.x.shape)}")
    print(f"  x:\n{g0.x}")
    print(f"  edge_index:\n{g0.edge_index}")
    print(f"  y (target node index, or len(nodes) for 'no fault'): {g0.y.item()}")
