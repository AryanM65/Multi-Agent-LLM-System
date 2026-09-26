"""Turn each trial record + its topology's structure into one PyG graph.

Feature vector per node (8-dim), see model-plan.md Step 2 for the full
explanation with a worked example:
    [lexical_unc, semantic_unc, jaccard_unc, has_semantic, has_jaccard,
     is_retriever, is_reasoner, is_writer]

Known, accepted data quirk (see btp-pipeline/model.md Section 3, item 5):
`retriever_c` in the `star` and `triple_retriever_fanin` topologies is never
a true fault target. We don't special-case it here (the graph is still built
normally, features and all) -- it's handled at evaluation time instead, by
excluding it from the candidate/target-node set for those two topologies.
"""
import torch
from torch_geometric.data import Data

ROLE_LIST = ["retriever", "reasoner", "writer"]

# Trials whose fault target is None (`is_control` / clean baseline) get this
# label. Handled as an extra "no fault" class -- see build_dataset()'s
# `num_classes` / class-index remapping below.
NO_FAULT_LABEL = "__no_fault__"


def node_feature_vector(trial: dict, node_id: str, role: str) -> list:
    lex = trial["uncertainties"].get(node_id)
    lex = lex if lex is not None else 0.0

    sem = trial.get("semantic_uncertainties", {}).get(node_id)
    has_sem = 1.0 if sem is not None else 0.0
    sem = sem if sem is not None else 0.0

    jac = trial.get("jaccard_uncertainties", {}).get(node_id)
    has_jac = 1.0 if jac is not None else 0.0
    jac = jac if jac is not None else 0.0

    role_onehot = [1.0 if role == r else 0.0 for r in ROLE_LIST]

    return [lex, sem, jac, has_sem, has_jac] + role_onehot


def trial_to_graph(trial: dict, topology: dict) -> Data:
    node_ids = list(topology["nodes"].keys())
    idx_of = {n: i for i, n in enumerate(node_ids)}

    x = [
        node_feature_vector(trial, n, topology["nodes"][n]["role"])
        for n in node_ids
    ]

    edges = topology["edges"]
    if edges:
        edge_index = [
            [idx_of[src] for src, dst in edges],
            [idx_of[dst] for src, dst in edges],
        ]
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


def build_dataset(trials: list, topologies: dict) -> list:
    """Returns a list of PyG Data graphs, one per trial, with `y` set to a
    per-graph node-class-index (0..num_nodes-1), or the index for
    "no fault" if the trial is a clean control -- see NO_FAULT_LABEL.

    Each graph's node-index space is local to that graph's own topology
    (graphs are NOT batched into one shared label space here).
    """
    graphs = []
    for trial in trials:
        topo = topologies[trial["topology_id"]]
        g = trial_to_graph(trial, topo)

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
