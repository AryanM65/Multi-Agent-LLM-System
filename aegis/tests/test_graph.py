import numpy as np

from aegis.src.graph import collate, edges_for


def test_edges_for_small():
    # turns: A, B, A -> agent_of_turn [0, 1, 0]; query node id = 3
    e = {tuple(r) for r in edges_for([0, 1, 0]).tolist()}
    assert (0, 1, 0) in e and (1, 0, 0) in e and (1, 2, 0) in e and (2, 1, 0) in e   # sequence
    assert (0, 2, 1) in e and (2, 0, 1) in e                                           # same agent
    assert (1, 1, 1) not in e and (0, 1, 1) not in e                                   # no cross-agent same edge
    for i in range(3):
        assert (3, i, 2) in e and (i, 3, 3) in e                                       # query links
    assert len(e) == 4 + 2 + 6


def test_collate_offsets_and_pool():
    def item(agents, n_agents):
        n = len(agents)
        return {"emb": np.zeros((n, 4), np.float32), "q": np.zeros(4, np.float32), "agent_of_turn": agents,
                "edges": edges_for(agents), "meta": np.zeros((n_agents, 2), np.float32),
                "y": np.zeros((n_agents, 14), np.float32), "n_agents": n_agents}

    b = collate([item([0, 1], 2), item([0, 0, 1], 2)])
    assert b["x"].shape == (3 + 4, 4)                      # (2 turns + query) + (3 turns + query)
    assert b["pool"].tolist() == [0, 1, -1, 2, 2, 3, -1]   # second trajectory's agents start at row 2
    assert b["n_rows"] == 4
    assert b["src"].max() < 7 and b["dst"].max() < 7
    assert b["src"][b["rel"] == 2].tolist() == [2, 2, 6, 6, 6]   # query nodes are the sources of q2t edges
