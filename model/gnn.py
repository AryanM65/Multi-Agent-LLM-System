"""GAT-based fault-localization GNN.

Outputs one logit per node in the input graph (a variable-length vector,
since graphs range from 3 to 7 nodes). An extra "no fault" logit is
concatenated on top, so the full output for a graph with N nodes is a
softmax over N+1 classes: N real nodes + "clean, no fault".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class FaultLocalizerGNN(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim))
        self.dropout = dropout

        self.node_head = nn.Linear(hidden_dim, 1)  # one "fault score" per node
        # a single learned scalar representing "no fault at all" for this graph,
        # computed from the graph's mean node embedding
        self.no_fault_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)

        node_logits = self.node_head(h).squeeze(-1)  # [num_nodes]
        graph_repr = h.mean(dim=0, keepdim=True)      # [1, hidden_dim]
        no_fault_logit = self.no_fault_head(graph_repr).squeeze(-1)  # [1]

        return torch.cat([node_logits, no_fault_logit], dim=0)  # [num_nodes + 1]
