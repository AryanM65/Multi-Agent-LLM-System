"""GAT-based fault-localization GNN.

Outputs one logit per node in the input graph (a variable-length vector,
since graphs range from 3 to 7 nodes). An extra "no fault" logit is
concatenated on top, so the full output for a graph with N nodes is a
softmax over N+1 classes: N real nodes + "clean, no fault".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


FAULT_TYPES = ["clean", "noise", "contamination", "ceiling"]

CONV_CLASSES = {"gat": GATConv, "gcn": GCNConv, "sage": SAGEConv}


class FaultLocalizerGNN(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=32, num_layers=2, dropout=0.2, multi_task=False, conv_type="gat"):
        super().__init__()
        # Input projection + dropout before the first conv layer -- added
        # after embeddings (384-dim) made in_dim jump from 12 to 396, feeding
        # raw high-dim input straight into a conv layer with no
        # regularization caused severe overfitting (train top1 0.30->0.54,
        # OOD flat). This compresses+regularizes the input before message
        # passing instead of relying on the conv layer's own (linear) input
        # weights.
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        # GAT (default) learns attention weights over neighbors; GCN and
        # SAGE use fixed/mean aggregation instead -- an architecture-level
        # ablation, not assumed to be better or worse than GAT. See
        # docs/model/futurework.md Section 6.
        ConvClass = CONV_CLASSES[conv_type]
        self.convs = nn.ModuleList()
        self.convs.append(ConvClass(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(ConvClass(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.multi_task = multi_task

        self.node_head = nn.Linear(hidden_dim, 1)  # one "fault score" per node
        # a single learned scalar representing "no fault at all" for this graph,
        # computed from the graph's mean node embedding
        self.no_fault_head = nn.Linear(hidden_dim, 1)

        if multi_task:
            # Auxiliary graph-level head: predict fault TYPE (clean/noise/
            # contamination/ceiling) alongside node localization. See
            # docs/model/futurework.md Section 6 -- an ablation, not assumed
            # to help; compare against multi_task=False before keeping it.
            self.fault_type_head = nn.Linear(hidden_dim, len(FAULT_TYPES))

    def forward(self, x, edge_index):
        h = self.input_dropout(F.relu(self.input_proj(x)))
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)

        node_logits = self.node_head(h).squeeze(-1)  # [num_nodes]
        graph_repr = h.mean(dim=0, keepdim=True)      # [1, hidden_dim]
        no_fault_logit = self.no_fault_head(graph_repr).squeeze(-1)  # [1]

        node_out = torch.cat([node_logits, no_fault_logit], dim=0)  # [num_nodes + 1]

        if self.multi_task:
            fault_type_logits = self.fault_type_head(graph_repr).squeeze(0)  # [4]
            return node_out, fault_type_logits
        return node_out
