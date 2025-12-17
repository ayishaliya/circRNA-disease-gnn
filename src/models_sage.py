import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv
class HeteroGraphSAGE(nn.Module):
    """
    2-layer heterogeneous GraphSAGE using HeteroConv.
    Returns embeddings for each node type in a dict: x_dict['circ'], x_dict['mir'], x_dict['dis'].
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=64, dropout=0.2):
        super().__init__()

        # Layer 1: Input → Hidden for each relation
        relations_1 = {
            ("circRNA", "interacts", "miRNA"): SAGEConv(in_channels, hidden_channels),
            ("miRNA", "interacts", "disease"): SAGEConv(in_channels, hidden_channels),
            ("circRNA", "associated", "disease"): SAGEConv(in_channels, hidden_channels),

            ("miRNA", "rev_interacts", "circRNA"): SAGEConv(in_channels, hidden_channels),
            ("disease", "rev_interacts", "miRNA"): SAGEConv(in_channels, hidden_channels),
            ("disease", "rev_associated", "circRNA"): SAGEConv(in_channels, hidden_channels),
        }
        # Layer 2: Hidden → Output for each relation
        relations_2 = {
            ("circRNA", "interacts", "miRNA"): SAGEConv(hidden_channels, out_channels),
            ("miRNA", "interacts", "disease"): SAGEConv(hidden_channels, out_channels),
            ("circRNA", "associated", "disease"): SAGEConv(hidden_channels, out_channels),

            ("miRNA", "rev_interacts", "circRNA"): SAGEConv(hidden_channels, out_channels),
            ("disease", "rev_interacts", "miRNA"): SAGEConv(hidden_channels, out_channels),
            ("disease", "rev_associated", "circRNA"): SAGEConv(hidden_channels, out_channels),
        }

        self.conv1 = HeteroConv(relations_1, aggr="mean")
        self.conv2 = HeteroConv(relations_2, aggr="mean")

        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connections to stabilize learning
        self.res_lin_circ = nn.Linear(in_channels, out_channels)
        self.res_lin_mir = nn.Linear(in_channels, out_channels)
        self.res_lin_dis = nn.Linear(in_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Layer 1
        hidden = self.conv1(x_dict, edge_index_dict)
        for k, v in hidden.items():
            hidden[k] = self.dropout(self.act(v))

        # Layer 2
        out = self.conv2(hidden, edge_index_dict)

        # Residual projections
        out_c = out["circRNA"] + self.res_lin_circ(x_dict["circRNA"])
        out_m = out["miRNA"]  + self.res_lin_mir(x_dict["miRNA"])
        out_d = out["disease"]  + self.res_lin_dis(x_dict["disease"])

        # Normalize embeddings
        out_c = nn.functional.normalize(out_c, p=2, dim=1)
        out_m = nn.functional.normalize(out_m, p=2, dim=1)
        out_d = nn.functional.normalize(out_d, p=2, dim=1)

        return {"circRNA": out_c, "miRNA": out_m, "disease": out_d}
