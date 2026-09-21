from __future__ import annotations

import torch
from torch import nn


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_size: int, edge_size: int):
        super().__init__()
        self.self_projection = nn.Linear(hidden_size, hidden_size)
        self.neighbor_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.edge_gate = nn.Sequential(nn.Linear(edge_size, hidden_size), nn.Sigmoid())
        self.activation = nn.Tanh()

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        # nodes: B,N,H; adjacency: B,N,N; edges: B,N,N,E
        neighbor_values = self.neighbor_projection(nodes).unsqueeze(1)
        gates = self.edge_gate(edges)
        messages = (neighbor_values * gates * adjacency.unsqueeze(-1)).sum(dim=2)
        degree = adjacency.sum(dim=2, keepdim=True).clamp_min(1.0)
        return self.activation(self.self_projection(nodes) + messages / degree)


class GraphCritic(nn.Module):
    """Training-only centralized graph critic without external GNN dependencies."""

    def __init__(self, node_size: int = 13, edge_size: int = 6, hidden_size: int = 64, layers: int = 2):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(node_size, hidden_size), nn.Tanh())
        self.layers = nn.ModuleList(MessagePassingLayer(hidden_size, edge_size) for _ in range(layers))
        self.value_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        if nodes.ndim == 2:
            nodes, adjacency, edges = nodes.unsqueeze(0), adjacency.unsqueeze(0), edges.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        hidden = self.node_encoder(nodes)
        for layer in self.layers:
            hidden = layer(hidden, adjacency, edges)
        pooled = hidden.mean(dim=1)
        value = self.value_head(pooled).squeeze(-1)
        return value.squeeze(0) if squeeze else value
