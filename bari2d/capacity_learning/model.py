from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CapacityGNNOutput:
    log_capacity: torch.Tensor
    critical_component_logits: torch.Tensor
    critical_edge_logits: torch.Tensor
    edge_log_capacity: torch.Tensor

    @property
    def capacity_kg(self) -> torch.Tensor:
        return torch.expm1(self.log_capacity).clamp_min(0.0)


class EdgeMessageLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_hidden: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        messages = self.message(torch.cat((nodes[source], nodes[target], edge_hidden), dim=-1))
        aggregate = torch.zeros_like(nodes)
        aggregate.index_add_(0, target, messages)
        degree = torch.zeros((nodes.shape[0], 1), device=nodes.device, dtype=nodes.dtype)
        degree.index_add_(0, target, torch.ones((target.shape[0], 1), device=nodes.device, dtype=nodes.dtype))
        aggregate = aggregate / degree.clamp_min(1.0)
        delta = self.update(torch.cat((nodes, aggregate), dim=-1))
        return self.norm(nodes + delta)


class CapacityGNN(nn.Module):
    """Physics-structured sparse GNN for variable-size bridge graphs.

    Failure is a weakest-connection event, so the network predicts a candidate
    log-capacity for every directed contact and combines those candidates using a
    differentiable soft minimum. This is a stronger inductive bias than ordinary
    mean pooling and mirrors the exact definition of bridge capacity.
    """

    def __init__(
        self,
        node_size: int,
        edge_size: int,
        *,
        hidden_size: int = 128,
        layers: int = 4,
        dropout: float = 0.05,
        softmin_temperature: float = 0.12,
    ) -> None:
        super().__init__()
        self.softmin_temperature = float(softmin_temperature)
        if self.softmin_temperature <= 0.0:
            raise ValueError("softmin_temperature must be positive")
        self.node_encoder = nn.Sequential(
            nn.Linear(node_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layers = nn.ModuleList(EdgeMessageLayer(hidden_size) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.pool_gate = nn.Linear(hidden_size, 1)
        self.component_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 6),
        )
        # Edge readout receives both endpoint states, the encoded edge itself,
        # and a pooled graph context. This works for both mechanics-informed and
        # geometry-only features.
        self.edge_capacity_head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.edge_failure_residual = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    @staticmethod
    def _attention_pool(nodes: torch.Tensor, node_graph: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        graph_count = int(node_graph.max().item()) + 1 if node_graph.numel() else 0
        weights = torch.sigmoid(gate)
        weighted = nodes * weights
        numerator = torch.zeros((graph_count, nodes.shape[1]), device=nodes.device, dtype=nodes.dtype)
        denominator = torch.zeros((graph_count, 1), device=nodes.device, dtype=nodes.dtype)
        numerator.index_add_(0, node_graph, weighted)
        denominator.index_add_(0, node_graph, weights)
        return numerator / denominator.clamp_min(1.0e-6)

    def _softmin_by_graph(self, values: torch.Tensor, edge_graph: torch.Tensor, graph_count: int) -> torch.Tensor:
        pooled: list[torch.Tensor] = []
        temperature = self.softmin_temperature
        for graph_id in range(graph_count):
            local = values[edge_graph == graph_id]
            if local.numel() == 0:
                pooled.append(values.new_tensor(0.0))
                continue
            weights = torch.softmax(-local / temperature, dim=0)
            pooled.append(torch.sum(weights * local))
        return torch.stack(pooled)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_graph: torch.Tensor,
    ) -> CapacityGNNOutput:
        node_hidden = self.node_encoder(node_features)
        edge_hidden = self.edge_encoder(edge_features)
        for layer in self.layers:
            node_hidden = layer(node_hidden, edge_index, edge_hidden)
            node_hidden = self.dropout(node_hidden)

        pooled = self._attention_pool(node_hidden, node_graph, self.pool_gate(node_hidden))
        component_logits = self.component_head(pooled)

        source, target = edge_index
        edge_graph = node_graph[source]
        graph_context = pooled[edge_graph]
        edge_readout = torch.cat(
            (node_hidden[source], node_hidden[target], edge_hidden, graph_context),
            dim=-1,
        )
        # Candidate is log(1 + edge failure mass) and must be nonnegative.
        edge_log_capacity = torch.nn.functional.softplus(
            self.edge_capacity_head(edge_readout).squeeze(-1)
        )
        graph_count = pooled.shape[0]
        log_capacity = self._softmin_by_graph(edge_log_capacity, edge_graph, graph_count)

        # Low predicted edge capacity should imply high failure probability. A
        # learned residual lets the auxiliary classifier use additional context.
        critical_edge_logits = (
            -edge_log_capacity / self.softmin_temperature
            + self.edge_failure_residual(edge_readout).squeeze(-1)
        )
        return CapacityGNNOutput(
            log_capacity=log_capacity,
            critical_component_logits=component_logits,
            critical_edge_logits=critical_edge_logits,
            edge_log_capacity=edge_log_capacity,
        )
