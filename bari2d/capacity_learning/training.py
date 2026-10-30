from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from bari2d.capacity_learning.model import CapacityGNN, CapacityGNNOutput


@dataclass
class CapacityLoss:
    total: torch.Tensor
    capacity: torch.Tensor
    component: torch.Tensor
    critical_edge: torch.Tensor
    edge_capacity: torch.Tensor


def capacity_multitask_loss(
    output: CapacityGNNOutput,
    batch: dict[str, torch.Tensor],
    *,
    component_weight: float = 0.15,
    critical_edge_weight: float = 0.10,
    edge_capacity_weight: float = 0.40,
) -> CapacityLoss:
    target_log_capacity = torch.log1p(batch["target_capacity_kg"])
    capacity_loss = F.smooth_l1_loss(output.log_capacity, target_log_capacity)
    component_loss = F.cross_entropy(output.critical_component_logits, batch["critical_component"])

    edge_target = batch["critical_edge_mask"]
    positives = edge_target.sum().clamp_min(1.0)
    negatives = (edge_target.numel() - edge_target.sum()).clamp_min(1.0)
    pos_weight = (negatives / positives).detach()
    edge_loss = F.binary_cross_entropy_with_logits(
        output.critical_edge_logits,
        edge_target,
        pos_weight=pos_weight,
    )
    edge_capacity_target = torch.log1p(batch["edge_capacity_kg"])
    edge_capacity_loss = F.smooth_l1_loss(output.edge_log_capacity, edge_capacity_target)
    total = (
        capacity_loss
        + component_weight * component_loss
        + critical_edge_weight * edge_loss
        + edge_capacity_weight * edge_capacity_loss
    )
    return CapacityLoss(total, capacity_loss, component_loss, edge_loss, edge_capacity_loss)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device | str) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


@torch.no_grad()
def capacity_metrics(
    model: CapacityGNN,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    output = model(
        batch["node_features"],
        batch["edge_index"],
        batch["edge_features"],
        batch["node_graph"],
    )
    prediction = output.capacity_kg
    target = batch["target_capacity_kg"]
    absolute = (prediction - target).abs()
    relative = absolute / target.clamp_min(0.1)
    component_accuracy = (
        output.critical_component_logits.argmax(dim=-1) == batch["critical_component"]
    ).float().mean()

    edge_graph = batch["edge_graph"]
    edge_correct = []
    graph_count = int(batch["target_capacity_kg"].shape[0])
    for graph_id in range(graph_count):
        mask = edge_graph == graph_id
        if not torch.any(mask):
            continue
        local_logits = output.critical_edge_logits[mask]
        local_target = batch["critical_edge_mask"][mask]
        selected = int(torch.argmax(local_logits).item())
        edge_correct.append(float(local_target[selected].item() > 0.5))

    return {
        "mae_kg": float(absolute.mean().item()),
        "median_ae_kg": float(absolute.median().item()),
        "mape": float(relative.mean().item()),
        "component_accuracy": float(component_accuracy.item()),
        "critical_edge_top1": float(sum(edge_correct) / max(len(edge_correct), 1)),
    }
