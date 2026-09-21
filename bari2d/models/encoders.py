from __future__ import annotations

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=2, padding=1),
            nn.Tanh(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=2),
            nn.Tanh(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # Input: batch, time, channels.
        encoded = self.network(values.transpose(1, 2))
        return encoded.mean(dim=-1)


class TrafficEncoder(nn.Module):
    def __init__(self, ray_count: int, hidden_size: int):
        super().__init__()
        self.temporal = TemporalEncoder(ray_count + 1, hidden_size)

    def forward(self, ir_history: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        return self.temporal(torch.cat((ir_history, action_history.unsqueeze(-1)), dim=-1))


class ConnectivityEncoder(nn.Module):
    def __init__(self, ray_count: int, hidden_size: int):
        super().__init__()
        self.temporal = nn.GRU(ray_count + 3, hidden_size, batch_first=True)

    def forward(
        self,
        ir_history: torch.Tensor,
        strain_history: torch.Tensor,
        action_history: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        repeated_velocity = velocity.unsqueeze(1).expand(-1, ir_history.shape[1], -1)
        sequence = torch.cat(
            (ir_history, strain_history.unsqueeze(-1), action_history.unsqueeze(-1), repeated_velocity), dim=-1
        )
        _, hidden = self.temporal(sequence)
        return hidden[-1]


class MechanicalEncoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.temporal = TemporalEncoder(2, hidden_size)

    def forward(self, strain_history: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        return self.temporal(torch.stack((strain_history, action_history), dim=-1))


class GoalEncoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, hidden_size), nn.Tanh())

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        return self.network(target)


class FiLMConditioner(nn.Module):
    def __init__(self, goal_size: int, feature_size: int):
        super().__init__()
        self.affine = nn.Linear(goal_size, feature_size * 2)
        nn.init.zeros_(self.affine.weight)
        with torch.no_grad():
            self.affine.bias[:feature_size].fill_(1.0)
            self.affine.bias[feature_size:].zero_()

    def forward(self, features: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(goal).chunk(2, dim=-1)
        return gamma * features + beta

