from __future__ import annotations

import torch
from torch import nn


class CentralizedCritic(nn.Module):
    def __init__(self, global_state_size: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(global_state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.network(global_state).squeeze(-1)

