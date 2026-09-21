from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from bari2d.env.bridge_env import ObservationLayout
from bari2d.models.encoders import (
    ConnectivityEncoder,
    FiLMConditioner,
    GoalEncoder,
    MechanicalEncoder,
    TrafficEncoder,
)
from bari2d.utils.config import ModelConfig


ARCHITECTURES = {
    "mlp",
    "gru",
    "gru_traffic",
    "gru_traffic_connectivity",
    "bio",
    "bio_film",
    "bio_heterogeneity",
}


@dataclass
class ActorOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    auxiliary: dict[str, torch.Tensor]
    latents: dict[str, torch.Tensor]


class SharedRecurrentActor(nn.Module):
    def __init__(self, layout: ObservationLayout, action_count: int, config: ModelConfig):
        super().__init__()
        if config.architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown actor architecture: {config.architecture}")
        self.layout = layout
        self.action_count = action_count
        self.config = config
        self.hidden_size = config.recurrent_hidden
        branch = config.branch_hidden
        self.use_traffic = config.architecture in {
            "gru_traffic", "gru_traffic_connectivity", "bio", "bio_film", "bio_heterogeneity"
        }
        self.use_connectivity = config.architecture in {
            "gru_traffic_connectivity", "bio", "bio_film", "bio_heterogeneity"
        }
        self.use_mechanical = config.architecture in {"bio", "bio_film", "bio_heterogeneity"}
        self.use_film = config.film or config.architecture == "bio_film"
        self.use_heterogeneity = config.use_heterogeneity or config.architecture == "bio_heterogeneity"
        self.is_recurrent = config.architecture != "mlp"

        self.goal_encoder = GoalEncoder(branch)
        if self.use_traffic:
            self.traffic_encoder = TrafficEncoder(layout.ray_count, branch)
            self.traffic_aux = nn.Linear(branch, 1)
        if self.use_connectivity:
            self.connectivity_encoder = ConnectivityEncoder(layout.ray_count, branch)
            self.connectivity_aux = nn.Linear(branch, 2)
        if self.use_mechanical:
            self.mechanical_encoder = MechanicalEncoder(branch)
            self.force_aux = nn.Linear(branch, 1)
        if self.use_film:
            if self.use_traffic:
                self.traffic_film = FiLMConditioner(branch, branch)
            if self.use_mechanical:
                self.mechanical_film = FiLMConditioner(branch, branch)

        if config.architecture in {"mlp", "gru"}:
            fused_input = layout.size
        else:
            fused_input = branch  # goal
            fused_input += branch if self.use_traffic else 0
            fused_input += branch if self.use_connectivity else 0
            fused_input += branch if self.use_mechanical else layout.history
            fused_input += 6
            if self.use_heterogeneity:
                fused_input += layout.latent.stop - layout.latent.start
        self.fused = nn.Sequential(nn.Linear(fused_input, config.fused_hidden), nn.Tanh())
        if self.is_recurrent:
            self.recurrent = nn.GRUCell(config.fused_hidden, config.recurrent_hidden)
            policy_input = config.recurrent_hidden
        else:
            policy_input = config.fused_hidden
        self.policy_head = nn.Linear(policy_input, action_count)

    def initial_hidden(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def _split(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = observation.shape[0]
        return {
            "ir": observation[:, self.layout.ir].reshape(batch, self.layout.history, self.layout.ray_count),
            "strain": observation[:, self.layout.strain],
            "actions": observation[:, self.layout.action_history],
            "internal": observation[:, self.layout.internal],
            "goal": observation[:, self.layout.goal],
            "latent": observation[:, self.layout.latent],
        }

    def forward_step(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> ActorOutput:
        if reset_mask is not None:
            hidden = hidden * (1.0 - reset_mask.float().reshape(-1, 1))
        parts = self._split(observation)
        latents: dict[str, torch.Tensor] = {}
        auxiliary: dict[str, torch.Tensor] = {}
        if self.config.architecture in {"mlp", "gru"}:
            fused_values = observation
        else:
            goal = self.goal_encoder(parts["goal"])
            latents["goal"] = goal
            values: list[torch.Tensor] = []
            if self.use_traffic:
                traffic = self.traffic_encoder(parts["ir"], parts["actions"])
                if self.use_film:
                    traffic = self.traffic_film(traffic, goal)
                latents["traffic"] = traffic
                auxiliary["traffic"] = self.traffic_aux(traffic).squeeze(-1)
                values.append(traffic)
            if self.use_connectivity:
                connectivity = self.connectivity_encoder(
                    parts["ir"], parts["strain"], parts["actions"], parts["internal"][:, 2:3]
                )
                latents["connectivity"] = connectivity
                connectivity_output = self.connectivity_aux(connectivity)
                auxiliary["connectivity"] = connectivity_output[:, 0]
                auxiliary["contact_persistence"] = connectivity_output[:, 1]
                values.append(connectivity)
            if self.use_mechanical:
                mechanical = self.mechanical_encoder(parts["strain"], parts["actions"])
                if self.use_film:
                    mechanical = self.mechanical_film(mechanical, goal)
                latents["mechanical"] = mechanical
                auxiliary["force_trend"] = self.force_aux(mechanical).squeeze(-1)
                values.append(mechanical)
            else:
                values.append(parts["strain"])
            values.extend((goal, parts["internal"]))
            if self.use_heterogeneity:
                values.append(parts["latent"])
            fused_values = torch.cat(values, dim=-1)
        fused = self.fused(fused_values)
        if self.is_recurrent:
            next_hidden = self.recurrent(fused, hidden)
            policy_features = next_hidden
        else:
            next_hidden = hidden
            policy_features = fused
        logits = self.policy_head(policy_features)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1.0e9)
        return ActorOutput(logits, next_hidden, auxiliary, latents)

    @torch.no_grad()
    def act(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ActorOutput]:
        output = self.forward_step(observation, hidden, action_mask, reset_mask)
        distribution = Categorical(logits=output.logits)
        action = output.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), output

    def evaluate_sequence(
        self,
        observations: torch.Tensor,
        initial_hidden: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor | None = None,
        reset_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        hidden = initial_hidden
        log_probabilities = []
        entropies = []
        auxiliary: dict[str, list[torch.Tensor]] = {}
        for step in range(observations.shape[0]):
            output = self.forward_step(
                observations[step],
                hidden,
                None if action_masks is None else action_masks[step],
                None if reset_masks is None else reset_masks[step],
            )
            hidden = output.hidden
            distribution = Categorical(logits=output.logits)
            log_probabilities.append(distribution.log_prob(actions[step]))
            entropies.append(distribution.entropy())
            for name, prediction in output.auxiliary.items():
                auxiliary.setdefault(name, []).append(prediction)
        stacked_auxiliary = {name: torch.stack(values) for name, values in auxiliary.items()}
        return torch.stack(log_probabilities), torch.stack(entropies), stacked_auxiliary

