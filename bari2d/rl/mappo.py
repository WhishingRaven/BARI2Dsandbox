from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch import nn

from bari2d.models.actor import SharedRecurrentActor
from bari2d.models.critic import CentralizedCritic
from bari2d.models.graph_critic import GraphCritic
from bari2d.rl.rollout_buffer import RolloutBatch
from bari2d.utils.config import TrainingConfig


class MAPPO:
    def __init__(
        self,
        actor: SharedRecurrentActor,
        critic: CentralizedCritic | GraphCritic,
        config: TrainingConfig,
        device: torch.device | str = "cpu",
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.config = config
        self.device = torch.device(device)
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)

    def value(
        self,
        global_state: torch.Tensor,
        graph_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if isinstance(self.critic, GraphCritic):
            if graph_state is None:
                raise ValueError("Graph critic requires graph state")
            return self.critic(*graph_state)
        return self.critic(global_state)

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        config = self.config
        robot_count = batch.observations.shape[1]
        minibatch_size = min(config.sequence_minibatch_agents, robot_count)
        metrics: dict[str, list[float]] = {"actor_loss": [], "critic_loss": [], "entropy": [], "auxiliary_loss": []}
        for _ in range(config.ppo_epochs):
            agent_order = torch.randperm(robot_count, device=self.device)
            for start in range(0, robot_count, minibatch_size):
                agents = agent_order[start : start + minibatch_size]
                log_probabilities, entropies, predictions = self.actor.evaluate_sequence(
                    batch.observations[:, agents],
                    batch.initial_hidden[agents],
                    batch.actions[:, agents],
                    batch.action_masks[:, agents],
                    batch.reset_masks[:, agents],
                )
                ratio = torch.exp(log_probabilities - batch.old_log_probabilities[:, agents])
                advantages = batch.advantages[:, None].expand_as(ratio)
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                entropy = entropies.mean()
                auxiliary_loss = torch.zeros((), device=self.device)
                for name, prediction in predictions.items():
                    target = batch.auxiliary_targets[name][:, agents]
                    auxiliary_loss = auxiliary_loss + nn.functional.mse_loss(prediction, target)
                actor_loss = policy_loss - config.entropy_coef * entropy + config.auxiliary_coef * auxiliary_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), config.max_grad_norm)
                self.actor_optimizer.step()
                metrics["actor_loss"].append(float(actor_loss.detach()))
                metrics["entropy"].append(float(entropy.detach()))
                metrics["auxiliary_loss"].append(float(auxiliary_loss.detach()))

            if isinstance(self.critic, GraphCritic):
                if batch.graph_nodes is None or batch.graph_adjacency is None or batch.graph_edges is None:
                    raise ValueError("Graph rollout state missing")
                values = self.critic(batch.graph_nodes, batch.graph_adjacency, batch.graph_edges)
            else:
                values = self.critic(batch.global_states)
            critic_loss = nn.functional.mse_loss(values, batch.returns)
            self.critic_optimizer.zero_grad()
            (config.value_coef * critic_loss).backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), config.max_grad_norm)
            self.critic_optimizer.step()
            metrics["critic_loss"].append(float(critic_loss.detach()))
        return {name: float(np.mean(values)) for name, values in metrics.items()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "training_config": asdict(self.config),
        }

    def load_state_dict(self, state: dict[str, Any], load_optimizers: bool = True) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
