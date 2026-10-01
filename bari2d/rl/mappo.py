from __future__ import annotations

import math
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from bari2d.models.actor import SharedRecurrentActor
from bari2d.models.critic import CentralizedCritic
from bari2d.models.graph_critic import GraphCritic
from bari2d.rl.rollout_buffer import RolloutBatch
from bari2d.utils.config import TrainingConfig

if TYPE_CHECKING:
    from bari2d.rl.demonstrations import DemonstrationSet


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

    @property
    def agent_conditioned(self) -> bool:
        return isinstance(self.critic, GraphCritic) and self.critic.agent_conditioned

    def value(
        self,
        global_state: torch.Tensor,
        graph_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Critic output: team value (...,) or, when agent-conditioned, per-robot values (..., N)."""
        if isinstance(self.critic, GraphCritic):
            if graph_state is None:
                raise ValueError("Graph critic requires graph state")
            return self.critic(*graph_state)
        return self.critic(global_state)

    def robot_values(
        self,
        global_states: torch.Tensor,
        graph_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        robot_count: int,
    ) -> torch.Tensor:
        """Values for a batch of states, always shaped (batch, robot_count)."""
        values = self.value(global_states, graph_state)
        if values.ndim == 1:
            values = values.unsqueeze(-1).expand(-1, robot_count)
        return values

    def _minibatch_count(self, sequences: int, robot_count: int) -> int:
        if self.config.num_minibatches > 0:
            return min(self.config.num_minibatches, sequences)
        per_minibatch = max(min(self.config.sequence_minibatch_agents, robot_count), 1)
        return min(math.ceil(robot_count / per_minibatch), sequences)

    def update(
        self,
        batch: RolloutBatch,
        demonstrations: DemonstrationSet | None = None,
        bc_coef: float = 0.0,
        train_actor: bool = True,
    ) -> dict[str, float]:
        config = self.config
        horizon, columns = batch.actions.shape
        chunk = config.recurrent_chunk if config.recurrent_chunk > 0 else horizon
        if horizon % chunk:
            raise ValueError(f"recurrent_chunk ({chunk}) must divide the rollout length ({horizon})")
        chunks = horizon // chunk

        def sequences(values: torch.Tensor) -> torch.Tensor:
            """(T, B, ...) -> (L, chunks * B, ...): every chunk of every column is one sequence."""
            shaped = values.reshape(chunks, chunk, columns, *values.shape[2:]).transpose(0, 1)
            return shaped.reshape(chunk, chunks * columns, *values.shape[2:])

        observations = sequences(batch.observations)
        actions = sequences(batch.actions)
        old_log_probabilities = sequences(batch.old_log_probabilities)
        action_masks = sequences(batch.action_masks)
        reset_masks = sequences(batch.reset_masks)
        advantages = sequences(batch.advantages)
        auxiliary_targets = {name: sequences(values) for name, values in batch.auxiliary_targets.items()}
        # Each chunk starts from the hidden state the actor had during the rollout.
        initial_hidden = batch.hidden_states.reshape(chunks, chunk, columns, -1)[:, 0].reshape(chunks * columns, -1)
        sequence_count = chunks * columns
        minibatches = self._minibatch_count(sequence_count, batch.robot_count)
        minibatch_size = math.ceil(sequence_count / minibatches)
        use_bc = demonstrations is not None and len(demonstrations) > 0 and bc_coef > 0.0

        metrics: dict[str, list[float]] = {
            "actor_loss": [], "policy_loss": [], "critic_loss": [], "entropy": [], "auxiliary_loss": [],
            "approx_kl": [], "clip_fraction": [], "bc_loss": [],
        }
        for _ in range(config.ppo_epochs):
            order = torch.randperm(sequence_count, device=self.device)
            for start in range(0, sequence_count if train_actor else 0, minibatch_size):
                index = order[start : start + minibatch_size]
                log_probabilities, entropies, predictions = self.actor.evaluate_sequence(
                    observations[:, index],
                    initial_hidden[index],
                    actions[:, index],
                    action_masks[:, index],
                    reset_masks[:, index],
                )
                log_ratio = log_probabilities - old_log_probabilities[:, index]
                ratio = torch.exp(log_ratio)
                minibatch_advantages = advantages[:, index]
                unclipped = ratio * minibatch_advantages
                clipped = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * minibatch_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                entropy = entropies.mean()
                auxiliary_loss = torch.zeros((), device=self.device)
                for name, prediction in predictions.items():
                    auxiliary_loss = auxiliary_loss + nn.functional.mse_loss(prediction, auxiliary_targets[name][:, index])
                actor_loss = policy_loss - config.entropy_coef * entropy + config.auxiliary_coef * auxiliary_loss
                bc_loss = torch.zeros((), device=self.device)
                if use_bc:
                    bc_loss = demonstrations.behaviour_cloning_loss(  # type: ignore[union-attr]
                        self.actor, config.bc_batch_episodes, config.bc_idle_weight, self.device, config.bc_window_steps
                    )
                    actor_loss = actor_loss + bc_coef * bc_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), config.max_grad_norm)
                self.actor_optimizer.step()
                with torch.no_grad():
                    metrics["approx_kl"].append(float(((ratio - 1.0) - log_ratio).mean()))
                    metrics["clip_fraction"].append(float(((ratio - 1.0).abs() > config.clip_ratio).float().mean()))
                metrics["actor_loss"].append(float(actor_loss.detach()))
                metrics["policy_loss"].append(float(policy_loss.detach()))
                metrics["entropy"].append(float(entropy.detach()))
                metrics["auxiliary_loss"].append(float(auxiliary_loss.detach()))
                metrics["bc_loss"].append(float(bc_loss.detach()))

            values = self._batch_values(batch)
            critic_loss = nn.functional.mse_loss(values, batch.returns)
            self.critic_optimizer.zero_grad()
            (config.value_coef * critic_loss).backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), config.max_grad_norm)
            self.critic_optimizer.step()
            metrics["critic_loss"].append(float(critic_loss.detach()))
        result = {name: float(np.mean(values)) if values else 0.0 for name, values in metrics.items()}
        result["actor_frozen"] = float(not train_actor)
        returns = batch.returns.flatten()
        variance = float(returns.var()) if returns.numel() > 1 else 0.0
        result["explained_variance"] = (
            float(1.0 - (returns - batch.values.flatten()).var() / variance) if variance > 1.0e-12 else 0.0
        )
        result["bc_coef"] = float(bc_coef if use_bc and train_actor else 0.0)
        return result

    def _batch_values(self, batch: RolloutBatch) -> torch.Tensor:
        """Critic values for every stored state, shaped like ``batch.returns`` (T, E, N)."""
        horizon, envs, robots = batch.returns.shape
        flat_states = batch.global_states.reshape(horizon * envs, -1)
        graph_state = None
        if isinstance(self.critic, GraphCritic):
            if batch.graph_nodes is None or batch.graph_adjacency is None or batch.graph_edges is None:
                raise ValueError("Graph rollout state missing")
            graph_state = (
                batch.graph_nodes.reshape(horizon * envs, robots, -1),
                batch.graph_adjacency.reshape(horizon * envs, robots, robots),
                batch.graph_edges.reshape(horizon * envs, robots, robots, -1),
            )
        return self.robot_values(flat_states, graph_state, robots).reshape(horizon, envs, robots)

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
