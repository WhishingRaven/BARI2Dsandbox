from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    global_states: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    action_masks: torch.Tensor
    reset_masks: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    agent_advantages: torch.Tensor
    initial_hidden: torch.Tensor
    auxiliary_targets: dict[str, torch.Tensor]
    graph_nodes: torch.Tensor | None = None
    graph_adjacency: torch.Tensor | None = None
    graph_edges: torch.Tensor | None = None


class RolloutBuffer:
    def __init__(
        self,
        horizon: int,
        robot_count: int,
        observation_size: int,
        global_state_size: int,
        action_count: int,
        hidden_size: int,
        use_graph: bool = False,
        graph_node_size: int = 13,
    ):
        self.horizon = horizon
        self.robot_count = robot_count
        self.observations = np.zeros((horizon, robot_count, observation_size), dtype=np.float32)
        self.global_states = np.zeros((horizon, global_state_size), dtype=np.float32)
        self.actions = np.zeros((horizon, robot_count), dtype=np.int64)
        self.log_probabilities = np.zeros((horizon, robot_count), dtype=np.float32)
        self.action_masks = np.zeros((horizon, robot_count, action_count), dtype=bool)
        self.reset_masks = np.zeros((horizon, robot_count), dtype=np.float32)
        self.rewards = np.zeros(horizon, dtype=np.float32)
        self.values = np.zeros(horizon, dtype=np.float32)
        self.dones = np.zeros(horizon, dtype=np.float32)
        self.returns = np.zeros(horizon, dtype=np.float32)
        self.advantages = np.zeros(horizon, dtype=np.float32)
        self.agent_rewards = np.zeros((horizon, robot_count), dtype=np.float32)
        self.agent_advantages = np.zeros((horizon, robot_count), dtype=np.float32)
        self.initial_hidden = np.zeros((robot_count, hidden_size), dtype=np.float32)
        self.auxiliary_targets = {
            "connectivity": np.zeros((horizon, robot_count), dtype=np.float32),
            "traffic": np.zeros((horizon, robot_count), dtype=np.float32),
            "contact_persistence": np.zeros((horizon, robot_count), dtype=np.float32),
            "force_trend": np.zeros((horizon, robot_count), dtype=np.float32),
        }
        self.graph_nodes = np.zeros((horizon, robot_count, graph_node_size), dtype=np.float32) if use_graph else None
        self.graph_adjacency = np.zeros((horizon, robot_count, robot_count), dtype=np.float32) if use_graph else None
        self.graph_edges = np.zeros((horizon, robot_count, robot_count, 6), dtype=np.float32) if use_graph else None
        self.position = 0

    def set_initial_hidden(self, hidden: np.ndarray) -> None:
        self.initial_hidden[:] = hidden

    def add(
        self,
        observation: np.ndarray,
        global_state: np.ndarray,
        actions: np.ndarray,
        log_probabilities: np.ndarray,
        action_masks: np.ndarray,
        reset_mask: np.ndarray,
        reward: float,
        agent_rewards: np.ndarray,
        value: float,
        done: bool,
        auxiliary_targets: dict[str, np.ndarray],
        graph_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        if self.position >= self.horizon:
            raise RuntimeError("Rollout buffer is full")
        index = self.position
        self.observations[index] = observation
        self.global_states[index] = global_state
        self.actions[index] = actions
        self.log_probabilities[index] = log_probabilities
        self.action_masks[index] = action_masks
        self.reset_masks[index] = reset_mask
        self.rewards[index] = reward
        self.agent_rewards[index] = agent_rewards
        self.values[index] = value
        self.dones[index] = done
        for name, values in auxiliary_targets.items():
            if name in self.auxiliary_targets:
                self.auxiliary_targets[name][index] = values
        if graph_state is not None:
            if self.graph_nodes is None or self.graph_adjacency is None or self.graph_edges is None:
                raise RuntimeError("Graph state provided to non-graph buffer")
            self.graph_nodes[index], self.graph_adjacency[index], self.graph_edges[index] = graph_state
        self.position += 1

    def compute_returns(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        if self.position != self.horizon:
            raise RuntimeError("Rollout must be full before computing returns")
        advantage = 0.0
        for step in reversed(range(self.horizon)):
            next_value = last_value if step == self.horizon - 1 else self.values[step + 1]
            continuation = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_value * continuation - self.values[step]
            advantage = delta + gamma * gae_lambda * continuation * advantage
            self.advantages[step] = advantage
        self.returns = self.advantages + self.values
        mean = float(self.advantages.mean())
        standard_deviation = float(self.advantages.std())
        self.advantages = (self.advantages - mean) / (standard_deviation + 1.0e-8)

        discounted_credit = np.zeros(self.robot_count, dtype=np.float32)
        for step in reversed(range(self.horizon)):
            continuation = 1.0 - self.dones[step]
            discounted_credit = self.agent_rewards[step] + gamma * continuation * discounted_credit
            self.agent_advantages[step] = discounted_credit
        credit_mean = float(self.agent_advantages.mean())
        credit_std = float(self.agent_advantages.std())
        self.agent_advantages = (self.agent_advantages - credit_mean) / (credit_std + 1.0e-8)

    def as_tensors(self, device: torch.device | str) -> RolloutBatch:
        tensor = lambda values, dtype=None: torch.as_tensor(values, dtype=dtype, device=device)
        return RolloutBatch(
            observations=tensor(self.observations),
            global_states=tensor(self.global_states),
            actions=tensor(self.actions, torch.long),
            old_log_probabilities=tensor(self.log_probabilities),
            action_masks=tensor(self.action_masks, torch.bool),
            reset_masks=tensor(self.reset_masks),
            returns=tensor(self.returns),
            advantages=tensor(self.advantages),
            agent_advantages=tensor(self.agent_advantages),
            initial_hidden=tensor(self.initial_hidden),
            auxiliary_targets={name: tensor(values) for name, values in self.auxiliary_targets.items()},
            graph_nodes=None if self.graph_nodes is None else tensor(self.graph_nodes),
            graph_adjacency=None if self.graph_adjacency is None else tensor(self.graph_adjacency),
            graph_edges=None if self.graph_edges is None else tensor(self.graph_edges),
        )
