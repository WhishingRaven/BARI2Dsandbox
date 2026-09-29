from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


AUXILIARY_TARGETS = ("connectivity", "traffic", "contact_persistence", "force_trend")


@dataclass
class RolloutBatch:
    """One rollout of ``env_count`` environments with ``robot_count`` robots each.

    Actor tensors are laid out (T, B) with B = env_count * robot_count
    (column b = env * robot_count + robot).  Critic tensors keep the
    environment axis: global states (T, E, ...), graph states (T, E, N, ...)
    and per-robot returns/values (T, E, N).
    """

    observations: torch.Tensor
    global_states: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    action_masks: torch.Tensor
    reset_masks: torch.Tensor
    hidden_states: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    auxiliary_targets: dict[str, torch.Tensor]
    env_count: int
    robot_count: int
    graph_nodes: torch.Tensor | None = None
    graph_adjacency: torch.Tensor | None = None
    graph_edges: torch.Tensor | None = None

    @property
    def initial_hidden(self) -> torch.Tensor:
        return self.hidden_states[0]


class RolloutBuffer:
    """Stores a rollout of parallel environments; GAE is computed per robot.

    Rewards are team rewards broadcast to every robot, but values (and the
    truncation bootstrap) can differ per robot when the critic is
    agent-conditioned, so advantages are per robot as well.  A team critic
    simply stores the same value for every robot.
    """

    def __init__(
        self,
        horizon: int,
        env_count: int,
        robot_count: int,
        observation_size: int,
        global_state_size: int,
        action_count: int,
        hidden_size: int,
        use_graph: bool = False,
        graph_node_size: int = 13,
    ):
        self.horizon = horizon
        self.env_count = env_count
        self.robot_count = robot_count
        shape = (horizon, env_count, robot_count)
        self.observations = np.zeros(shape + (observation_size,), dtype=np.float32)
        self.global_states = np.zeros((horizon, env_count, global_state_size), dtype=np.float32)
        self.actions = np.zeros(shape, dtype=np.int64)
        self.log_probabilities = np.zeros(shape, dtype=np.float32)
        self.action_masks = np.zeros(shape + (action_count,), dtype=bool)
        self.reset_masks = np.zeros(shape, dtype=np.float32)
        self.hidden_states = np.zeros(shape + (hidden_size,), dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros((horizon, env_count), dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.auxiliary_targets = {name: np.zeros(shape, dtype=np.float32) for name in AUXILIARY_TARGETS}
        self.graph_nodes = np.zeros(shape + (graph_node_size,), dtype=np.float32) if use_graph else None
        self.graph_adjacency = np.zeros(shape + (robot_count,), dtype=np.float32) if use_graph else None
        self.graph_edges = np.zeros(shape + (robot_count, 6), dtype=np.float32) if use_graph else None
        self.position = 0

    def add(
        self,
        observations: np.ndarray,
        global_states: np.ndarray,
        actions: np.ndarray,
        log_probabilities: np.ndarray,
        action_masks: np.ndarray,
        reset_masks: np.ndarray,
        hidden_states: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        auxiliary_targets: dict[str, np.ndarray],
        graph_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Add one step of every environment; per-robot arrays are (E, N, ...), ``dones`` is (E,)."""
        if self.position >= self.horizon:
            raise RuntimeError("Rollout buffer is full")
        index = self.position
        self.observations[index] = observations
        self.global_states[index] = global_states
        self.actions[index] = actions
        self.log_probabilities[index] = log_probabilities
        self.action_masks[index] = action_masks
        self.reset_masks[index] = reset_masks
        self.hidden_states[index] = hidden_states
        self.rewards[index] = rewards
        self.values[index] = values
        self.dones[index] = dones
        for name, value in auxiliary_targets.items():
            if name in self.auxiliary_targets:
                self.auxiliary_targets[name][index] = value
        if graph_state is not None:
            if self.graph_nodes is None or self.graph_adjacency is None or self.graph_edges is None:
                raise RuntimeError("Graph state provided to non-graph buffer")
            self.graph_nodes[index], self.graph_adjacency[index], self.graph_edges[index] = graph_state
        self.position += 1

    def compute_returns(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        """GAE per robot; ``last_values`` (E, N) are the values of the states after the rollout."""
        if self.position != self.horizon:
            raise RuntimeError("Rollout must be full before computing returns")
        advantage = np.zeros((self.env_count, self.robot_count), dtype=np.float32)
        for step in reversed(range(self.horizon)):
            next_values = last_values if step == self.horizon - 1 else self.values[step + 1]
            continuation = (1.0 - self.dones[step])[:, None]
            delta = self.rewards[step] + gamma * next_values * continuation - self.values[step]
            advantage = delta + gamma * gae_lambda * continuation * advantage
            self.advantages[step] = advantage
        self.returns = self.advantages + self.values
        mean = float(self.advantages.mean())
        standard_deviation = float(self.advantages.std())
        self.advantages = (self.advantages - mean) / (standard_deviation + 1.0e-8)

    def as_tensors(self, device: torch.device | str) -> RolloutBatch:
        horizon, columns = self.horizon, self.env_count * self.robot_count

        def tensor(values: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
            return torch.as_tensor(values, dtype=dtype, device=device)

        def flat(values: np.ndarray) -> np.ndarray:
            return values.reshape(horizon, columns, *values.shape[3:])

        return RolloutBatch(
            observations=tensor(flat(self.observations)),
            global_states=tensor(self.global_states),
            actions=tensor(flat(self.actions), torch.long),
            old_log_probabilities=tensor(flat(self.log_probabilities)),
            action_masks=tensor(flat(self.action_masks), torch.bool),
            reset_masks=tensor(flat(self.reset_masks)),
            hidden_states=tensor(flat(self.hidden_states)),
            returns=tensor(self.returns),
            values=tensor(self.values),
            advantages=tensor(flat(self.advantages)),
            auxiliary_targets={name: tensor(flat(values)) for name, values in self.auxiliary_targets.items()},
            env_count=self.env_count,
            robot_count=self.robot_count,
            graph_nodes=None if self.graph_nodes is None else tensor(self.graph_nodes),
            graph_adjacency=None if self.graph_adjacency is None else tensor(self.graph_adjacency),
            graph_edges=None if self.graph_edges is None else tensor(self.graph_edges),
        )
