from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bari2d.env.bridge_env import BridgeEnv
from bari2d.models.actor import SharedRecurrentActor
from bari2d.models.critic import CentralizedCritic
from bari2d.models.graph_critic import GraphCritic
from bari2d.rl.demonstrations import DemonstrationSet
from bari2d.rl.mappo import MAPPO
from bari2d.rl.rollout_buffer import AUXILIARY_TARGETS, RolloutBuffer
from bari2d.rl.vec_env import make_vector_env
from bari2d.utils.config import ExperimentConfig
from bari2d.utils.logging import EpisodeLogger, PolicyStatistics


class CurriculumScheduler:
    """Advances the stage when the recent success rate at the current stage is high enough.

    With parallel environments, episodes that were still running from an
    earlier stage when the stage advanced are not counted.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config.training
        self.stage = int(config.environment.curriculum_stage)
        self.results: deque[bool] = deque(maxlen=self.config.curriculum_window)

    def record(self, success: bool, stage: int | None = None) -> bool:
        if not self.config.curriculum_enabled:
            return False
        if stage is not None and int(stage) != self.stage:
            return False
        self.results.append(success)
        if (
            len(self.results) == self.results.maxlen
            and np.mean(self.results) >= self.config.curriculum_success_threshold
            and self.stage < self.config.curriculum_max_stage
        ):
            self.stage += 1
            self.results.clear()
            return True
        return False


class Trainer:
    def __init__(self, config: ExperimentConfig, device: str | torch.device = "cpu"):
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(config.environment.seed)
        # Reference environment for sizes and layout; rollouts use self.vector_env.
        self.env = BridgeEnv(config.environment)
        self.actor = SharedRecurrentActor(self.env.layout, self.env.action_count, config.model)
        training = config.training
        if training.critic == "graph":
            self.critic: CentralizedCritic | GraphCritic = GraphCritic(
                node_size=self.env.graph_node_size, agent_conditioned=training.critic_agent_conditioned
            )
        elif training.critic == "mlp":
            if training.critic_agent_conditioned:
                raise ValueError("critic_agent_conditioned requires critic: graph")
            self.critic = CentralizedCritic(self.env.global_state_size)
        else:
            raise ValueError(f"Unknown critic: {training.critic}")
        if training.recurrent_chunk > 0 and training.rollout_steps % training.recurrent_chunk:
            raise ValueError("training.recurrent_chunk must divide training.rollout_steps")
        self.use_graph = isinstance(self.critic, GraphCritic)
        self.algorithm = MAPPO(self.actor, self.critic, training, self.device)
        self.logger = EpisodeLogger(training.output_dir)
        self.curriculum = CurriculumScheduler(config)
        self.output_dir = Path(training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.completed_updates = 0
        self.demonstrations: DemonstrationSet | None = None
        if training.demo_path and (training.bc_coef > 0.0 or training.bc_coef_final > 0.0):
            self.demonstrations = DemonstrationSet.load(training.demo_path)

    # ------------------------------------------------------------ checkpoints
    def load_checkpoint(self, path: str | Path) -> int:
        """Restore policy, critic, optimizer state and curriculum stage before continuing training.

        Environment state is deliberately reset at the next rollout boundary:
        this avoids restoring partially completed physics episodes while still
        preserving all learnable MAPPO state and the absolute update number.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "algorithm" not in checkpoint:
            raise ValueError(f"Unsupported training checkpoint: {checkpoint_path}")
        update = checkpoint.get("update")
        if not isinstance(update, int) or update < 0:
            raise ValueError(f"Checkpoint has an invalid update number: {checkpoint_path}")
        self.algorithm.load_state_dict(checkpoint["algorithm"])
        self.completed_updates = update
        stage = checkpoint.get("curriculum_stage")
        if isinstance(stage, int) and stage >= 1:
            self.curriculum.stage = stage
            self.env.config.curriculum_stage = stage
        return update

    def load_pretrained(self, path: str | Path) -> dict[str, Any]:
        """Initialise the actor from a behaviour-cloning checkpoint (scripts/pretrain_bc.py)."""
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        state = checkpoint.get("algorithm", {}).get("actor") if isinstance(checkpoint, dict) else None
        if state is None:
            raise ValueError(f"Checkpoint has no actor weights: {path}")
        self.actor.load_state_dict(state)
        return dict(checkpoint.get("pretrain", {}))

    def save_checkpoint(self, update: int) -> Path:
        path = self.output_dir / f"checkpoint_{update:06d}.pt"
        torch.save(
            {
                "algorithm": self.algorithm.state_dict(),
                "experiment_config": self.config.to_dict(),
                "update": update,
                "curriculum_stage": self.curriculum.stage,
            },
            path,
        )
        return path

    # ------------------------------------------------------------------ utils
    def bc_coefficient(self, update: int) -> float:
        training = self.config.training
        if self.demonstrations is None:
            return 0.0
        if training.bc_decay_updates <= 0:
            return training.bc_coef
        fraction = min(max(update - 1, 0) / training.bc_decay_updates, 1.0)
        return training.bc_coef + (training.bc_coef_final - training.bc_coef) * fraction

    def _values(self, global_states: np.ndarray, graph_states: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> np.ndarray:
        """Per-robot critic values (E, N) for a stack of environment states."""
        robots = self.env.config.robot.count
        with torch.no_grad():
            states = torch.as_tensor(global_states, device=self.device)
            graph = None if graph_states is None else tuple(torch.as_tensor(item, device=self.device) for item in graph_states)
            return self.algorithm.robot_values(states, graph, robots).cpu().numpy()

    @staticmethod
    def _stack(states: list[dict[str, Any]]) -> dict[str, Any]:
        stacked = {
            "observation": np.stack([state["observation"] for state in states]).astype(np.float32),
            "action_masks": np.stack([state["action_masks"] for state in states]),
            "global_state": np.stack([state["global_state"] for state in states]).astype(np.float32),
            "graph_state": None,
        }
        if states[0]["graph_state"] is not None:
            stacked["graph_state"] = tuple(
                np.stack([state["graph_state"][part] for state in states]).astype(np.float32) for part in range(3)
            )
        return stacked

    # --------------------------------------------------------------- training
    def train(self, updates: int | None = None) -> list[dict[str, float]]:
        training = self.config.training
        target_update = updates if updates is not None else training.total_updates
        if target_update <= self.completed_updates:
            raise ValueError(
                f"Requested target update {target_update} is not after restored update {self.completed_updates}"
            )
        environment_config = self.config.environment
        environment_config.curriculum_stage = self.curriculum.stage
        vector_env = make_vector_env(environment_config, training.num_envs, training.vector_env, self.use_graph)
        try:
            return self._train(vector_env, target_update)
        finally:
            vector_env.close()

    def _train(self, vector_env, target_update: int) -> list[dict[str, float]]:
        training = self.config.training
        envs, robots = vector_env.num_envs, self.env.config.robot.count
        hidden_size = self.actor.hidden_size
        state = self._stack(vector_env.reset())
        hidden = self.actor.initial_hidden(envs * robots, self.device)
        episode_start = np.ones((envs, robots), dtype=np.float32)
        statistics = [PolicyStatistics(self.env.action_count) for _ in range(envs)]
        episode_returns = np.zeros(envs, dtype=np.float64)
        episode_components: list[dict[str, float]] = [{} for _ in range(envs)]
        history: list[dict[str, float]] = []
        episode_number = 0
        for update in range(self.completed_updates + 1, target_update + 1):
            buffer = RolloutBuffer(
                training.rollout_steps,
                envs,
                robots,
                self.env.observation_size,
                self.env.global_state_size,
                self.env.action_count,
                hidden_size,
                use_graph=self.use_graph,
                graph_node_size=self.env.graph_node_size,
            )
            episodes_this_update = 0
            for _ in range(training.rollout_steps):
                observation_tensor = torch.as_tensor(state["observation"].reshape(envs * robots, -1), device=self.device)
                mask_tensor = torch.as_tensor(state["action_masks"].reshape(envs * robots, -1), device=self.device)
                reset_tensor = torch.as_tensor(episode_start.reshape(-1), device=self.device)
                hidden_in = hidden.detach().cpu().numpy().reshape(envs, robots, hidden_size)
                with torch.no_grad():
                    actions, log_probabilities, _, output = self.actor.act(
                        observation_tensor, hidden, mask_tensor, reset_mask=reset_tensor
                    )
                hidden = output.hidden
                values = self._values(state["global_state"], state["graph_state"])
                action_values = actions.cpu().numpy().reshape(envs, robots)
                results = vector_env.step(action_values)

                rewards = np.array([result["reward"] for result in results], dtype=np.float64)
                terminated = np.array([result["terminated"] for result in results])
                truncated = np.array([result["truncated"] for result in results])
                dones = terminated | truncated
                stored_rewards = np.repeat(rewards[:, None], robots, axis=1).astype(np.float32)
                bootstrap = [index for index in range(envs) if truncated[index] and not terminated[index]]
                if bootstrap and training.bootstrap_truncation:
                    final_global = np.stack([results[index]["final_global_state"] for index in bootstrap]).astype(np.float32)
                    final_graph = None
                    if self.use_graph:
                        final_graph = tuple(
                            np.stack([results[index]["final_graph_state"][part] for index in bootstrap]).astype(np.float32)
                            for part in range(3)
                        )
                    final_values = self._values(final_global, final_graph)
                    for row, index in enumerate(bootstrap):
                        stored_rewards[index] += training.gamma * final_values[row]
                auxiliary = {
                    name: np.stack([np.asarray(result["auxiliary_targets"][name], dtype=np.float32) for result in results])
                    for name in AUXILIARY_TARGETS
                }
                buffer.add(
                    state["observation"],
                    state["global_state"],
                    action_values,
                    log_probabilities.cpu().numpy().reshape(envs, robots),
                    state["action_masks"],
                    episode_start,
                    hidden_in,
                    stored_rewards,
                    values,
                    dones.astype(np.float32),
                    auxiliary,
                    state["graph_state"],
                )
                for index, result in enumerate(results):
                    statistics[index].add(
                        action_values[index],
                        {name: value[index * robots : (index + 1) * robots] for name, value in output.latents.items()},
                    )
                    episode_returns[index] += result["reward"]
                    for name, component in result["reward_components"].items():
                        episode_components[index][name] = episode_components[index].get(name, 0.0) + float(component)
                    if not dones[index]:
                        continue
                    episode_number += 1
                    episodes_this_update += 1
                    info = result["episode_info"]
                    episode_log: dict[str, Any] = dict(info)
                    episode_log.update(statistics[index].summarize())
                    episode_log.update(episode=episode_number, update=update, env_index=index)
                    episode_log.update(
                        episode_return=float(episode_returns[index]),
                        episode_reward_components=episode_components[index],
                    )
                    advanced = self.curriculum.record(bool(info["success"]), info.get("curriculum_stage"))
                    episode_log["curriculum_advanced"] = advanced
                    self.logger.log(episode_log)
                    if advanced:
                        vector_env.set_stage(self.curriculum.stage)
                        self.env.config.curriculum_stage = self.curriculum.stage
                    episode_returns[index] = 0.0
                    episode_components[index] = {}
                    statistics[index] = PolicyStatistics(self.env.action_count)
                state = self._stack(results)
                episode_start = np.repeat(dones[:, None], robots, axis=1).astype(np.float32)
            last_values = self._values(state["global_state"], state["graph_state"])
            buffer.compute_returns(last_values, training.gamma, training.gae_lambda)
            bc_coef = self.bc_coefficient(update)
            train_actor = update > self.config.training.critic_warmup_updates
            metrics = self.algorithm.update(buffer.as_tensors(self.device), self.demonstrations, bc_coef, train_actor)
            metrics["update"] = float(update)
            metrics["curriculum_stage"] = float(self.curriculum.stage)
            metrics["episodes"] = float(episodes_this_update)
            history.append(metrics)
            with (self.output_dir / "updates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(
                f"Update {update}/{target_update} "
                f"actor_loss={metrics['actor_loss']:.4f} "
                f"critic_loss={metrics['critic_loss']:.4f} "
                f"entropy={metrics['entropy']:.4f} "
                f"kl={metrics['approx_kl']:.4f} "
                f"ev={metrics['explained_variance']:.2f} "
                f"stage={self.curriculum.stage}",
                flush=True,
            )
            if update % training.checkpoint_interval == 0 or update == target_update:
                self.save_checkpoint(update)
        self.completed_updates = target_update
        return history
