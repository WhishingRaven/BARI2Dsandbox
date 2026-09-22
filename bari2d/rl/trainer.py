from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bari2d.env.bridge_env import BridgeEnv
from bari2d.models.actor import SharedRecurrentActor
from bari2d.models.critic import CentralizedCritic
from bari2d.models.graph_critic import GraphCritic
from bari2d.rl.mappo import MAPPO
from bari2d.rl.rollout_buffer import RolloutBuffer
from bari2d.utils.config import ExperimentConfig
from bari2d.utils.logging import EpisodeLogger, PolicyStatistics


class CurriculumScheduler:
    def __init__(self, config: ExperimentConfig):
        self.config = config.training
        self.results: deque[bool] = deque(maxlen=self.config.curriculum_window)

    def record(self, success: bool, env: BridgeEnv) -> bool:
        if not self.config.curriculum_enabled:
            return False
        self.results.append(success)
        if (
            len(self.results) == self.results.maxlen
            and np.mean(self.results) >= self.config.curriculum_success_threshold
            and env.config.curriculum_stage < self.config.curriculum_max_stage
        ):
            env.config.curriculum_stage += 1
            self.results.clear()
            return True
        return False


class Trainer:
    def __init__(self, config: ExperimentConfig, device: str | torch.device = "cpu"):
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(config.environment.seed)
        self.env = BridgeEnv(config.environment)
        self.actor = SharedRecurrentActor(self.env.layout, self.env.action_count, config.model)
        if config.training.critic == "graph":
            self.critic: CentralizedCritic | GraphCritic = GraphCritic(node_size=self.env.graph_node_size)
        elif config.training.critic == "mlp":
            self.critic = CentralizedCritic(self.env.global_state_size)
        else:
            raise ValueError(f"Unknown critic: {config.training.critic}")
        self.algorithm = MAPPO(self.actor, self.critic, config.training, self.device)
        self.logger = EpisodeLogger(config.training.output_dir)
        self.curriculum = CurriculumScheduler(config)
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, updates: int | None = None) -> list[dict[str, float]]:
        training = self.config.training
        update_count = updates if updates is not None else training.total_updates
        observation, _ = self.env.reset()
        hidden = self.actor.initial_hidden(self.env.config.robot.count, self.device)
        episode_start = np.ones(self.env.config.robot.count, dtype=np.float32)
        policy_statistics = PolicyStatistics(self.env.action_count)
        history: list[dict[str, float]] = []
        episode_number = 0
        last_done = False
        for update in range(1, update_count + 1):
            buffer = RolloutBuffer(
                training.rollout_steps,
                self.env.config.robot.count,
                self.env.observation_size,
                self.env.global_state_size,
                self.env.action_count,
                self.actor.hidden_size,
                use_graph=isinstance(self.critic, GraphCritic),
                graph_node_size=self.env.graph_node_size,
            )
            buffer.set_initial_hidden(hidden.detach().cpu().numpy())
            for _ in range(training.rollout_steps):
                observation_tensor = torch.as_tensor(observation, device=self.device)
                action_masks = self.env.action_masks()
                mask_tensor = torch.as_tensor(action_masks, device=self.device)
                reset_tensor = torch.as_tensor(episode_start, device=self.device)
                global_state = self.env.global_state()
                graph_state = self.env.dense_graph_state() if isinstance(self.critic, GraphCritic) else None
                with torch.no_grad():
                    actions, log_probabilities, _, output = self.actor.act(
                        observation_tensor, hidden, mask_tensor, reset_mask=reset_tensor
                    )
                    hidden = output.hidden
                    if graph_state is None:
                        value = self.algorithm.value(torch.as_tensor(global_state, device=self.device))
                    else:
                        graph_tensors = tuple(torch.as_tensor(item, device=self.device) for item in graph_state)
                        value = self.algorithm.value(torch.as_tensor(global_state, device=self.device), graph_tensors)
                action_values = actions.cpu().numpy()
                policy_statistics.add(action_values, output.latents)
                next_observation, reward, terminated, truncated, info = self.env.step(action_values)
                done = terminated or truncated
                buffer.add(
                    observation,
                    global_state,
                    action_values,
                    log_probabilities.cpu().numpy(),
                    action_masks,
                    episode_start,
                    reward,
                    float(value.cpu()),
                    done,
                    info["auxiliary_targets"],
                    graph_state,
                )
                observation = next_observation
                last_done = done
                episode_start = np.zeros(self.env.config.robot.count, dtype=np.float32)
                if done:
                    episode_number += 1
                    episode_log: dict[str, Any] = dict(info)
                    episode_log.update(policy_statistics.summarize())
                    episode_log.update(episode=episode_number, update=update, curriculum_stage=self.env.config.curriculum_stage)
                    advanced = self.curriculum.record(bool(info["success"]), self.env)
                    episode_log["curriculum_advanced"] = advanced
                    self.logger.log(episode_log)
                    observation, _ = self.env.reset()
                    hidden = self.actor.initial_hidden(self.env.config.robot.count, self.device)
                    episode_start = np.ones(self.env.config.robot.count, dtype=np.float32)
                    policy_statistics = PolicyStatistics(self.env.action_count)
            with torch.no_grad():
                if last_done:
                    last_value = 0.0
                elif isinstance(self.critic, GraphCritic):
                    graph_tensors = tuple(
                        torch.as_tensor(item, device=self.device) for item in self.env.dense_graph_state()
                    )
                    last_value = float(
                        self.algorithm.value(torch.as_tensor(self.env.global_state(), device=self.device), graph_tensors).cpu()
                    )
                else:
                    last_value = float(
                        self.algorithm.value(torch.as_tensor(self.env.global_state(), device=self.device)).cpu()
                    )
            buffer.compute_returns(last_value, training.gamma, training.gae_lambda)
            metrics = self.algorithm.update(buffer.as_tensors(self.device))
            metrics["update"] = float(update)
            metrics["curriculum_stage"] = float(self.env.config.curriculum_stage)
            history.append(metrics)
            print(
                f"Update {update}/{update_count} "
                f"actor_loss={metrics['actor_loss']:.4f} "
                f"critic_loss={metrics['critic_loss']:.4f} "
                f"entropy={metrics['entropy']:.4f}",
                flush=True,
            )
            if update % training.checkpoint_interval == 0 or update == update_count:
                self.save_checkpoint(update)
        return history

    def save_checkpoint(self, update: int) -> Path:
        path = self.output_dir / f"checkpoint_{update:06d}.pt"
        torch.save(
            {
                "algorithm": self.algorithm.state_dict(),
                "experiment_config": self.config.to_dict(),
                "update": update,
            },
            path,
        )
        return path
