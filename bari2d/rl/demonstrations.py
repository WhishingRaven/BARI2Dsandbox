"""Demonstrations from the scripted teacher and the behaviour-cloning loss.

Demonstration-guided RL (reverse curriculum + behaviour cloning):

1. Start states come from the start-state curriculum (a gripping root, a
   walker on it, ... down to no preparation) so the teacher shows every
   construction step, including the late ones that exploration rarely reaches.
2. The teacher's action labels every robot at every step.  With probability
   ``epsilon`` the executed action is replaced by a random valid one, so the
   data covers a "tube" around the teacher's path and how to recover from it.
3. Observations are always produced by the simulator (never synthesized), so
   sensor histories and the recurrent state stay physically consistent.

The policy is pretrained with L_BC = -E[w * log pi(a_teacher | o, h)] and the
same term is kept, with an annealed weight, during PPO fine-tuning.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.robot import DiscreteAction
from bari2d.rl.rollout_buffer import AUXILIARY_TARGETS
from bari2d.rl.teacher import ScriptedTeacher
from bari2d.utils.config import EnvironmentConfig

IDLE = int(DiscreteAction.IDLE)
PROTECTED_ACTIONS = (int(DiscreteAction.ANCHOR), int(DiscreteAction.RELEASE))


@dataclass
class DemoEpisode:
    observations: np.ndarray  # (T, N, D)
    actions: np.ndarray  # (T, N) teacher labels
    action_masks: np.ndarray  # (T, N, A)
    active: np.ndarray  # (T, N) the robot the teacher was moving
    auxiliary_targets: dict[str, np.ndarray]  # (T, N)
    stage: int
    success: bool
    ignore: np.ndarray | None = None  # (T,) steps without loss weight (window burn-in)

    def __len__(self) -> int:
        return int(self.actions.shape[0])


def record_episode(
    config: EnvironmentConfig, stage: int, seed: int, epsilon: float = 0.1, idle_epsilon_scale: float = 0.25
) -> DemoEpisode:
    """Run the teacher once from the start state of ``stage``."""
    environment_config = deepcopy(config)
    environment_config.curriculum_stage = int(stage)
    environment_config.seed = int(seed)
    env = BridgeEnv(environment_config)
    observation, _ = env.reset(seed=seed)
    teacher = ScriptedTeacher(env)
    rng = np.random.default_rng(seed + 7919)
    observations, labels, masks, active = [], [], [], []
    auxiliary = {name: [] for name in AUXILIARY_TARGETS}
    info: dict = {"success": False}
    for _ in range(environment_config.max_steps):
        action_masks = env.action_masks()
        teacher_actions, worker = teacher.actions()
        executed = teacher_actions.copy()
        for robot_id in range(len(executed)):
            # The moving robot is perturbed with probability epsilon, waiting robots
            # with epsilon * idle_epsilon_scale (default a quarter).  Only motion, CLIMB and IDLE are injected:
            # a random ANCHOR/RELEASE would fix or undo structure rather than show a
            # recoverable deviation.
            rate = epsilon if robot_id == worker else epsilon * idle_epsilon_scale
            if rng.random() < rate:
                choices = [a for a in np.flatnonzero(action_masks[robot_id]) if a not in PROTECTED_ACTIONS]
                if choices:
                    executed[robot_id] = int(rng.choice(choices))
        observations.append(observation)
        labels.append(teacher_actions)
        masks.append(action_masks)
        flags = np.zeros(len(executed), dtype=bool)
        if worker is not None:
            flags[worker] = True
        active.append(flags)
        observation, _, terminated, truncated, info = env.step(executed)
        for name in AUXILIARY_TARGETS:
            auxiliary[name].append(np.asarray(info["auxiliary_targets"][name], dtype=np.float32))
        if terminated or truncated:
            break
    return DemoEpisode(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(labels, dtype=np.int64),
        action_masks=np.asarray(masks, dtype=bool),
        active=np.asarray(active, dtype=bool),
        auxiliary_targets={name: np.asarray(values, dtype=np.float32) for name, values in auxiliary.items()},
        stage=int(stage),
        success=bool(info["success"]),
    )


def _record_task(arguments: tuple[EnvironmentConfig, int, int, float, float]) -> DemoEpisode:
    return record_episode(*arguments)


def generate_demonstrations(
    config: EnvironmentConfig,
    stages: list[int],
    episodes: int,
    epsilon: float = 0.1,
    seed: int = 0,
    workers: int = 1,
    progress: bool = False,
    idle_epsilon_scale: float = 0.25,
) -> list[DemoEpisode]:
    """``episodes`` teacher episodes, stage ``stages[i % len(stages)]`` for episode i (parallel when workers > 1)."""
    tasks = [(config, stages[index % len(stages)], seed + index, epsilon, idle_epsilon_scale) for index in range(episodes)]
    report_every = max(len(tasks) // 10, 1)
    results: list[DemoEpisode] = []

    def collect(episode: DemoEpisode) -> None:
        results.append(episode)
        if progress and (len(results) % report_every == 0 or len(results) == len(tasks)):
            successes = sum(item.success for item in results)
            print(f"  {len(results)}/{len(tasks)} teacher episodes done ({successes} successful)", flush=True)

    if workers <= 1:
        for task in tasks:
            collect(_record_task(task))
        return results
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(workers) as pool:
        for episode in pool.imap(_record_task, tasks):
            collect(episode)
    return results


class DemonstrationSet:
    def __init__(self, episodes: list[DemoEpisode]):
        self.episodes = episodes
        self._rng = random.Random(0)

    def __len__(self) -> int:
        return len(self.episodes)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"episodes": [episode.__dict__ for episode in self.episodes]}, path)

    @classmethod
    def load(cls, path: str | Path) -> "DemonstrationSet":
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls([DemoEpisode(**episode) for episode in data["episodes"]])

    def summary(self) -> dict[int, tuple[int, int]]:
        """stage -> (episodes, teacher successes)."""
        result: dict[int, tuple[int, int]] = {}
        for episode in self.episodes:
            count, successes = result.get(episode.stage, (0, 0))
            result[episode.stage] = (count + 1, successes + int(episode.success))
        return dict(sorted(result.items()))

    def batch(
        self,
        episodes: list[DemoEpisode],
        idle_weight: float,
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        """Pad episodes into (T_max, episodes * N, ...) sequences with per-step loss weights."""
        steps = max(len(episode) for episode in episodes)
        robots = episodes[0].actions.shape[1]
        size = episodes[0].observations.shape[2]
        action_count = episodes[0].action_masks.shape[2]
        columns = len(episodes) * robots
        observations = np.zeros((steps, columns, size), dtype=np.float32)
        actions = np.full((steps, columns), IDLE, dtype=np.int64)
        masks = np.zeros((steps, columns, action_count), dtype=bool)
        masks[..., IDLE] = True
        weights = np.zeros((steps, columns), dtype=np.float32)
        auxiliary = {name: np.zeros((steps, columns), dtype=np.float32) for name in AUXILIARY_TARGETS}
        for index, episode in enumerate(episodes):
            length, span = len(episode), slice(index * robots, (index + 1) * robots)
            observations[:length, span] = episode.observations
            actions[:length, span] = episode.actions
            masks[:length, span] = episode.action_masks
            weights[:length, span] = np.where(episode.active | (episode.actions != IDLE), 1.0, idle_weight)
            if episode.ignore is not None:
                weights[:length, span][episode.ignore] = 0.0
            for name in AUXILIARY_TARGETS:
                auxiliary[name][:length, span] = episode.auxiliary_targets[name]
        return {
            "observations": torch.as_tensor(observations, device=device),
            "actions": torch.as_tensor(actions, device=device),
            "action_masks": torch.as_tensor(masks, device=device),
            "weights": torch.as_tensor(weights, device=device),
            "auxiliary_targets": {name: torch.as_tensor(values, device=device) for name, values in auxiliary.items()},
        }

    def sample(self, count: int) -> list[DemoEpisode]:
        return [self._rng.choice(self.episodes) for _ in range(count)]

    @staticmethod
    def losses(actor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Weighted cross-entropy on teacher actions, auxiliary MSE on valid steps, and accuracy stats."""
        observations = batch["observations"]
        hidden = actor.initial_hidden(observations.shape[1], observations.device)
        logits, predictions = actor.sequence_logits(observations, hidden, batch["action_masks"], None)
        log_probabilities = torch.log_softmax(logits, dim=-1).gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)
        weights = batch["weights"]
        bc_loss = -(weights * log_probabilities).sum() / weights.sum().clamp_min(1.0e-8)
        valid = (weights > 0).float()
        auxiliary_loss = torch.zeros((), device=observations.device)
        for name, prediction in predictions.items():
            error = (prediction - batch["auxiliary_targets"][name]) ** 2
            auxiliary_loss = auxiliary_loss + (error * valid).sum() / valid.sum().clamp_min(1.0)
        with torch.no_grad():
            correct = (logits.argmax(dim=-1) == batch["actions"]).float()
            moving = valid * (batch["actions"] != IDLE).float()
            stats = {
                "accuracy": float((correct * valid).sum() / valid.sum().clamp_min(1.0)),
                "move_accuracy": float((correct * moving).sum() / moving.sum().clamp_min(1.0)),
            }
        return bc_loss, auxiliary_loss, stats

    def sample_windows(self, count: int, window: int, burn_in: int) -> list[DemoEpisode]:
        """``count`` random stretches of at most ``window`` steps (burn-in steps carry no loss weight)."""
        windows = []
        for episode in self.sample(count):
            length = len(episode)
            if length <= window:
                windows.append(episode)
                continue
            start = self._rng.randrange(0, length - window + 1)
            cut = slice(start, start + window)
            active = episode.active[cut].copy()
            labels = episode.actions[cut].copy()
            ignore = np.zeros(window, dtype=bool)
            ignore[: burn_in if start > 0 else 0] = True
            windows.append(
                DemoEpisode(
                    observations=episode.observations[cut],
                    actions=labels,
                    action_masks=episode.action_masks[cut],
                    active=active,
                    auxiliary_targets={name: values[cut] for name, values in episode.auxiliary_targets.items()},
                    stage=episode.stage,
                    success=episode.success,
                    ignore=ignore,
                )
            )
        return windows

    def behaviour_cloning_loss(
        self,
        actor,
        batch_episodes: int,
        idle_weight: float,
        device: torch.device | str,
        window: int = 0,
    ) -> torch.Tensor:
        """BC term used during PPO: whole episodes, or random windows of ``window`` steps.

        A window starting mid-episode begins from a zero recurrent state, so its
        first quarter (at most 32 steps) only warms the GRU up and carries no loss.
        """
        if window > 0:
            episodes = self.sample_windows(batch_episodes, window, min(32, window // 4))
        else:
            episodes = self.sample(batch_episodes)
        bc_loss, _, _ = self.losses(actor, self.batch(episodes, idle_weight, device))
        return bc_loss
