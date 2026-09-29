"""Parallel rollouts: several BridgeEnv instances stepped together.

``SyncVectorEnv`` steps the environments one after another in this process
(more diverse data per update, same wall-clock cost per step).
``SubprocVectorEnv`` runs one worker process per environment so the physics
(sensors, equilibrium LPs) of all environments runs in parallel.  Workers use
the ``spawn`` start method, which is what Windows supports, so scripts that
train with it must keep the ``if __name__ == "__main__":`` guard.

Both backends auto-reset an environment when its episode ends and return, for
that step, the final (pre-reset) critic state and the full episode info.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from copy import deepcopy
from typing import Any

import numpy as np

from bari2d.env.bridge_env import BridgeEnv
from bari2d.utils.config import EnvironmentConfig

VECTOR_BACKENDS = ("sync", "subprocess")


class EnvRunner:
    """One environment with auto-reset; returns only what training needs each step."""

    def __init__(self, config: EnvironmentConfig, use_graph: bool):
        self.env = BridgeEnv(config)
        self.use_graph = use_graph
        self.episode_stage = self.env.config.curriculum_stage

    def _state(self, observation: np.ndarray) -> dict[str, Any]:
        return {
            "observation": observation,
            "action_masks": self.env.action_masks(),
            "global_state": self.env.global_state(),
            "graph_state": self.env.dense_graph_state() if self.use_graph else None,
        }

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        observation, _ = self.env.reset(seed=seed)
        self.episode_stage = self.env.config.curriculum_stage
        return self._state(observation)

    def step(self, actions: np.ndarray) -> dict[str, Any]:
        observation, reward, terminated, truncated, info = self.env.step(actions)
        result: dict[str, Any] = {
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": bool(info["success"]),
            "reward_components": info.get("reward_components", {}),
            "auxiliary_targets": info["auxiliary_targets"],
        }
        if terminated or truncated:
            result["final_global_state"] = self.env.global_state()
            result["final_graph_state"] = self.env.dense_graph_state() if self.use_graph else None
            info["curriculum_stage"] = self.episode_stage
            result["episode_info"] = info
            observation, _ = self.env.reset()
            self.episode_stage = self.env.config.curriculum_stage
        result.update(self._state(observation))
        return result

    def set_stage(self, stage: int) -> None:
        """Takes effect at the next reset; the running episode keeps its stage."""
        self.env.config.curriculum_stage = int(stage)


class SyncVectorEnv:
    def __init__(self, configs: list[EnvironmentConfig], use_graph: bool):
        self.runners = [EnvRunner(config, use_graph) for config in configs]

    @property
    def num_envs(self) -> int:
        return len(self.runners)

    def reset(self) -> list[dict[str, Any]]:
        return [runner.reset() for runner in self.runners]

    def step(self, actions: np.ndarray) -> list[dict[str, Any]]:
        return [runner.step(action) for runner, action in zip(self.runners, actions)]

    def set_stage(self, stage: int) -> None:
        for runner in self.runners:
            runner.set_stage(stage)

    def close(self) -> None:
        self.runners = []


def _worker(remote, parent_remote, config: EnvironmentConfig, use_graph: bool) -> None:
    parent_remote.close()
    try:
        runner = EnvRunner(config, use_graph)
    except Exception:  # pragma: no cover - reported to the parent
        remote.send((False, traceback.format_exc()))
        return
    remote.send((True, None))
    while True:
        try:
            command, payload = remote.recv()
        except (EOFError, KeyboardInterrupt):
            break
        try:
            if command == "step":
                remote.send((True, runner.step(payload)))
            elif command == "reset":
                remote.send((True, runner.reset(payload)))
            elif command == "set_stage":
                runner.set_stage(payload)
                remote.send((True, None))
            elif command == "close":
                remote.send((True, None))
                break
            else:
                remote.send((False, f"Unknown command {command!r}"))
        except Exception:
            remote.send((False, traceback.format_exc()))
    remote.close()


class SubprocVectorEnv:
    def __init__(self, configs: list[EnvironmentConfig], use_graph: bool):
        context = mp.get_context("spawn")
        self.remotes = []
        self.processes = []
        for config in configs:
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, parent, config, use_graph), daemon=True)
            process.start()
            child.close()
            self.remotes.append(parent)
            self.processes.append(process)
        for remote in self.remotes:
            self._receive(remote)
        self.closed = False

    @property
    def num_envs(self) -> int:
        return len(self.remotes)

    @staticmethod
    def _receive(remote) -> Any:
        ok, value = remote.recv()
        if not ok:
            raise RuntimeError(f"Environment worker failed:\n{value}")
        return value

    def _broadcast(self, command: str, payloads: list[Any]) -> list[Any]:
        for remote, payload in zip(self.remotes, payloads):
            remote.send((command, payload))
        return [self._receive(remote) for remote in self.remotes]

    def reset(self) -> list[dict[str, Any]]:
        return self._broadcast("reset", [None] * self.num_envs)

    def step(self, actions: np.ndarray) -> list[dict[str, Any]]:
        return self._broadcast("step", list(actions))

    def set_stage(self, stage: int) -> None:
        self._broadcast("set_stage", [int(stage)] * self.num_envs)

    def close(self) -> None:
        if getattr(self, "closed", True):
            return
        self.closed = True
        for remote in self.remotes:
            try:
                remote.send(("close", None))
                remote.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def environment_configs(config: EnvironmentConfig, num_envs: int) -> list[EnvironmentConfig]:
    """Per-environment configs: environment 0 keeps the base seed, the others are offset."""
    configs = []
    for index in range(num_envs):
        copy = deepcopy(config)
        copy.seed = config.seed + 1000 * index
        configs.append(copy)
    return configs


def make_vector_env(config: EnvironmentConfig, num_envs: int, backend: str, use_graph: bool):
    if num_envs < 1:
        raise ValueError("num_envs must be at least 1")
    if backend not in VECTOR_BACKENDS:
        raise ValueError(f"Unknown vector_env backend: {backend} (choose from {VECTOR_BACKENDS})")
    configs = environment_configs(config, num_envs)
    if backend == "subprocess" and num_envs > 1:
        return SubprocVectorEnv(configs, use_graph)
    return SyncVectorEnv(configs, use_graph)
