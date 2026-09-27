"""Reward shaping (frontier progress) and GAE truncation bootstrapping."""

from __future__ import annotations

import numpy as np
import pytest

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.robot import DiscreteAction, RobotState
from bari2d.rl import trainer as trainer_module
from bari2d.rl.trainer import Trainer
from bari2d.utils.config import EnvironmentConfig, load_config

IDLE = int(DiscreteAction.IDLE)


def frontier_env(count: int = 1, measure: str = "frontier") -> BridgeEnv:
    config = EnvironmentConfig()
    config.robot.count = count
    config.physics.model = "static"
    config.gap.curriculum_widths = [1.2]
    config.load.curriculum_targets = [1.0]
    config.reward.progress_measure = measure
    env = BridgeEnv(config)
    env.reset(seed=3)
    return env


def place(env: BridgeEnv, robot_id: int, longitudinal: float, layer: int = 0) -> RobotState:
    left, _ = env.field.boundaries(0.0)
    state = RobotState(robot_id, env.field.center + env.field.normal * (left + longitudinal), env.field.orientation, layer=layer)
    state.latent = np.zeros(env.config.latent_dim, dtype=np.float32)
    return state


def test_frontier_progress_counts_a_nose_over_the_edge() -> None:
    frontier = frontier_env(measure="frontier")
    frontier.set_robot_states([place(frontier, 0, -0.3)])
    # Front end 0.15 past the edge of a 1.2 gap.
    assert frontier.current_progress == pytest.approx(0.15 / 1.2, abs=1e-6)

    center = frontier_env(measure="center")
    center.set_robot_states([place(center, 0, -0.3)])
    assert center.current_progress == 0.0


def test_frontier_progress_ignores_robots_that_fall() -> None:
    env = frontier_env()
    env.set_robot_states([place(env, 0, 0.1)])
    info = env.step(np.array([IDLE]))[4]
    assert env.robots[0].fallen
    assert info["span_progress"] == 0.0


def test_frontier_progress_grows_when_walking_out_on_a_gripped_robot() -> None:
    env = frontier_env(count=2)
    # The walker's front already reaches past the gripped root's front end.
    env.set_robot_states([place(env, 0, -0.03), place(env, 1, 0.0, layer=1)])
    env.contact_model._attach(env.robots[0], "left_bank", env.robots)  # type: ignore[attr-defined]
    before = env.step(np.array([IDLE, IDLE]))[4]["span_progress"]
    after = env.step(np.array([IDLE, int(DiscreteAction.FORWARD)]))[4]
    assert not env.robots[1].fallen and env.robots[1].layer == 1
    assert after["span_progress"] > before
    assert after["reward_components"]["span"] > 0.0


def test_unknown_progress_measure_is_rejected() -> None:
    with pytest.raises(ValueError):
        frontier_env(measure="median")


@pytest.mark.parametrize("bootstrap", [True, False])
def test_time_limit_truncation_is_bootstrapped(monkeypatch: pytest.MonkeyPatch, tmp_path, bootstrap: bool) -> None:
    config = load_config("configs/beacon_structured_mappo_static_v2.yaml")
    config.environment.robot.count = 3
    config.environment.max_steps = 4
    config.training.rollout_steps = 8
    config.training.total_updates = 1
    config.training.output_dir = str(tmp_path / "run")
    config.training.bootstrap_truncation = bootstrap
    stored: list[tuple[float, bool]] = []
    original_add = trainer_module.RolloutBuffer.add

    def capture(self, observation, global_state, actions, log_probabilities, action_masks, reset_mask, reward, value, done, *rest):
        stored.append((float(reward), bool(done)))
        return original_add(self, observation, global_state, actions, log_probabilities, action_masks, reset_mask, reward, value, done, *rest)

    monkeypatch.setattr(trainer_module.RolloutBuffer, "add", capture)
    monkeypatch.setattr(Trainer, "_state_value", lambda self: 10.0)
    trainer = Trainer(config, "cpu")
    trainer.train()
    episodes = [line for line in (tmp_path / "run" / "episodes.jsonl").read_text().splitlines() if line]
    assert len(episodes) == 2
    import json

    returns = [json.loads(line)["episode_return"] for line in episodes]
    stored_returns = [sum(reward for reward, _ in stored[:4]), sum(reward for reward, _ in stored[4:])]
    expected_bonus = config.training.gamma * 10.0 if bootstrap else 0.0
    for logged, buffered in zip(returns, stored_returns):
        assert buffered == pytest.approx(logged + expected_bonus, abs=1e-5)
