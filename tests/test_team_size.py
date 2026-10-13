"""v11: more modules and a random team size per episode (robot.active_min)."""

from __future__ import annotations

import numpy as np
import pytest

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.robot import DiscreteAction
from bari2d.rl.trainer import Trainer
from bari2d.utils.config import load_config

V10 = "configs/beacon_structured_mappo_static_v10.yaml"
V11 = "configs/beacon_structured_mappo_static_v11.yaml"


def v11_env(stage: int, seed: int) -> BridgeEnv:
    config = load_config(V11).environment
    config.curriculum_stage = stage
    env = BridgeEnv(config)
    env.reset(seed=seed)
    return env


def test_v11_config() -> None:
    config = load_config(V11)
    environment = config.environment
    assert environment.robot.count == 16 and environment.robot.active_min == 12
    assert environment.spawn_edge_distance == [4.0, 4.0, 4.0, 4.0, 5.5, 5.5, 5.5, 7.5, 0.0]
    assert environment.reward.collapse_penalty == 0.5
    assert config.training.output_dir.endswith("_v11")
    assert config.training.demo_path == load_config(V10).training.demo_path


def test_each_episode_draws_a_team_of_12_to_16() -> None:
    sizes = set()
    for seed in range(20):
        env = v11_env(stage=7, seed=seed)
        info = env.info()
        absent = [robot for robot in env.robots if robot.absent]
        assert 12 <= info["active_robots"] <= 16
        assert info["active_robots"] == 16 - len(absent)
        assert info["fallen_robots"] == 0  # absent robots are not counted as falls
        assert all(robot.fallen for robot in absent)
        masks = env.action_masks()
        for robot in absent:
            assert masks[robot.robot_id].tolist() == [a == DiscreteAction.IDLE for a in DiscreteAction]
        sizes.add(info["active_robots"])
    assert len(sizes) >= 3


def test_absent_robots_stay_out_and_cost_nothing() -> None:
    env = v11_env(stage=7, seed=3)
    absent = [robot for robot in env.robots if robot.absent]
    assert absent
    before = [robot.position.copy() for robot in absent]
    rng = np.random.default_rng(0)
    for _ in range(20):
        masks = env.action_masks()
        actions = np.array([rng.choice(np.flatnonzero(row)) for row in masks])
        _, _, _, _, info = env.step(actions)
        assert info["active_robots"] == 16 - len(absent)
    for robot, position in zip(absent, before):
        assert np.allclose(robot.position, position) and not robot.anchored
    assert info["fallen_robots"] == sum(r.fallen and not r.absent for r in env.robots)


def test_prepared_robots_always_take_part() -> None:
    for seed in range(10):
        env = v11_env(stage=6, seed=seed)  # row_spanning: robots 0-2 form the row
        assert not any(robot.absent for robot in env.robots[:3])
        assert env._structure_spans()


def test_default_keeps_every_robot() -> None:
    config = load_config(V10).environment
    config.curriculum_stage = 7
    env = BridgeEnv(config)
    env.reset(seed=0)
    assert env.info()["active_robots"] == 10 and not any(robot.absent for robot in env.robots)
    config.robot.active_min = 11
    with pytest.raises(ValueError):
        BridgeEnv(config)


def _small(path: str, output_dir: str):
    config = load_config(path)
    config.environment.max_steps = 16
    config.training.num_envs = 1
    config.training.vector_env = "sync"
    config.training.rollout_steps = 8
    config.training.recurrent_chunk = 8
    config.training.num_minibatches = 1
    config.training.ppo_epochs = 1
    config.training.demo_path = ""
    config.training.checkpoint_interval = 1000
    config.training.output_dir = output_dir
    return config


def test_v11_resumes_a_10_robot_v10_checkpoint(tmp_path) -> None:
    v10 = Trainer(_small(V10, str(tmp_path / "v10")))
    v10.curriculum.stage = 7
    path = v10.save_checkpoint(3000)

    v11 = Trainer(_small(V11, str(tmp_path / "v11")))
    assert v11.env.config.robot.count == 16
    assert v11.load_checkpoint(path) == 3000
    history = v11.train(updates=3001)
    assert len(history) == 1 and history[-1]["curriculum_stage"] == 7.0
