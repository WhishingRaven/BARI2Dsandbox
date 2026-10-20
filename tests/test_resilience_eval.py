"""scripts/evaluate_resilience.py: the four injected faults do what they say."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.robot import DiscreteAction
from bari2d.utils.config import load_config

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_resilience.py"
spec = importlib.util.spec_from_file_location("evaluate_resilience", SCRIPT)
resilience = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resilience)  # type: ignore[union-attr]

V14 = "configs/beacon_structured_mappo_static_v14.yaml"


def arguments(**overrides) -> argparse.Namespace:
    values = dict(fault_window=[5, 5], fraction=[0.10, 0.20], delay=[50, 100], power_mode="hold")
    values.update(overrides)
    return argparse.Namespace(**values)


def env_at(stage: int = 11, seed: int = 0) -> BridgeEnv:
    config = load_config(V14).environment
    config.curriculum_stage = stage
    env = BridgeEnv(config)
    env.reset(seed=seed)
    return env


def run(env: BridgeEnv, injector, steps: int, rng: np.random.Generator | None):
    """Step with random valid actions (or all IDLE when rng is None)."""
    observation = env.observations()
    seen = []
    for _ in range(steps):
        injector.before_step()
        seen.append(injector.observations(observation))
        masks = env.action_masks()
        if rng is None:
            actions = np.full(len(env.robots), int(DiscreteAction.IDLE))
        else:
            actions = np.array([rng.choice(np.flatnonzero(row)) for row in masks])
        actions = injector.actions(actions)
        observation, *_ = env.step(actions)
    return seen


def test_power_cut_stops_10_to_20_percent_of_the_team() -> None:
    env = env_at()
    injector = resilience.FaultInjector(env, "power", np.random.default_rng(1), arguments(), (5, 5))
    run(env, injector, 6, np.random.default_rng(0))
    assert injector.injected_at == 5
    team = sum(not robot.absent for robot in env.robots)
    assert 0.10 * team - 1 <= len(injector.affected) <= 0.20 * team + 1
    positions = {r: env.robots[r].position.copy() for r in injector.affected}
    actions = injector.actions(np.full(len(env.robots), int(DiscreteAction.FORWARD)))
    assert all(actions[r] == int(DiscreteAction.IDLE) for r in injector.affected)
    run(env, injector, 10, np.random.default_rng(2))
    for robot_id, position in positions.items():
        robot = env.robots[robot_id]
        if not robot.fallen:
            assert np.linalg.norm(robot.position - position) < 0.2  # only pushed, never driven


def test_comms_delay_feeds_an_old_beacon_signal() -> None:
    env = env_at(stage=10, seed=2)
    injector = resilience.FaultInjector(env, "comms", np.random.default_rng(3), arguments(fault_window=[60, 60]), (60, 60))
    seen = run(env, injector, 130, np.random.default_rng(0))
    beacon = env.layout.beacon
    delay = injector.delay
    assert 50 <= delay <= 100 and injector.injected_at == 60
    true_history = injector.beacon_history  # the last delay + 1 true signals
    assert np.allclose(seen[-1][injector.affected, beacon], true_history[0][injector.affected])


def test_sensor_failure_zeroes_the_imu_like_channels() -> None:
    env = env_at()
    injector = resilience.FaultInjector(env, "sensor", np.random.default_rng(4), arguments(), (5, 5))
    seen = run(env, injector, 8, np.random.default_rng(0))
    assert injector.injected_at == 5
    internal = env.layout.internal
    for channel in resilience.IMU_CHANNELS:
        assert np.all(seen[-1][injector.affected, internal.start + channel] == 0.0)


def test_decouple_releases_a_joint_of_the_prepared_row() -> None:
    env = env_at()
    row_joints = {robot.robot_id: robot.anchor_partner for robot in env.robots[1:4]}
    injector = resilience.FaultInjector(env, "decouple", np.random.default_rng(5), arguments(), (3, 3))
    run(env, injector, 4, None)
    assert injector.injected_at == 3 and len(injector.affected) == 1
    owner = injector.affected[0]
    assert owner in row_joints and f"released robot {owner} from {row_joints[owner]}" == injector.detail
    released = env.robots[owner]
    assert not (released.anchored and released.anchor_partner == row_joints[owner])


def test_wilson_interval() -> None:
    low, high = resilience.wilson(30, 50)
    assert low == pytest.approx(0.462, abs=0.01) and high == pytest.approx(0.724, abs=0.01)
