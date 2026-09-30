"""Design-matched bank grip (v6): the belly IR sits ahead of the centre and a
ground-level robot grips the bank only when that sensor sees no floor."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.sensors import _downward_ir_distance, downward_ir_point, front_over_cliff
from bari2d.rl.demonstrations import generate_demonstrations, record_episode
from bari2d.rl.trainer import Trainer
from bari2d.utils.config import load_config

V5 = "configs/beacon_structured_mappo_static_v5.yaml"
V6 = "configs/beacon_structured_mappo_static_v6.yaml"


def make_env(path: str, stage: int = 4, seed: int = 1) -> BridgeEnv:
    config = load_config(path).environment
    config.curriculum_stage = stage
    env = BridgeEnv(config)
    env.reset(seed=seed)
    return env


def place(env: BridgeEnv, behind_edge: float, theta: float, robot_id: int = 0) -> None:
    """Put a free ground robot with its centre ``behind_edge`` before the left cliff edge."""
    robot = env.robots[robot_id]
    left, _ = env.field.boundaries(0.0)
    robot.position = env.field.center + env.field.normal * (left - behind_edge)
    robot.theta = theta
    robot.layer = 0
    robot.anchored = False
    robot.anchor_partner = None
    robot.fallen = False


def test_v6_config_matches_the_design() -> None:
    config = load_config(V6)
    assert config.environment.sensor.downward_ir_offset == pytest.approx(0.45 - 0.9 * 4.0 / 15.0)
    assert config.environment.contact.anchor_rule == "front_downward_ir"
    assert config.training.bc_coef_final > 0.0
    assert config.training.demo_path != load_config(V5).training.demo_path
    assert config.training.output_dir.endswith("_v6")
    trainer = SimpleNamespace(config=config, demonstrations=object())
    assert Trainer.bc_coefficient(trainer, 1) == pytest.approx(config.training.bc_coef)
    assert Trainer.bc_coefficient(trainer, 1000) == pytest.approx(config.training.bc_coef_final)


@pytest.mark.parametrize(
    ("behind_edge", "theta", "expected"),
    [
        (0.10, 0.0, True),  # front feet over the lip, facing the gap
        (0.20, 0.0, True),
        (0.35, 0.0, False),  # front still on the bank
        (0.60, 0.0, False),  # the old rule allowed this
        (-0.05, 0.0, False),  # centre already past the edge
        (0.10, np.pi / 2.0, False),  # sideways
        (0.10, np.pi, False),  # backwards
    ],
)
def test_bank_grip_needs_the_front_belly_ir_over_the_cliff(behind_edge: float, theta: float, expected: bool) -> None:
    env = make_env(V6)
    place(env, behind_edge, theta)
    assert env._anchor_action_available(env.robots[0]) is expected


def test_old_edge_band_rule_is_unchanged_by_default() -> None:
    env = make_env(V5)
    assert env.config.contact.anchor_rule == "edge_band"
    place(env, 0.60, 0.0)
    assert env._anchor_action_available(env.robots[0])  # 0.6 behind the edge
    place(env, 0.10, np.pi / 2.0)
    assert env._anchor_action_available(env.robots[0])  # sideways


def test_belly_ir_reading_follows_the_sensor_position() -> None:
    env = make_env(V6)
    robot = env.robots[0]
    place(env, 0.10, 0.0)
    sensor = env.config.sensor
    assert np.allclose(downward_ir_point(robot, sensor), robot.position + robot.heading * 0.21)
    assert _downward_ir_distance(robot, env.robots, env.field, env.config.robot, sensor) == 1.0  # no floor
    assert front_over_cliff(robot, env.robots, env.field, env.config.robot, sensor)
    place(env, 0.35, 0.0)
    assert _downward_ir_distance(robot, env.robots, env.field, env.config.robot, sensor) == 0.0  # ground

    centre = make_env(V5)
    place(centre, 0.10, 0.0)
    reading = _downward_ir_distance(centre.robots[0], centre.robots, centre.field, centre.config.robot, centre.config.sensor)
    assert reading == 0.0  # the centre sensor still sees the bank


def test_raised_robots_do_not_grip_the_bank() -> None:
    env = make_env(V6)
    place(env, 0.10, 0.0)
    env.robots[0].layer = 1
    assert not front_over_cliff(env.robots[0], env.robots, env.field, env.config.robot, env.config.sensor)


def test_front_rule_requires_a_forward_belly_sensor() -> None:
    config = load_config(V6).environment
    config.sensor.downward_ir_offset = 0.0
    with pytest.raises(ValueError):
        BridgeEnv(config)
    config = load_config(V6).environment
    config.contact.anchor_rule = "anywhere"
    with pytest.raises(ValueError):
        BridgeEnv(config)


def test_teacher_still_builds_the_bridge_from_scratch() -> None:
    episode = record_episode(load_config(V6).environment, stage=4, seed=11, epsilon=0.0)
    assert episode.success


def test_demonstrations_follow_the_weighted_stage_list(capsys) -> None:
    episodes = generate_demonstrations(load_config(V6).environment, [1, 2, 1], 3, epsilon=0.0, progress=True)
    assert [episode.stage for episode in episodes] == [1, 2, 1]
    assert "3/3 teacher episodes done" in capsys.readouterr().out
