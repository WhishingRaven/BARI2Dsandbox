"""v7: 10 robots, stages 5-7 that need a reinforced (two-row) bridge, the
multi-row teacher, climb credit kept on extension, windowed BC and the
vectorised IR ray march."""

from __future__ import annotations

import numpy as np
import pytest

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.field import LEFT_BANK, RIGHT_BANK
from bari2d.env.robot import RobotState
from bari2d.env.sensors import _point_in_robot, cliff_sensor_direction, ir_distances
from bari2d.rl.demonstrations import DemonstrationSet, generate_demonstrations
from bari2d.utils.config import load_config
from bari2d.utils.scripted import assemble_static_chain

V7 = "configs/beacon_structured_mappo_static_v7.yaml"


def v7_env(stage: int, seed: int = 0) -> BridgeEnv:
    config = load_config(V7).environment
    config.curriculum_stage = stage
    config.randomization_stage = 99  # deterministic anchor strength for the capacity checks
    env = BridgeEnv(config)
    env.reset(seed=seed)
    return env


def _reference_ir(robot, robots, field, robot_config, sensor_config) -> list[float]:
    """The original scalar ray march (noise-free planar rays)."""
    readings = []
    origin_bank = field.bank_at(robot.position)
    for angle_deg in sensor_config.ir_angles_deg:
        angle = robot.theta + np.deg2rad(angle_deg)
        direction = np.array([np.cos(angle), np.sin(angle)])
        measured = sensor_config.ir_range
        for distance in np.arange(sensor_config.ir_step, sensor_config.ir_range + sensor_config.ir_step, sensor_config.ir_step):
            point = robot.position + direction * distance
            if not field.inside(point):
                measured = distance
                break
            bank = field.bank_at(point)
            if bank != origin_bank and (bank is None or origin_bank is None):
                measured = distance
                break
            if any(
                other.robot_id != robot.robot_id and not other.fallen and abs(other.layer - robot.layer) <= 1
                and _point_in_robot(point, other, robot_config)
                for other in robots
            ):
                measured = distance
                break
        readings.append(float(np.clip(measured / sensor_config.ir_range, 0.0, 1.0)))
    return readings


def test_vectorised_ray_march_matches_the_scalar_march() -> None:
    rng = np.random.default_rng(0)
    for seed in range(4):
        env = v7_env(stage=[1, 4, 6, 7][seed], seed=seed)
        for robot in env.robots:
            if rng.random() < 0.6:
                robot.position = robot.position + rng.normal(0.0, 0.6, 2)
                robot.theta = float(rng.uniform(-np.pi, np.pi))
        for robot in env.robots:
            fast = ir_distances(robot, env.robots, env.field, env.config.robot, env.config.sensor, np.random.default_rng(1))
            planar = fast[: len(env.config.sensor.ir_angles_deg)]
            assert list(planar) == _reference_ir(robot, env.robots, env.field, env.config.robot, env.config.sensor)
            cliff = cliff_sensor_direction(robot, env.robots, env.field, env.config.robot, env.config.sensor)
            if cliff is not None:
                assert env.field.bank_at(robot.position) in (LEFT_BANK, RIGHT_BANK)


def test_v7_task_needs_reinforcement_from_stage_5() -> None:
    config = load_config(V7)
    assert config.environment.robot.count == 10
    assert config.training.entropy_coef == pytest.approx(0.03)
    for stage in (5, 6, 7):
        single, double = [], []
        for rows, store in ((1, single), (2, double)):
            env = v7_env(stage)
            assemble_static_chain(env, rows=rows)
            store.append(env.current_capacity)
        target = config.environment.load.curriculum_targets[stage - 1]
        assert single[0] * 1.2 < target <= double[0] * 0.8, (stage, single, double, target)
    env = v7_env(4)
    assemble_static_chain(env, rows=1)
    assert env.current_capacity >= config.environment.load.curriculum_targets[3]


def test_climb_credit_is_kept_when_the_climber_extends() -> None:
    env = v7_env(stage=3)
    root = env.robots[0]
    assert root.anchored
    climber = env.robots[1]
    left, _ = env.field.boundaries(0.0)
    states = [RobotState(robot.robot_id, robot.position.copy(), robot.theta, layer=robot.layer, latent=robot.latent) for robot in env.robots]
    states[1] = RobotState(1, root.position.copy(), root.theta, layer=1, latent=climber.latent)
    env.contact_model.reset()
    env.set_robot_states(states)
    env.contact_model._attach(env.robots[0], LEFT_BANK, env.robots)  # type: ignore[attr-defined]
    env.set_robot_states(env.robots)
    standing = env._climb_potential()
    assert standing == 1.0
    # Extend: robot 1 now grips robot 0 end-to-end over the gap.
    states = [RobotState(robot.robot_id, robot.position.copy(), robot.theta, layer=robot.layer, latent=robot.latent) for robot in env.robots]
    states[1] = RobotState(1, root.position + root.heading * env.config.robot.length, root.theta, layer=0, latent=climber.latent)
    env.contact_model.reset()
    env.set_robot_states(states)
    env.contact_model._attach(env.robots[0], LEFT_BANK, env.robots)  # type: ignore[attr-defined]
    env.contact_model._attach(env.robots[1], 0, env.robots)  # type: ignore[attr-defined]
    env.set_robot_states(env.robots)
    assert env.structure_metrics()["extended"] == 1
    assert env._climb_potential() == 1.0
    env.config.reward.climb_counts_extended = False
    assert env._climb_potential() == 0.0  # the old potential gave it back


def test_teacher_builds_a_second_row_when_one_is_not_enough() -> None:
    from bari2d.rl.teacher import ScriptedTeacher

    env = v7_env(stage=5, seed=31)
    teacher = ScriptedTeacher(env)
    info: dict = {}
    for _ in range(env.config.max_steps):
        actions, _ = teacher.actions()
        _, _, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break
    assert info["success"]
    rows = teacher.chains()
    assert len(rows) == 2 and all(teacher.spans(row) for row in rows)
    assert env.current_capacity >= env.target_load > 15.0


def test_bc_windows_cut_long_demonstrations_and_skip_the_burn_in() -> None:
    config = load_config(V7).environment
    config.robot.count = 5
    config.max_steps = 120
    episodes = generate_demonstrations(config, [4], 2, epsilon=0.1)
    demonstrations = DemonstrationSet(episodes)
    windows = demonstrations.sample_windows(6, window=40, burn_in=10)
    assert all(len(window) <= 40 for window in windows)
    batch = demonstrations.batch(windows, 0.2, "cpu")
    for index, window in enumerate(windows):
        if window.ignore is not None and window.ignore.any():
            columns = slice(index * 5, (index + 1) * 5)
            assert float(batch["weights"][:10, columns].abs().sum()) == 0.0
