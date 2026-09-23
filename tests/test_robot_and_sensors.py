from __future__ import annotations

import numpy as np
import pytest

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.robot import DiscreteAction, RobotState
from bari2d.env.sensors import ir_distances, ir_ray_count
from bari2d.utils.config import EnvironmentConfig


def test_reset_randomizes_robot_positions_headings_and_terrain_reproducibly() -> None:
    config = EnvironmentConfig()
    first = BridgeEnv(config)
    first.reset(seed=21)
    first_positions = np.stack([robot.position for robot in first.robots])
    first_headings = np.asarray([robot.theta for robot in first.robots])
    first_terrain = (first.field.gap_width, first.field.orientation, first.field.irregularity)

    repeated = BridgeEnv(config)
    repeated.reset(seed=21)
    different = BridgeEnv(config)
    different.reset(seed=22)

    assert np.allclose(first_positions, np.stack([robot.position for robot in repeated.robots]))
    assert np.allclose(first_headings, [robot.theta for robot in repeated.robots])
    assert first_terrain == (
        repeated.field.gap_width,
        repeated.field.orientation,
        repeated.field.irregularity,
    )
    assert not np.allclose(first_positions, np.stack([robot.position for robot in different.robots]))
    assert not np.allclose(first_headings, [robot.theta for robot in different.robots])
    assert first_terrain != (
        different.field.gap_width,
        different.field.orientation,
        different.field.irregularity,
    )
    assert np.std(first_headings) > 0.5


def test_robot_moves_and_steers(env) -> None:
    robot = env.robots[0]
    initial_position = robot.position.copy()
    initial_angle = robot.theta
    robot.step_kinematics(
        int(DiscreteAction.FORWARD_LEFT), env.config.robot, env.config.time_step, env.rng
    )
    assert np.linalg.norm(robot.position - initial_position) > 0.0
    assert robot.theta > initial_angle
    assert robot.velocity > 0.0


def test_ir_detects_robot_and_gap_edge(env) -> None:
    left, _ = env.field.boundaries(0.0)
    edge = env.field.center + env.field.normal * left
    first = RobotState(0, edge - env.field.normal * 1.0, env.field.orientation)
    second = RobotState(1, first.position + env.field.normal * 1.2, env.field.orientation)
    robots = [first, second]
    robot_reading = ir_distances(first, robots, env.field, env.config.robot, env.config.sensor, env.rng)[0]
    edge_reading = ir_distances(first, [first], env.field, env.config.robot, env.config.sensor, env.rng)[0]
    assert 0.0 < robot_reading < 1.0
    assert 0.0 < edge_reading < 1.0
    assert robot_reading < edge_reading


def test_default_ir_has_cardinal_and_downward_rays(env) -> None:
    readings = ir_distances(
        env.robots[0], env.robots, env.field, env.config.robot, env.config.sensor, env.rng
    )

    assert env.config.sensor.ir_angles_deg == [0.0, 180.0, 90.0, -90.0]
    assert ir_ray_count(env.config.sensor) == 5
    assert env.observation_size == 39
    assert readings.shape == (5,)
    assert readings[-1] == 0.0


def test_downward_ir_reports_a_cliff_when_no_surface_is_below(env) -> None:
    robot = env.robots[0]
    robot.position = env.field.center.copy()

    readings = ir_distances(
        robot, env.robots, env.field, env.config.robot, env.config.sensor, env.rng
    )

    assert readings[-1] == 1.0


def test_contact_enables_climb(env) -> None:
    base = env.robots[0]
    support = env.robots[1]
    support.position = base.position + base.heading * (env.config.robot.length * 0.8)
    support.theta = base.theta
    assert env.action_masks()[0, DiscreteAction.CLIMB]
    actions = np.full(len(env.robots), int(DiscreteAction.IDLE))
    actions[0] = int(DiscreteAction.CLIMB)
    env.step(actions)
    assert base.head_lifted
    assert base.layer == support.layer + 1


def test_elevated_robot_automatically_descends_after_losing_support_overlap() -> None:
    config = EnvironmentConfig()
    config.robot.count = 2
    config.max_steps = 10
    environment = BridgeEnv(config)
    environment.reset(seed=3)
    climber, support = environment.robots
    climber.layer = 1
    support.layer = 0
    support.position = climber.position - climber.heading * (config.robot.length - 0.03)
    support.theta = climber.theta

    actions = np.full(2, int(DiscreteAction.IDLE))
    actions[0] = int(DiscreteAction.FORWARD)
    environment.step(actions)

    assert climber.layer == 0


def test_climb_uses_a_same_layer_robot_to_reach_layer_two() -> None:
    config = EnvironmentConfig()
    config.robot.count = 3
    config.max_steps = 10
    environment = BridgeEnv(config)
    environment.reset(seed=5)
    climber, lower_support, same_layer_support = environment.robots
    origin = np.array([2.0, environment.field.width / 2.0])
    climber.position = origin.copy()
    climber.theta = 0.0
    climber.layer = 1
    lower_support.position = origin + np.array([0.1, 0.0])
    lower_support.layer = 0
    same_layer_support.position = origin + np.array([0.8, 0.0])
    same_layer_support.layer = 1
    environment.set_robot_states(environment.robots)

    actions = np.full(3, int(DiscreteAction.IDLE))
    actions[climber.robot_id] = int(DiscreteAction.CLIMB)
    environment.step(actions)

    assert climber.layer == 2


def test_same_layer_climb_also_works_above_layer_two() -> None:
    config = EnvironmentConfig()
    config.robot.count = 8
    config.max_steps = 10
    environment = BridgeEnv(config)
    environment.reset(seed=8)
    climber, same_layer_support = environment.robots[:2]
    origin = np.array([2.0, environment.field.width / 2.0])
    target = origin + np.array([0.8, 0.0])
    climber.position = origin.copy()
    climber.theta = 0.0
    climber.layer = 3
    same_layer_support.position = target.copy()
    same_layer_support.layer = 3
    for robot, layer in zip(environment.robots[2:5], range(3)):
        robot.position = origin + np.array([0.1, 0.0])
        robot.layer = layer
    for robot, layer in zip(environment.robots[5:], range(3)):
        robot.position = target.copy()
        robot.layer = layer
    environment.set_robot_states(environment.robots)

    actions = np.full(config.robot.count, int(DiscreteAction.IDLE))
    actions[climber.robot_id] = int(DiscreteAction.CLIMB)
    environment.step(actions)

    assert climber.layer == 4


@pytest.mark.parametrize(
    ("starting_layer", "overlapping_support_layer", "expected_layer"),
    [(2, 0, 1), (4, 2, 3)],
)
def test_elevated_robot_falls_to_highest_overlapping_support(
    starting_layer: int, overlapping_support_layer: int, expected_layer: int
) -> None:
    config = EnvironmentConfig()
    config.robot.count = 3 + overlapping_support_layer
    config.max_steps = 10
    environment = BridgeEnv(config)
    environment.reset(seed=6)
    falling, nearby_without_overlap, overlapping_support = environment.robots[:3]
    origin = np.array([2.0, environment.field.width / 2.0])
    falling.position = origin.copy()
    falling.layer = starting_layer
    nearby_without_overlap.position = origin + np.array([1.0, 0.0])
    nearby_without_overlap.layer = starting_layer - 1
    overlapping_support.position = origin.copy()
    overlapping_support.layer = overlapping_support_layer
    for robot, layer in zip(environment.robots[3:], range(overlapping_support_layer)):
        robot.position = origin.copy()
        robot.layer = layer
    environment.set_robot_states(environment.robots)

    environment.step(np.full(config.robot.count, int(DiscreteAction.IDLE)))

    assert falling.layer == expected_layer


def test_anchored_robot_still_falls_without_overlapping_support() -> None:
    config = EnvironmentConfig()
    config.robot.count = 2
    config.max_steps = 10
    environment = BridgeEnv(config)
    environment.reset(seed=7)
    falling, distant_robot = environment.robots
    falling.position = np.array([2.0, environment.field.width / 2.0])
    falling.layer = 4
    falling.anchored = True
    distant_robot.position = falling.position + np.array([2.0, 0.0])
    distant_robot.layer = 0
    environment.set_robot_states(environment.robots)

    environment.step(np.full(2, int(DiscreteAction.IDLE)))

    assert falling.layer == 0


def test_downward_ir_uses_highest_overlapping_robot_surface(env) -> None:
    robot, nearby_without_overlap, overlapping_support = env.robots[:3]
    robot.position = env.field.center.copy()
    robot.layer = 2
    nearby_without_overlap.position = robot.position + np.array([1.0, 0.0])
    nearby_without_overlap.layer = 1
    overlapping_support.position = robot.position.copy()
    overlapping_support.layer = 0

    readings = ir_distances(
        robot,
        [robot, nearby_without_overlap, overlapping_support],
        env.field,
        env.config.robot,
        env.config.sensor,
        env.rng,
    )

    expected = 2 * env.config.robot.climb_height / env.config.sensor.ir_range
    assert readings[-1] == pytest.approx(expected)
