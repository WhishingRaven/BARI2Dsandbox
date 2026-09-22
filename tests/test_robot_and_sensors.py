from __future__ import annotations

import numpy as np

from bari2d.env.robot import DiscreteAction, RobotState
from bari2d.env.sensors import ir_distances, ir_ray_count


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


def test_elevated_robot_automatically_descends_after_leaving_support_range() -> None:
    from bari2d.env.bridge_env import BridgeEnv
    from bari2d.utils.config import EnvironmentConfig

    config = EnvironmentConfig()
    config.robot.count = 2
    config.max_steps = 10
    environment = BridgeEnv(config)
    environment.reset(seed=3)
    climber, support = environment.robots
    climber.layer = 1
    support.layer = 0
    support.position = climber.position - climber.heading * (config.robot.length * 1.25 - 0.03)

    actions = np.full(2, int(DiscreteAction.IDLE))
    actions[0] = int(DiscreteAction.FORWARD)
    environment.step(actions)

    assert climber.layer == 0
