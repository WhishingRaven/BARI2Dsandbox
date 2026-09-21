from __future__ import annotations

import numpy as np

from bari2d.env.robot import DiscreteAction, RobotState
from bari2d.env.sensors import ir_distances


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

