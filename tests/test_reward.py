from __future__ import annotations

import numpy as np

from bari2d.env.robot import DiscreteAction
from bari2d.utils.scripted import assemble_parallel_bridge


def test_efficiency_reward_components_have_expected_signs(env) -> None:
    actions = np.full(len(env.robots), int(DiscreteAction.IDLE))
    actions[0] = int(DiscreteAction.FORWARD)
    _, reward, _, _, info = env.step(actions)
    components = info["reward_components"]
    assert components["time"] < 0.0
    assert components["energy"] < 0.0
    assert np.isclose(reward, sum(components.values()))


def test_moving_is_preferred_to_every_robot_idling(environment_config) -> None:
    from bari2d.env.bridge_env import BridgeEnv

    idle_env = BridgeEnv(environment_config)
    idle_env.reset(seed=17)
    _, idle_reward, _, _, _ = idle_env.step(
        np.full(len(idle_env.robots), int(DiscreteAction.IDLE))
    )

    moving_env = BridgeEnv(environment_config)
    moving_env.reset(seed=17)
    _, moving_reward, _, _, _ = moving_env.step(
        np.full(len(moving_env.robots), int(DiscreteAction.FORWARD))
    )

    assert moving_reward > idle_reward


def test_new_anchor_is_penalized(env) -> None:
    actions = np.full(len(env.robots), int(DiscreteAction.IDLE))
    actions[0] = int(DiscreteAction.ANCHOR)
    _, _, _, _, info = env.step(actions)
    assert info["reward_components"]["anchor"] < 0.0


def test_capacity_objective_does_not_stop_when_bridge_first_spans(env) -> None:
    assemble_parallel_bridge(env, rows=1)
    actions = np.full(len(env.robots), int(DiscreteAction.IDLE))
    _, _, terminated, truncated, info = env.step(actions)
    assert not terminated
    assert not truncated
    assert info["span"]
    assert not info["success"]


def test_capacity_objective_ends_at_fixed_horizon(environment_config) -> None:
    from bari2d.env.bridge_env import BridgeEnv

    environment_config.max_steps = 1
    environment = BridgeEnv(environment_config)
    environment.reset(seed=13)
    actions = np.full(len(environment.robots), int(DiscreteAction.IDLE))
    _, _, terminated, truncated, _ = environment.step(actions)
    assert not terminated
    assert truncated


def test_new_rightward_progress_in_gap_is_rewarded_once(env) -> None:
    assemble_parallel_bridge(env, rows=1, anchor=False)
    robot = next(robot for robot in env.robots if env.field.is_gap(robot.position))
    robot.theta = env.field.orientation
    env.set_robot_states(env.robots)
    actions = np.full(len(env.robots), int(DiscreteAction.IDLE))
    actions[robot.robot_id] = int(DiscreteAction.FORWARD)

    _, _, _, _, info = env.step(actions)

    assert info["gap_progress_by_robot"][robot.robot_id] > 0.0
    assert info["reward_components"]["gap_progress"] > 0.0
    assert info["agent_rewards"][robot.robot_id] > 0.0
