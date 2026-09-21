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


def test_new_anchor_is_penalized(env) -> None:
    actions = np.full(len(env.robots), int(DiscreteAction.IDLE))
    actions[0] = int(DiscreteAction.ANCHOR)
    _, _, _, _, info = env.step(actions)
    assert info["reward_components"]["anchor"] < 0.0


def test_success_requires_span_and_target_capacity(env) -> None:
    env.target_load = 3.0
    assemble_parallel_bridge(env, rows=1)
    actions = np.full(len(env.robots), int(DiscreteAction.IDLE))
    _, _, terminated, _, info = env.step(actions)
    assert terminated
    assert info["success"]
    assert info["reward_components"]["success"] > 0.0

    env.reset(seed=13)
    env.target_load = 8.0
    assemble_parallel_bridge(env, rows=1)
    _, _, terminated, _, info = env.step(actions)
    assert not terminated
    assert not info["success"]
    assert info["span"]
    assert info["capacity"] < info["target_load"]
