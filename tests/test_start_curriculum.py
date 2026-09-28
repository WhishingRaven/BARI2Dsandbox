"""Start-state curriculum, local structure cues, climb potential and structure metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bari2d.env.bridge_env import BridgeEnv, make_observation_layout
from bari2d.env.contact_model import oriented_boxes_overlap
from bari2d.env.robot import DiscreteAction
from bari2d.models.actor import SharedRecurrentActor
from bari2d.utils.config import load_config

V4 = "configs/beacon_structured_mappo_static_v4.yaml"
IDLE = int(DiscreteAction.IDLE)


def v4_env(stage: int, seed: int = 5) -> tuple[BridgeEnv, np.ndarray, dict]:
    config = load_config(V4).environment
    config.curriculum_stage = stage
    env = BridgeEnv(config)
    observation, info = env.reset(seed=seed)
    return env, observation, info


def act(env: BridgeEnv, robot_id: int, action: DiscreteAction):
    actions = np.full(env.config.robot.count, IDLE)
    actions[robot_id] = int(action)
    return env.step(actions)


@pytest.mark.parametrize("stage, setup", [(1, "walker_on_root"), (2, "walker_behind_root"), (3, "root_only"), (4, "none")])
def test_prepared_start_matches_the_stage(stage: int, setup: str) -> None:
    env, _, info = v4_env(stage)
    assert env._start_setup() == setup
    root, walker = env.robots[0], env.robots[1]
    left, _ = env.field.boundaries(float(env.field.coordinates(root.position)[1]))
    if setup == "none":
        assert info["structure"]["edge_grips"] == 0
        return
    assert root.anchored and root.anchor_partner == "left_bank"
    assert env.field.coordinates(root.position)[0] == pytest.approx(left - 0.05, abs=1e-6)
    assert walker.layer == (1 if setup == "walker_on_root" else 0)
    for first in env.robots:
        for second in env.robots:
            if first.robot_id < second.robot_id and first.layer == second.layer:
                assert not oriented_boxes_overlap(first, second, env.config.robot)
    after = env.step(np.full(env.config.robot.count, IDLE))[4]
    assert after["fallen_robots"] == 0 and root.anchored


def test_structure_cues_report_the_anchored_neighbour() -> None:
    env, observation, _ = v4_env(1)
    assert observation[1, env.layout.cues].tolist() == [0.0, 1.0]  # standing on the gripping robot
    env, observation, _ = v4_env(2)
    assert observation[1, env.layout.cues].tolist() == [1.0, 0.0]  # facing the gripping robot
    assert observation[0, env.layout.cues].tolist() == [0.0, 0.0]


def test_cues_are_absent_unless_enabled() -> None:
    v3 = load_config("configs/beacon_structured_mappo_static_v3.yaml").environment
    layout = make_observation_layout(v3)
    assert layout.cues.stop == layout.cues.start
    v4 = load_config(V4).environment
    assert make_observation_layout(v4).size == layout.size + 2


def test_walking_out_and_gripping_completes_stage_one() -> None:
    env, _, _ = v4_env(1)
    root, walker = env.robots[0], env.robots[1]
    for _ in range(20):
        if float((walker.position - root.position) @ root.heading) > 0.05:
            break
        act(env, 1, DiscreteAction.FORWARD)
    _, _, terminated, _, info = act(env, 1, DiscreteAction.ANCHOR)
    assert terminated and info["success"]
    assert info["structure"]["extended"] == 1
    assert info["reward_components"]["climb"] == pytest.approx(-env.config.reward.climb_delta)


def test_climbing_onto_the_gripping_robot_earns_the_climb_potential() -> None:
    env, _, _ = v4_env(2)
    info = act(env, 1, DiscreteAction.CLIMB)[4]
    assert env.robots[1].layer == 1
    assert info["reward_components"]["climb"] == pytest.approx(env.config.reward.climb_delta)
    assert info["structure"]["on_structure"] == 1


def test_randomization_waits_for_its_stage() -> None:
    env, _, _ = v4_env(4)
    assert env._episode_randomization["anchor_strength_scale"] == 1.0
    env, _, _ = v4_env(7, seed=11)
    assert env._episode_randomization["sensor_noise"] > 0.0


def test_unknown_start_setup_is_rejected() -> None:
    config = load_config(V4).environment
    config.start_setups = ["floating_bridge"]
    with pytest.raises(ValueError):
        BridgeEnv(config).reset(seed=1)


def test_actor_accepts_cue_observations() -> None:
    config = load_config(V4)
    env = BridgeEnv(config.environment)
    observation, _ = env.reset(seed=2)
    actor = SharedRecurrentActor(env.layout, env.action_count, config.model)
    hidden = actor.initial_hidden(env.config.robot.count)
    action, _, _, output = actor.act(torch.as_tensor(observation), hidden, torch.as_tensor(env.action_masks()))
    assert action.shape == (env.config.robot.count,)
    assert output.logits.shape == (env.config.robot.count, env.action_count)
