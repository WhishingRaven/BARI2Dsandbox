"""Static (gravity + rigid-body equilibrium) physics: checks with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.contact_model import ContactModel
from bari2d.env.robot import DiscreteAction, RobotState
from bari2d.env.static_physics import StaticContactModel
from bari2d.utils.config import EnvironmentConfig, load_config
from bari2d.utils.scripted import assemble_static_chain, build_by_grip_and_extend, park_robots_on_left_bank

IDLE = int(DiscreteAction.IDLE)


def static_env(count: int, gap: float = 1.2, target: float = 1.0, bending: float | None = None) -> BridgeEnv:
    config = EnvironmentConfig()
    config.robot.count = count
    config.max_steps = 400
    config.physics.model = "static"
    config.gap.curriculum_widths = [gap]
    config.load.curriculum_targets = [target]
    if bending is not None:
        config.physics.anchor_bending_limit = bending
        config.physics.anchor_torsion_limit = bending
    env = BridgeEnv(config)
    env.reset(seed=3)
    return env


def left_edge(env: BridgeEnv) -> np.ndarray:
    left, _ = env.field.boundaries(0.0)
    return env.field.center + env.field.normal * left


def pose(env: BridgeEnv, robot_id: int, longitudinal: float, layer: int = 0) -> RobotState:
    """Robot facing the gap with its center ``longitudinal`` past the left cliff edge."""
    state = RobotState(robot_id, left_edge(env) + env.field.normal * longitudinal, env.field.orientation, layer=layer)
    state.latent = np.zeros(env.config.latent_dim, dtype=np.float32)
    return state


def idle_step(env: BridgeEnv) -> dict:
    return env.step(np.full(env.config.robot.count, IDLE))[4]


def test_legacy_physics_remains_the_default() -> None:
    config = EnvironmentConfig()
    assert config.physics.model == "legacy"
    env = BridgeEnv(config)
    assert type(env.contact_model) is ContactModel


def test_unknown_physics_model_is_rejected() -> None:
    config = EnvironmentConfig()
    config.physics.model = "rigid"
    with pytest.raises(ValueError):
        BridgeEnv(config)


def test_robot_tips_once_its_center_passes_the_cliff_edge() -> None:
    env = static_env(1)
    env.set_robot_states([pose(env, 0, -0.05)])
    idle_step(env)
    assert not env.robots[0].fallen

    env.set_robot_states([pose(env, 0, 0.05)])
    idle_step(env)
    assert env.robots[0].fallen


def test_touching_robots_without_anchors_cannot_bridge() -> None:
    # The legacy shortcut: two robots meet end to end with a corner on each bank.
    env = static_env(2, gap=1.2)
    env.set_robot_states([pose(env, 0, 0.15), pose(env, 1, 1.05)])
    assert env.current_capacity == 0.0
    assert not env._structure_spans()
    info = idle_step(env)
    assert all(robot.fallen for robot in env.robots)
    assert not info["success"]


@pytest.mark.parametrize("bending, survives", [(1.9, True), (1.7, False)])
def test_cantilever_joint_breaks_at_the_analytic_moment(bending: float, survives: bool) -> None:
    # Robot 0 lies on the bank with its front at the edge; robots 1 and 2 hang
    # end to end beyond it.  The joint at the edge carries
    # M = W * (0.45 + 1.35) = 1.8 robot-weight lengths.
    env = static_env(3, gap=3.0, bending=bending)
    length = env.config.robot.length
    env.set_robot_states([pose(env, 0, -length / 2), pose(env, 1, length / 2 + 1e-6), pose(env, 2, 1.5 * length + 1e-6)])
    model = env.contact_model
    assert isinstance(model, StaticContactModel)
    model._attach(env.robots[0], "left_bank", env.robots)
    model._attach(env.robots[1], 0, env.robots)
    model._attach(env.robots[2], 1, env.robots)
    idle_step(env)
    assert env.robots[0].anchored and not env.robots[0].fallen
    assert (not env.robots[1].fallen) is survives
    assert (not env.robots[2].fallen) is survives
    assert (model.anchor_failures == 0) is survives


def test_overhanging_rider_leans_on_the_bank_but_falls_into_the_gap() -> None:
    length = 0.9
    # Over the bank a first-layer rider past its support leans on the ground (tilted climb pose).
    env = static_env(2)
    env.set_robot_states([pose(env, 0, -3.0), pose(env, 1, -3.0 + 0.6, layer=1)])
    idle_step(env)
    assert env.robots[1].layer == 1 and not env.robots[1].fallen

    # A second-layer rider cannot reach the ground: the same overhang lands it on the bank.
    env = static_env(3)
    env.set_robot_states([pose(env, 0, -3.0), pose(env, 1, -3.0, layer=1), pose(env, 2, -3.0 + 0.6, layer=2)])
    idle_step(env)
    assert env.robots[2].layer == 0 and not env.robots[2].fallen

    # Over the gap: the same overhang drops the rider into the gap.
    env = static_env(2, gap=3.0)
    env.set_robot_states([pose(env, 0, -length / 2), pose(env, 1, -length / 2 + 0.6, layer=1)])
    env.contact_model._attach(env.robots[0], "left_bank", env.robots)  # type: ignore[attr-defined]
    idle_step(env)
    assert env.robots[1].fallen


def test_upper_layer_robot_grips_the_robot_below_not_the_ground() -> None:
    env = static_env(2)
    env.set_robot_states([pose(env, 0, -0.3), pose(env, 1, -0.4, layer=1)])
    assert env.contact_model.candidate_anchor(env.robots[1], env.robots, env.field) == 0
    assert env.contact_model.candidate_anchor(env.robots[0], env.robots, env.field) == "left_bank"


def test_capacity_falls_with_gap_width_and_grows_with_rows() -> None:
    capacities = []
    for gap in (2.3, 3.0, 4.0):
        env = static_env(20, gap=gap, bending=2.0)
        assemble_static_chain(env, rows=1)
        capacities.append(env.current_capacity)
    assert capacities[0] > capacities[1] > capacities[2] > 0.0

    env = static_env(20, gap=4.0, bending=2.0)
    assemble_static_chain(env, rows=2)
    assert env.current_capacity == pytest.approx(2.0 * capacities[2], rel=1.0e-6)
    result = env._evaluate_load(accurate=True)
    assert result.failure_mode == "anchor_bending"
    assert result.load_bearing_path[0] == "left_bank" and result.load_bearing_path[-1] == "right_bank"


def test_grip_and_extend_builds_a_bridge_through_the_action_interface() -> None:
    env = static_env(5, gap=1.7, target=1.5)
    result = build_by_grip_and_extend(env)
    assert result["success"], result["failure"]
    assert result["robots_used"] == 3
    chain = [robot for robot in env.robots if robot.anchored]
    assert [robot.layer for robot in chain] == [0, 0, 0]


def test_weak_anchors_break_under_a_walker_on_the_cantilever() -> None:
    env = static_env(5, gap=2.3, target=2.2, bending=2.0)
    result = build_by_grip_and_extend(env)
    assert not result["success"]
    assert env.contact_model.anchor_failures >= 1
    assert any(robot.fallen for robot in env.robots)


def test_parked_robots_do_not_overlap() -> None:
    env = static_env(20)
    states = park_robots_on_left_bank(env, list(range(20)))
    env.set_robot_states(states)
    info = idle_step(env)
    assert info["fallen_robots"] == 0
    assert all(env.field.bank_at(robot.position) == "left_bank" for robot in env.robots)


def test_static_training_config_loads() -> None:
    config = load_config("configs/beacon_structured_mappo_static.yaml")
    assert config.environment.physics.model == "static"
    env = BridgeEnv(config.environment)
    observation, info = env.reset(seed=1)
    assert observation.shape == (config.environment.robot.count, env.observation_size)
    assert info["fallen_robots"] == 0
