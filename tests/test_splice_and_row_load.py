"""v8: the row_midspan load test (no reward for angled rows or robots on top)
and splice grips (a robot lying across a joint grips both robots below)."""

from __future__ import annotations

import numpy as np
import pytest

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.field import LEFT_BANK
from bari2d.env.robot import RobotState
from bari2d.rl.teacher import ScriptedTeacher
from bari2d.utils.config import load_config
from bari2d.utils.scripted import assemble_static_chain

V8 = "configs/beacon_structured_mappo_static_v8.yaml"


def v8_env(stage: int, seed: int = 0, **physics) -> BridgeEnv:
    config = load_config(V8).environment
    config.curriculum_stage = stage
    config.randomization_stage = 99
    config.load.max_target = 60.0
    for name, value in physics.items():
        setattr(config.physics, name, value)
    env = BridgeEnv(config)
    env.reset(seed=seed)
    return env


def angled_row(env: BridgeEnv, angle_deg: float, links: int = 3) -> float:
    left, _ = env.field.boundaries(0.0)
    theta = np.deg2rad(angle_deg)
    heading = np.array([np.cos(theta), np.sin(theta)])
    root = env.field.center + env.field.normal * (left - 0.05)
    length = env.config.robot.length
    states = [RobotState(k, root + heading * length * k, theta) for k in range(links)]
    states += [
        RobotState(k, env.field.center + env.field.normal * (left - 4.0) + env.field.tangent * ((k - links) * 0.6 - 2.0), 0.0)
        for k in range(links, env.config.robot.count)
    ]
    env.contact_model.reset()
    env.set_robot_states(states)
    env.contact_model._attach(env.robots[0], LEFT_BANK, env.robots)  # type: ignore[attr-defined]
    for k in range(1, links):
        env.contact_model._attach(env.robots[k], k - 1, env.robots)  # type: ignore[attr-defined]
    env.set_robot_states(env.robots)
    return env.current_capacity


def row_with_tops(env: BridgeEnv, rows: int, offsets: tuple[float, ...]) -> float:
    """Straight rows plus robots gripping on top of row 0 at ``offsets`` from the root centre."""
    assemble_static_chain(env, rows=rows)
    anchors = [(robot.robot_id, robot.anchor_partner) for robot in env.robots if robot.anchored]
    states = [RobotState(r.robot_id, r.position.copy(), r.theta, layer=r.layer, latent=r.latent) for r in env.robots]
    root = env.robots[0]
    tops = []
    for index, offset in enumerate(offsets):
        robot_id = len(anchors) + index
        tops.append(robot_id)
        states[robot_id] = RobotState(robot_id, root.position + root.heading * offset, root.theta, layer=1)
    env.contact_model.reset()
    env.set_robot_states(states)
    for robot_id, partner in anchors:
        env.contact_model._attach(env.robots[robot_id], partner, env.robots)  # type: ignore[attr-defined]
    for robot_id in tops:
        partner = env.contact_model.support_partner(env.robots[robot_id], env.robots)
        env.contact_model._attach(env.robots[robot_id], partner, env.robots)  # type: ignore[attr-defined]
    env.set_robot_states(env.robots)
    return env.current_capacity


def test_row_midspan_ignores_the_angle_of_a_row() -> None:
    straight = angled_row(v8_env(5), 0.0)
    assert angled_row(v8_env(5), 16.0) == pytest.approx(straight)
    assert angled_row(v8_env(5), 28.0) == pytest.approx(straight)
    uniform = v8_env(5)
    uniform.config.load.load_protocol = "uniform"
    assert angled_row(uniform, 16.0) > 1.5 * straight  # the v7 artefact


def test_splice_grip_strengthens_the_joint_it_covers() -> None:
    single = row_with_tops(v8_env(5), 1, ())
    resting = row_with_tops(v8_env(5, splice_grip=False), 1, (0.45,))
    spliced = row_with_tops(v8_env(5), 1, (0.45,))
    assert resting == pytest.approx(single)  # a robot merely lying on top only adds weight
    assert spliced > 1.4 * single


def test_v8_targets_need_reinforcement_and_both_kinds_reach_them() -> None:
    config = load_config(V8)
    assert config.environment.load.load_protocol == "row_midspan"
    assert config.environment.physics.splice_grip
    for stage in (5, 6, 7):
        target = config.environment.load.curriculum_targets[stage - 1]
        # Stage 7 randomises anchor strength by +-20%.
        margin = 1.2 if stage == 7 else 1.1
        assert row_with_tops(v8_env(stage), 1, ()) * margin < target
        assert row_with_tops(v8_env(stage), 2, ()) / margin >= target
    for stage in (5, 6):
        target = config.environment.load.curriculum_targets[stage - 1]
        assert row_with_tops(v8_env(stage), 1, (0.45,)) >= target


def test_teacher_reinforces_by_splicing_the_root_joint() -> None:
    env = v8_env(5, seed=31)
    env.config.load.max_target = 40.0
    teacher = ScriptedTeacher(env, reinforcement="splice")
    info: dict = {}
    for _ in range(env.config.max_steps):
        actions, _ = teacher.actions()
        _, _, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break
    assert info["success"]
    rows = teacher.chains()
    assert len(rows) == 1 and len(teacher.splices(rows)) == 1


def test_legacy_evaluator_accepts_the_new_protocol() -> None:
    config = load_config("configs/baseline.yaml").environment
    config.load.load_protocol = "row_midspan"
    BridgeEnv(config).reset(seed=0)
