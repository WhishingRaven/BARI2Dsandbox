"""v9: standing robots capped in the climb reward, the row_spanning start
setup and the split curriculum (span 1.7 -> reinforce a prepared row -> both)."""

from __future__ import annotations

import numpy as np
import pytest

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.field import LEFT_BANK
from bari2d.env.robot import RobotState
from bari2d.rl.teacher import ScriptedTeacher
from bari2d.utils.config import load_config

V9 = "configs/beacon_structured_mappo_static_v9.yaml"


def v9_env(stage: int, seed: int = 0) -> BridgeEnv:
    config = load_config(V9).environment
    config.curriculum_stage = stage
    env = BridgeEnv(config)
    env.reset(seed=seed)
    return env


def test_v9_curriculum_lists_line_up() -> None:
    config = load_config(V9)
    environment = config.environment
    stages = config.training.curriculum_max_stage
    assert stages == 9
    for values in (environment.start_setups, environment.spawn_edge_distance,
                   environment.gap.curriculum_widths, environment.load.curriculum_targets):
        assert len(values) == stages
    assert environment.gap.curriculum_widths[4] == 1.7 and environment.load.curriculum_targets[4] == 1.0
    assert environment.start_setups[5] == "row_spanning" and environment.load.curriculum_targets[5] == 18.0
    assert environment.randomization_stage == stages
    assert set(config.training.demo_stages) == set(range(1, stages + 1))


@pytest.mark.parametrize(("stage", "links"), [(6, 3)])
def test_row_spanning_start_is_an_anchored_row_that_needs_reinforcement(stage: int, links: int) -> None:
    for seed in range(3):
        env = v9_env(stage, seed)
        row = env.robots[:links]
        assert row[0].anchor_partner == LEFT_BANK
        assert [robot.anchor_partner for robot in row[1:]] == list(range(links - 1))
        assert env._structure_spans()
        assert 0.0 < env.current_capacity < env.target_load
        _, _, terminated, _, info = env.step(np.full(env.config.robot.count, 9))
        assert not terminated and not any(robot.fallen for robot in row)


def test_climb_reward_counts_one_standing_robot_and_every_extended_link() -> None:
    env = v9_env(stage=3)
    root = env.robots[0]
    states = [RobotState(robot.robot_id, robot.position.copy(), robot.theta, layer=robot.layer) for robot in env.robots]
    # Two robots standing on the root.
    states[1] = RobotState(1, root.position - root.heading * 0.2, root.theta, layer=1)
    states[2] = RobotState(2, root.position + root.heading * 0.2, root.theta, layer=1)
    env.contact_model.reset()
    env.set_robot_states(states)
    env.contact_model._attach(env.robots[0], LEFT_BANK, env.robots)  # type: ignore[attr-defined]
    env.set_robot_states(env.robots)
    assert env.structure_metrics()["on_structure"] == 2
    assert env._climb_potential() == 1.0
    env.config.reward.climb_standing_cap = -1  # v7/v8 rule: both standing robots counted
    assert env._climb_potential() == 2.0


@pytest.mark.parametrize("reinforcement", ["splice", "rows"])
def test_teacher_reinforces_the_prepared_row(reinforcement: str) -> None:
    env = v9_env(stage=6, seed=31)
    teacher = ScriptedTeacher(env, reinforcement=reinforcement)
    info: dict = {}
    for _ in range(env.config.max_steps):
        actions, _ = teacher.actions()
        _, _, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break
    assert info["success"]
    rows = teacher.chains()
    if reinforcement == "splice":
        assert len(rows) == 1 and len(teacher.splices(rows)) == 1
    else:
        assert len(rows) == 2
