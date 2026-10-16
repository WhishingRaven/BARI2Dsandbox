"""v14: 24 modules, a design load of 24, and robots crossing the finished bridge
(climbs land on the robot climbed, robots step down off a splice)."""

from __future__ import annotations

import numpy as np
import pytest

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.field import RIGHT_BANK
from bari2d.env.robot import DiscreteAction
from bari2d.rl.teacher import ScriptedTeacher
from bari2d.rl.trainer import Trainer
from bari2d.utils.config import load_config

V13 = "configs/beacon_structured_mappo_static_v13.yaml"
V14 = "configs/beacon_structured_mappo_static_v14.yaml"


def v14_env(stage: int, seed: int, **physics) -> BridgeEnv:
    config = load_config(V14).environment
    config.curriculum_stage = stage
    for name, value in physics.items():
        setattr(config.physics, name, value)
    env = BridgeEnv(config)
    env.reset(seed=seed)
    return env


def test_v14_stages() -> None:
    config = load_config(V14)
    environment = config.environment
    stages = config.training.curriculum_max_stage
    assert stages == 17 and environment.randomization_stage == 17
    lists = (environment.start_setups, environment.spawn_edge_distance, environment.gap.curriculum_widths,
             environment.load.curriculum_targets, environment.load.curriculum_crossers, environment.curriculum_max_steps)
    assert all(len(values) == stages for values in lists)
    table = list(zip(environment.gap.curriculum_widths, environment.load.curriculum_targets,
                     environment.start_setups, environment.load.curriculum_crossers, environment.curriculum_max_steps))
    assert table[10:] == [
        (2.3, 18.0, "row_spanning", 0, 1500), (2.3, 18.0, "none", 0, 1500),
        (2.3, 24.0, "row_spanning", 0, 1500), (2.3, 24.0, "none", 0, 1500),
        (2.3, 1.0, "row_spanning", 3, 2000), (2.3, 24.0, "none", 3, 2000), (3.0, 24.0, "none", 3, 2000),
    ]
    v13 = load_config(V13).environment
    for stage in range(10):  # stages 1-10 ask for the same task as v13
        assert table[stage][:3] == (v13.gap.curriculum_widths[stage], v13.load.curriculum_targets[stage], v13.start_setups[stage])
    assert environment.robot.count == 24 and environment.robot.active_min == 18
    assert environment.physics.climb_onto_support and environment.physics.step_down_onto_lower
    assert environment.reward.cross_delta == 3.0
    assert config.training.output_dir.endswith("_v14") and config.training.total_updates == 10000


def test_stage_time_limits_and_defaults() -> None:
    assert v14_env(10, 0)._max_steps() == 1500 and v14_env(15, 0)._max_steps() == 2000
    config = load_config(V13).environment
    config.curriculum_stage = 10
    env = BridgeEnv(config)
    env.reset(seed=0)
    assert env._max_steps() == 1500 and env._required_crossers() == 0
    assert not env.config.physics.climb_onto_support and not env.config.physics.step_down_onto_lower


def _queue_behind_root(env: BridgeEnv, count: int, splice: bool = False) -> list:
    """Clear the bank, put a splice over the root joint (optional) and ``count`` robots queued behind the root."""
    root, length = env.robots[0], env.config.robot.length
    free = [robot for robot in env.robots[4:] if not robot.absent]
    for index, robot in enumerate(free[count:]):
        robot.position = np.array([0.8 + 0.7 * (index % 6), 0.8 + 0.6 * (index // 6)])
        robot.theta, robot.layer = 0.0, 0
    if splice:
        free[count].position = root.position + root.heading * 0.45
        free[count].theta, free[count].layer = root.theta, 1
    for index, robot in enumerate(free[:count]):
        robot.position = root.position - root.heading * ((index + 1) * (length + 0.08))
        robot.theta, robot.layer = root.theta, 0
    env.set_robot_states(env.robots)
    if splice:
        env.contact_model._attach(free[count], 0, env.robots)  # type: ignore[attr-defined]
        env.set_robot_states(env.robots)
    return free[:count]


def _walk_across(env: BridgeEnv, crossers: list) -> dict:
    info: dict = {}
    for _ in range(env._max_steps()):
        actions = np.full(len(env.robots), int(DiscreteAction.IDLE))
        for robot in crossers:
            if robot.fallen or (robot.layer == 0 and env.field.bank_at(robot.position) == RIGHT_BANK):
                continue
            support = env._climb_support(robot)
            climb = support is not None and support.anchored
            actions[robot.robot_id] = int(DiscreteAction.CLIMB if climb else DiscreteAction.FORWARD)
        _, _, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break
    return info


def test_a_climb_lands_on_the_robot_climbed() -> None:
    for onto, inside in ((True, True), (False, False)):
        env = v14_env(15, 3, climb_onto_support=onto)
        walker = _queue_behind_root(env, 1)[0]
        root = env.robots[0]
        actions = np.full(len(env.robots), int(DiscreteAction.IDLE))
        actions[walker.robot_id] = int(DiscreteAction.CLIMB)
        env.step(actions)
        behind = float((root.position - walker.position) @ root.heading)
        assert walker.layer == 1
        assert (behind <= 0.25 * env.config.robot.length + 0.06) is inside


def test_robots_cross_a_spliced_bridge_and_count() -> None:
    env = v14_env(15, 4)
    assert env._required_crossers() == 3 and env.structure_metrics()["crossed"] == 0  # the far link is anchored
    crossers = _queue_behind_root(env, 3, splice=True)
    info = _walk_across(env, crossers)
    assert info["success"] and info["structure"]["crossed"] == 3 and info["fallen_robots"] == 0
    assert all(robot.layer == 0 and env.field.bank_at(robot.position) == RIGHT_BANK for robot in crossers)


def test_without_the_new_climb_physics_a_splice_blocks_the_deck() -> None:
    env = v14_env(15, 4, climb_onto_support=False, step_down_onto_lower=False)
    crossers = _queue_behind_root(env, 3, splice=True)
    info = _walk_across(env, crossers)
    assert not info["success"] and info["structure"]["crossed"] < 3


def test_success_waits_for_the_crossers_and_crossing_pays() -> None:
    env = v14_env(15, 3)
    assert env._structure_spans() and env.current_capacity >= env.target_load
    _, _, terminated, _, info = env.step(np.full(len(env.robots), int(DiscreteAction.IDLE)))
    assert not terminated and not info["success"]
    crossers = _queue_behind_root(env, 3)
    paid = 0.0
    for _ in range(env._max_steps()):
        actions = np.full(len(env.robots), int(DiscreteAction.IDLE))
        for robot in crossers:
            if robot.layer == 0 and env.field.bank_at(robot.position) == RIGHT_BANK:
                continue
            support = env._climb_support(robot)
            actions[robot.robot_id] = int(DiscreteAction.CLIMB if support is not None and support.anchored else DiscreteAction.FORWARD)
        _, _, terminated, truncated, info = env.step(actions)
        paid += info["reward_components"]["cross"]
        if terminated or truncated:
            break
    assert info["success"] and paid == pytest.approx(3 * 3.0)


def test_teacher_reaches_the_load_of_24() -> None:
    env = v14_env(13, 31)
    teacher = ScriptedTeacher(env, reinforcement="rows")
    info: dict = {}
    for _ in range(env._max_steps()):
        actions, _ = teacher.actions()
        _, _, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break
    assert info["success"] and env.current_capacity >= 24.0


def test_a_16_robot_v13_checkpoint_resumes_with_24_robots_at_stage_10(tmp_path) -> None:
    def small(path: str, output_dir: str):
        config = load_config(path)
        config.environment.max_steps = 16
        config.environment.curriculum_max_steps = []
        config.training.num_envs = 1
        config.training.vector_env = "sync"
        config.training.rollout_steps = 8
        config.training.recurrent_chunk = 8
        config.training.num_minibatches = 1
        config.training.ppo_epochs = 1
        config.training.demo_path = ""
        config.training.checkpoint_interval = 1000
        config.training.output_dir = output_dir
        return config

    v13 = Trainer(small(V13, str(tmp_path / "v13")))
    v13.curriculum.stage = 10
    path = v13.save_checkpoint(7500)
    v14 = Trainer(small(V14, str(tmp_path / "v14")))
    assert v14.env.config.robot.count == 24
    assert v14.load_checkpoint(path) == 7500 and v14.curriculum.stage == 10
    history = v14.train(updates=7501)
    assert history[-1]["curriculum_stage"] == 10.0
