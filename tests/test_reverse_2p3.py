"""v13: reverse curriculum at 2.3 - rows prepared one and two links short of
the far bank (row_minus_1, row_minus_2) before the empty start."""

from __future__ import annotations

import numpy as np
import pytest

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.field import LEFT_BANK
from bari2d.env.robot import DiscreteAction
from bari2d.rl.teacher import ScriptedTeacher
from bari2d.rl.trainer import Trainer
from bari2d.utils.config import load_config

V12 = "configs/beacon_structured_mappo_static_v12.yaml"
V13 = "configs/beacon_structured_mappo_static_v13.yaml"


def v13_env(stage: int, seed: int) -> BridgeEnv:
    config = load_config(V13).environment
    config.curriculum_stage = stage
    env = BridgeEnv(config)
    env.reset(seed=seed)
    return env


def test_v13_stages() -> None:
    config = load_config(V13)
    environment = config.environment
    stages = config.training.curriculum_max_stage
    assert stages == 13 and environment.randomization_stage == 13
    for values in (environment.start_setups, environment.spawn_edge_distance,
                   environment.gap.curriculum_widths, environment.load.curriculum_targets):
        assert len(values) == stages
    table = list(zip(environment.gap.curriculum_widths, environment.load.curriculum_targets, environment.start_setups))
    assert table[7:12] == [
        (2.3, 1.0, "row_minus_1"), (2.3, 1.0, "row_minus_2"), (2.3, 1.0, "none"),
        (2.3, 18.0, "row_spanning"), (2.3, 18.0, "none"),
    ]
    assert table[12] == (3.0, 18.0, "none")
    assert set(config.training.demo_stages) == set(range(1, stages + 1))
    assert config.training.output_dir.endswith("_v13") and config.training.total_updates == 7500
    v12 = load_config(V12)
    assert environment.reward == v12.environment.reward and environment.robot == v12.environment.robot
    assert environment.max_steps == v12.environment.max_steps == 1500


@pytest.mark.parametrize(("stage", "row"), [(8, 3), (9, 2)])
def test_rows_prepared_short_of_the_far_bank(stage: int, row: int) -> None:
    for seed in range(3):
        env = v13_env(stage, seed)
        prepared = env.robots[:row]
        assert prepared[0].anchor_partner == LEFT_BANK
        assert [robot.anchor_partner for robot in prepared[1:]] == list(range(row - 1))
        assert not any(robot.absent for robot in prepared)
        assert not any(robot.anchored for robot in env.robots[row:])
        assert not env._structure_spans()
        _, _, terminated, _, info = env.step(np.full(len(env.robots), int(DiscreteAction.IDLE)))
        assert not terminated and info["fallen_robots"] == 0


def test_the_full_row_is_the_short_row_plus_its_missing_links() -> None:
    lengths = {}
    for setup in ("row_spanning", "row_minus_1", "row_minus_2"):
        config = load_config(V13).environment
        config.curriculum_stage = 11  # 2.3, row_spanning
        config.start_setups = list(config.start_setups)
        config.start_setups[10] = setup
        env = BridgeEnv(config)
        env.reset(seed=4)
        lengths[setup] = env._prepared_row_length
    assert lengths == {"row_spanning": 4, "row_minus_1": 3, "row_minus_2": 2}


@pytest.mark.parametrize("stage", [8, 9])
def test_teacher_completes_the_short_row(stage: int) -> None:
    env = v13_env(stage, seed=5)
    teacher = ScriptedTeacher(env, reinforcement="splice")
    info: dict = {}
    for _ in range(env.config.max_steps):
        actions, _ = teacher.actions()
        _, _, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break
    assert info["success"]


def test_a_v12_stage_8_checkpoint_resumes_at_the_one_link_short_stage(tmp_path) -> None:
    def small(path: str, output_dir: str):
        config = load_config(path)
        config.environment.max_steps = 16
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

    v12 = Trainer(small(V12, str(tmp_path / "v12")))
    v12.curriculum.stage = 8
    path = v12.save_checkpoint(6000)
    v13 = Trainer(small(V13, str(tmp_path / "v13")))
    assert v13.load_checkpoint(path) == 6000 and v13.curriculum.stage == 8
    history = v13.train(updates=6001)
    assert history[-1]["curriculum_stage"] == 8.0
