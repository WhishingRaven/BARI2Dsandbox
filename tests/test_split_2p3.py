"""v12: the 2.3 gap split like 1.7 (span -> reinforce a prepared row -> both)
and a 1500-step time limit."""

from __future__ import annotations

import numpy as np

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.field import LEFT_BANK
from bari2d.env.robot import DiscreteAction
from bari2d.rl.teacher import ScriptedTeacher
from bari2d.rl.trainer import Trainer
from bari2d.utils.config import load_config

V11 = "configs/beacon_structured_mappo_static_v11.yaml"
V12 = "configs/beacon_structured_mappo_static_v12.yaml"


def v12_env(stage: int, seed: int) -> BridgeEnv:
    config = load_config(V12).environment
    config.curriculum_stage = stage
    env = BridgeEnv(config)
    env.reset(seed=seed)
    return env


def test_v12_stages() -> None:
    config = load_config(V12)
    environment = config.environment
    stages = config.training.curriculum_max_stage
    assert stages == 11 and environment.randomization_stage == 11
    for values in (environment.start_setups, environment.spawn_edge_distance,
                   environment.gap.curriculum_widths, environment.load.curriculum_targets):
        assert len(values) == stages
    table = list(zip(environment.gap.curriculum_widths, environment.load.curriculum_targets, environment.start_setups))
    assert table[4:7] == [(1.7, 1.0, "none"), (1.7, 18.0, "row_spanning"), (1.7, 18.0, "none")]
    assert table[7:10] == [(2.3, 1.0, "none"), (2.3, 18.0, "row_spanning"), (2.3, 18.0, "none")]
    assert table[10] == (3.0, 18.0, "none")
    assert environment.max_steps == 1500
    assert set(config.training.demo_stages) == set(range(1, stages + 1))
    assert config.training.output_dir.endswith("_v12") and config.training.total_updates == 6000
    v11 = load_config(V11)
    assert config.training.demo_path == v11.training.demo_path
    assert environment.robot.count == v11.environment.robot.count == 16
    assert environment.reward == v11.environment.reward


def test_stage_9_starts_from_a_spanning_4_robot_row_that_needs_reinforcement() -> None:
    for seed in range(3):
        env = v12_env(stage=9, seed=seed)
        row = env.robots[:4]
        assert row[0].anchor_partner == LEFT_BANK
        assert [robot.anchor_partner for robot in row[1:]] == [0, 1, 2]
        assert not any(robot.absent for robot in row)
        assert env._structure_spans() and 0.0 < env.current_capacity < env.target_load
        _, _, terminated, _, info = env.step(np.full(len(env.robots), int(DiscreteAction.IDLE)))
        assert not terminated and info["fallen_robots"] == 0


def test_teacher_reinforces_the_prepared_4_robot_row() -> None:
    env = v12_env(stage=9, seed=31)
    teacher = ScriptedTeacher(env, reinforcement="splice")
    info: dict = {}
    for _ in range(env.config.max_steps):
        actions, _ = teacher.actions()
        _, _, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break
    assert info["success"] and env.current_capacity >= 18.0


def test_a_v11_stage_8_checkpoint_resumes_at_the_new_2p3_span_stage(tmp_path) -> None:
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

    v11 = Trainer(small(V11, str(tmp_path / "v11")))
    v11.curriculum.stage = 8
    path = v11.save_checkpoint(4500)
    v12 = Trainer(small(V12, str(tmp_path / "v12")))
    assert v12.load_checkpoint(path) == 4500 and v12.curriculum.stage == 8
    env = BridgeEnv(v12.config.environment)
    env.config.curriculum_stage = 8
    env.reset(seed=0)
    assert env.field.gap_width == 2.3 and env.target_load == 1.0
    history = v12.train(updates=4501)
    assert history[-1]["curriculum_stage"] == 8.0
