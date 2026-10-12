"""v10: v9 continued from its last checkpoint with a stronger fall penalty."""

from __future__ import annotations

import numpy as np
import pytest

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.robot import DiscreteAction
from bari2d.rl.trainer import Trainer
from bari2d.utils.config import load_config

V9 = "configs/beacon_structured_mappo_static_v9.yaml"
V10 = "configs/beacon_structured_mappo_static_v10.yaml"


def test_v10_only_changes_the_fall_penalty_and_the_run() -> None:
    v9, v10 = load_config(V9).to_dict(), load_config(V10).to_dict()
    assert v10["environment"]["reward"]["collapse_penalty"] == 2.0
    assert v9["environment"]["reward"]["collapse_penalty"] == 0.5
    assert v10["training"]["total_updates"] == 3000
    assert v10["training"]["output_dir"].endswith("_v10")
    assert v10["training"]["demo_path"] == v9["training"]["demo_path"]
    for section in ("environment", "model", "training"):
        for key in set(v9[section]) | set(v10[section]):
            if (section, key) in {("environment", "reward"), ("training", "total_updates"), ("training", "output_dir")}:
                continue
            assert v9[section].get(key) == v10[section].get(key), (section, key)
    rewards = {k: v for k, v in v10["environment"]["reward"].items() if k != "collapse_penalty"}
    assert rewards == {k: v for k, v in v9["environment"]["reward"].items() if k != "collapse_penalty"}


def test_a_fall_costs_the_new_penalty_in_the_step_it_happens() -> None:
    config = load_config(V10).environment
    config.curriculum_stage = 7
    env = BridgeEnv(config)
    env.reset(seed=0)
    robot = env.robots[-1]
    robot.position = env.field.center.copy()  # free robot in the middle of the gap
    robot.layer, robot.anchored, robot.anchor_partner = 0, False, None
    _, _, _, _, info = env.step(np.full(len(env.robots), int(DiscreteAction.IDLE)))
    assert robot.fallen
    fallen = sum(r.fallen for r in env.robots)
    assert info["reward_components"]["collapse"] == pytest.approx(-2.0 * fallen)


def _small(path: str, output_dir: str):
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


def test_v10_resumes_a_v9_checkpoint_at_its_update_and_stage(tmp_path) -> None:
    v9 = Trainer(_small(V9, str(tmp_path / "v9")))
    v9.curriculum.stage = 7
    path = v9.save_checkpoint(1500)

    v10 = Trainer(_small(V10, str(tmp_path / "v10")))
    assert v10.load_checkpoint(path) == 1500
    assert v10.curriculum.stage == 7
    history = v10.train(updates=1501)
    assert len(history) == 1
    assert history[-1]["curriculum_stage"] == 7.0
