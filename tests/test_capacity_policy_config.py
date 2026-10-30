from __future__ import annotations

from pathlib import Path

from bari2d.env.bridge_env import BridgeEnv
from bari2d.models.actor import SharedRecurrentActor
from bari2d.utils.config import load_config


def test_canonical_capacity_policy_config_loads() -> None:
    config = load_config(Path("configs/capacity_local.yaml"))
    environment = BridgeEnv(config.environment)
    actor = SharedRecurrentActor(environment.layout, environment.action_count, config.model)

    assert config.environment.episode_objective == "capacity_horizon"
    assert config.model.architecture == "gru"
    assert config.training.critic == "graph"
    assert config.training.capacity_teacher.enabled
    assert config.training.capacity_teacher.backend == "exact"
    assert config.training.agent_credit_coef == 1.0
    assert not actor.use_heterogeneity
    assert environment.observation_size == 39


def test_old_experiment_yaml_variants_are_removed() -> None:
    yaml_files = sorted(path.name for path in Path("configs").glob("*.yaml"))
    assert yaml_files == ["capacity_local.yaml"]
