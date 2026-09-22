from __future__ import annotations

from pathlib import Path

import pytest

from bari2d.env.bridge_env import BridgeEnv
from bari2d.models.actor import SharedRecurrentActor
from bari2d.utils.config import load_config


CASES = (
    ("baseline.yaml", "gru", "mlp", False),
    ("mlp.yaml", "mlp", "mlp", False),
    ("gru_traffic.yaml", "gru_traffic", "mlp", False),
    ("gru_traffic_connectivity.yaml", "gru_traffic_connectivity", "mlp", False),
    ("bio.yaml", "bio", "graph", False),
    ("bio_film.yaml", "bio_film", "graph", False),
    ("bio_heterogeneity.yaml", "bio_heterogeneity", "graph", True),
    ("curriculum.yaml", "gru", "mlp", False),
)


@pytest.mark.parametrize(("filename", "architecture", "critic", "heterogeneous"), CASES)
def test_experiment_config_is_loadable_and_builds_its_actor(
    filename: str, architecture: str, critic: str, heterogeneous: bool
) -> None:
    config = load_config(Path("configs") / filename)
    environment = BridgeEnv(config.environment)
    actor = SharedRecurrentActor(environment.layout, environment.action_count, config.model)

    assert config.model.architecture == architecture
    assert config.training.critic == critic
    assert actor.use_heterogeneity is heterogeneous
    assert environment.observation_size == 39
