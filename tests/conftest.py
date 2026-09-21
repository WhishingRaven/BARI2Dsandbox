from __future__ import annotations

import pytest

from bari2d.env.bridge_env import BridgeEnv
from bari2d.utils.config import EnvironmentConfig


@pytest.fixture
def environment_config() -> EnvironmentConfig:
    config = EnvironmentConfig()
    config.robot.count = 20
    config.max_steps = 20
    config.seed = 13
    config.sensor.sensor_noise = 0.0
    config.actuator_noise = 0.0
    return config


@pytest.fixture
def env(environment_config: EnvironmentConfig) -> BridgeEnv:
    environment = BridgeEnv(environment_config)
    environment.reset(seed=13)
    return environment

