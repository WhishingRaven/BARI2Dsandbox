from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class RobotConfig:
    count: int = 20
    length: float = 0.9
    width: float = 0.42
    mass: float = 1.0
    max_speed: float = 0.6
    max_steering_deg: float = 20.0
    turn_rate: float = 1.2
    anchor_range: float = 1.05
    climb_height: float = 0.35
    max_layer: int = 2


@dataclass
class SensorConfig:
    ir_range: float = 3.0
    ir_angles_deg: list[float] = field(default_factory=lambda: [0.0])
    ir_step: float = 0.025
    history: int = 4
    sensor_noise: float = 0.0


@dataclass
class ContactConfig:
    contact_stiffness: float = 80.0
    contact_damping: float = 2.0
    friction: float = 0.75
    contact_capacity: float = 5.0
    bank_capacity: float = 12.0
    anchor_tension_limit: float = 12.0
    anchor_compression_limit: float = 12.0
    anchor_shear_limit: float = 8.0
    anchor_rotational_stiffness: float = 0.0
    anchor_failure_probability: float = 0.0


@dataclass
class GapConfig:
    field_length: float = 18.0
    field_width: float = 10.0
    width_min: float = 3.0
    width_max: float = 5.0
    orientation_deg: float = 0.0
    orientation_jitter_deg: float = 0.0
    irregularity: float = 0.0
    irregularity_frequency: float = 0.8


@dataclass
class LoadConfig:
    target_min: float = 3.0
    target_max: float = 10.0
    max_target: float = 15.0
    fast_mode: bool = True
    load_step: float = 0.25
    displacement_limit: float = 2.0
    load_protocol: str = "uniform"


@dataclass
class RewardConfig:
    span_delta: float = 2.0
    mechanical_delta: float = 2.0
    time_penalty: float = 0.002
    energy_penalty: float = 0.01
    anchor_penalty: float = 0.015
    robot_use_penalty: float = 0.01
    collapse_penalty: float = 1.0
    success_reward: float = 10.0


@dataclass
class EnvironmentConfig:
    max_steps: int = 400
    time_step: float = 0.1
    curriculum_stage: int = 1
    seed: int = 0
    actuator_noise: float = 0.0
    latent_sigma: float = 0.0
    latent_dim: int = 4
    robot: RobotConfig = field(default_factory=RobotConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    gap: GapConfig = field(default_factory=GapConfig)
    load: LoadConfig = field(default_factory=LoadConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class ModelConfig:
    architecture: str = "gru"
    branch_hidden: int = 32
    fused_hidden: int = 64
    recurrent_hidden: int = 64
    film: bool = False
    use_heterogeneity: bool = False


@dataclass
class TrainingConfig:
    total_updates: int = 1000
    rollout_steps: int = 400
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    auxiliary_coef: float = 0.1
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    sequence_minibatch_agents: int = 5
    checkpoint_interval: int = 50
    output_dir: str = "runs/default"
    critic: str = "mlp"
    curriculum_enabled: bool = True
    curriculum_window: int = 25
    curriculum_success_threshold: float = 0.7
    curriculum_max_stage: int = 4


@dataclass
class ExperimentConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _update_dataclass(instance: T, values: dict[str, Any]) -> T:
    names = {item.name for item in fields(instance)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"Unknown configuration keys for {type(instance).__name__}: {sorted(unknown)}")
    for key, value in values.items():
        current = getattr(instance, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"Expected mapping for {key}")
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    config = ExperimentConfig()
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise TypeError("Configuration root must be a mapping")
    return _update_dataclass(config, values)
