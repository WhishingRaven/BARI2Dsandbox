from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


BEACON_FEATURE_SIZE = 6


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
    max_layer: int = 10


@dataclass
class SensorConfig:
    ir_range: float = 3.0
    ir_angles_deg: list[float] = field(default_factory=lambda: [0.0, 180.0, 90.0, -90.0])
    downward_ir_enabled: bool = True
    ir_step: float = 0.025
    history: int = 4
    sensor_noise: float = 0.0
    # Two local contact cues appended to each observation (off by default so
    # existing checkpoints keep their input size): the robot directly ahead is
    # anchored (touch / short-range signal), and the robot beneath is anchored.
    structure_cues: bool = False


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
    # Optional training constraint: a bank anchor needs to be near its cliff
    # edge, while a robot-to-robot anchor needs a real physical contact.
    anchor_requires_edge_or_contact: bool = False
    anchor_edge_band: float = 1.2


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
    # When supplied, one deterministic gap width is used for each curriculum
    # stage (the final value is reused for later stages).
    curriculum_widths: list[float] = field(default_factory=list)


@dataclass
class LoadConfig:
    target_min: float = 3.0
    target_max: float = 10.0
    max_target: float = 15.0
    fast_mode: bool = True
    load_step: float = 0.25
    displacement_limit: float = 2.0
    load_protocol: str = "uniform"
    # Optional per-stage loads, analogous to GapConfig.curriculum_widths.
    curriculum_targets: list[float] = field(default_factory=list)


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
    beacon_discovery_reward: float = 0.0
    beacon_gather_delta: float = 0.0
    # Applied per non-anchored robot after a local beacon is available and the
    # robot has remained idle longer than the grace period.
    idle_penalty: float = 0.0
    idle_grace_steps: int = 5
    # How span progress (the ``span`` reward potential) is measured:
    # ``center``   - farthest robot center connected to the left bank (original);
    # ``frontier`` - farthest robot front end among robots supported from the
    #                left bank (load-bearing structure under static physics), so a
    #                robot gripping the edge or walking out on the chain already counts.
    # ``anchored_frontier`` - like ``frontier`` but only for robots held by a
    #                structure that grips the left bank (bank anchor, then anchors
    #                and robots standing on it); a robot merely nosing over the
    #                edge earns nothing until it grips.
    progress_measure: str = "center"
    # For the frontier measures, progress starts counting this far behind the
    # left cliff edge (0 = at the edge), so a grip close to the edge already
    # earns part of the span reward and a closer grip earns more.
    progress_baseline: float = 0.0
    # Potential reward for robots standing on an anchored robot (at most
    # ``climb_cap`` count); stepping down again gives it back.
    climb_delta: float = 0.0
    climb_cap: int = 1


@dataclass
class BeaconConfig:
    """Local, multi-hop cliff-beacon communication; no global actor input."""

    enabled: bool = False
    communication_range: float = 2.5
    max_hops: int = 4


@dataclass
class PhysicsConfig:
    """Structural model selection.

    ``legacy`` keeps the original BARI graph-flow physics unchanged.  ``static``
    adds gravity and rigid-body equilibrium (bari2d/env/statics.py): a robot
    whose weight is not supported tips over, plain side contact carries no
    weight, anchors carry vertical force and bending/torsion couples up to the
    limits below, and load capacity is the largest equilibrium load.  Forces are
    in robot weights (mass * gravity) and moments in weight * length units.
    """

    model: str = "legacy"
    gravity: float = 1.0
    # Vertical shear capacity of an anchor; ``None`` reuses contact.anchor_shear_limit.
    anchor_vertical_limit: float | None = None
    anchor_bending_limit: float = 5.0
    anchor_torsion_limit: float = 5.0
    # Gripping while standing on the front half of an aligned robot steps off
    # its front end and attaches end-to-end (walk over the bridge, extend it).
    extend_on_anchor: bool = True
    extend_alignment: float = 0.8
    # Robots up to this layer also rest on the ground where their footprint is
    # over a bank.  Layer 1 covers the tilted climbing pose (rear wheels on the
    # ground, front on another robot) that the 2.5D layers cannot draw.
    ground_contact_max_layer: int = 1


@dataclass
class EnvironmentConfig:
    max_steps: int = 1000
    time_step: float = 0.1
    curriculum_stage: int = 1
    seed: int = 0
    # Per curriculum stage, the farthest a robot center may start from the
    # left cliff edge (the last value is reused; <= 0 means anywhere on the bank).
    spawn_edge_distance: list[float] = field(default_factory=list)
    # Per curriculum stage, a prepared start (the last value is reused):
    # "none", "root_only" (robot 0 grips the bank at the cliff edge),
    # "walker_behind_root" (robot 1 also waits right behind it, facing it) or
    # "walker_on_root" (robot 1 stands on the rear half of the gripping robot).
    # A training scaffold for learning the late steps first; evaluate on "none".
    start_setups: list[str] = field(default_factory=list)
    # First curriculum stage with physical domain randomization (friction,
    # mass, anchor strength, noise).
    randomization_stage: int = 4
    actuator_noise: float = 0.0
    latent_sigma: float = 0.0
    latent_dim: int = 4
    robot: RobotConfig = field(default_factory=RobotConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    gap: GapConfig = field(default_factory=GapConfig)
    load: LoadConfig = field(default_factory=LoadConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    beacon: BeaconConfig = field(default_factory=BeaconConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)


@dataclass
class ModelConfig:
    architecture: str = "structured_mappo"
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
    # A time-limit cut is not a real ending: add gamma * V(final state) to the
    # last reward instead of treating the cut as terminal in GAE.
    bootstrap_truncation: bool = True


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


def config_from_dict(values: dict[str, Any]) -> ExperimentConfig:
    """Build an experiment configuration embedded in a checkpoint."""
    if not isinstance(values, dict):
        raise TypeError("Checkpoint configuration must be a mapping")
    # Checkpoints created before the downward sensor was introduced must retain
    # their original observation layout so their actor input layer still fits.
    restored = dict(values)
    environment = restored.get("environment")
    if isinstance(environment, dict) and isinstance(environment.get("sensor"), dict):
        restored_environment = dict(environment)
        restored_sensor = dict(restored_environment["sensor"])
        restored_sensor.setdefault("downward_ir_enabled", False)
        restored_environment["sensor"] = restored_sensor
        restored["environment"] = restored_environment
    return _update_dataclass(ExperimentConfig(), restored)
