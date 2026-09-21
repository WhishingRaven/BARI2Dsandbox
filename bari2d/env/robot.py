from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from bari2d.utils.config import RobotConfig


class DiscreteAction(IntEnum):
    FORWARD = 0
    BACKWARD = 1
    FORWARD_LEFT = 2
    FORWARD_RIGHT = 3
    BACKWARD_LEFT = 4
    BACKWARD_RIGHT = 5
    CLIMB = 6
    ANCHOR = 7
    RELEASE = 8
    IDLE = 9


ACTION_COUNT = len(DiscreteAction)


@dataclass
class RobotState:
    robot_id: int
    position: np.ndarray
    theta: float
    velocity: float = 0.0
    steering: float = 0.0
    head_lifted: bool = False
    anchored: bool = False
    strain: float = 0.0
    previous_action: int = int(DiscreteAction.IDLE)
    layer: int = 0
    fallen: bool = False
    energy: float = 0.0
    anchor_partner: int | str | None = None
    recurrent_hidden: np.ndarray | None = None
    latent: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def corners(self, config: RobotConfig) -> np.ndarray:
        half = np.array([config.length / 2.0, config.width / 2.0])
        local = np.array(
            [[-half[0], -half[1]], [half[0], -half[1]], [half[0], half[1]], [-half[0], half[1]]]
        )
        cosine, sine = np.cos(self.theta), np.sin(self.theta)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        return local @ rotation.T + self.position

    @property
    def heading(self) -> np.ndarray:
        return np.array([np.cos(self.theta), np.sin(self.theta)])

    def step_kinematics(
        self,
        action: int,
        config: RobotConfig,
        dt: float,
        rng: np.random.Generator,
        actuator_noise: float = 0.0,
    ) -> None:
        self.previous_action = int(action)
        speed_command = 0.0
        steering_command = 0.0
        if action in (DiscreteAction.FORWARD, DiscreteAction.FORWARD_LEFT, DiscreteAction.FORWARD_RIGHT):
            speed_command = 1.0
        elif action in (DiscreteAction.BACKWARD, DiscreteAction.BACKWARD_LEFT, DiscreteAction.BACKWARD_RIGHT):
            speed_command = -1.0
        if action in (DiscreteAction.FORWARD_LEFT, DiscreteAction.BACKWARD_LEFT):
            steering_command = 1.0
        elif action in (DiscreteAction.FORWARD_RIGHT, DiscreteAction.BACKWARD_RIGHT):
            steering_command = -1.0

        speed_command += float(rng.normal(0.0, actuator_noise))
        steering_command += float(rng.normal(0.0, actuator_noise))
        speed_command = float(np.clip(speed_command, -1.0, 1.0))
        steering_command = float(np.clip(steering_command, -1.0, 1.0))
        self.velocity = speed_command * config.max_speed
        self.steering = np.deg2rad(config.max_steering_deg) * steering_command
        self.theta = float((self.theta + config.turn_rate * steering_command * dt + np.pi) % (2 * np.pi) - np.pi)
        displacement = self.heading * self.velocity * dt
        self.position = self.position + displacement
        self.energy += abs(self.velocity) * dt + 0.1 * abs(self.steering) * dt

