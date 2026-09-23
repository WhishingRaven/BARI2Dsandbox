from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bari2d.utils.config import GapConfig


LEFT_BANK = "left_bank"
RIGHT_BANK = "right_bank"


@dataclass
class GapField:
    length: float
    width: float
    gap_width: float
    center: np.ndarray
    orientation: float
    irregularity: float = 0.0
    irregularity_frequency: float = 0.8
    phase: float = 0.0
    obstacles: tuple[tuple[float, float, float], ...] = ()

    @property
    def normal(self) -> np.ndarray:
        return np.array([np.cos(self.orientation), np.sin(self.orientation)])

    @property
    def tangent(self) -> np.ndarray:
        return np.array([-np.sin(self.orientation), np.cos(self.orientation)])

    def coordinates(self, point: np.ndarray) -> tuple[float, float]:
        relative = np.asarray(point, dtype=float) - self.center
        return float(relative @ self.normal), float(relative @ self.tangent)

    def boundaries(self, tangent_coordinate: float) -> tuple[float, float]:
        perturbation = self.irregularity * np.sin(
            self.irregularity_frequency * tangent_coordinate + self.phase
        )
        return -self.gap_width / 2.0 + perturbation, self.gap_width / 2.0 + perturbation

    def bank_at(self, point: np.ndarray) -> str | None:
        x, y = np.asarray(point, dtype=float)
        if x < 0.0 or x > self.length or y < 0.0 or y > self.width:
            return None
        longitudinal, tangent = self.coordinates(point)
        left, right = self.boundaries(tangent)
        if longitudinal <= left:
            return LEFT_BANK
        if longitudinal >= right:
            return RIGHT_BANK
        return None

    def is_gap(self, point: np.ndarray) -> bool:
        x, y = np.asarray(point, dtype=float)
        return 0.0 <= x <= self.length and 0.0 <= y <= self.width and self.bank_at(point) is None

    def inside(self, point: np.ndarray) -> bool:
        x, y = np.asarray(point, dtype=float)
        return 0.0 <= x <= self.length and 0.0 <= y <= self.width

    def bank_contact(self, corners: np.ndarray) -> set[str]:
        return {bank for corner in corners if (bank := self.bank_at(corner)) is not None}

    def normalized_progress(self, point: np.ndarray) -> float:
        longitudinal, tangent = self.coordinates(point)
        left, right = self.boundaries(tangent)
        return float(np.clip((longitudinal - left) / max(right - left, 1.0e-6), 0.0, 1.0))


class GapGenerator:
    def __init__(self, config: GapConfig):
        self.config = config

    def generate(self, rng: np.random.Generator, stage: int = 1) -> GapField:
        config = self.config
        difficulty = float(np.clip(stage, 1, 4)) / 4.0
        width_max = config.width_min + difficulty * (config.width_max - config.width_min)
        gap_width = (
            float(rng.uniform(config.width_min, width_max))
            if width_max > config.width_min
            else config.width_min
        )
        orientation_jitter = difficulty * config.orientation_jitter_deg
        orientation = np.deg2rad(
            config.orientation_deg + rng.uniform(-orientation_jitter, orientation_jitter)
        )
        irregularity = float(rng.uniform(0.0, difficulty * config.irregularity))
        max_feasible = config.field_length * 0.45
        gap_width = min(gap_width, max_feasible)
        return GapField(
            length=config.field_length,
            width=config.field_width,
            gap_width=gap_width,
            center=np.array([config.field_length / 2.0, config.field_width / 2.0]),
            orientation=orientation,
            irregularity=irregularity,
            irregularity_frequency=config.irregularity_frequency,
            phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
