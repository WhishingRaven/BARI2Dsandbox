from __future__ import annotations

import numpy as np

from bari2d.env.field import LEFT_BANK, RIGHT_BANK, GapGenerator
from bari2d.utils.config import GapConfig


def test_stage_one_gap_is_straight_and_feasible() -> None:
    config = GapConfig(width_min=3.0, width_max=5.0, irregularity=0.4)
    field = GapGenerator(config).generate(np.random.default_rng(2), stage=1)
    assert field.gap_width == 3.0
    assert field.irregularity == 0.0
    assert field.gap_width < field.length * 0.5
    assert field.bank_at(np.array([1.0, 5.0])) == LEFT_BANK
    assert field.bank_at(np.array([17.0, 5.0])) == RIGHT_BANK
    assert field.is_gap(field.center)


def test_later_curriculum_randomizes_geometry() -> None:
    config = GapConfig(
        width_min=3.0,
        width_max=5.0,
        orientation_jitter_deg=20.0,
        irregularity=0.35,
    )
    generator = GapGenerator(config)
    fields = [generator.generate(np.random.default_rng(seed), stage=3) for seed in range(5)]
    assert len({round(field.gap_width, 4) for field in fields}) > 1
    assert len({round(field.orientation, 4) for field in fields}) > 1
    assert all(field.irregularity == 0.35 for field in fields)

