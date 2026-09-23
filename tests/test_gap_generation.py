from __future__ import annotations

import numpy as np

from bari2d.env.field import LEFT_BANK, RIGHT_BANK, GapGenerator
from bari2d.utils.config import GapConfig


def test_stage_one_gap_is_randomized_within_easy_bounds() -> None:
    config = GapConfig(
        width_min=3.0,
        width_max=5.0,
        orientation_jitter_deg=20.0,
        irregularity=0.4,
    )
    fields = [GapGenerator(config).generate(np.random.default_rng(seed), stage=1) for seed in range(5)]

    assert len({round(field.gap_width, 4) for field in fields}) > 1
    assert len({round(field.orientation, 4) for field in fields}) > 1
    assert len({round(field.irregularity, 4) for field in fields}) > 1
    assert all(3.0 <= field.gap_width <= 3.5 for field in fields)
    assert all(abs(np.rad2deg(field.orientation)) <= 5.0 for field in fields)
    assert all(0.0 <= field.irregularity <= 0.1 for field in fields)
    assert all(field.gap_width < field.length * 0.5 for field in fields)
    assert all(field.bank_at(np.array([1.0, 5.0])) == LEFT_BANK for field in fields)
    assert all(field.bank_at(np.array([17.0, 5.0])) == RIGHT_BANK for field in fields)
    assert all(field.is_gap(field.center) for field in fields)


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
    assert len({round(field.irregularity, 4) for field in fields}) > 1
    assert all(0.0 <= field.irregularity <= 0.35 * 0.75 for field in fields)
