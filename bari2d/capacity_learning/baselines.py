from __future__ import annotations

from math import inf

import numpy as np

from bari2d.capacity_learning.dataset import CapacityGraphSample


def exact_capacity_from_mechanics_features(sample: CapacityGraphSample) -> float:
    """Recover the exact capacity encoded by a mechanics-informed sample.

    The final 12 edge features are [w_base / w_max, w_1kg / w_max]. For every
    component the failure condition is |a + m b| = 1. Taking the minimum
    nonnegative solution across directed edges/components reproduces the statics
    target (directed duplicates are harmless).

    This is intentionally a diagnostic baseline: if a mechanics-informed GNN
    cannot approach this result, the learning pipeline/model is at fault. It is
    not a learned model and it is not available in geometry-only mode.
    """

    if sample.metadata.get("feature_mode") != "mechanics_informed":
        raise ValueError("Sample does not contain mechanics-informed wrench features")
    values = sample.edge_features.detach().cpu().numpy()
    base = values[:, -12:-6]
    unit = values[:, -6:]
    best = inf
    for a_row, b_row in zip(base, unit):
        for a, b in zip(a_row, b_row):
            if abs(a) > 1.0 + 1.0e-8:
                candidate = 0.0
            elif abs(b) <= 1.0e-12:
                candidate = inf
            elif b > 0.0:
                candidate = (1.0 - a) / b
            else:
                candidate = (-1.0 - a) / b
            if candidate >= -1.0e-7:
                best = min(best, max(float(candidate), 0.0))
    return float(best)
