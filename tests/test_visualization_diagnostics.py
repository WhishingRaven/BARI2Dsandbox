from __future__ import annotations

import numpy as np

from bari2d.mechanics import LocalWrenchLimit, bridge_center_payload_3d
from bari2d.utils.scripted import assemble_parallel_bridge
from bari2d.utils.visualization import contact_utilization_at_payload


def test_capacity_utilization_marks_the_critical_contact(env) -> None:
    assemble_parallel_bridge(env, rows=1, anchor=True)
    limits = LocalWrenchLimit(
        axial=200.0,
        shear_u=120.0,
        shear_v=120.0,
        torsion=40.0,
        bending_u=80.0,
        bending_v=80.0,
    )
    payload = bridge_center_payload_3d(env.robots, env.graph, env.field, env.config.robot)
    result = env.local_spatial_payload_capacity(payload, limits)
    assert np.isfinite(result.max_payload_kg)
    assert result.critical_contact is not None
    utilization = contact_utilization_at_payload(result, limits)
    critical_key = frozenset(result.critical_contact)
    assert critical_key in utilization
    assert utilization[critical_key] >= 1.0 - 1.0e-6
