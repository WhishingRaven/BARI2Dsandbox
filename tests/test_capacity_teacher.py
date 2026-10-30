from __future__ import annotations

import numpy as np
import torch

from bari2d.capacity_learning.model import CapacityGNN

from bari2d.rl.capacity_teacher import CapacityTeacher
from bari2d.utils.config import CapacityTeacherConfig
from bari2d.utils.scripted import assemble_parallel_bridge


def test_exact_capacity_teacher_scores_a_scripted_bridge(env) -> None:
    assemble_parallel_bridge(env, rows=1, anchor=True)
    teacher = CapacityTeacher(CapacityTeacherConfig(), env.config)
    result = teacher.evaluate(env)

    assert result.backend == "exact"
    assert result.capacity_kg > 0.0
    assert result.terminal_reward > 0.0
    assert result.critical_component in {
        "Axial", "ShearU", "ShearV", "Torsion", "BendingU", "BendingV"
    }
    assert result.agent_credits.shape == (len(env.robots),)
    assert np.any(np.abs(result.agent_credits) > 0.0)


def test_capacity_teacher_penalizes_non_spanning_structure(env) -> None:
    config = CapacityTeacherConfig(no_span_penalty=-3.5)
    teacher = CapacityTeacher(config, env.config)
    result = teacher.evaluate(env)

    assert result.capacity_kg == 0.0
    assert result.terminal_reward == -3.5
    assert np.allclose(result.agent_credits, 0.0)


def test_geometry_only_gnn_teacher_uses_capacity_checkpoint(env, tmp_path) -> None:
    assemble_parallel_bridge(env, rows=1, anchor=True)
    model = CapacityGNN(node_size=11, edge_size=30, hidden_size=16, layers=2)
    checkpoint = tmp_path / "capacity_gnn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_size": 11,
            "edge_size": 30,
            "hidden_size": 16,
            "layers": 2,
            "feature_mode": "geometry_only",
            "payload_mode": "center",
            "wrench_frame": "contact_local",
            "schema_version": 2,
        },
        checkpoint,
    )
    config = CapacityTeacherConfig(
        backend="gnn",
        checkpoint=str(checkpoint),
        counterfactual_credit=False,
    )
    teacher = CapacityTeacher(config, env.config)
    result = teacher.evaluate(env)
    assert result.backend == "gnn"
    assert np.isfinite(result.capacity_kg)
    assert result.capacity_kg >= 0.0
