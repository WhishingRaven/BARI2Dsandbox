from __future__ import annotations

import numpy as np
import torch

from bari2d.bridge_generation import RandomBridgeConfig, RandomBridgeGenerator
from bari2d.capacity_learning import (
    CapacityDatasetBuilder,
    exact_capacity_from_mechanics_features,
    CapacityGNN,
    collate_capacity_graphs,
    load_capacity_dataset,
    save_capacity_dataset,
)
from bari2d.capacity_learning.training import capacity_multitask_loss


def test_random_bridge_generator_always_spans_banks() -> None:
    generator = RandomBridgeGenerator(
        config=RandomBridgeConfig(rows_range=(1, 2), stack_probability=0.3),
        seed=4,
    )
    for _ in range(8):
        bridge = generator.generate()
        assert bridge.graph.spans
        assert bridge.robot_count >= 3
        assert bridge.edge_count >= bridge.robot_count - 1
        assert bridge.payload_candidates


def test_capacity_graph_sample_contains_physics_and_failure_labels() -> None:
    builder = CapacityDatasetBuilder(seed=5, feature_mode="mechanics_informed")
    sample = builder.generate_sample()

    assert sample.node_features.shape[1] == len(builder.node_feature_names) == 11
    assert sample.edge_features.shape[1] == len(builder.edge_feature_names) == 42
    assert sample.edge_index.shape == (2, sample.edge_count)
    assert np.isfinite(float(sample.target_capacity_kg))
    assert float(sample.target_capacity_kg) > 0.0
    # Every physical connection is represented in both directed message-passing directions.
    assert int(sample.critical_edge_mask.sum().item()) == 2
    assert sample.edge_capacity_kg.shape == (sample.edge_count,)
    assert torch.all(sample.edge_capacity_kg >= 0.0)
    assert 0 <= int(sample.critical_component.item()) < 6


def test_geometry_only_mode_omits_solved_wrench_features() -> None:
    builder = CapacityDatasetBuilder(seed=6, feature_mode="geometry_only")
    sample = builder.generate_sample()
    assert sample.edge_features.shape[1] == len(builder.BASE_EDGE_FEATURE_NAMES) == 30


def test_capacity_gnn_batches_variable_graphs_and_backpropagates() -> None:
    builder = CapacityDatasetBuilder(seed=7)
    samples = builder.generate(3)
    batch = collate_capacity_graphs(samples)
    model = CapacityGNN(
        node_size=batch["node_features"].shape[1],
        edge_size=batch["edge_features"].shape[1],
        hidden_size=32,
        layers=2,
        dropout=0.0,
    )
    output = model(
        batch["node_features"],
        batch["edge_index"],
        batch["edge_features"],
        batch["node_graph"],
    )
    assert output.log_capacity.shape == (3,)
    assert output.critical_component_logits.shape == (3, 6)
    assert output.critical_edge_logits.shape == (batch["edge_features"].shape[0],)

    loss = capacity_multitask_loss(output, batch)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_capacity_dataset_round_trip(tmp_path) -> None:
    builder = CapacityDatasetBuilder(seed=8)
    samples = builder.generate(2)
    path = save_capacity_dataset(tmp_path / "capacity.pt", samples, metadata={"mode": "test"})
    restored, metadata = load_capacity_dataset(path)
    assert metadata["mode"] == "test"
    assert len(restored) == 2
    torch.testing.assert_close(restored[0].node_features, samples[0].node_features)
    torch.testing.assert_close(restored[0].edge_features, samples[0].edge_features)
    torch.testing.assert_close(restored[0].target_capacity_kg, samples[0].target_capacity_kg)
    torch.testing.assert_close(restored[0].edge_capacity_kg, samples[0].edge_capacity_kg)


def test_mechanics_features_recover_exact_statics_capacity() -> None:
    builder = CapacityDatasetBuilder(seed=9, feature_mode="mechanics_informed")
    for _ in range(4):
        sample = builder.generate_sample()
        recovered = exact_capacity_from_mechanics_features(sample)
        np.testing.assert_allclose(recovered, float(sample.target_capacity_kg), rtol=2.0e-5, atol=2.0e-5)
