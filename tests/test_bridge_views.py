from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')

from bari2d.capacity_learning.dataset import CapacityDatasetBuilder
from bari2d.capacity_learning.bridge_views import draw_bridge_topdown, save_paired_view, robot_layers, _layer_roster


def test_topdown_snapshot_matches_payload_and_contacts(tmp_path: Path) -> None:
    builder = CapacityDatasetBuilder(seed=23, feature_mode="mechanics_informed")
    sample = builder.generate_sample()
    view = sample.metadata["topdown_view"]
    assert view["robot_length_m"] == builder.robot_config.length
    assert view["robot_width_m"] == builder.robot_config.width
    assert np.isfinite(view["payload_xy"]).all()
    assert len(view["contacts"]) == sample.edge_count // 2
    actual_layers = robot_layers(sample)
    assert actual_layers == {int(k): int(v) for k, v in view["robot_layers"].items()}
    assert set(actual_layers.values()).issubset({0, 1, 2})
    assert len([i for i in actual_layers]) == int(sample.metadata["robot_count"])
    assert _layer_roster(sample)
    pairs = {tuple(sorted([e['source'], e['target']])) for e in view['contacts']}
    assert len(pairs) == sample.edge_count // 2
    png = save_paired_view(sample, tmp_path / "paired.png")
    assert png.exists() and png.stat().st_size > 5000


def test_old_sample_without_new_snapshot_still_renders(tmp_path: Path) -> None:
    sample = CapacityDatasetBuilder(seed=24).generate_sample()
    expected_layers = robot_layers(sample)
    sample.metadata.pop("topdown_view")
    assert robot_layers(sample) == expected_layers
    png = save_paired_view(sample, tmp_path / "legacy.png")
    assert png.exists() and png.stat().st_size > 5000


def test_layer_labels_include_actual_stacks(tmp_path: Path) -> None:
    builder = CapacityDatasetBuilder(seed=47)
    sample = builder.generate_sample()
    # Verify labels use actual integer layer, not normalized feature 8.
    features = sample.node_features.detach().cpu().numpy()
    for i, layer in robot_layers(sample).items():
        assert int(round(float(features[i, 5]) - 0.5)) == layer
    assert save_paired_view(sample, tmp_path / "actual_layers.png").exists()
