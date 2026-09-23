from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import torch

from bari2d.env.bridge_env import BridgeEnv
from bari2d.models.actor import SharedRecurrentActor
from bari2d.utils.config import ExperimentConfig, config_from_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.infer import InferenceSession, InferenceViewer, discover_checkpoints, load_inference_model


def _checkpoint(path: Path) -> Path:
    config = ExperimentConfig()
    config.environment.robot.count = 4
    config.environment.max_steps = 4
    config.model.fused_hidden = 16
    config.model.recurrent_hidden = 16
    env = BridgeEnv(config.environment)
    actor = SharedRecurrentActor(env.layout, env.action_count, config.model)
    torch.save({"algorithm": {"actor": actor.state_dict()}, "experiment_config": config.to_dict()}, path)
    return path


def test_checkpoint_discovery_deduplicates_explicit_and_directory_paths(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt")
    found = discover_checkpoints([checkpoint], tmp_path)
    assert found == [checkpoint.resolve()]


def test_inference_session_runs_policy_only_step(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "model.pt")
    loaded = load_inference_model(checkpoint, None, torch.device("cpu"))
    session = InferenceSession(loaded, torch.device("cpu"), seed=3)
    assert session.preview_prediction is not None
    assert session.preview_prediction.actions.shape == (4,)
    session.step()
    assert session.env.step_count == 1
    assert session.last_prediction is not None


def test_inference_reset_advances_to_a_new_randomized_episode(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "model.pt")
    loaded = load_inference_model(checkpoint, None, torch.device("cpu"))
    session = InferenceSession(loaded, torch.device("cpu"), seed=3)
    first_positions = np.stack([robot.position.copy() for robot in session.env.robots])

    session.reset()

    second_positions = np.stack([robot.position.copy() for robot in session.env.robots])
    assert not np.allclose(first_positions, second_positions)


def test_viewer_refresh_draws_frame_immediately(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "model.pt")
    viewer = InferenceViewer(
        [checkpoint],
        fallback_config=None,
        device=torch.device("cpu"),
        seed=3,
        stage=None,
        target_load=None,
        deterministic=False,
        interval_ms=150,
    )
    viewer.figure.canvas.draw = Mock()
    viewer.figure.canvas.draw_idle = Mock()

    viewer.refresh()

    viewer.figure.canvas.draw.assert_called_once_with()
    viewer.figure.canvas.draw_idle.assert_not_called()
    plt.close(viewer.figure)


def test_legacy_checkpoint_config_keeps_its_original_sensor_layout() -> None:
    serialized = ExperimentConfig().to_dict()
    serialized["environment"]["sensor"].pop("downward_ir_enabled")
    serialized["environment"]["sensor"]["ir_angles_deg"] = [0.0]

    restored = config_from_dict(serialized)
    environment = BridgeEnv(restored.environment)

    assert not restored.environment.sensor.downward_ir_enabled
    assert environment.observation_size == 23
