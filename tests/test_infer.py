from __future__ import annotations

from pathlib import Path
import sys

import torch

from bari2d.env.bridge_env import BridgeEnv
from bari2d.models.actor import SharedRecurrentActor
from bari2d.utils.config import ExperimentConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.infer import InferenceSession, discover_checkpoints, load_inference_model


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
