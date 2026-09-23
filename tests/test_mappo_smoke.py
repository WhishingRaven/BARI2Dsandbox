from __future__ import annotations

import json

from bari2d.rl.trainer import Trainer
from bari2d.utils.config import ExperimentConfig


def test_recurrent_mappo_completes_one_update(tmp_path, capsys) -> None:
    config = ExperimentConfig()
    config.environment.robot.count = 4
    config.environment.max_steps = 4
    config.model.architecture = "gru"
    config.model.fused_hidden = 16
    config.model.recurrent_hidden = 16
    config.training.rollout_steps = 4
    config.training.ppo_epochs = 1
    config.training.sequence_minibatch_agents = 2
    config.training.checkpoint_interval = 1
    config.training.output_dir = str(tmp_path)
    trainer = Trainer(config)
    history = trainer.train(updates=1)
    output = capsys.readouterr().out
    assert len(history) == 1
    assert "Update 1/1" in output
    assert "actor_loss=" in output
    assert (tmp_path / "checkpoint_000001.pt").exists()
    episode_path = tmp_path / "episodes.jsonl"
    assert episode_path.exists()
    episode = json.loads(episode_path.read_text(encoding="utf-8").splitlines()[0])
    assert "episode_reward" in episode
    assert "gap_progress" in episode["reward_component_totals"]
    assert len(episode["agent_rewards"]) == config.environment.robot.count
