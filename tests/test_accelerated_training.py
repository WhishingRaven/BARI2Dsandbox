"""Sequence GRU, richer connectivity branch, agent-conditioned critic, parallel
rollouts and demonstration-guided (behaviour-cloning) training."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bari2d.env.bridge_env import BridgeEnv
from bari2d.models.actor import SharedRecurrentActor
from bari2d.models.graph_critic import GraphCritic
from bari2d.rl.demonstrations import DemonstrationSet, record_episode
from bari2d.rl.rollout_buffer import RolloutBuffer
from bari2d.rl.teacher import ScriptedTeacher
from bari2d.rl.trainer import CurriculumScheduler, Trainer
from bari2d.rl.vec_env import EnvRunner, SubprocVectorEnv, SyncVectorEnv, environment_configs
from bari2d.utils.config import ModelConfig, load_config

V5 = "configs/beacon_structured_mappo_static_v5.yaml"


def v5_env(stage: int = 1, seed: int = 3) -> BridgeEnv:
    config = load_config(V5).environment
    config.curriculum_stage = stage
    env = BridgeEnv(config)
    env.reset(seed=seed)
    return env


def rollout(env: BridgeEnv, steps: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    observations, masks, actions = [], [], []
    observation = env.observations()
    for _ in range(steps):
        mask = env.action_masks()
        action = np.array([rng.choice(np.flatnonzero(row)) for row in mask])
        observations.append(observation)
        masks.append(mask)
        actions.append(action)
        observation, *_ = env.step(action)
    return torch.as_tensor(np.array(observations)), torch.as_tensor(np.array(masks)), torch.as_tensor(np.array(actions))


def v5_actor(env: BridgeEnv) -> SharedRecurrentActor:
    torch.manual_seed(0)
    return SharedRecurrentActor(env.layout, env.action_count, load_config(V5).model)


def test_sequence_gru_matches_step_by_step_with_resets() -> None:
    env = v5_env()
    actor = v5_actor(env)
    observations, masks, actions = rollout(env, 24)
    resets = torch.zeros(24, 5)
    resets[0] = 1.0
    resets[9, :] = 1.0
    resets[15, 2] = 1.0
    resets[16, 2] = 1.0
    hidden = torch.randn(5, actor.hidden_size) * 0.3
    log_probabilities, entropies, _ = actor.evaluate_sequence(observations, hidden, actions, masks, resets)
    step_hidden = hidden
    for step in range(24):
        output = actor.forward_step(observations[step], step_hidden, masks[step], resets[step])
        step_hidden = output.hidden
        expected = torch.distributions.Categorical(logits=output.logits).log_prob(actions[step])
        assert torch.allclose(log_probabilities[step], expected, atol=1.0e-5)
    assert torch.isfinite(entropies).all()


def test_legacy_grucell_checkpoints_still_load() -> None:
    env = v5_env()
    actor = v5_actor(env)
    state = actor.state_dict()
    legacy = {key.replace("_l0", ""): value for key, value in state.items()}
    assert "recurrent.weight_ih" in legacy
    restored = SharedRecurrentActor(env.layout, env.action_count, load_config(V5).model)
    restored.load_state_dict(legacy)
    for name, value in restored.state_dict().items():
        assert torch.equal(value, state[name])


def test_rich_connectivity_branch_reads_strain_action_and_cues() -> None:
    env = v5_env()
    rich = v5_actor(env)
    plain_config = ModelConfig(**{**load_config(V5).model.__dict__, "connectivity_inputs": "ir"})
    plain = SharedRecurrentActor(env.layout, env.action_count, plain_config)
    channels = env.layout.ray_count + 1 + env.action_count + 2
    assert rich.connectivity_encoder.network[0].in_channels == channels
    assert plain.connectivity_encoder.network[0].in_channels == env.layout.ray_count
    observation = torch.as_tensor(env.observations())
    changed = observation.clone()
    changed[:, env.layout.strain] += 0.5
    changed[:, env.layout.cues] = 1.0 - changed[:, env.layout.cues]
    for actor, should_change in ((rich, True), (plain, False)):
        before = actor.encode(observation)[2]["connectivity"]
        after = actor.encode(changed)[2]["connectivity"]
        assert (not torch.allclose(before, after)) == should_change


def test_agent_conditioned_critic_gives_one_value_per_robot() -> None:
    env = v5_env()
    nodes, adjacency, edges = (torch.as_tensor(item) for item in env.dense_graph_state())
    critic = GraphCritic(agent_conditioned=True)
    single = critic(nodes, adjacency, edges)
    batch = critic(nodes.expand(3, -1, -1), adjacency.expand(3, -1, -1), edges.expand(3, -1, -1, -1))
    assert single.shape == (5,)
    assert batch.shape == (3, 5)
    assert not torch.allclose(single, single[0].expand(5))
    assert GraphCritic()(nodes, adjacency, edges).ndim == 0


def test_per_robot_gae_matches_a_reference_loop() -> None:
    rng = np.random.default_rng(1)
    horizon, envs, robots = 6, 2, 3
    buffer = RolloutBuffer(horizon, envs, robots, observation_size=4, global_state_size=2, action_count=10, hidden_size=3)
    rewards = rng.normal(size=(horizon, envs, robots)).astype(np.float32)
    values = rng.normal(size=(horizon, envs, robots)).astype(np.float32)
    dones = np.zeros((horizon, envs), dtype=np.float32)
    dones[2, 0] = 1.0
    for step in range(horizon):
        buffer.add(
            np.zeros((envs, robots, 4)), np.zeros((envs, 2)), np.zeros((envs, robots), dtype=np.int64),
            np.zeros((envs, robots)), np.ones((envs, robots, 10), dtype=bool), np.zeros((envs, robots)),
            np.zeros((envs, robots, 3)), rewards[step], values[step], dones[step], {},
        )
    last = rng.normal(size=(envs, robots)).astype(np.float32)
    buffer.compute_returns(last, gamma=0.9, gae_lambda=0.8)
    for env_index in range(envs):
        for robot in range(robots):
            advantage, expected = 0.0, np.zeros(horizon)
            for step in reversed(range(horizon)):
                next_value = last[env_index, robot] if step == horizon - 1 else values[step + 1, env_index, robot]
                keep = 1.0 - dones[step, env_index]
                delta = rewards[step, env_index, robot] + 0.9 * next_value * keep - values[step, env_index, robot]
                advantage = delta + 0.9 * 0.8 * keep * advantage
                expected[step] = advantage
            assert np.allclose(buffer.returns[:, env_index, robot], expected + values[:, env_index, robot], atol=1.0e-5)


def small_v5(tmp_path, backend: str = "sync"):
    config = load_config(V5)
    config.environment.max_steps = 12
    config.training.output_dir = str(tmp_path / "run")
    config.training.num_envs = 2
    config.training.vector_env = backend
    config.training.rollout_steps = 8
    config.training.recurrent_chunk = 4
    config.training.num_minibatches = 2
    config.training.ppo_epochs = 1
    config.training.checkpoint_interval = 1
    config.training.demo_path = ""
    config.training.critic_warmup_updates = 0
    return config


def test_subprocess_and_sync_vector_envs_agree(tmp_path) -> None:
    config = small_v5(tmp_path).environment
    configs = environment_configs(config, 2)
    sync = SyncVectorEnv(configs, use_graph=True)
    workers = SubprocVectorEnv(configs, use_graph=True)
    try:
        for first, second in zip(sync.reset(), workers.reset()):
            assert np.allclose(first["observation"], second["observation"])
        rng = np.random.default_rng(0)
        for _ in range(14):  # crosses the 12-step time limit, so both auto-reset
            actions = rng.integers(0, 10, size=(2, 5))
            for first, second in zip(sync.step(actions), workers.step(actions)):
                assert first["reward"] == pytest.approx(second["reward"])
                assert first["truncated"] == second["truncated"]
                assert np.allclose(first["observation"], second["observation"])
                assert ("episode_info" in first) == ("episode_info" in second)
        workers.set_stage(2)
        sync.set_stage(2)
    finally:
        workers.close()


def test_runner_reports_the_stage_the_episode_was_played_in() -> None:
    config = load_config(V5).environment
    config.max_steps = 2
    runner = EnvRunner(config, use_graph=False)
    runner.reset()
    runner.set_stage(3)
    idle = np.full(5, 9)
    runner.step(idle)
    result = runner.step(idle)
    assert result["episode_info"]["curriculum_stage"] == 1
    assert runner.env.config.curriculum_stage == 3 and runner.episode_stage == 3


def test_curriculum_ignores_episodes_from_an_earlier_stage() -> None:
    config = load_config(V5)
    config.training.curriculum_window = 2
    scheduler = CurriculumScheduler(config)
    assert not scheduler.record(True, 1)
    assert scheduler.record(True, 1) and scheduler.stage == 2
    assert not scheduler.record(True, 1)  # still running from stage 1
    assert len(scheduler.results) == 0


@pytest.mark.parametrize("backend", ["sync", "subprocess"])
def test_parallel_training_with_agent_values_and_chunks(tmp_path, backend: str) -> None:
    config = small_v5(tmp_path, backend)
    trainer = Trainer(config)
    history = trainer.train(updates=2)
    assert len(history) == 2
    for key in ("approx_kl", "clip_fraction", "explained_variance", "policy_loss"):
        assert np.isfinite(history[-1][key])
    checkpoint = torch.load(tmp_path / "run" / "checkpoint_000002.pt", weights_only=False)
    assert checkpoint["curriculum_stage"] == 1
    resumed = Trainer(config)
    checkpoint["curriculum_stage"] = 3
    torch.save(checkpoint, tmp_path / "run" / "patched.pt")
    assert resumed.load_checkpoint(tmp_path / "run" / "patched.pt") == 2
    assert resumed.curriculum.stage == 3


def test_teacher_builds_the_bridge_from_prepared_starts() -> None:
    for stage, limit in ((1, 20), (2, 30)):
        env = v5_env(stage)
        teacher = ScriptedTeacher(env)
        for _ in range(limit):
            actions, worker = teacher.actions()
            *_, terminated, _, info = env.step(actions)
            if terminated:
                break
        assert info["success"], stage
        assert worker == 1


def test_behaviour_cloning_learns_teacher_actions(tmp_path) -> None:
    config = load_config(V5)
    episodes = [record_episode(config.environment, stage, seed, epsilon=0.1) for stage, seed in ((1, 0), (2, 1))]
    assert all(episode.success for episode in episodes)
    assert episodes[0].observations.shape[1:] == (5, v5_env().observation_size)
    demonstrations = DemonstrationSet(episodes)
    demonstrations.save(tmp_path / "demos.pt")
    demonstrations = DemonstrationSet.load(tmp_path / "demos.pt")
    actor = v5_actor(v5_env())
    optimizer = torch.optim.Adam(actor.parameters(), lr=3.0e-3)
    batch = demonstrations.batch(demonstrations.episodes, idle_weight=0.2, device="cpu")
    first, _, _ = demonstrations.losses(actor, batch)
    for _ in range(40):
        loss, auxiliary, _ = demonstrations.losses(actor, batch)
        optimizer.zero_grad()
        (loss + 0.1 * auxiliary).backward()
        optimizer.step()
    last, _, stats = demonstrations.losses(actor, batch)
    assert float(last) < 0.5 * float(first)
    assert stats["move_accuracy"] > 0.8

    config.environment.max_steps = 12
    config.training.output_dir = str(tmp_path / "run")
    config.training.rollout_steps = 8
    config.training.recurrent_chunk = 4
    config.training.num_envs = 1
    config.training.ppo_epochs = 1
    config.training.demo_path = str(tmp_path / "demos.pt")
    config.training.bc_coef, config.training.bc_coef_final, config.training.bc_decay_updates = 0.5, 0.0, 2
    config.training.critic_warmup_updates = 0
    torch.save({"algorithm": {"actor": actor.state_dict()}, "pretrain": {"epochs": 40}}, tmp_path / "bc.pt")
    trainer = Trainer(config)
    assert trainer.load_pretrained(tmp_path / "bc.pt") == {"epochs": 40}
    assert trainer.bc_coefficient(1) == pytest.approx(0.5) and trainer.bc_coefficient(3) == pytest.approx(0.0)
    history = trainer.train(updates=1)
    assert history[0]["bc_coef"] == pytest.approx(0.5)
    assert history[0]["bc_loss"] > 0.0


def test_critic_warmup_keeps_the_actor_fixed(tmp_path) -> None:
    config = small_v5(tmp_path)
    config.training.critic_warmup_updates = 1
    trainer = Trainer(config)
    before = {name: value.clone() for name, value in trainer.actor.state_dict().items()}
    history = trainer.train(updates=1)
    assert history[0]["actor_frozen"] == 1.0
    assert all(torch.equal(before[name], value) for name, value in trainer.actor.state_dict().items())
