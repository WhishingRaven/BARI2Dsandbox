#!/usr/bin/env python3
"""Behaviour-cloning pretraining from scripted-teacher demonstrations.

    python scripts/pretrain_bc.py --config configs/beacon_structured_mappo_static_v6.yaml
    python scripts/train.py --config configs/beacon_structured_mappo_static_v6.yaml \
        --pretrained runs/beacon_structured_mappo_static_v6/bc_pretrain.pt

1. The scripted teacher (bari2d/rl/teacher.py) builds bridges from the start
   states of the curriculum stages; with probability --epsilon each executed
   action is replaced by a random valid one while the label stays the
   teacher's action.  Observations always come from the simulator.
2. The shared actor is trained to imitate the teacher (weighted cross-entropy,
   IDLE labels of waiting robots down-weighted) plus the auxiliary cue losses.
3. The demonstrations are saved (training.demo_path) so PPO can keep an
   annealed behaviour-cloning term; the actor is saved for --pretrained.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from bari2d.env.bridge_env import BridgeEnv
from bari2d.models.actor import SharedRecurrentActor
from bari2d.rl.demonstrations import DemonstrationSet, generate_demonstrations
from bari2d.utils.config import ExperimentConfig, load_config


def evaluate(actor: SharedRecurrentActor, config: ExperimentConfig, stage: int, episodes: int, seed: int) -> float:
    """Success rate of the (stochastic) policy from the start state of ``stage``."""
    environment_config = deepcopy(config.environment)
    environment_config.curriculum_stage = stage
    env = BridgeEnv(environment_config)
    successes = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        hidden = actor.initial_hidden(env.config.robot.count)
        while True:
            actions, _, _, output = actor.act(torch.as_tensor(observation), hidden, torch.as_tensor(env.action_masks()))
            hidden = output.hidden
            observation, _, terminated, truncated, info = env.step(actions.numpy())
            if terminated or truncated:
                successes += int(info["success"])
                break
    return successes / max(episodes, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, help="teacher episodes, spread over --stages (default: training.demo_episodes or 240)")
    parser.add_argument(
        "--stages", type=int, nargs="+",
        help="curriculum stages to demonstrate, in turn; repeat one to weight it (default: training.demo_stages or 1..curriculum_max_stage)",
    )
    parser.add_argument("--epsilon", type=float, default=0.1, help="probability of a random executed action per robot and step")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="processes for demonstration generation")
    parser.add_argument("--demos", help="demonstration file (default: training.demo_path or <output_dir>/demos.pt)")
    parser.add_argument("--reuse-demos", action="store_true", help="load --demos instead of generating new ones")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--output", help="actor checkpoint (default: <output_dir>/bc_pretrain.pt)")
    parser.add_argument("--eval-episodes", type=int, default=0, help="policy episodes per stage after training (0 = skip)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    training = config.training
    output_dir = Path(training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_path = Path(arguments.demos or training.demo_path or output_dir / "demos.pt")
    output = Path(arguments.output or output_dir / "bc_pretrain.pt")
    stages = arguments.stages or list(training.demo_stages) or list(range(1, training.curriculum_max_stage + 1))
    episode_count = arguments.episodes or training.demo_episodes or 240
    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)

    start = time.time()
    if arguments.reuse_demos and demo_path.is_file():
        demonstrations = DemonstrationSet.load(demo_path)
        print(f"Loaded {len(demonstrations)} demonstrations from {demo_path}", flush=True)
    else:
        print(f"Generating {episode_count} teacher episodes over stages {stages} with {arguments.workers} worker(s)...", flush=True)
        episodes = generate_demonstrations(
            config.environment, stages, episode_count, arguments.epsilon, arguments.seed, arguments.workers, progress=True
        )
        demonstrations = DemonstrationSet(episodes)
        demonstrations.save(demo_path)
        print(f"Saved {len(demonstrations)} demonstrations to {demo_path} ({time.time() - start:.0f}s)", flush=True)
    for stage, (count, successes) in demonstrations.summary().items():
        lengths = [len(episode) for episode in demonstrations.episodes if episode.stage == stage]
        print(f"  stage {stage}: {count} episodes, teacher success {successes}/{count}, mean length {np.mean(lengths):.0f} steps")

    env = BridgeEnv(config.environment)
    device = torch.device(arguments.device)
    actor = SharedRecurrentActor(env.layout, env.action_count, config.model).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=arguments.learning_rate)
    episodes = list(demonstrations.episodes)
    stats: dict[str, float] = {}
    for epoch in range(1, arguments.epochs + 1):
        random.shuffle(episodes)
        totals = {"loss": 0.0, "bc": 0.0, "aux": 0.0, "accuracy": 0.0, "move_accuracy": 0.0}
        batches = 0
        for first in range(0, len(episodes), arguments.batch_episodes):
            batch = demonstrations.batch(episodes[first : first + arguments.batch_episodes], training.bc_idle_weight, device)
            bc_loss, auxiliary_loss, accuracy = demonstrations.losses(actor, batch)
            loss = bc_loss + training.auxiliary_coef * auxiliary_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), training.max_grad_norm)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["bc"] += float(bc_loss.detach())
            totals["aux"] += float(auxiliary_loss.detach())
            totals["accuracy"] += accuracy["accuracy"]
            totals["move_accuracy"] += accuracy["move_accuracy"]
            batches += 1
        stats = {name: value / max(batches, 1) for name, value in totals.items()}
        print(
            f"Epoch {epoch}/{arguments.epochs} loss={stats['loss']:.4f} bc={stats['bc']:.4f} aux={stats['aux']:.4f} "
            f"accuracy={stats['accuracy']:.3f} move_accuracy={stats['move_accuracy']:.3f}",
            flush=True,
        )

    pretrain_info = {
        "demonstrations": str(demo_path),
        "episodes": len(demonstrations),
        "stages": stages,
        "epochs": arguments.epochs,
        **{f"final_{name}": value for name, value in stats.items()},
    }
    if arguments.eval_episodes > 0:
        actor.eval()
        for stage in sorted(set(stages)):
            rate = evaluate(actor, config, stage, arguments.eval_episodes, seed=10_000 + arguments.seed)
            pretrain_info[f"success_stage_{stage}"] = rate
            print(f"  BC policy, stage {stage}: success {rate:.0%} over {arguments.eval_episodes} episodes", flush=True)
    torch.save(
        {
            "algorithm": {"actor": actor.state_dict()},
            "experiment_config": config.to_dict(),
            "update": 0,
            "pretrain": pretrain_info,
        },
        output,
    )
    print(f"Saved behaviour-cloned actor to {output}", flush=True)


if __name__ == "__main__":
    main()
