#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json

import numpy as np
import torch

from bari2d.env.bridge_env import BridgeEnv
from bari2d.models.actor import SharedRecurrentActor
from bari2d.rl.capacity_teacher import CapacityTeacher
from bari2d.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a shared decentralized policy")
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="configs/capacity_local.yaml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stage", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    config.environment.curriculum_stage = arguments.stage
    env = BridgeEnv(config.environment)
    actor = SharedRecurrentActor(env.layout, env.action_count, config.model).to(arguments.device)
    checkpoint = torch.load(arguments.checkpoint, map_location=arguments.device, weights_only=False)
    actor.load_state_dict(checkpoint["algorithm"]["actor"])
    actor.eval()
    teacher_config = replace(
        config.training.capacity_teacher,
        enabled=True,
        backend="exact",
        checkpoint=None,
        counterfactual_credit=False,
    )
    capacity_teacher = CapacityTeacher(teacher_config, config.environment, device=arguments.device)
    results = []
    for episode in range(arguments.episodes):
        observation, _ = env.reset(seed=config.environment.seed + episode)
        hidden = actor.initial_hidden(env.config.robot.count, arguments.device)
        while True:
            with torch.no_grad():
                actions, _, _, output = actor.act(
                    torch.as_tensor(observation, device=arguments.device),
                    hidden,
                    torch.as_tensor(env.action_masks(), device=arguments.device),
                    deterministic=True,
                )
                hidden = output.hidden
            observation, _, terminated, truncated, info = env.step(actions.cpu().numpy())
            if terminated or truncated:
                teacher = capacity_teacher.evaluate(env)
                info = dict(info)
                info["exact_capacity_kg"] = teacher.capacity_kg
                info["success"] = bool(env.graph.spans and teacher.capacity_kg >= env.target_load)
                results.append(info)
                break
    summary = {
        "episodes": len(results),
        "success_rate": float(np.mean([result["success"] for result in results])),
        "mean_capacity_kg": float(np.mean([result["exact_capacity_kg"] for result in results])),
        "median_capacity_kg": float(np.median([result["exact_capacity_kg"] for result in results])),
        "mean_energy": float(np.mean([result["energy"] for result in results])),
        "mean_anchored_robots": float(np.mean([result["anchored_robots"] for result in results])),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

