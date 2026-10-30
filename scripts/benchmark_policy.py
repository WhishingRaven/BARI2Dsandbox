#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bari2d.env.bridge_env import BridgeEnv
from bari2d.models.actor import SharedRecurrentActor
from bari2d.rl.capacity_teacher import CapacityTeacher
from bari2d.utils.config import config_from_dict, load_config


def load_experiment(checkpoint: dict, fallback: str):
    if "experiment_config" in checkpoint:
        return config_from_dict(checkpoint["experiment_config"])
    return load_config(fallback)


def exact_teacher(config, device: str) -> CapacityTeacher:
    teacher_config = replace(
        config.training.capacity_teacher,
        enabled=True,
        backend="exact",
        checkpoint=None,
        counterfactual_credit=False,
    )
    return CapacityTeacher(teacher_config, config.environment, device=device)


def run_trained(actor, env: BridgeEnv, seed: int, device: str) -> dict:
    observation, _ = env.reset(seed=seed)
    hidden = actor.initial_hidden(env.config.robot.count, device)
    while True:
        with torch.no_grad():
            actions, _, _, output = actor.act(
                torch.as_tensor(observation, device=device),
                hidden,
                torch.as_tensor(env.action_masks(), device=device),
                deterministic=True,
            )
            hidden = output.hidden
        observation, _, terminated, truncated, info = env.step(actions.cpu().numpy())
        if terminated or truncated:
            return dict(info)


def run_random(env: BridgeEnv, seed: int) -> dict:
    observation, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 99173)
    while True:
        masks = env.action_masks()
        actions = np.asarray([rng.choice(np.flatnonzero(row)) for row in masks], dtype=np.int64)
        observation, _, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            return dict(info)


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def bootstrap_mean_ci(values: np.ndarray, seed: int = 12345, samples: int = 5000) -> list[float]:
    if len(values) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a trained local policy against a random valid-action baseline")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", default="configs/capacity_local.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--stage", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = load_experiment(checkpoint, args.config)
    config.environment.curriculum_stage = args.stage
    actor_env = BridgeEnv(config.environment)
    random_env = BridgeEnv(config.environment)
    actor = SharedRecurrentActor(actor_env.layout, actor_env.action_count, config.model).to(args.device)
    actor.load_state_dict(checkpoint["algorithm"]["actor"])
    actor.eval()
    teacher = exact_teacher(config, args.device)

    rows = []
    trained_caps = []
    random_caps = []
    trained_success = []
    random_success = []
    for index in range(args.episodes):
        seed = args.seed_start + index
        trained_info = run_trained(actor, actor_env, seed, args.device)
        trained_eval = teacher.evaluate(actor_env)
        random_info = run_random(random_env, seed)
        random_eval = teacher.evaluate(random_env)
        trained_ok = bool(actor_env.graph.spans and trained_eval.capacity_kg >= actor_env.target_load)
        random_ok = bool(random_env.graph.spans and random_eval.capacity_kg >= random_env.target_load)
        trained_caps.append(trained_eval.capacity_kg)
        random_caps.append(random_eval.capacity_kg)
        trained_success.append(trained_ok)
        random_success.append(random_ok)
        rows.append({
            "seed": seed,
            "trained_capacity_kg": trained_eval.capacity_kg,
            "random_capacity_kg": random_eval.capacity_kg,
            "trained_success": int(trained_ok),
            "random_success": int(random_ok),
            "trained_energy": trained_info.get("energy", 0.0),
            "random_energy": random_info.get("energy", 0.0),
            "trained_anchors": trained_info.get("anchored_robots", 0),
            "random_anchors": random_info.get("anchored_robots", 0),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    trained = np.asarray(trained_caps, dtype=float)
    random_values = np.asarray(random_caps, dtype=float)
    paired = trained - random_values
    summary = {
        "checkpoint": str(args.checkpoint),
        "episodes": args.episodes,
        "stage": args.stage,
        "seed_start": args.seed_start,
        "trained_capacity_kg": stats(trained),
        "random_capacity_kg": stats(random_values),
        "paired_capacity_gain_kg": stats(paired),
        "paired_mean_gain_95pct_bootstrap_ci_kg": bootstrap_mean_ci(paired),
        "trained_success_rate": float(np.mean(trained_success)),
        "random_success_rate": float(np.mean(random_success)),
        "trained_beats_random_fraction": float(np.mean(trained > random_values)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.boxplot([trained, random_values], labels=["trained", "random valid-action"], showmeans=True)
    ax.set_ylabel("Exact maximum payload (kg)")
    ax.set_title(f"Held-out policy benchmark — stage {args.stage}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "capacity_comparison.png", dpi=170)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(args.output_dir)


if __name__ == "__main__":
    main()
