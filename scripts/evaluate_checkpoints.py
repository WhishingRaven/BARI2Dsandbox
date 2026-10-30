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


def update_from_name(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


def evaluate_actor(actor, env: BridgeEnv, teacher: CapacityTeacher, seeds: list[int], device: str) -> tuple[float, float, float]:
    capacities: list[float] = []
    successes: list[float] = []
    for seed in seeds:
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
            observation, _, terminated, truncated, _ = env.step(actions.cpu().numpy())
            if terminated or truncated:
                result = teacher.evaluate(env)
                capacities.append(float(result.capacity_kg))
                successes.append(float(env.graph.spans and result.capacity_kg >= env.target_load))
                break
    values = np.asarray(capacities, dtype=float)
    return float(values.mean()), float(np.median(values)), float(np.mean(successes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved policy checkpoints on identical held-out seeds using exact statics")
    parser.add_argument("run_dir", type=Path, nargs="?", default=Path("runs/capacity_local"))
    parser.add_argument("--config", default="configs/capacity_local.yaml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stage", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--every", type=int, default=1, help="Evaluate every Nth saved checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    checkpoints = sorted(args.run_dir.glob("checkpoint_*.pt"), key=update_from_name)
    checkpoints = [path for index, path in enumerate(checkpoints) if index % max(args.every, 1) == 0]
    if not checkpoints:
        raise SystemExit(f"No checkpoint_*.pt files found under {args.run_dir}")
    first = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    config = config_from_dict(first["experiment_config"]) if "experiment_config" in first else load_config(args.config)
    config.environment.curriculum_stage = args.stage
    env = BridgeEnv(config.environment)
    teacher_config = replace(
        config.training.capacity_teacher,
        enabled=True,
        backend="exact",
        checkpoint=None,
        counterfactual_credit=False,
    )
    teacher = CapacityTeacher(teacher_config, config.environment, device=args.device)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    rows = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        actor = SharedRecurrentActor(env.layout, env.action_count, config.model).to(args.device)
        actor.load_state_dict(checkpoint["algorithm"]["actor"])
        actor.eval()
        mean_capacity, median_capacity, success_rate = evaluate_actor(actor, env, teacher, seeds, args.device)
        row = {
            "update": int(checkpoint.get("update", update_from_name(checkpoint_path))),
            "checkpoint": str(checkpoint_path),
            "mean_exact_capacity_kg": mean_capacity,
            "median_exact_capacity_kg": median_capacity,
            "success_rate": success_rate,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    output = args.output_dir or (args.run_dir / "checkpoint_evaluation")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "checkpoints.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda row: row["mean_exact_capacity_kg"])
    summary = {
        "held_out_episodes_per_checkpoint": args.episodes,
        "stage": args.stage,
        "seed_start": args.seed_start,
        "best_checkpoint": best,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot([r["update"] for r in rows], [r["mean_exact_capacity_kg"] for r in rows], marker="o", label="mean exact capacity")
    ax.plot([r["update"] for r in rows], [r["median_exact_capacity_kg"] for r in rows], marker=".", label="median exact capacity")
    ax.set_xlabel("Training update")
    ax.set_ylabel("Maximum payload (kg)")
    ax.set_title(f"Held-out exact capacity across checkpoints — stage {args.stage}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "capacity_by_checkpoint.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(output)


if __name__ == "__main__":
    main()
