#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation

from bari2d.env.bridge_env import BridgeEnv
from bari2d.models.actor import SharedRecurrentActor
from bari2d.mechanics import SpatialStaticSolver, bridge_center_payload_3d
from bari2d.rl.capacity_teacher import CapacityTeacher
from bari2d.utils.config import config_from_dict, load_config
from bari2d.utils.visualization import draw_capacity_result, draw_environment


def load_experiment(checkpoint: dict, fallback: str):
    if "experiment_config" in checkpoint:
        return config_from_dict(checkpoint["experiment_config"])
    return load_config(fallback)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a trained decentralized bridge-construction policy")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", default="configs/capacity_local.yaml")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--stage", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of deterministic argmax actions")
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output-prefix", type=Path, default=Path("policy_episode"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = load_experiment(checkpoint, args.config)
    config.environment.curriculum_stage = args.stage
    env = BridgeEnv(config.environment)
    actor = SharedRecurrentActor(env.layout, env.action_count, config.model).to(args.device)
    actor.load_state_dict(checkpoint["algorithm"]["actor"])
    actor.eval()

    observation, _ = env.reset(seed=args.seed)
    hidden = actor.initial_hidden(env.config.robot.count, args.device)
    frame_images: list[np.ndarray] = []
    action_counts = np.zeros(env.action_count, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(9, 5))
    draw_environment(env, ax, show_action_summary=True)
    fig.canvas.draw()
    frame_images.append(np.asarray(fig.canvas.buffer_rgba()).copy())
    final_info = env.info()
    while True:
        with torch.no_grad():
            actions, _, _, output = actor.act(
                torch.as_tensor(observation, device=args.device),
                hidden,
                torch.as_tensor(env.action_masks(), device=args.device),
                deterministic=not args.stochastic,
            )
            hidden = output.hidden
        action_values = actions.cpu().numpy()
        action_counts += np.bincount(action_values, minlength=env.action_count)
        observation, _, terminated, truncated, final_info = env.step(action_values)
        if env.step_count % max(args.frame_skip, 1) == 0 or terminated or truncated:
            draw_environment(env, ax, show_action_summary=True)
            fig.canvas.draw()
            frame_images.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        if terminated or truncated:
            break
    plt.close(fig)

    teacher_config = replace(
        config.training.capacity_teacher,
        enabled=True,
        backend="exact",
        checkpoint=None,
        counterfactual_credit=False,
    )
    teacher = CapacityTeacher(teacher_config, config.environment, device=args.device)
    teacher_eval = teacher.evaluate(env)
    capacity_result = None
    if env.graph.spans:
        try:
            payload = bridge_center_payload_3d(env.robots, env.graph, env.field, env.config.robot)
            solver = SpatialStaticSolver(
                env.robots,
                env.graph,
                env.field,
                env.config.robot,
                verify_rank=bool(teacher_config.exact_verify_rank),
            )
            capacity_result = solver.capacity_local(payload, teacher.limit, include_self_weight=True)
        except (ValueError, np.linalg.LinAlgError):
            capacity_result = None

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    gif_path = prefix.with_suffix(".gif")
    movie_fig, movie_ax = plt.subplots(figsize=(9, 5))
    movie_ax.axis("off")
    image = movie_ax.imshow(frame_images[0])

    def update(index: int):
        image.set_data(frame_images[index])
        return (image,)

    movie = animation.FuncAnimation(movie_fig, update, frames=len(frame_images), interval=1000 / max(args.fps, 1), blit=True)
    movie.save(gif_path, writer=animation.PillowWriter(fps=max(args.fps, 1)))
    plt.close(movie_fig)

    final_path = prefix.with_name(prefix.name + "_final").with_suffix(".png")
    final_fig, final_ax = plt.subplots(figsize=(10, 5.5))
    if capacity_result is None:
        draw_environment(env, final_ax, exact_capacity_kg=teacher_eval.capacity_kg, show_action_summary=True)
    else:
        draw_capacity_result(env, capacity_result, teacher.limit, final_ax, payload_kg=teacher_eval.capacity_kg)
    final_fig.tight_layout()
    final_fig.savefig(final_path, dpi=180)
    plt.close(final_fig)

    names = [str(index) for index in range(env.action_count)]
    try:
        from bari2d.env.robot import DiscreteAction
        names = [DiscreteAction(index).name for index in range(env.action_count)]
    except Exception:
        pass
    total_actions = max(int(action_counts.sum()), 1)
    result = {
        "seed": args.seed,
        "stage": args.stage,
        "steps": env.step_count,
        "span": bool(env.graph.spans),
        "exact_capacity_kg": teacher_eval.capacity_kg,
        "target_load_kg": env.target_load,
        "success": bool(env.graph.spans and teacher_eval.capacity_kg >= env.target_load),
        "critical_component": teacher_eval.critical_component,
        "critical_contact": None if capacity_result is None else capacity_result.critical_contact,
        "energy": final_info.get("energy", 0.0),
        "anchored_robots": final_info.get("anchored_robots", 0),
        "fallen_robots": final_info.get("fallen_robots", 0),
        "action_distribution": {name: float(action_counts[i] / total_actions) for i, name in enumerate(names)},
        "gif": str(gif_path),
        "final_png": str(final_path),
    }
    json_path = prefix.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
