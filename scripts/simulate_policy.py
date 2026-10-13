#!/usr/bin/env python3
"""Run a trained policy for a few episodes, print what it built, and save GIFs.

    # stage-5 episodes of the final v8 policy, GIF of every episode
    python scripts/simulate_policy.py --checkpoint runs/beacon_structured_mappo_static_v8/checkpoint_001500.pt ^
        --stage 5 --seeds 0 5 --gif-dir runs/beacon_structured_mappo_static_v8/sim

    # the same policy started from a prepared spanning row (v9 stage 6)
    python scripts/simulate_policy.py --checkpoint runs/beacon_structured_mappo_static_v8/checkpoint_001500.pt ^
        --config configs/beacon_structured_mappo_static_v9.yaml --stage 6 --seeds 200 212

* The environment comes from the checkpoint (its saved experiment config)
  unless --config is given; --config lets you try a policy on another
  version's stages (the observation layout must match).
* Actions are sampled from the policy (as during training).  --deterministic
  takes the most likely action instead; trained policies mostly IDLE then.
* A seed fixes the start state and the sampled actions, so a seed always
  replays the same episode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from bari2d.env.bridge_env import BridgeEnv  # noqa: E402
from bari2d.env.robot import DiscreteAction  # noqa: E402
from bari2d.models.actor import SharedRecurrentActor  # noqa: E402
from bari2d.utils.config import ExperimentConfig, config_from_dict, load_config  # noqa: E402
from bari2d.utils.visualization import draw_environment  # noqa: E402

ACTION_NAMES = [action.name for action in DiscreteAction]


def load_policy(checkpoint_path: Path, config_path: Path | None) -> tuple[ExperimentConfig, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if config_path is not None:
        config = load_config(config_path)
    elif isinstance(checkpoint.get("experiment_config"), dict):
        config = config_from_dict(checkpoint["experiment_config"])
    else:
        raise SystemExit("The checkpoint has no saved config; pass --config.")
    state = checkpoint.get("algorithm", checkpoint)
    if "actor" not in state:
        raise SystemExit(f"No actor weights in {checkpoint_path}")
    return config, state["actor"]


def top_grips(env: BridgeEnv) -> int:
    """Robots gripping while standing on another robot (splices / stacked grips)."""
    return sum(1 for robot in env.robots if robot.anchored and robot.layer >= 1 and not robot.fallen)


def fallen(env: BridgeEnv) -> int:
    """Robots that fell (robots absent from the episode are not counted)."""
    return int(sum(robot.fallen and not robot.absent for robot in env.robots))


def active(env: BridgeEnv) -> int:
    """Robots taking part in the episode (robot.active_min randomises this)."""
    return int(sum(not robot.absent for robot in env.robots))


class GifRecorder:
    """Draws the gap region every ``every`` steps and writes an animated GIF.

    The layout is fixed once (no per-frame tight_layout) and every frame shares
    one colour palette, so the picture does not jump or flicker between frames.
    """

    def __init__(self, env: BridgeEnv, label: str, seed: int, every: int, max_frames: int):
        self.env, self.label, self.seed = env, label, seed
        self.every, self.max_frames = max(every, 1), max_frames
        self.figure, self.axis = plt.subplots(figsize=(7.2, 6.4))
        self.figure.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.12)
        # Action counts live in the figure (below the axes) so long lines never move the plot.
        self.action_text = self.figure.text(0.02, 0.015, "", fontsize=8, va="bottom")
        self.frames: list[np.ndarray] = []

    def snap(self, actions: np.ndarray | None, info: dict | None, force: bool = False) -> None:
        env = self.env
        if not force and env.step_count % self.every:
            return
        draw_environment(env, self.axis)
        for text in list(self.axis.texts):
            text.remove()
        center = env.field.center[0]
        self.axis.set_xlim(center - 4.2, center + 2.6)
        self.axis.set_ylim(0.0, env.field.width)
        metrics = env.structure_metrics()
        self.axis.set_title(
            f"{self.label} | seed {self.seed}\n"
            f"step {env.step_count}  edge grips {metrics['edge_grips']}  on structure {metrics['on_structure']}  "
            f"extended {metrics['extended']}  top grips {top_grips(env)}  "
            f"fallen {fallen(env)}  robots {active(env)}  capacity {env.current_capacity:.1f}/{env.target_load:.0f}",
            fontsize=9,
        )
        summary = ""
        if actions is not None:
            names, counts = np.unique([ACTION_NAMES[action] for action in actions], return_counts=True)
            summary = "actions: " + " | ".join(f"{name} {count}" for name, count in zip(names, counts))
        self.action_text.set_text(summary)
        if info and info.get("success"):
            self.axis.text(0.5, 0.95, "SUCCESS", transform=self.axis.transAxes, ha="center",
                           fontsize=14, color="#1a9850", weight="bold")
        self.figure.canvas.draw()
        self.frames.append(np.asarray(self.figure.canvas.buffer_rgba())[..., :3].copy())

    def save(self, path: Path) -> None:
        from PIL import Image

        plt.close(self.figure)
        frames = self.frames + [self.frames[-1]] * 8  # hold the last frame
        if len(frames) > self.max_frames:  # keep the start and the end
            frames = frames[: self.max_frames // 2] + frames[-(self.max_frames - self.max_frames // 2):]
        # One palette for the whole GIF, built from frames spread over the episode
        # (per-frame palettes make the background colours flicker).
        median_cut = getattr(getattr(Image, "Quantize", None), "MEDIANCUT", 0)
        no_dither = getattr(getattr(Image, "Dither", None), "NONE", 0)
        samples = frames[:: max(1, len(frames) // 12)]
        palette = Image.fromarray(np.concatenate(samples, axis=0)).quantize(colors=255, method=median_cut)
        images = [Image.fromarray(frame).quantize(palette=palette, dither=no_dither) for frame in frames]
        path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(path, save_all=True, append_images=images[1:], duration=110, loop=0, optimize=True)
        Image.fromarray(frames[-1]).save(path.with_suffix(".png"))


def run_episode(config: ExperimentConfig, actor: SharedRecurrentActor, seed: int, deterministic: bool,
                recorder_args: dict | None) -> tuple[dict, GifRecorder | None]:
    env = BridgeEnv(config.environment)
    generator = torch.Generator().manual_seed(seed)
    observation, _ = env.reset(seed=seed)
    hidden = actor.initial_hidden(env.config.robot.count)
    recorder = GifRecorder(env, seed=seed, **recorder_args) if recorder_args else None
    if recorder:
        recorder.snap(None, None, force=True)
    info: dict = {}
    for _ in range(env.config.max_steps):
        with torch.no_grad():
            output = actor.forward_step(torch.as_tensor(observation), hidden, torch.as_tensor(env.action_masks()))
        hidden = output.hidden
        if deterministic:
            actions = output.logits.argmax(dim=-1)
        else:
            actions = torch.multinomial(torch.softmax(output.logits, dim=-1), 1, generator=generator).squeeze(-1)
        actions = actions.numpy()
        observation, _, terminated, truncated, info = env.step(actions)
        if recorder:
            recorder.snap(actions, info, force=terminated or truncated)
        if terminated or truncated:
            break
    result = {
        "seed": seed,
        "success": bool(info.get("success")),
        "steps": env.step_count,
        "capacity": round(float(env.current_capacity), 2),
        "target": float(env.target_load),
        **env.structure_metrics(),
        "top_grips": top_grips(env),
        "fallen": fallen(env),
        "robots": active(env),
    }
    return result, recorder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="environment/model config (default: the one saved in the checkpoint)")
    parser.add_argument("--stage", type=int, help="curriculum stage (default: the config's first stage)")
    parser.add_argument("--seeds", type=int, nargs=2, default=[0, 5], metavar=("FIRST", "STOP"),
                        help="episode seeds range(FIRST, STOP)")
    parser.add_argument("--deterministic", action="store_true", help="most likely action instead of sampling")
    parser.add_argument("--gif-dir", type=Path, help="write one GIF (+ final-frame PNG) per episode here")
    parser.add_argument("--every", type=int, default=0, help="steps per GIF frame (0 = automatic)")
    parser.add_argument("--max-frames", type=int, default=220)
    parser.add_argument("--json", type=Path, help="also write the per-episode results as JSON")
    arguments = parser.parse_args()

    config, actor_state = load_policy(arguments.checkpoint, arguments.config)
    if arguments.stage is not None:
        config.environment.curriculum_stage = arguments.stage
    env = BridgeEnv(config.environment)
    env.reset(seed=arguments.seeds[0])
    actor = SharedRecurrentActor(env.layout, env.action_count, config.model)
    actor.load_state_dict(actor_state)
    actor.eval()
    torch.set_num_threads(max(torch.get_num_threads(), 1))

    stage = config.environment.curriculum_stage
    label = f"{arguments.checkpoint.parent.name}/{arguments.checkpoint.stem} | stage {stage} (gap {env.field.gap_width:.1f})"
    print(f"{label}: target {env.target_load:.1f}, {config.environment.robot.count} robots, "
          f"{'argmax' if arguments.deterministic else 'sampled'} actions", flush=True)
    results = []
    for seed in range(*arguments.seeds):
        recorder_args = None
        if arguments.gif_dir is not None:
            every = arguments.every or max(1, config.environment.max_steps // 200)
            path = arguments.gif_dir / f"{arguments.checkpoint.stem}_stage{stage}_seed{seed}.gif"
            recorder_args = {"label": label, "every": every, "max_frames": arguments.max_frames, "path": path}
        result, recorder = run_episode(config, actor, seed, arguments.deterministic,
                                       {k: v for k, v in recorder_args.items() if k != "path"} if recorder_args else None)
        if recorder:
            recorder.save(recorder_args["path"])
            result["gif"] = str(recorder_args["path"])
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    successes = sum(result["success"] for result in results)
    print(f"success {successes}/{len(results)}  mean steps {np.mean([r['steps'] for r in results]):.0f}  "
          f"mean capacity {np.mean([r['capacity'] for r in results]):.1f}")
    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
