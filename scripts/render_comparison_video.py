#!/usr/bin/env python3
"""Render the eight final policies as a synchronized 2x4 MP4 comparison."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from infer import InferenceSession, load_inference_model, resolved_device
from bari2d.utils.visualization import draw_environment


RUNS = (
    "baseline",
    "bio",
    "bio_film",
    "bio_heterogeneity",
    "curriculum",
    "gru_traffic",
    "gru_traffic_connectivity",
    "mlp",
)


def configuration_label(session: InferenceSession) -> str:
    config = session.model.config
    model = config.model
    environment = config.environment
    training = config.training
    return "\n".join(
        (
            f"arch={model.architecture}  critic={training.critic}",
            f"FiLM={'on' if model.film else 'off'}  hetero={'on' if model.use_heterogeneity else 'off'} (sigma={environment.latent_sigma:g})",
            f"sensor/act noise={environment.sensor.sensor_noise:g}/{environment.actuator_noise:g}",
        )
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--render-every",
        type=int,
        default=5,
        help="Render one video frame after this many simulation steps (all steps still execute).",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--target-load", type=float, default=8.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax actions instead of sampling the learned policy distribution.",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/inference_comparison_2x4_1000steps.mp4"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    device = resolved_device(arguments.device)
    sessions: list[tuple[str, InferenceSession]] = []
    for run in RUNS:
        checkpoint = Path("runs") / run / "checkpoint_001000.pt"
        model = load_inference_model(checkpoint, None, device)
        sessions.append(
            (
                run,
                InferenceSession(
                    model,
                    device=device,
                    seed=arguments.seed,
                    deterministic=arguments.deterministic,
                    stage=arguments.stage,
                    target_load=arguments.target_load,
                ),
            )
        )

    figure, axes = plt.subplots(2, 4, figsize=(16, 9), dpi=120)
    figure.subplots_adjust(left=0.025, right=0.99, top=0.90, bottom=0.035, wspace=0.11, hspace=0.19)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1920x1080",
        "-r",
        str(arguments.fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(arguments.output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert encoder.stdin is not None
    try:
        if arguments.render_every < 1:
            raise ValueError("--render-every must be at least 1")
        for simulation_step in range(0, arguments.steps + 1, arguments.render_every):
            figure.suptitle(
                "BARI2D policy inference comparison (final checkpoint)"
                f"  |  shared test: seed={arguments.seed}, stage={arguments.stage}, target load={arguments.target_load:g}"
                f"  |  simulation step={simulation_step}/{arguments.steps}"
                f"  |  policy={'argmax' if arguments.deterministic else 'sampled'}"
                f"  |  {arguments.render_every}x playback",
                fontsize=15,
                fontweight="bold",
            )
            for axis, (run, session) in zip(axes.flat, sessions):
                draw_environment(session.env, axis)
                info = session.env.info()
                axis.set_title(
                    f"{run}  |  capacity={session.env.current_capacity:.2f}  progress={session.env.current_progress:.2f}",
                    fontsize=9,
                    fontweight="bold",
                )
                axis.text(
                    0.015,
                    0.985,
                    configuration_label(session),
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.9,
                    linespacing=1.22,
                    bbox={"facecolor": "white", "alpha": 0.83, "edgecolor": "#444444", "pad": 2.5},
                    zorder=30,
                )
                axis.text(
                    0.985,
                    0.015,
                    f"reward={session.last_reward:+.3f}\nphase={info['phase']}",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=6.5,
                    bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
                    zorder=30,
                )
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            encoder.stdin.write(np.ascontiguousarray(rgba[:, :, :3]).tobytes())
            for _ in range(min(arguments.render_every, arguments.steps - simulation_step)):
                for _run, session in sessions:
                    session.step()
    finally:
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding the comparison video")
        plt.close(figure)
    print(arguments.output)


if __name__ == "__main__":
    main()
