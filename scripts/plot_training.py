#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, min(int(window), len(values)))
    kernel = np.ones(window, dtype=float) / window
    prefix = np.full(window - 1, np.nan)
    return np.concatenate((prefix, np.convolve(values, kernel, mode="valid")))


def save_line(x, ys, labels, title, ylabel, path: Path, *, horizontal=None) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for y, label in zip(ys, labels):
        ax.plot(x, y, label=label)
    if horizontal is not None:
        ax.axhline(horizontal, linestyle="--", linewidth=1.0, label=f"target={horizontal:g}")
    ax.set_title(title)
    ax.set_xlabel("Episode" if "episode" in path.stem else "Update")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if len(labels) > 1 or horizontal is not None:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot capacity-policy training diagnostics")
    parser.add_argument("run_dir", nargs="?", default="runs/capacity_local")
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output = args.output_dir or (run_dir / "plots")
    output.mkdir(parents=True, exist_ok=True)
    episodes = read_jsonl(run_dir / "episodes.jsonl")
    updates = read_jsonl(run_dir / "updates.jsonl")
    if not episodes:
        raise SystemExit(f"No episodes found in {run_dir / 'episodes.jsonl'}")

    e = np.asarray([row.get("episode", i + 1) for i, row in enumerate(episodes)], dtype=float)
    capacity = np.asarray([row.get("capacity_teacher", {}).get("capacity_kg", 0.0) for row in episodes], dtype=float)
    success = np.asarray([float(row.get("success", False)) for row in episodes], dtype=float)
    reward = np.asarray([row.get("episode_reward", 0.0) for row in episodes], dtype=float)
    target = np.asarray([row.get("target_load", np.nan) for row in episodes], dtype=float)
    energy = np.asarray([row.get("energy", 0.0) for row in episodes], dtype=float)
    anchors = np.asarray([row.get("anchored_robots", 0.0) for row in episodes], dtype=float)
    fallen = np.asarray([row.get("fallen_robots", 0.0) for row in episodes], dtype=float)

    save_line(e, [capacity, rolling(capacity, args.window)], ["exact/teacher capacity", f"rolling mean ({args.window})"],
              "Bridge capacity during training", "Capacity (kg)", output / "episode_capacity.png",
              horizontal=float(np.nanmedian(target)) if np.any(np.isfinite(target)) else None)
    save_line(e, [rolling(success, args.window)], [f"rolling success rate ({args.window})"],
              "Held objective success during training", "Success rate", output / "episode_success.png")
    save_line(e, [reward, rolling(reward, args.window)], ["episode reward", f"rolling mean ({args.window})"],
              "Episode return", "Reward", output / "episode_reward.png")
    save_line(e, [rolling(energy, args.window), rolling(anchors, args.window), rolling(fallen, args.window)],
              ["energy", "anchored robots", "fallen robots"], "Construction behavior", "Value",
              output / "episode_behavior.png")

    if updates:
        u = np.asarray([row.get("update", i + 1) for i, row in enumerate(updates)], dtype=float)
        for key, title in [
            ("entropy", "Policy entropy"),
            ("critic_loss", "Critic loss"),
            ("approx_kl", "Approximate KL divergence"),
            ("clip_fraction", "PPO clip fraction"),
            ("explained_variance", "Critic explained variance"),
        ]:
            values = np.asarray([row.get(key, np.nan) for row in updates], dtype=float)
            save_line(u, [values], [key], title, key, output / f"update_{key}.png")

    summary = {
        "episodes": len(episodes),
        "last_window": min(args.window, len(episodes)),
        "capacity_mean_last_window_kg": float(np.mean(capacity[-args.window:])),
        "capacity_median_last_window_kg": float(np.median(capacity[-args.window:])),
        "success_rate_last_window": float(np.mean(success[-args.window:])),
        "reward_mean_last_window": float(np.mean(reward[-args.window:])),
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(output)


if __name__ == "__main__":
    main()
