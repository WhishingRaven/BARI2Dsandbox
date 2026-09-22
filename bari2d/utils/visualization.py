from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, patches, transforms

from bari2d.env.bridge_env import BridgeEnv


def robot_layer_color(layer: int, max_layer: int) -> tuple[float, float, float, float]:
    """Map each discrete height layer to a stable, high-contrast color."""
    normalized = np.clip(layer / max(max_layer, 1), 0.0, 1.0)
    return tuple(plt.colormaps["plasma"](normalized))


def draw_environment(env: BridgeEnv, axis: plt.Axes | None = None) -> plt.Axes:
    axis = axis or plt.subplots(figsize=(11, 6))[1]
    axis.clear()
    field = env.field
    x_values = np.linspace(0.0, field.length, 480)
    y_values = np.linspace(0.0, field.width, 280)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    relative_x = x_grid - field.center[0]
    relative_y = y_grid - field.center[1]
    longitudinal = relative_x * field.normal[0] + relative_y * field.normal[1]
    tangent = relative_x * field.tangent[0] + relative_y * field.tangent[1]
    perturbation = field.irregularity * np.sin(field.irregularity_frequency * tangent + field.phase)
    left_boundary = -field.gap_width / 2.0 + perturbation
    right_boundary = field.gap_width / 2.0 + perturbation
    gap_mask = (longitudinal > left_boundary) & (longitudinal < right_boundary)
    background = np.empty((*gap_mask.shape, 4), dtype=float)
    background[:] = (0.725, 0.604, 0.420, 0.65)
    background[gap_mask] = (0.494, 0.714, 0.812, 0.35)
    axis.imshow(background, origin="lower", extent=(0.0, field.length, 0.0, field.width), aspect="auto")

    for edge in env.graph.edges.values():
        if not isinstance(edge.source, int) or not isinstance(edge.target, int):
            continue
        first = env.robots[edge.source].position
        second = env.robots[edge.target].position
        color = "#b2182b" if edge.kind == "anchor" else "#555555"
        width = 2.2 if edge.kind == "anchor" else 0.8
        axis.plot([first[0], second[0]], [first[1], second[1]], color=color, linewidth=width, alpha=0.75)

    load_path = [] if env.last_load_test is None else env.last_load_test.load_bearing_path
    for first_node, second_node in zip(load_path, load_path[1:]):
        if isinstance(first_node, int) and isinstance(second_node, int):
            first, second = env.robots[first_node].position, env.robots[second_node].position
            axis.plot([first[0], second[0]], [first[1], second[1]], color="#ffd92f", linewidth=4.0, alpha=0.8)

    for robot in env.robots:
        color = robot_layer_color(robot.layer, env.config.robot.max_layer)
        if robot.fallen:
            color = "#444444"
        rectangle = patches.Rectangle(
            (-env.config.robot.length / 2.0, -env.config.robot.width / 2.0),
            env.config.robot.length,
            env.config.robot.width,
            facecolor=color,
            edgecolor="#d7191c" if robot.anchored else "black",
            linewidth=2.2 if robot.anchored else 0.8,
            alpha=0.5 if robot.layer > 0 else 0.9,
        )
        transform = transforms.Affine2D().rotate(robot.theta).translate(*robot.position) + axis.transData
        rectangle.set_transform(transform)
        axis.add_patch(rectangle)
        heading = robot.heading * env.config.robot.length * 0.55
        axis.plot(
            [robot.position[0], robot.position[0] + heading[0]],
            [robot.position[1], robot.position[1] + heading[1]],
            color="black",
            linewidth=0.8,
        )
        for angle_deg in env.config.sensor.ir_angles_deg:
            angle = robot.theta + np.deg2rad(angle_deg)
            end = robot.position + np.array([np.cos(angle), np.sin(angle)]) * env.config.sensor.ir_range
            axis.plot([robot.position[0], end[0]], [robot.position[1], end[1]], color="#2c7bb6", alpha=0.12)
        if env.config.sensor.downward_ir_enabled:
            axis.scatter(
                [robot.position[0]],
                [robot.position[1]],
                marker="v",
                color="#7b3294",
                s=16,
                zorder=9,
                linewidths=0.0,
            )

    axis.set(xlim=(0.0, field.length), ylim=(0.0, field.width), aspect="equal")
    axis.set_title(
        f"step={env.step_count} target={env.target_load:.2f} capacity={env.current_capacity:.2f} "
        f"progress={env.current_progress:.2f} span={env.graph.spans}"
    )
    axis.text(
        0.01,
        0.01,
        f"본체색: layer 0 (어두움) → {env.config.robot.max_layer} (밝음); 보라색: 하향 IR",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    return axis


def save_frame(env: BridgeEnv, path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 6))
    draw_environment(env, axis)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def record_episode(env: BridgeEnv, policy, path: str | Path, max_steps: int | None = None) -> dict:
    observation, _ = env.reset()
    frame_images: list[np.ndarray] = []
    figure, axis = plt.subplots(figsize=(11, 6))
    hidden = None
    limit = max_steps or env.config.max_steps
    final_info = env.info()
    for _ in range(limit):
        action, hidden = policy(observation, hidden, env.action_masks())
        observation, _, terminated, truncated, final_info = env.step(action)
        draw_environment(env, axis)
        figure.canvas.draw()
        frame_images.append(np.asarray(figure.canvas.buffer_rgba()).copy())
        if terminated or truncated:
            break
    plt.close(figure)
    output_figure, output_axis = plt.subplots(figsize=(11, 6))
    output_axis.axis("off")
    image = output_axis.imshow(frame_images[0])

    def update(frame_index: int):
        image.set_data(frame_images[frame_index])
        return (image,)

    movie = animation.FuncAnimation(output_figure, update, frames=len(frame_images), interval=80, blit=True)
    movie.save(path, writer=animation.PillowWriter(fps=12))
    plt.close(output_figure)
    return final_info
