from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colors, patches, transforms

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.robot import DiscreteAction
from bari2d.mechanics import LocalWrenchLimit, SpatialCapacityResult


def robot_layer_color(layer: int, max_layer: int) -> tuple[float, float, float, float]:
    """Map layers from red through violet in visible-spectrum order."""
    normalized = np.clip(layer / max(max_layer, 1), 0.0, 1.0)
    red, green, blue = colors.hsv_to_rgb((normalized * 0.75, 1.0, 1.0))
    return float(red), float(green), float(blue), 1.0


def robot_layer_zorder(layer: int, detail: int = 0) -> float:
    """Reserve a drawing band so every higher layer covers lower details."""
    return 10.0 + max(layer, 0) * 10.0 + detail


def draw_environment(
    env: BridgeEnv,
    axis: plt.Axes | None = None,
    *,
    contact_utilization: dict[frozenset[int | str], float] | None = None,
    critical_contact: tuple[int | str, int | str] | None = None,
    exact_capacity_kg: float | None = None,
    show_action_summary: bool = False,
) -> plt.Axes:
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

    critical_key = None if critical_contact is None else frozenset(critical_contact)
    utilization_map = contact_utilization or {}
    utilization_cmap = plt.get_cmap("RdYlGn_r")
    for edge in env.graph.edges.values():
        if not isinstance(edge.source, int) or not isinstance(edge.target, int):
            continue
        first = env.robots[edge.source].position
        second = env.robots[edge.target].position
        key = frozenset((edge.source, edge.target))
        utilization = utilization_map.get(key)
        if utilization is None:
            color = "#b2182b" if edge.kind == "anchor" else "#555555"
            width = 2.2 if edge.kind == "anchor" else 0.8
        else:
            color = utilization_cmap(float(np.clip(utilization, 0.0, 1.0)))
            width = 1.5 + 3.0 * float(np.clip(utilization, 0.0, 1.0))
        if key == critical_key:
            color = "#d73027"
            width = max(width, 5.0)
        axis.plot([first[0], second[0]], [first[1], second[1]], color=color, linewidth=width, alpha=0.9)

    load_path = [] if env.last_load_test is None else env.last_load_test.load_bearing_path
    for first_node, second_node in zip(load_path, load_path[1:]):
        if isinstance(first_node, int) and isinstance(second_node, int):
            first, second = env.robots[first_node].position, env.robots[second_node].position
            axis.plot([first[0], second[0]], [first[1], second[1]], color="#ffd92f", linewidth=4.0, alpha=0.8)

    for robot in sorted(env.robots, key=lambda item: (item.layer, item.robot_id)):
        layer_zorder = robot_layer_zorder(robot.layer)
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
            alpha=1.0,
            zorder=layer_zorder + 2,
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
            zorder=layer_zorder + 3,
        )
        for angle_deg in env.config.sensor.ir_angles_deg:
            angle = robot.theta + np.deg2rad(angle_deg)
            end = robot.position + np.array([np.cos(angle), np.sin(angle)]) * env.config.sensor.ir_range
            axis.plot(
                [robot.position[0], end[0]],
                [robot.position[1], end[1]],
                color="#2c7bb6",
                alpha=0.12,
                zorder=layer_zorder + 1,
            )
        if env.config.sensor.downward_ir_enabled:
            axis.scatter(
                [robot.position[0]],
                [robot.position[1]],
                marker="v",
                color="#7b3294",
                s=16,
                zorder=layer_zorder + 3,
                linewidths=0.0,
            )

    axis.set(xlim=(0.0, field.length), ylim=(0.0, field.width), aspect="equal")
    capacity_text = f"proxy_capacity={env.current_capacity:.2f}"
    if exact_capacity_kg is not None:
        capacity_text += f" exact_capacity={exact_capacity_kg:.2f} kg"
    axis.set_title(
        f"step={env.step_count} target={env.target_load:.2f} {capacity_text} "
        f"progress={env.current_progress:.2f} span={env.graph.spans}"
    )
    axis.text(
        0.01,
        0.01,
        f"body color: layer 0 (red) -> {env.config.robot.max_layer} (violet); triangle: downward IR",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    if show_action_summary:
        counts: dict[str, int] = {}
        for robot in env.robots:
            name = DiscreteAction(int(robot.previous_action)).name
            counts[name] = counts.get(name, 0) + 1
        summary = " | ".join(f"{name}:{count}" for name, count in sorted(counts.items()) if count)
        axis.text(
            0.99,
            0.01,
            summary,
            transform=axis.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    return axis


def contact_utilization_at_payload(
    result: SpatialCapacityResult,
    limits: LocalWrenchLimit,
    payload_kg: float | None = None,
) -> dict[frozenset[int | str], float]:
    """Return maximum local component utilization for each solved contact.

    Utilization 1.0 means one local axial/shear/torsion/bending component has
    reached its configured connector limit. Values above one indicate the
    requested payload exceeds that component limit.
    """
    mass = result.max_payload_kg if payload_kg is None else float(payload_kg)
    if not np.isfinite(mass):
        mass = 0.0
    base = result.base_response.contact_by_key()
    unit = result.unit_payload_response.contact_by_key()
    maxima = limits.as_array()
    output: dict[frozenset[int | str], float] = {}
    for key in set(base) | set(unit):
        base_wrench = base[key].local_wrench_on_source.as_array() if key in base else np.zeros(6)
        unit_wrench = unit[key].local_wrench_on_source.as_array() if key in unit else np.zeros(6)
        wrench = base_wrench + mass * unit_wrench
        ratios = np.zeros(6, dtype=float)
        finite = np.isfinite(maxima) & (maxima > 0.0)
        ratios[finite] = np.abs(wrench[finite]) / maxima[finite]
        output[key] = float(np.max(ratios))
    return output


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


def draw_capacity_result(
    env: BridgeEnv,
    result: SpatialCapacityResult,
    limits: LocalWrenchLimit,
    axis: plt.Axes | None = None,
    *,
    payload_kg: float | None = None,
) -> plt.Axes:
    """Draw the final bridge with connector utilization from exact statics."""
    mass = result.max_payload_kg if payload_kg is None else float(payload_kg)
    utilization = contact_utilization_at_payload(result, limits, mass)
    axis = draw_environment(
        env,
        axis,
        contact_utilization=utilization,
        critical_contact=result.critical_contact,
        exact_capacity_kg=mass,
        show_action_summary=True,
    )
    cmap = plt.get_cmap("RdYlGn_r")
    critical_key = None if result.critical_contact is None else frozenset(result.critical_contact)
    response_by_key = result.base_response.contact_by_key()
    response_by_key.update(result.unit_payload_response.contact_by_key())
    for key, response in response_by_key.items():
        if isinstance(response.source, int) and isinstance(response.target, int):
            continue
        robot_id = response.source if isinstance(response.source, int) else response.target
        if not isinstance(robot_id, int):
            continue
        start = env.robots[robot_id].position
        point = np.asarray(response.point[:2], dtype=float)
        value = utilization.get(key, 0.0)
        color = "#d73027" if key == critical_key else cmap(float(np.clip(value, 0.0, 1.0)))
        width = 5.0 if key == critical_key else 1.5 + 3.0 * float(np.clip(value, 0.0, 1.0))
        axis.plot([start[0], point[0]], [start[1], point[1]], color=color, linewidth=width, alpha=0.95)
    critical = "none"
    if result.critical_contact is not None:
        critical = f"{result.critical_contact} / {result.critical_component}"
    axis.text(
        0.5,
        0.99,
        f"connector utilization at {mass:.2f} kg; critical={critical}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )
    return axis
