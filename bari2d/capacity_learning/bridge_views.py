"""Paired graph/top-down view for capacity-network evaluation.

The right panel depicts the *actual planar footprints* from the BARI2D
2.5-D robot state. It is not an image input to the GNN and is not a 3-D
physics render. Coordinates are expressed in the gap-aligned local frame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


COMPONENT_NAMES = ("Axial", "ShearU", "ShearV", "Torsion", "BendingU", "BendingV")


def _undirected(sample) -> set[tuple[int, int]]:
    edge_index = sample.edge_index.detach().cpu().numpy()
    return {tuple(sorted((int(a), int(b)))) for a, b in edge_index.T}


def _pair(sample, index: int) -> tuple[int, int] | None:
    if not 0 <= index < sample.edge_count:
        return None
    source, target = sample.edge_index[:, index].tolist()
    return tuple(sorted((int(source), int(target))))


def _critical_pairs(sample) -> set[tuple[int, int]]:
    mask = sample.critical_edge_mask.detach().cpu().numpy() > 0.5
    return {_pair(sample, i) for i in np.flatnonzero(mask)} - {None}


def _node_xy(sample) -> np.ndarray:
    return sample.node_features.detach().cpu().numpy()[:, 3:5]


def robot_layers(sample) -> dict[int, int]:
    """Actual integer BARI2D layers keyed by graph node index.

    Prefer the original RobotState.layer captured in the top-down snapshot.
    Backward compatibility: for schema-v2 datasets without the snapshot,
    node feature 5 is z / robot_height = layer + 0.5 exactly by construction.
    Do not use feature 8: it is layer / max_layer, not an integer layer.
    """
    features = sample.node_features.detach().cpu().numpy()
    robot_indices = np.flatnonzero(features[:, 0] > 0.5)
    stored = (sample.metadata.get("topdown_view") or {}).get("robot_layers", {})
    return {
        int(i): int(stored[str(i)]) if str(i) in stored
        else max(0, int(np.rint(float(features[i, 5]) - 0.5)))
        for i in robot_indices
    }


def _layer_colors(layers: dict[int, int]) -> dict[int, str]:
    # Distinct categorical colors for actual physical layers, not a continuous
    # palette based on normalized layer height.
    palette = ("#f6dfad", "#e6a271", "#a7cddb", "#bfb3d8", "#9abeac")
    return {level: palette[level % len(palette)] for level in set(layers.values())}


def _layer_roster(sample) -> list[str]:
    """Unambiguous legend including robots hidden underneath higher layers."""
    layers = robot_layers(sample)
    return [
        f"Layer {level}:  " + ", ".join(str(i) for i in sorted(layers) if layers[i] == level)
        for level in sorted(set(layers.values()))
    ]


def _payload_xy(sample) -> np.ndarray:
    view = sample.metadata.get("topdown_view") or {}
    if "payload_xy" in view:
        return np.asarray(view["payload_xy"], dtype=float)
    nodes = sample.node_features.detach().cpu().numpy()
    chosen = np.flatnonzero(nodes[:, 9] > 0.5)
    return nodes[int(chosen[0]), 3:5] if len(chosen) else np.array([0.0, 0.0])


def _contact_xy(sample, pair: tuple[int, int]) -> np.ndarray:
    view = sample.metadata.get("topdown_view") or {}
    for contact in view.get("contacts", []):
        if tuple(sorted((int(contact["source"]), int(contact["target"])))) == pair:
            return np.asarray(contact["xy"], dtype=float)
    pts = _node_xy(sample)
    a, b = pair
    # Bank virtual nodes represent an entire support, not a physical point;
    # use the robot footprint near the bank as a fallback.
    if sample.node_features[a, 0] < 0.5:
        return pts[b].copy()
    if sample.node_features[b, 0] < 0.5:
        return pts[a].copy()
    return (pts[a] + pts[b]) / 2.0


def _pair_capacities(sample, edge_capacities: np.ndarray | None) -> dict[tuple[int, int], float]:
    if edge_capacities is None:
        return {}
    result: dict[tuple[int, int], float] = {}
    for idx, value in enumerate(edge_capacities):
        pair = _pair(sample, idx)
        if pair is not None:
            result[pair] = min(float(value), result.get(pair, float("inf")))
    return result


def draw_capacity_graph(ax, sample, *, predicted_pair=None, edge_capacities=None, exact_critical=True) -> None:
    """Simplified node-link representation of precisely the same bridge."""
    nodes = sample.node_features.detach().cpu().numpy()
    positions = nodes[:, 3:5]
    per_pair = _pair_capacities(sample, edge_capacities)
    values = np.asarray(list(per_pair.values()), dtype=float)
    low, high = (float(np.percentile(values, 10)), float(np.percentile(values, 90))) if len(values) else (0., 1.)
    if high <= low:
        high = low + 1.
    cmap = plt.get_cmap("RdYlGn")
    for pair in sorted(_undirected(sample)):
        a, b = pair
        xy = positions[[a, b]]
        strength = per_pair.get(pair)
        color = cmap(np.clip((strength - low) / (high - low), 0, 1)) if strength is not None else "#8b9aa3"
        ax.plot(xy[:, 0], xy[:, 1], lw=2., c=color, zorder=1, alpha=.9)
    if exact_critical:
        for a, b in _critical_pairs(sample):
            xy = positions[[a, b]]
            ax.plot(xy[:, 0], xy[:, 1], color="#d42b4b", lw=5., zorder=2)
    if predicted_pair is not None:
        a, b = predicted_pair
        xy = positions[[a, b]]
        ax.plot(xy[:, 0], xy[:, 1], color="#376ce5", lw=2.8, ls="--", zorder=3)
    layers = robot_layers(sample)
    layer_colors = _layer_colors(layers)
    for i in sorted(layers, key=lambda idx: (layers[idx], idx)):
        level = layers[i]
        ax.scatter([positions[i, 0]], [positions[i, 1]], s=185,
                   c=layer_colors[level], edgecolor="#353535", lw=.9,
                   zorder=4 + level * .2)
        ax.annotate(f"{i}\nL{level}", positions[i], ha="center", va="center",
                    fontsize=6.2, linespacing=.85, color="#222222",
                    zorder=5 + level * .2)
    for i, row in enumerate(nodes):
        if row[1] > 0.5 or row[2] > 0.5:
            ax.scatter([positions[i, 0]], [positions[i, 1]], marker="s", s=195, c="#4b5055", zorder=5)
            ax.annotate("L" if row[1] > .5 else "R", positions[i], c="white", weight="bold", fontsize=10, ha="center", va="center", zorder=6)
    payload = _payload_xy(sample)
    ax.scatter([payload[0]], [payload[1]], marker="*", s=280, c="#f7c744", edgecolors="black", zorder=8)
    ax.set_title("Contact graph (network representation)", fontsize=13, pad=12)
    ax.set_xlabel("Longitudinal / gap width", fontsize=10)
    ax.set_ylabel("Tangent / gap width", fontsize=10)
    ax.grid(alpha=.15, zorder=0)


def draw_bridge_topdown(ax, sample, *, predicted_pair=None, show_robot_ids=True) -> None:
    """Real robot bodies, orientation and overlapping layers seen from above.

    This is reconstructed from the same node poses used to construct the graph.
    Geometry metadata (if available) preserves exact physical load/contact points.
    """
    features = sample.node_features.detach().cpu().numpy()
    view = sample.metadata.get("topdown_view") or {}
    gap = float(view.get("gap_width_m", sample.metadata.get("gap_width", 4.5)))
    length = float(view.get("robot_length_m", 0.9)) / max(gap, 1e-9)
    width = float(view.get("robot_width_m", 0.42)) / max(gap, 1e-9)
    is_robot = features[:, 0] > 0.5
    positions = features[:, 3:5]
    robot_positions = positions[is_robot]
    y_min = float(robot_positions[:, 1].min()) if len(robot_positions) else -.3
    y_max = float(robot_positions[:, 1].max()) if len(robot_positions) else .3
    pad = max(.10, 1.2 * width)
    y_low, y_high = min(-.25, y_min - pad), max(.25, y_max + pad)
    x_low = min(-.75, float(robot_positions[:, 0].min()) - .16)
    x_high = max(.75, float(robot_positions[:, 0].max()) + .16)
    ax.set_facecolor("#f8f8f6")
    ax.axvspan(x_low, -.5, color="#e2dac9", alpha=1., zorder=-5)
    ax.axvspan(-.5, .5, color="#dcebf0", alpha=1., zorder=-5)
    ax.axvspan(.5, x_high, color="#e2dac9", alpha=1., zorder=-5)
    for bank_x in (-.5, .5):
        ax.plot([bank_x, bank_x], [y_low, y_high], color="#7c8c93", ls="--", lw=1.5, zorder=0)
    layers = robot_layers(sample)
    layer_colors = _layer_colors(layers)
    order = sorted(np.flatnonzero(is_robot), key=lambda i: (layers[i], i))
    for idx in order:
        node = features[idx]
        px, py = positions[idx]
        yaw = float(np.arctan2(node[7], node[6]))
        co, si = np.cos(yaw), np.sin(yaw)
        corners = np.array([[-length / 2, -width / 2], [length / 2, -width / 2], [length / 2, width / 2], [-length / 2, width / 2]])
        rotated = corners @ np.array([[co, si], [-si, co]]) + np.array([px, py])
        # Highest actual robot layer is drawn on top of lower, occluded bodies.
        layer = layers[idx]
        fill = layer_colors[layer]
        ax.add_patch(patches.Polygon(rotated, closed=True, facecolor=fill, edgecolor="#37434c", linewidth=.95, zorder=5 + layer * 3))
        # Heading tip is a genuine visual indication of robot yaw.
        arrow_length = length * .37
        ax.plot([px, px + co * arrow_length], [py, py + si * arrow_length],
                color="#33424b", lw=1.2, zorder=6 + layer * 3)
        if show_robot_ids:
            ax.annotate(f"{idx}\nL{layer}", (px, py), ha="center", va="center",
                        fontsize=6.1, linespacing=.82, color="#1b282f",
                        weight="medium", zorder=7 + layer * 3,
                        bbox={"boxstyle": "round,pad=0.10", "facecolor": "white",
                              "edgecolor": "none", "alpha": 0.72})
    for pair, color, radius, label in [
        *((pair, "#d42b4b", 65, "Exact critical") for pair in sorted(_critical_pairs(sample))),
        *(([predicted_pair, "#376ce5", 30, "GNN critical"],) if predicted_pair is not None else ()),
    ]:
        point = _contact_xy(sample, pair)
        ax.scatter([point[0]], [point[1]], s=radius, facecolors="none", edgecolors=color, linewidths=2.3, zorder=20, label=label)
    payload = _payload_xy(sample)
    ax.scatter([payload[0]], [payload[1]], marker="*", s=300, color="#f7c744", edgecolors="black", linewidth=1.1, zorder=22, label="Payload")
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.set_title("Top-down view (ID / actual layer)", fontsize=13, pad=12)
    ax.set_xlabel("Longitudinal / gap width", fontsize=10)
    ax.set_ylabel("Tangent / gap width", fontsize=10)
    ax.grid(alpha=.1, zorder=-1)
    ax.set_aspect("equal", adjustable="box")


def save_paired_view(sample, path: str | Path, *, predicted_kg: float | None = None,
                     predicted_pair: tuple[int, int] | None = None,
                     edge_capacities: np.ndarray | None = None,
                     title: str | None = None) -> Path:
    """Save aligned graph and physical top-down robot view for one bridge."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A dedicated roster avoids losing robot IDs underneath overlapping layers.
    fig = plt.figure(figsize=(18.5, 6.3))
    layout = fig.add_gridspec(2, 2, height_ratios=[8.8, 1.6], hspace=.36,
                             wspace=.19, left=.045, right=.985, top=.88, bottom=.055)
    axes = (fig.add_subplot(layout[0, 0]), fig.add_subplot(layout[0, 1]))
    roster_ax = fig.add_subplot(layout[1, :])
    draw_capacity_graph(axes[0], sample, predicted_pair=predicted_pair, edge_capacities=edge_capacities)
    draw_bridge_topdown(axes[1], sample, predicted_pair=predicted_pair)
    axes[0].set_xlim(*axes[1].get_xlim())
    axes[0].set_ylim(*axes[1].get_ylim())
    axes[0].set_aspect("equal", adjustable="box")
    actual = float(sample.target_capacity_kg.item())
    if title is None:
        title = f"Same bridge: exact {actual:.2f} kg"
        if predicted_kg is not None:
            title += f" / GNN {predicted_kg:.2f} kg"
    fig.suptitle(title, fontsize=14, weight="medium", y=.975)
    roster_ax.axis("off")
    roster_ax.text(0, 1.0, "Actual robot layers (ID by layer; includes hidden lower robots)",
                   fontsize=9.3, weight="bold", va="top", transform=roster_ax.transAxes)
    for row_idx, row in enumerate(_layer_roster(sample)):
        roster_ax.text(.015, .78 - row_idx * .24, row, fontsize=8.4,
                       va="top", transform=roster_ax.transAxes)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path
