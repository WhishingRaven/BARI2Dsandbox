from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from bari2d.capacity_learning import CapacityDatasetBuilder, CapacityGNN, load_capacity_dataset
from bari2d.capacity_learning.bridge_views import save_paired_view


COMPONENT_NAMES = ("Axial", "ShearU", "ShearV", "Torsion", "BendingU", "BendingV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained bridge-capacity GNN against exact statics labels and "
            "generate publication/presentation-friendly diagnostic figures."
        )
    )
    parser.add_argument("checkpoint", type=Path, help="CapacityGNN checkpoint, e.g. mechanics_gnn.pt")
    parser.add_argument("--dataset", type=Path, default=None, help="Optional saved held-out capacity dataset (.pt)")
    parser.add_argument("--samples", type=int, default=200, help="Held-out procedural samples when --dataset is omitted")
    parser.add_argument("--seed", type=int, default=20000, help="Held-out procedural seed")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/capacity/evaluation"))
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: Path, device: torch.device) -> tuple[CapacityGNN, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = ("model_state_dict", "node_size", "edge_size", "hidden_size", "layers")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Not a CapacityGNN checkpoint; missing keys: {missing}")
    model = CapacityGNN(
        node_size=int(checkpoint["node_size"]),
        edge_size=int(checkpoint["edge_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        layers=int(checkpoint["layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_samples(args: argparse.Namespace, checkpoint: dict[str, Any]):
    feature_mode = str(checkpoint.get("feature_mode", "mechanics_informed"))
    payload_mode = str(checkpoint.get("payload_mode", "random"))
    if args.dataset is not None:
        samples, metadata = load_capacity_dataset(args.dataset)
        if not samples:
            raise ValueError("Dataset is empty")
        dataset_feature_mode = metadata.get("feature_mode", samples[0].metadata.get("feature_mode"))
        dataset_payload_mode = metadata.get("payload_mode", samples[0].metadata.get("payload_mode"))
        if dataset_feature_mode is not None and str(dataset_feature_mode) != feature_mode:
            raise ValueError(
                f"Dataset feature_mode={dataset_feature_mode!r} does not match checkpoint feature_mode={feature_mode!r}"
            )
        if dataset_payload_mode is not None and str(dataset_payload_mode) != payload_mode:
            print(
                f"warning: dataset payload_mode={dataset_payload_mode!r} differs from checkpoint payload_mode={payload_mode!r}"
            )
        return samples, metadata

    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    builder = CapacityDatasetBuilder(
        feature_mode=feature_mode,
        payload_mode=payload_mode,
        seed=args.seed,
    )
    samples = builder.generate(args.samples)
    metadata = {
        "source": "procedural_held_out",
        "seed": args.seed,
        "feature_mode": feature_mode,
        "payload_mode": payload_mode,
        "schema_version": 2,
        "wrench_frame": "contact_local",
    }
    return samples, metadata


def sample_forward(model: CapacityGNN, sample, device: torch.device) -> dict[str, Any]:
    node_features = sample.node_features.to(device)
    edge_index = sample.edge_index.to(device)
    edge_features = sample.edge_features.to(device)
    node_graph = torch.zeros(sample.node_count, dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(node_features, edge_index, edge_features, node_graph)
    prediction = float(output.capacity_kg[0].cpu().item())
    component_pred = int(output.critical_component_logits[0].argmax().cpu().item())
    edge_logits = output.critical_edge_logits.detach().cpu().numpy()
    edge_capacity_pred = torch.expm1(output.edge_log_capacity).clamp_min(0.0).detach().cpu().numpy()
    predicted_edge = int(np.argmax(edge_logits)) if edge_logits.size else -1
    return {
        "prediction_kg": prediction,
        "component_pred": component_pred,
        "edge_logits": edge_logits,
        "edge_capacity_pred": edge_capacity_pred,
        "predicted_edge": predicted_edge,
    }


def pair_for_edge(sample, edge_idx: int) -> tuple[int, int] | None:
    if edge_idx < 0 or edge_idx >= sample.edge_count:
        return None
    source = int(sample.edge_index[0, edge_idx].item())
    target = int(sample.edge_index[1, edge_idx].item())
    return (source, target) if source <= target else (target, source)


def exact_critical_pairs(sample) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    mask = sample.critical_edge_mask.detach().cpu().numpy() > 0.5
    for edge_idx in np.flatnonzero(mask):
        pair = pair_for_edge(sample, int(edge_idx))
        if pair is not None:
            pairs.add(pair)
    return pairs


def compute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = np.asarray([row["exact_kg"] for row in rows], dtype=float)
    pred = np.asarray([row["predicted_kg"] for row in rows], dtype=float)
    error = pred - exact
    abs_error = np.abs(error)
    rel = abs_error / np.maximum(exact, 0.1)
    ss_res = float(np.sum((pred - exact) ** 2))
    ss_tot = float(np.sum((exact - exact.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    pearson = float(np.corrcoef(exact, pred)[0, 1]) if len(rows) > 1 and np.std(exact) > 0 and np.std(pred) > 0 else float("nan")
    component_accuracy = float(np.mean([row["component_correct"] for row in rows]))
    critical_edge_accuracy = float(np.mean([row["edge_correct"] for row in rows]))
    return {
        "samples": len(rows),
        "exact_capacity_mean_kg": float(exact.mean()),
        "predicted_capacity_mean_kg": float(pred.mean()),
        "mae_kg": float(abs_error.mean()),
        "median_ae_kg": float(np.median(abs_error)),
        "rmse_kg": float(np.sqrt(np.mean(error**2))),
        "mape_percent": float(100.0 * rel.mean()),
        "r2": r2,
        "pearson_r": pearson,
        "critical_component_accuracy": component_accuracy,
        "critical_edge_top1": critical_edge_accuracy,
        "overprediction_fraction": float(np.mean(pred > exact)),
        "overprediction_gt_10pct_fraction": float(np.mean(pred > 1.10 * exact)),
        "overprediction_gt_20pct_fraction": float(np.mean(pred > 1.20 * exact)),
        "mean_signed_error_kg": float(error.mean()),
        "max_overprediction_kg": float(np.max(error)),
    }


def plot_parity(rows: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    exact = np.asarray([row["exact_kg"] for row in rows])
    pred = np.asarray([row["predicted_kg"] for row in rows])
    robot_count = np.asarray([row["robot_count"] for row in rows])
    low = 0.0
    high = float(max(exact.max(), pred.max()) * 1.05)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    scatter = ax.scatter(exact, pred, c=robot_count, cmap="viridis", alpha=0.78, edgecolors="none", s=34)
    ax.plot([low, high], [low, high], "--", color="black", linewidth=1.2, label="Ideal: prediction = exact")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Exact maximum payload (kg)")
    ax.set_ylabel("GNN predicted maximum payload (kg)")
    ax.set_title("Bridge capacity prediction on held-out bridges")
    ax.grid(alpha=0.2)
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("Number of robots")
    text = (
        f"N={summary['samples']}\n"
        f"MAE={summary['mae_kg']:.2f} kg\n"
        f"MAPE={summary['mape_percent']:.1f}%\n"
        f"R²={summary['r2']:.3f}\n"
        f"critical edge={100*summary['critical_edge_top1']:.1f}%"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_error_vs_size(rows: list[dict[str, Any]], path: Path) -> None:
    counts = np.asarray([row["robot_count"] for row in rows], dtype=int)
    exact = np.asarray([row["exact_kg"] for row in rows], dtype=float)
    pred = np.asarray([row["predicted_kg"] for row in rows], dtype=float)
    rel_percent = 100.0 * np.abs(pred - exact) / np.maximum(exact, 0.1)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.scatter(counts, rel_percent, alpha=0.45, s=28, label="Held-out bridge")
    unique = np.unique(counts)
    medians = [float(np.median(rel_percent[counts == value])) for value in unique]
    ax.plot(unique, medians, "o-", linewidth=1.8, markersize=4.5, label="Median at robot count")
    ax.set_xlabel("Number of robots in bridge")
    ax.set_ylabel("Absolute relative capacity error (%)")
    ax.set_title("Generalization with bridge size")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_component_confusion(rows: list[dict[str, Any]], path: Path) -> None:
    matrix = np.zeros((6, 6), dtype=int)
    for row in rows:
        matrix[int(row["component_exact"]), int(row["component_pred"])] += 1
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(6), COMPONENT_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(6), COMPONENT_NAMES)
    ax.set_xlabel("Predicted critical component")
    ax.set_ylabel("Exact critical component")
    ax.set_title("Critical failure component")
    threshold = matrix.max() * 0.55 if matrix.size else 0
    for i in range(6):
        for j in range(6):
            value = int(matrix[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > threshold else "black")
    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def unique_edge_predictions(sample, edge_capacity_pred: np.ndarray) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], list[float]] = {}
    for edge_idx in range(sample.edge_count):
        pair = pair_for_edge(sample, edge_idx)
        if pair is None:
            continue
        values.setdefault(pair, []).append(float(edge_capacity_pred[edge_idx]))
    return {pair: float(min(candidates)) for pair, candidates in values.items()}


def draw_bridge_example(sample, result: dict[str, Any], path: Path, label: str) -> None:
    features = sample.node_features.detach().cpu().numpy()
    positions = features[:, 3:5]
    is_robot = features[:, 0] > 0.5
    is_left = features[:, 1] > 0.5
    is_right = features[:, 2] > 0.5
    is_payload = features[:, 9] > 0.5
    exact_pairs = exact_critical_pairs(sample)
    pred_pair = pair_for_edge(sample, int(result["predicted_edge"]))
    edge_predictions = unique_edge_predictions(sample, result["edge_capacity_pred"])

    unique_pairs: set[tuple[int, int]] = set()
    for edge_idx in range(sample.edge_count):
        pair = pair_for_edge(sample, edge_idx)
        if pair is not None:
            unique_pairs.add(pair)

    finite_caps = np.asarray(list(edge_predictions.values()), dtype=float)
    if finite_caps.size:
        cap_low = float(np.percentile(finite_caps, 10))
        cap_high = float(np.percentile(finite_caps, 90))
        if cap_high <= cap_low:
            cap_high = cap_low + 1.0
    else:
        cap_low, cap_high = 0.0, 1.0
    cmap = plt.get_cmap("RdYlGn")

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    for pair in sorted(unique_pairs):
        a, b = pair
        x = [positions[a, 0], positions[b, 0]]
        y = [positions[a, 1], positions[b, 1]]
        cap = edge_predictions.get(pair, cap_high)
        normalized = np.clip((cap - cap_low) / max(cap_high - cap_low, 1.0e-9), 0.0, 1.0)
        ax.plot(x, y, color=cmap(normalized), linewidth=2.2, alpha=0.8, zorder=1)

    for pair in exact_pairs:
        a, b = pair
        ax.plot(
            [positions[a, 0], positions[b, 0]],
            [positions[a, 1], positions[b, 1]],
            color="crimson",
            linewidth=6.0,
            alpha=0.9,
            zorder=3,
            label="Exact critical contact",
        )
    if pred_pair is not None:
        a, b = pred_pair
        ax.plot(
            [positions[a, 0], positions[b, 0]],
            [positions[a, 1], positions[b, 1]],
            color="royalblue",
            linewidth=3.2,
            linestyle="--",
            alpha=0.95,
            zorder=4,
            label="GNN predicted critical contact",
        )

    robot_indices = np.flatnonzero(is_robot)
    ax.scatter(positions[robot_indices, 0], positions[robot_indices, 1], s=95, c="white", edgecolors="black", zorder=5)
    for idx in robot_indices:
        ax.text(positions[idx, 0], positions[idx, 1], str(idx), ha="center", va="center", fontsize=7, zorder=6)
    payload_indices = np.flatnonzero(is_payload)
    if payload_indices.size:
        idx = int(payload_indices[0])
        ax.scatter([positions[idx, 0]], [positions[idx, 1]], marker="*", s=300, c="gold", edgecolors="black", zorder=7, label="Payload location")
    bank_indices = np.flatnonzero(is_left | is_right)
    for idx in bank_indices:
        label_text = "L" if is_left[idx] else "R"
        ax.scatter([positions[idx, 0]], [positions[idx, 1]], marker="s", s=180, c="0.3", zorder=5)
        ax.text(positions[idx, 0], positions[idx, 1], label_text, color="white", ha="center", va="center", fontweight="bold", zorder=6)

    exact_component = COMPONENT_NAMES[int(sample.critical_component.item())]
    pred_component = COMPONENT_NAMES[int(result["component_pred"])]
    exact_kg = float(sample.target_capacity_kg.item())
    pred_kg = float(result["prediction_kg"])
    ax.set_title(
        f"{label}: exact {exact_kg:.2f} kg vs GNN {pred_kg:.2f} kg\n"
        f"critical component: exact {exact_component} / predicted {pred_component}"
    )
    ax.set_xlabel("Gap-longitudinal coordinate / gap width")
    ax.set_ylabel("Gap-tangent coordinate / gap width")
    ax.grid(alpha=0.15)
    ax.set_aspect("equal", adjustable="datalim")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    if dedup:
        ax.legend(dedup.values(), dedup.keys(), loc="best", fontsize=8)
    ax.text(
        0.01,
        0.01,
        "Edge color: GNN-predicted local failure capacity (red=lower, green=higher)",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint, device)
    samples, dataset_metadata = load_samples(args, checkpoint)

    feature_mode = str(checkpoint.get("feature_mode", "unknown"))
    payload_mode = str(checkpoint.get("payload_mode", "unknown"))
    print(f"device: {device}")
    print(f"feature_mode: {feature_mode}")
    print(f"payload_mode: {payload_mode}")
    if feature_mode == "mechanics_informed":
        print(
            "NOTE: mechanics_informed includes solved wrench response in the input. "
            "Treat this checkpoint as a mechanics/control baseline, not evidence that geometry-only mechanics were learned."
        )
    if checkpoint.get("test_metrics") is not None:
        print(f"checkpoint stored test_metrics: {checkpoint['test_metrics']}")

    rows: list[dict[str, Any]] = []
    forward_results: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        result = sample_forward(model, sample, device)
        forward_results.append(result)
        exact = float(sample.target_capacity_kg.item())
        predicted = float(result["prediction_kg"])
        component_exact = int(sample.critical_component.item())
        component_pred = int(result["component_pred"])
        pred_pair = pair_for_edge(sample, int(result["predicted_edge"]))
        exact_pairs = exact_critical_pairs(sample)
        row = {
            "index": index,
            "exact_kg": exact,
            "predicted_kg": predicted,
            "signed_error_kg": predicted - exact,
            "absolute_error_kg": abs(predicted - exact),
            "relative_error_percent": 100.0 * abs(predicted - exact) / max(exact, 0.1),
            "robot_count": int(sample.metadata.get("robot_count", int((sample.node_features[:, 0] > 0.5).sum().item()))),
            "edge_count": int(sample.metadata.get("undirected_edge_count", sample.edge_count // 2)),
            "component_exact": component_exact,
            "component_pred": component_pred,
            "component_correct": component_exact == component_pred,
            "edge_correct": pred_pair in exact_pairs if pred_pair is not None else False,
        }
        rows.append(row)
        if (index + 1) % 50 == 0 or index + 1 == len(samples):
            print(f"evaluated {index + 1}/{len(samples)}")

    summary = compute_summary(rows)
    summary["checkpoint"] = str(args.checkpoint)
    summary["feature_mode"] = feature_mode
    summary["payload_mode"] = payload_mode
    summary["wrench_frame"] = checkpoint.get("wrench_frame")
    summary["schema_version"] = checkpoint.get("schema_version")
    summary["checkpoint_stored_test_metrics"] = checkpoint.get("test_metrics")
    summary["dataset_metadata"] = dataset_metadata

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)
    fieldnames = list(rows[0].keys())
    with (args.output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_parity(rows, summary, args.output_dir / "01_exact_vs_gnn.png")
    plot_error_vs_size(rows, args.output_dir / "02_error_vs_robot_count.png")
    plot_component_confusion(rows, args.output_dir / "03_critical_component_confusion.png")

    absolute_errors = np.asarray([row["absolute_error_kg"] for row in rows])
    median_target = float(np.median(absolute_errors))
    representative_idx = int(np.argmin(np.abs(absolute_errors - median_target)))
    signed_errors = np.asarray([row["signed_error_kg"] for row in rows])
    worst_over_idx = int(np.argmax(signed_errors))
    draw_bridge_example(
        samples[representative_idx],
        forward_results[representative_idx],
        args.output_dir / "04_representative_bridge.png",
        "Representative held-out bridge",
    )
    draw_bridge_example(
        samples[worst_over_idx],
        forward_results[worst_over_idx],
        args.output_dir / "05_worst_overprediction_bridge.png",
        "Worst GNN overprediction in held-out set",
    )

    # Each original network plot now has a companion showing the same bridge
    # as real simulator robot rectangles, bank terrain and payload location.
    save_paired_view(
        samples[representative_idx],
        args.output_dir / "04_representative_bridge_dual_view.png",
        predicted_kg=float(forward_results[representative_idx]["prediction_kg"]),
        predicted_pair=pair_for_edge(samples[representative_idx], int(forward_results[representative_idx]["predicted_edge"])),
        edge_capacities=forward_results[representative_idx]["edge_capacity_pred"],
    )
    save_paired_view(
        samples[worst_over_idx],
        args.output_dir / "05_worst_overprediction_bridge_dual_view.png",
        predicted_kg=float(forward_results[worst_over_idx]["prediction_kg"]),
        predicted_pair=pair_for_edge(samples[worst_over_idx], int(forward_results[worst_over_idx]["predicted_edge"])),
        edge_capacities=forward_results[worst_over_idx]["edge_capacity_pred"],
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"saved figures and metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()
