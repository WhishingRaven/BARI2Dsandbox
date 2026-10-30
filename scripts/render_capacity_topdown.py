"""Render the physical top-down bridge geometry alongside network graph images.

Example (reconstruct the *same* cases as a single-worker evaluation):

  python scripts/render_capacity_topdown.py --seed 20000 --indices 73 105 \
      --graph-images 04_representative_bridge.png 05_worst_overprediction_bridge.png

For ordinary evaluations use scripts/evaluate_capacity_gnn.py, which now saves
paired graph/top-down figures automatically, using actual GNN predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from bari2d.capacity_learning.dataset import CapacityDatasetBuilder
from bari2d.capacity_learning.bridge_views import draw_bridge_topdown, save_paired_view, _layer_roster


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce seeded capacity samples and show real 2.5-D robot top-down footprints")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--indices", type=int, nargs="+", default=[73, 105])
    parser.add_argument("--feature-mode", choices=("mechanics_informed", "geometry_only"), default="mechanics_informed")
    parser.add_argument("--payload-mode", choices=("center", "random"), default="random")
    parser.add_argument("--graph-images", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/capacity/topdown"))
    args = parser.parse_args()
    if not args.indices or min(args.indices) < 0:
        parser.error("--indices must list one or more nonnegative sample indices")
    if args.graph_images and len(args.graph_images) != len(args.indices):
        parser.error("--graph-images must contain exactly one PNG for each requested index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    builder = CapacityDatasetBuilder(feature_mode=args.feature_mode, payload_mode=args.payload_mode, seed=args.seed)
    chosen = set(args.indices)
    graph_files = dict(zip(args.indices, args.graph_images))
    for index in range(max(chosen) + 1):
        sample = builder.generate_sample()
        if index not in chosen:
            continue
        out = args.output_dir
        png = out / f"bridge_{index:03d}_topdown.png"
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        draw_bridge_topdown(ax, sample)
        fig.suptitle(f"Bridge #{index} | exact maximum payload {float(sample.target_capacity_kg):.2f} kg", fontsize=12)
        fig.subplots_adjust(left=.08, right=.98, top=.88, bottom=.31)
        for row_idx, row in enumerate(_layer_roster(sample)):
            fig.text(.095, .205 - row_idx * .052, row, ha="left", va="center", fontsize=8)
        fig.savefig(png, dpi=210, facecolor="white")
        plt.close(fig)
        save_paired_view(sample, out / f"bridge_{index:03d}_graph_and_topdown.png")
        if index in graph_files:
            image_path = graph_files[index]
            if not image_path.exists():
                parser.error(f"Graph image does not exist: {image_path}")
            left = Image.open(image_path).convert("RGB")
            right = Image.open(png).convert("RGB")
            # Rescale without cropping either original visualization.
            height = 1000
            left = left.resize((round(left.width * height / left.height), height), Image.Resampling.LANCZOS)
            right = right.resize((round(right.width * height / right.height), height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (left.width + right.width + 32, height), "white")
            canvas.paste(left, (0, 0))
            canvas.paste(right, (left.width + 32, 0))
            canvas.save(out / f"bridge_{index:03d}_original_graph_vs_real_topdown.png")
        print(f"bridge {index}: {sample.metadata['robot_count']} robots, {float(sample.target_capacity_kg):.3f} kg; saved to {out}")


if __name__ == "__main__":
    main()
