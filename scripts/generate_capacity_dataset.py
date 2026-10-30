from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from pathlib import Path

from bari2d.capacity_learning import CapacityDatasetBuilder, save_capacity_dataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate wrench-limited random-bridge graph data")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("runs/capacity/dataset.pt"))
    parser.add_argument(
        "--feature-mode",
        choices=("mechanics_informed", "geometry_only"),
        default="mechanics_informed",
    )
    parser.add_argument("--payload-mode", choices=("center", "random"), default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="Parallel generator processes")
    return parser.parse_args()



def _generate_chunk(count: int, seed: int, feature_mode: str, payload_mode: str):
    builder = CapacityDatasetBuilder(
        seed=seed,
        feature_mode=feature_mode,
        payload_mode=payload_mode,
    )
    return builder.generate(count)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logger.info(
        "Starting dataset generation: samples=%d feature_mode=%s payload_mode=%s seed=%d workers=%d output=%s",
        args.samples,
        args.feature_mode,
        args.payload_mode,
        args.seed,
        args.workers,
        args.output,
    )
    builder = CapacityDatasetBuilder(
        seed=args.seed,
        feature_mode=args.feature_mode,
        payload_mode=args.payload_mode,
    )
    samples = []
    if args.workers <= 1 or args.samples <= 1:
        for index in range(args.samples):
            samples.append(builder.generate_sample())
            if (index + 1) % max(min(args.samples // 10, 100), 1) == 0 or index + 1 == args.samples:
                logger.info("Generated %d/%d samples", index + 1, args.samples)
    else:
        workers = min(args.workers, args.samples)
        base = args.samples // workers
        remainder = args.samples % workers
        counts = [base + (1 if worker < remainder else 0) for worker in range(workers)]
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _generate_chunk,
                    count,
                    args.seed + 1_000_003 * (worker + 1),
                    args.feature_mode,
                    args.payload_mode,
                )
                for worker, count in enumerate(counts)
                if count > 0
            ]
            for future in as_completed(futures):
                chunk = future.result()
                samples.extend(chunk)
                completed += len(chunk)
                logger.info("Generated %d/%d samples", completed, args.samples)

    capacities = [float(sample.target_capacity_kg) for sample in samples]
    metadata = {
        "schema_version": 2,
        "wrench_frame": "contact_local",
        "feature_mode": args.feature_mode,
        "payload_mode": args.payload_mode,
        "seed": args.seed,
        "node_feature_names": builder.node_feature_names,
        "edge_feature_names": builder.edge_feature_names,
        "capacity_min_kg": min(capacities) if capacities else None,
        "capacity_max_kg": max(capacities) if capacities else None,
        "capacity_mean_kg": sum(capacities) / len(capacities) if capacities else None,
    }
    output = save_capacity_dataset(args.output, samples, metadata=metadata)
    logger.info("Saved dataset: %s", output)
    if samples:
        logger.info(
            "Dataset summary: node_features=%d edge_features=%d capacity_kg(min=%.4f mean=%.4f max=%.4f)",
            samples[0].node_features.shape[1],
            samples[0].edge_features.shape[1],
            metadata["capacity_min_kg"],
            metadata["capacity_mean_kg"],
            metadata["capacity_max_kg"],
        )


if __name__ == "__main__":
    main()
