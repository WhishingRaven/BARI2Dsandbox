from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from bari2d.capacity_learning import (
    CapacityGNN,
    InfiniteBridgeDataset,
    collate_capacity_graphs,
    load_capacity_dataset,
)
from bari2d.capacity_learning.training import capacity_metrics, capacity_multitask_loss, move_batch

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the bridge payload-capacity GNN")
    parser.add_argument("--dataset", type=Path, default=None, help="Optional finite .pt dataset")
    parser.add_argument("--feature-mode", choices=("mechanics_informed", "geometry_only"), default="mechanics_informed")
    parser.add_argument("--payload-mode", choices=("center", "random"), default="random")
    parser.add_argument("--steps", type=int, default=2000, help="Optimizer steps for infinite mode")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs for finite-dataset mode")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0, help="Procedural DataLoader workers")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("runs/capacity/capacity_gnn.pt"))
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_samples(samples, seed: int):
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    n = len(indices)
    n_train = max(1, int(0.8 * n))
    n_val = max(1, int(0.1 * n)) if n >= 10 else max(0, n - n_train)
    train = [samples[i] for i in indices[:n_train]]
    val = [samples[i] for i in indices[n_train : n_train + n_val]]
    test = [samples[i] for i in indices[n_train + n_val :]]
    if not val:
        val = train[: min(len(train), 4)]
    if not test:
        test = val
    return train, val, test


def make_model(first_sample, args, device):
    model = CapacityGNN(
        node_size=first_sample.node_features.shape[1],
        edge_size=first_sample.edge_features.shape[1],
        hidden_size=args.hidden_size,
        layers=args.layers,
    )
    return model.to(device)


def forward_loss(model, batch):
    output = model(
        batch["node_features"],
        batch["edge_index"],
        batch["edge_features"],
        batch["node_graph"],
    )
    return capacity_multitask_loss(output, batch)


def evaluate(model, loader, device, max_batches: int | None = None):
    model.eval()
    totals = {"mae_kg": 0.0, "median_ae_kg": 0.0, "mape": 0.0, "component_accuracy": 0.0, "critical_edge_top1": 0.0}
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            metrics = capacity_metrics(model, batch)
            for key, value in metrics.items():
                totals[key] += value
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    return {key: value / max(batches, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = choose_device(args.device)
    logger.info(
        "Starting capacity GNN training: mode=%s payload_mode=%s batch_size=%d device=%s seed=%d",
        "finite" if args.dataset is not None else "online",
        args.payload_mode,
        args.batch_size,
        device,
        args.seed,
    )

    if args.dataset is not None:
        samples, metadata = load_capacity_dataset(args.dataset)
        if not samples:
            raise ValueError("Dataset is empty")
        logger.info("Loaded %d samples from %s", len(samples), args.dataset)
        train_samples, val_samples, test_samples = split_samples(samples, args.seed)
        train_loader = DataLoader(train_samples, batch_size=args.batch_size, shuffle=True, collate_fn=collate_capacity_graphs)
        val_loader = DataLoader(val_samples, batch_size=args.batch_size, shuffle=False, collate_fn=collate_capacity_graphs)
        test_loader = DataLoader(test_samples, batch_size=args.batch_size, shuffle=False, collate_fn=collate_capacity_graphs)
        first_sample = samples[0]
        if metadata.get("feature_mode") is not None:
            args.feature_mode = str(metadata["feature_mode"])
        if metadata.get("payload_mode") is not None:
            args.payload_mode = str(metadata["payload_mode"])
        wrench_frame = str(metadata.get("wrench_frame", samples[0].metadata.get("wrench_frame", "unknown")))
        schema_version = int(metadata.get("schema_version", samples[0].metadata.get("schema_version", 1)))
        total_epochs = args.epochs
        online = False
    else:
        dataset = InfiniteBridgeDataset(
            seed=args.seed,
            feature_mode=args.feature_mode,
            payload_mode=args.payload_mode,
        )
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate_capacity_graphs,
            num_workers=max(args.workers, 0),
        )
        iterator = iter(train_loader)
        first_sample = next(iterator)["node_features"]
        # Build one direct sample to obtain feature dimensions without relying on
        # the packed batch's graph boundaries.
        probe_dataset = InfiniteBridgeDataset(seed=args.seed, feature_mode=args.feature_mode, payload_mode=args.payload_mode)
        first_sample_object = next(iter(probe_dataset))
        first_sample = first_sample_object
        total_epochs = 1
        online = True
        validation_stream = iter(
            InfiniteBridgeDataset(
                seed=args.seed + 999,
                feature_mode=args.feature_mode,
                payload_mode=args.payload_mode,
            )
        )
        validation_samples = [next(validation_stream) for _ in range(128)]
        val_loader = DataLoader(
            validation_samples,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_capacity_graphs,
        )
        test_loader = val_loader
        wrench_frame = "contact_local"
        schema_version = 2
        logger.info(
            "Online data: feature_mode=%s payload_mode=%s steps=%d workers=%d validation_samples=128",
            args.feature_mode,
            args.payload_mode,
            args.steps,
            args.workers,
        )

    model = make_model(first_sample, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    step = 0
    if online:
        log_interval = 100
        train_iterator = iter(train_loader)
        model.train()
        while step < args.steps:
            batch = move_batch(next(train_iterator), device)
            optimizer.zero_grad(set_to_none=True)
            loss = forward_loss(model, batch)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1
            if step == 1 or step % log_interval == 0 or step == args.steps:
                val = evaluate(model, val_loader, device)
                logger.info(
                    "step=%d/%d loss=%.5f val_mae=%.4fkg val_mape=%.2f%% component=%.1f%% edge=%.1f%%",
                    step,
                    args.steps,
                    loss.total.item(),
                    val["mae_kg"],
                    100 * val["mape"],
                    100 * val["component_accuracy"],
                    100 * val["critical_edge_top1"],
                )
                model.train()
    else:
        for epoch in range(1, total_epochs + 1):
            model.train()
            running = 0.0
            batches = 0
            log_interval = max(1, (len(train_loader) + 9) // 10)
            for batch in train_loader:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = forward_loss(model, batch)
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                running += float(loss.total.item())
                batches += 1
                step += 1
                if batches % log_interval == 0 or batches == len(train_loader):
                    logger.info(
                        "epoch=%d/%d batches=%d/%d train_loss=%.5f",
                        epoch,
                        total_epochs,
                        batches,
                        len(train_loader),
                        running / batches,
                    )
            val = evaluate(model, val_loader, device)
            logger.info(
                "epoch=%d/%d complete train_loss=%.5f val_mae=%.4fkg val_mape=%.2f%% component=%.1f%% edge=%.1f%%",
                epoch,
                total_epochs,
                running / max(batches, 1),
                val["mae_kg"],
                100 * val["mape"],
                100 * val["component_accuracy"],
                100 * val["critical_edge_top1"],
            )

    test = evaluate(model, test_loader, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "node_size": int(first_sample.node_features.shape[1]),
        "edge_size": int(first_sample.edge_features.shape[1]),
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "feature_mode": args.feature_mode,
        "payload_mode": args.payload_mode,
        "wrench_frame": wrench_frame,
        "schema_version": schema_version,
        "test_metrics": test,
    }
    torch.save(checkpoint, args.output)
    logger.info("Test metrics: %s", test)
    logger.info("Saved checkpoint: %s", args.output)


if __name__ == "__main__":
    main()
