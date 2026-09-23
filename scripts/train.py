#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bari2d.rl.trainer import Trainer
from bari2d.utils.config import config_from_dict, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train decentralized bridge construction with recurrent MAPPO")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Training configuration (ignored when --resume is used).")
    parser.add_argument("--updates", type=int, help="Final update number; must exceed the resumed checkpoint update.")
    parser.add_argument("--resume", type=Path, help="Resume model and optimizer state from this training checkpoint.")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    arguments = parser.parse_args()
    if arguments.resume is None:
        trainer = Trainer(load_config(arguments.config), arguments.device)
    else:
        checkpoint = torch.load(arguments.resume, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "experiment_config" not in checkpoint:
            raise ValueError(f"Checkpoint has no experiment configuration: {arguments.resume}")
        trainer = Trainer(config_from_dict(checkpoint["experiment_config"]), arguments.device)
        resumed_update = trainer.load_checkpoint(arguments.resume)
        print(f"Resumed {arguments.resume} after update {resumed_update}", flush=True)
    history = trainer.train(arguments.updates)
    print(history[-1] if history else {})


if __name__ == "__main__":
    main()
