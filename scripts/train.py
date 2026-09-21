#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from bari2d.rl.trainer import Trainer
from bari2d.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train decentralized bridge construction with recurrent MAPPO")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    arguments = parser.parse_args()
    trainer = Trainer(load_config(arguments.config), arguments.device)
    history = trainer.train(arguments.updates)
    print(history[-1] if history else {})


if __name__ == "__main__":
    main()

