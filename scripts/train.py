#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from bari2d.rl.trainer import Trainer
from bari2d.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train decentralized bridge construction with recurrent MAPPO")
    parser.add_argument("--config", default="configs/structured_mappo.yaml")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument(
        "--resume",
        help="Checkpoint to resume from (weights, optimizers, update number and curriculum stage). "
        "--updates is the final target update, not additional updates.",
    )
    parser.add_argument(
        "--pretrained",
        help="Behaviour-cloning checkpoint from scripts/pretrain_bc.py; initialises the actor before PPO.",
    )
    parser.add_argument("--num-envs", type=int, help="Override training.num_envs.")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.num_envs is not None:
        config.training.num_envs = arguments.num_envs
    trainer = Trainer(config, arguments.device)
    if arguments.pretrained and arguments.resume:
        parser.error("use either --pretrained (start from a BC policy) or --resume (continue a run), not both")
    if arguments.pretrained:
        info = trainer.load_pretrained(arguments.pretrained)
        print(f"Actor initialised from {arguments.pretrained} {info}", flush=True)
    if arguments.resume:
        restored_update = trainer.load_checkpoint(arguments.resume)
        print(f"Resumed from {arguments.resume} at update {restored_update}, stage {trainer.curriculum.stage}", flush=True)
    history = trainer.train(arguments.updates)
    print(history[-1] if history else {})


if __name__ == "__main__":
    main()
