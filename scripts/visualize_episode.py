#!/usr/bin/env python3
from __future__ import annotations

import argparse

from bari2d.env.bridge_env import BridgeEnv
from bari2d.utils.config import load_config
from bari2d.utils.scripted import assemble_parallel_bridge
from bari2d.utils.visualization import save_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a deterministic scripted bridge fixture")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--output", default="bridge.png")
    arguments = parser.parse_args()
    env = BridgeEnv(load_config(arguments.config).environment)
    env.reset()
    assemble_parallel_bridge(env, arguments.rows)
    env.last_load_test = env.accurate_evaluator.evaluate(env.graph)
    env.current_capacity = env.last_load_test.capacity
    save_frame(env, arguments.output)
    print(arguments.output)


if __name__ == "__main__":
    main()
