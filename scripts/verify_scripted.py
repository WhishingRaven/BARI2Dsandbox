#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.robot import DiscreteAction
from bari2d.utils.config import ExperimentConfig
from bari2d.utils.scripted import assemble_parallel_bridge


def main() -> None:
    config = ExperimentConfig()
    env = BridgeEnv(config.environment)
    env.reset(seed=11)
    start = env.robots[0].position.copy()
    actions = np.full(env.config.robot.count, int(DiscreteAction.IDLE))
    actions[0] = int(DiscreteAction.FORWARD_LEFT)
    env.step(actions)
    assert not np.allclose(start, env.robots[0].position)
    assert env.robots[0].theta != 0.0

    env.reset(seed=11)
    assemble_parallel_bridge(env, rows=1)
    weak = env.accurate_evaluator.evaluate(env.graph).capacity
    assert env.graph.spans and weak > 0.0
    env.reset(seed=11)
    assemble_parallel_bridge(env, rows=2)
    strong = env.accurate_evaluator.evaluate(env.graph).capacity
    assert strong > weak
    print({"motion": True, "span": True, "weak_capacity": weak, "strong_capacity": strong})


if __name__ == "__main__":
    main()
