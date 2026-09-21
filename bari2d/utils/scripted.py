from __future__ import annotations

import numpy as np

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.robot import RobotState


def assemble_parallel_bridge(env: BridgeEnv, rows: int = 1, anchor: bool = True) -> list[RobotState]:
    """Build a topology-neutral scripted fixture for simulator verification only.

    This helper is never available to learned actors and is not used by rewards.
    """
    if rows < 1:
        raise ValueError("rows must be positive")
    robot_config = env.config.robot
    chain_length = int(np.ceil(env.field.gap_width / (robot_config.length * 0.82))) + 1
    required = rows * chain_length
    if required > robot_config.count:
        raise ValueError(f"Need {required} robots for {rows} rows, only {robot_config.count} configured")
    states: list[RobotState] = []
    left, right = env.field.boundaries(0.0)
    longitudinal_positions = np.linspace(left - robot_config.length * 0.3, right + robot_config.length * 0.3, chain_length)
    row_spacing = robot_config.width * 1.15
    for row in range(rows):
        tangent = (row - (rows - 1) / 2.0) * row_spacing
        for longitudinal in longitudinal_positions:
            position = env.field.center + env.field.normal * longitudinal + env.field.tangent * tangent
            states.append(
                RobotState(
                    len(states),
                    position,
                    env.field.orientation,
                    latent=np.zeros(env.config.latent_dim, dtype=np.float32),
                )
            )
    while len(states) < robot_config.count:
        robot_id = len(states)
        column = robot_id - required
        position = env.field.center - env.field.normal * (env.field.gap_width / 2.0 + 1.5 + column * 1.1)
        position[0] = np.clip(position[0], robot_config.length / 2.0, env.field.length - robot_config.length / 2.0)
        position[1] = np.clip(position[1], robot_config.width / 2.0, env.field.width - robot_config.width / 2.0)
        states.append(
            RobotState(
                robot_id,
                position,
                env.field.orientation,
                latent=np.zeros(env.config.latent_dim, dtype=np.float32),
            )
        )
    env.contact_model.reset()
    env.set_robot_states(states)
    if anchor:
        for robot in env.robots[:required]:
            env.contact_model.anchor(robot, env.robots, env.field)
        env.graph = env.contact_model.build_graph(env.robots, env.field, env.rng)
        env.current_progress = env.graph.spanning_progress(env.robots, env.field)
        env.current_capacity = env.fast_evaluator.evaluate(env.graph, env.config.load).capacity
    return states
