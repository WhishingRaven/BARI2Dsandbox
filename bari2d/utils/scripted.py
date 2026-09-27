from __future__ import annotations

import numpy as np

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.robot import DiscreteAction, RobotState


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


def park_robots_on_left_bank(env: BridgeEnv, robot_ids: list[int], keep_clear: float = 0.0) -> list[RobotState]:
    """Place spare robots in a non-overlapping grid on the left bank, away from the gap."""
    robot_config = env.config.robot
    left, _ = env.field.boundaries(0.0)
    edge = env.field.center + env.field.normal * left
    column_step = robot_config.length * 1.25
    row_step = robot_config.width * 1.6
    rows_available = max(int(env.field.width / row_step) - 1, 1)
    states: list[RobotState] = []
    for slot, robot_id in enumerate(robot_ids):
        column, row = divmod(slot, rows_available)
        tangent = (row - (rows_available - 1) / 2.0) * row_step
        if abs(tangent) < keep_clear:
            tangent = np.sign(tangent or 1.0) * (keep_clear + row_step)
        position = edge - env.field.normal * (2.0 + column * column_step) + env.field.tangent * tangent
        states.append(
            RobotState(robot_id, position, env.field.orientation, latent=np.zeros(env.config.latent_dim, dtype=np.float32))
        )
    return states


def assemble_static_chain(env: BridgeEnv, rows: int = 1, setback: float = 0.01, bearing: float = 0.05) -> list[RobotState]:
    """Build the bridge that grip-and-extend construction produces, for static physics checks.

    Row ``r`` starts with a robot whose center sits ``setback`` behind the left
    cliff edge and grips the bank.  Every further robot is attached end-to-end
    to the previous one until the chain rests at least ``bearing`` on the right
    bank.  Spare robots are parked on the left bank.  Never used by rewards.
    """
    if not env.static_physics:
        raise ValueError("assemble_static_chain requires environment.physics.model == 'static'")
    if rows < 1:
        raise ValueError("rows must be positive")
    robot_config = env.config.robot
    length = robot_config.length
    left, right = env.field.boundaries(0.0)
    chain_length = int(np.ceil((right - left + setback + bearing - length / 2.0) / length)) + 1
    required = rows * chain_length
    if required > robot_config.count:
        raise ValueError(f"Need {required} robots for {rows} rows, only {robot_config.count} configured")
    row_spacing = robot_config.width * 1.15
    states: list[RobotState] = []
    for row in range(rows):
        tangent = (row - (rows - 1) / 2.0) * row_spacing
        for link in range(chain_length):
            longitudinal = left - setback + link * length
            position = env.field.center + env.field.normal * longitudinal + env.field.tangent * tangent
            states.append(
                RobotState(len(states), position, env.field.orientation, latent=np.zeros(env.config.latent_dim, dtype=np.float32))
            )
    keep_clear = rows * row_spacing / 2.0 + robot_config.width
    states += park_robots_on_left_bank(env, list(range(required, robot_config.count)), keep_clear)
    env.contact_model.reset()
    env.set_robot_states(states)
    for row in range(rows):
        for link in range(chain_length):
            robot = env.robots[row * chain_length + link]
            partner = env.field.bank_contact(robot.corners(robot_config)) if link == 0 else None
            target = sorted(partner)[0] if partner else row * chain_length + link - 1
            env.contact_model._attach(robot, target, env.robots)  # type: ignore[attr-defined]
    env.set_robot_states(env.robots)
    return env.robots


def build_by_grip_and_extend(env: BridgeEnv, setback: float = 0.01, max_walk_steps: int = 200) -> dict[str, object]:
    """Construct a bridge through the real action interface (static physics only).

    Robot 0 drives to the cliff edge and grips the bank.  Each further robot
    is brought behind the chain root (standing in for the approach drive),
    then uses CLIMB, FORWARD and ANCHOR: it walks along the top of the chain to
    the tip and grips-and-extends.  Every step runs the full equilibrium check,
    so walker weight on a growing cantilever can break anchors.  This is a
    feasibility probe for the task, never used by rewards or policies.
    """
    if not env.static_physics:
        raise ValueError("build_by_grip_and_extend requires environment.physics.model == 'static'")
    robot_config = env.config.robot
    count = robot_config.count
    left, _ = env.field.boundaries(0.0)
    normal = env.field.normal
    root_position = env.field.center + normal * (left - setback)
    states = [RobotState(0, root_position.copy(), env.field.orientation, latent=np.zeros(env.config.latent_dim, dtype=np.float32))]
    states += park_robots_on_left_bank(env, list(range(1, count)), keep_clear=robot_config.width * 2.0)
    env.contact_model.reset()
    env.set_robot_states(states)
    idle = int(DiscreteAction.IDLE)
    result: dict[str, object] = {"success": False, "robots_used": 1, "steps": 0, "failure": None}

    def act(robot_id: int, action: DiscreteAction) -> bool:
        actions = np.full(count, idle, dtype=np.int64)
        actions[robot_id] = int(action)
        _, _, terminated, truncated, info = env.step(actions)
        result["steps"] = int(result["steps"]) + 1
        result["info"] = info
        if info["fallen_robots"] or info["anchor_failures"]:
            result["failure"] = f"fallen={info['fallen_robots']} anchor_failures={info['anchor_failures']}"
        return bool(terminated or truncated)

    def _finish(outcome: dict[str, object]) -> dict[str, object]:
        info = outcome["info"]
        outcome["success"] = bool(info["success"])  # type: ignore[index]
        outcome["capacity"] = float(info["capacity"])  # type: ignore[index]
        if not outcome["success"]:
            outcome["failure"] = outcome["failure"] or "episode ended without a bridge"
        return outcome

    if act(0, DiscreteAction.ANCHOR) or not env.robots[0].anchored:
        result["failure"] = result["failure"] or "root could not grip the bank"
        return result
    chain = [0]
    for walker_id in range(1, count):
        walker = env.robots[walker_id]
        walker.position = env.robots[0].position - normal * (robot_config.length + 0.02)
        walker.theta = env.field.orientation
        walker.layer = 0
        walker.velocity = 0.0
        if act(walker_id, DiscreteAction.CLIMB) or walker.layer != 1:
            result["failure"] = result["failure"] or f"robot {walker_id} could not climb"
            return result
        frontier = env.robots[chain[-1]]
        for _ in range(max_walk_steps):
            if float((walker.position - frontier.position) @ frontier.heading) >= 0.1:
                break
            if act(walker_id, DiscreteAction.FORWARD):
                result["robots_used"] = len(chain) + 1
                return _finish(result)
            if walker.fallen or result["failure"]:
                result["failure"] = result["failure"] or f"robot {walker_id} lost footing"
                return result
        done = act(walker_id, DiscreteAction.ANCHOR)
        if result["failure"] or not walker.anchored:
            result["failure"] = result["failure"] or f"robot {walker_id} could not extend"
            return result
        chain.append(walker_id)
        result["robots_used"] = len(chain)
        if done:
            return _finish(result)
    result["failure"] = result["failure"] or "ran out of robots"
    return result
