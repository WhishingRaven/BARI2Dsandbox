"""A privileged scripted builder used only to generate demonstrations.

The teacher is a "weak but valid" construction procedure: one robot works at
a time and every other robot waits.

1. No robot grips the left bank: the robot closest to the cliff drives to the
   edge (via a staging point one robot length behind it), faces the gap and
   grips (ANCHOR).
2. A chain exists but does not reach the far bank: a robot already standing on
   the chain walks to its front end and grips-and-extends (FORWARD, ANCHOR);
   otherwise the nearest free robot drives behind the chain root, faces it and
   climbs on (CLIMB).
3. Repeat until the bridge spans the gap.  Grips that are not part of a
   useful bridge (anchored away from the edge, stray robot-to-robot anchors)
   are released first.

It reads the full simulator state (positions, anchors, the gap geometry),
which the learned policy never sees.  It is never used by rewards; its only
role is to label observations with a reasonable action (behaviour cloning).
"""

from __future__ import annotations

import numpy as np

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.field import LEFT_BANK
from bari2d.env.robot import DiscreteAction, RobotState

IDLE = int(DiscreteAction.IDLE)


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class ScriptedTeacher:
    def __init__(self, env: BridgeEnv, staging_distance: float = 0.9):
        self.env = env
        self.staging_distance = staging_distance
        self.worker: int | None = None

    def reset(self) -> None:
        self.worker = None

    # ------------------------------------------------------------------ state
    def _usable(self, robot: RobotState) -> bool:
        return not robot.fallen

    def _valid_root(self, robot: RobotState) -> bool:
        """A left-bank grip whose front end reaches the cliff edge (a useful bridge root)."""
        if not (self._usable(robot) and robot.anchored and robot.anchor_partner == LEFT_BANK and robot.layer == 0):
            return False
        longitudinal, tangent = self.env.field.coordinates(robot.position)
        left, _ = self.env.field.boundaries(tangent)
        return longitudinal + self.env.config.robot.length / 2.0 >= left - 0.05

    def chain(self) -> list[int]:
        """Robots of the bridge, from the one gripping the left bank to the front end."""
        roots = [robot for robot in self.env.robots if self._valid_root(robot)]
        if not roots:
            return []
        normal = self.env.field.normal
        root = max(roots, key=lambda robot: float(robot.position @ normal))
        chain = [root.robot_id]
        while True:
            followers = [
                robot for robot in self.env.robots
                if self._usable(robot) and robot.anchored and robot.anchor_partner == chain[-1] and robot.layer == 0
                and robot.robot_id not in chain
            ]
            if not followers:
                return chain
            chain.append(max(followers, key=lambda robot: float(robot.position @ normal)).robot_id)

    def _standing_on_chain(self, chain: list[int]) -> RobotState | None:
        members = set(chain)
        riders = []
        for robot in self.env.robots:
            if not self._usable(robot) or robot.anchored or robot.layer == 0:
                continue
            below = [other for other in self.env.robots if other.robot_id in members and other.layer == robot.layer - 1]
            if any(float(np.linalg.norm(robot.position - other.position)) < self.env.config.robot.length for other in below):
                riders.append(robot)
        if not riders:
            return None
        normal = self.env.field.normal
        return max(riders, key=lambda robot: float(robot.position @ normal))

    def _free_ground_robots(self, exclude: set[int]) -> list[RobotState]:
        return [
            robot for robot in self.env.robots
            if self._usable(robot) and not robot.anchored and robot.layer == 0 and robot.robot_id not in exclude
        ]

    # ------------------------------------------------------------ navigation
    def _rotate(self, robot: RobotState, error: float) -> int:
        """Turn on the spot by alternating forward and backward steering moves."""
        if error > 0:
            forward, backward = DiscreteAction.FORWARD_LEFT, DiscreteAction.BACKWARD_LEFT
        else:
            forward, backward = DiscreteAction.FORWARD_RIGHT, DiscreteAction.BACKWARD_RIGHT
        return int(backward if robot.previous_action == int(forward) else forward)

    def _drive_to(self, robot: RobotState, target: np.ndarray, tolerance: float) -> int | None:
        offset = target - robot.position
        distance = float(np.linalg.norm(offset))
        if distance <= tolerance:
            return None
        error = _wrap(float(np.arctan2(offset[1], offset[0])) - robot.theta)
        if abs(error) > 0.3:
            return self._rotate(robot, error)
        if abs(error) > 0.08:
            return int(DiscreteAction.FORWARD_LEFT if error > 0 else DiscreteAction.FORWARD_RIGHT)
        return int(DiscreteAction.FORWARD)

    def _approach(self, robot: RobotState, goal: np.ndarray, heading: float) -> int | None:
        """Reach ``goal`` facing ``heading`` along a straight final approach; None when there."""
        direction = np.array([np.cos(heading), np.sin(heading)])
        side = np.array([-direction[1], direction[0]])
        along = float((robot.position - goal) @ direction)
        lateral = float((robot.position - goal) @ side)
        heading_error = _wrap(heading - robot.theta)
        in_corridor = abs(lateral) <= 0.12 and -self.staging_distance - 0.25 <= along <= 0.1
        if in_corridor and abs(heading_error) <= 0.1:
            if along < -0.03:
                return int(DiscreteAction.FORWARD)
            if along > 0.03:
                return int(DiscreteAction.BACKWARD)
            return None
        staging = goal - direction * self.staging_distance
        move = self._drive_to(robot, staging, 0.1)
        if move is not None:
            return move
        if abs(heading_error) > 0.06:
            return self._rotate(robot, heading_error)
        return int(DiscreteAction.FORWARD)

    def _edge_goal(self, robot: RobotState) -> tuple[np.ndarray, float]:
        field = self.env.field
        _, tangent = field.coordinates(robot.position)
        half_span = max(field.width / 2.0 - 1.5, 0.0)
        tangent = float(np.clip(tangent, -half_span, half_span))
        left, _ = field.boundaries(tangent)
        goal = field.center + field.normal * (left - 0.05) + field.tangent * tangent
        return goal, float(np.arctan2(field.normal[1], field.normal[0]))

    # --------------------------------------------------------------- policy
    def actions(self) -> tuple[np.ndarray, int | None]:
        """Teacher action for every robot and the id of the robot it is moving."""
        env = self.env
        actions = np.full(len(env.robots), IDLE, dtype=np.int64)
        masks = env.action_masks()
        chain = self.chain()
        worker, action = None, IDLE
        stray = [
            robot for robot in env.robots
            if self._usable(robot) and robot.anchored and robot.robot_id not in chain
        ]
        if stray:
            # Undo grips that are not part of a useful bridge (e.g. anchored away from the edge).
            worker, action = stray[0].robot_id, int(DiscreteAction.RELEASE)
        elif not chain:
            worker, action = self._grip_edge()
        else:
            rider = self._standing_on_chain(chain)
            if rider is not None:
                worker, action = rider.robot_id, self._extend(rider, chain)
            else:
                worker, action = self._join(chain)
        if worker is not None:
            if not masks[worker, action]:
                action = IDLE
            actions[worker] = action
        self.worker = worker
        return actions, worker

    def _pick(self, candidates: list[RobotState], cost) -> RobotState | None:
        if not candidates:
            return None
        if self.worker is not None and any(robot.robot_id == self.worker for robot in candidates):
            return self.env.robots[self.worker]
        return min(candidates, key=cost)

    def _grip_edge(self) -> tuple[int | None, int]:
        def cost(robot: RobotState) -> float:
            goal, _ = self._edge_goal(robot)
            return float(np.linalg.norm(goal - robot.position))

        robot = self._pick(self._free_ground_robots(set()), cost)
        if robot is None:
            return None, IDLE
        goal, heading = self._edge_goal(robot)
        move = self._approach(robot, goal, heading)
        if move is not None:
            return robot.robot_id, move
        if self.env.action_masks()[robot.robot_id, DiscreteAction.ANCHOR]:
            return robot.robot_id, int(DiscreteAction.ANCHOR)
        return robot.robot_id, int(DiscreteAction.FORWARD)

    def _extend(self, rider: RobotState, chain: list[int]) -> int:
        front = self.env.robots[chain[-1]]
        if float((rider.position - front.position) @ front.heading) < 0.1:
            return int(DiscreteAction.FORWARD)
        return int(DiscreteAction.ANCHOR)

    def _join(self, chain: list[int]) -> tuple[int | None, int]:
        root = self.env.robots[chain[0]]
        length = self.env.config.robot.length
        goal = root.position - root.heading * (length + 0.02)

        def cost(robot: RobotState) -> float:
            return float(np.linalg.norm(goal - robot.position))

        robot = self._pick(self._free_ground_robots(set(chain)), cost)
        if robot is None:
            return None, IDLE
        move = self._approach(robot, goal, root.theta)
        if move is not None:
            return robot.robot_id, move
        support = self.env._climb_support(robot)  # noqa: SLF001 - privileged teacher
        if support is not None and support.robot_id in chain:
            return robot.robot_id, int(DiscreteAction.CLIMB)
        return robot.robot_id, int(DiscreteAction.FORWARD)
