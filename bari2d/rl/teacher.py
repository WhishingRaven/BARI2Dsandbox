"""A privileged scripted builder used only to generate demonstrations.

The teacher is a "weak but valid" construction procedure: one robot works at
a time, up to ``helpers`` more robots queue behind the root of the row being
built (so the next climber is already there), and every other robot waits.

1. No robot grips the left bank: the robot closest to the cliff drives to the
   edge (via a staging point one robot length behind it), faces the gap and
   grips (ANCHOR).
2. A chain exists but does not reach the far bank: a robot already standing on
   the chain walks to its front end and grips-and-extends (FORWARD, ANCHOR);
   otherwise the nearest free robot drives behind the chain root, faces it and
   climbs on (CLIMB).
3. Repeat until the row spans the gap.  Grips that are not part of a
   useful bridge (anchored away from the edge, stray robot-to-robot anchors)
   are released first.
4. Reinforcement: when every row spans but the capacity is still below the
   target load, a free robot grips the edge next to the outermost root
   (``row_spacing`` to the side) and a second row is built the same way.

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
    def __init__(
        self, env: BridgeEnv, staging_distance: float = 0.9, row_spacing: float | None = None, helpers: int = 2
    ):
        self.env = env
        # Robots that queue behind the root of the row being built (0 = off).
        self.helpers = helpers
        self.staging_distance = staging_distance
        # Centre-to-centre distance between neighbouring rows.  Wider than the
        # 1.15 x width of scripted.assemble_static_chain: a root may end up a few
        # degrees off the normal, and its row then drifts sideways link by link.
        self.row_spacing = env.config.robot.width * 1.6 if row_spacing is None else row_spacing
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

    def chains(self) -> list[list[int]]:
        """Every row of the bridge, each from the robot gripping the left bank to its front end."""
        normal = self.env.field.normal
        rows: list[list[int]] = []
        used: set[int] = set()
        roots = sorted((robot for robot in self.env.robots if self._valid_root(robot)), key=lambda robot: robot.robot_id)
        for root in roots:
            chain = [root.robot_id]
            used.add(root.robot_id)
            while True:
                followers = [
                    robot for robot in self.env.robots
                    if self._usable(robot) and robot.anchored and robot.anchor_partner == chain[-1] and robot.layer == 0
                    and robot.robot_id not in used
                ]
                if not followers:
                    break
                follower = max(followers, key=lambda robot: float(robot.position @ normal))
                chain.append(follower.robot_id)
                used.add(follower.robot_id)
            rows.append(chain)
        return rows

    def spans(self, chain: list[int]) -> bool:
        """The front robot of this row reaches the right bank."""
        front = self.env.robots[chain[-1]]
        longitudinal, tangent = self.env.field.coordinates(front.position)
        _, right = self.env.field.boundaries(tangent)
        reach = abs(float(front.heading @ self.env.field.normal)) * self.env.config.robot.length / 2.0
        return longitudinal + reach >= right

    def chain(self) -> list[int]:
        """The row being built: the longest row that does not span yet (else the longest row)."""
        rows = self.chains()
        if not rows:
            return []
        open_rows = [row for row in rows if not self.spans(row)]
        return max(open_rows or rows, key=len)

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

    def _edge_goal(self, robot: RobotState, tangent: float | None = None) -> tuple[np.ndarray, float]:
        field = self.env.field
        if tangent is None:
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
        rows = self.chains()
        members = {robot_id for row in rows for robot_id in row}
        worker, action = None, IDLE
        stray = [
            robot for robot in env.robots
            if self._usable(robot) and robot.anchored and robot.robot_id not in members
        ]
        open_rows = [row for row in rows if not self.spans(row)]
        if stray:
            # Undo grips that are not part of a useful bridge (e.g. anchored away from the edge).
            worker, action = stray[0].robot_id, int(DiscreteAction.RELEASE)
        elif not rows:
            worker, action = self._grip_edge()
        elif open_rows:
            chain = max(open_rows, key=len)
            rider = self._standing_on_chain(chain)
            if rider is not None:
                worker, action = rider.robot_id, self._extend(rider, chain)
            else:
                worker, action = self._join(chain, members)
            # Pipelining: the next robots queue at staging points behind the
            # root (no climbing) while the rider or the joiner works.
            busy = set(members) | ({worker} if worker is not None else set())
            first_slot = 0 if rider is not None else 1
            for slot in range(first_slot, first_slot + self.helpers):
                helper = self._preposition(chain, busy, slot)
                if helper is None:
                    break
                busy.add(helper[0])
                if helper[1] != IDLE and masks[helper[0], helper[1]]:
                    actions[helper[0]] = helper[1]
        elif env.current_capacity < env.target_load:
            worker, action = self._grip_edge(self._new_row_tangent(rows), members)
        if worker is not None:
            if not masks[worker, action]:
                action = IDLE
            actions[worker] = action
        self.worker = worker
        return actions, worker

    def _new_row_tangent(self, rows: list[list[int]]) -> float:
        """Lateral position of the next root: one row spacing beside the outermost root, on the side of the free robots."""
        field = self.env.field
        tangents = [field.coordinates(self.env.robots[row[0]].position)[1] for row in rows]
        members = {robot_id for row in rows for robot_id in row}
        free = self._free_ground_robots(members)
        free_side = np.mean([field.coordinates(robot.position)[1] for robot in free]) if free else 0.0
        half_span = max(field.width / 2.0 - 1.5, 0.0)
        candidates = [max(tangents) + self.row_spacing, min(tangents) - self.row_spacing]
        candidates = [value for value in candidates if abs(value) <= half_span] or candidates
        return min(candidates, key=lambda value: abs(value - free_side))

    def _pick(self, candidates: list[RobotState], cost) -> RobotState | None:
        if not candidates:
            return None
        if self.worker is not None and any(robot.robot_id == self.worker for robot in candidates):
            return self.env.robots[self.worker]
        return min(candidates, key=cost)

    def _grip_edge(self, tangent: float | None = None, exclude: set[int] | None = None) -> tuple[int | None, int]:
        def cost(robot: RobotState) -> float:
            goal, _ = self._edge_goal(robot, tangent)
            return float(np.linalg.norm(goal - robot.position))

        robot = self._pick(self._free_ground_robots(exclude or set()), cost)
        if robot is None:
            return None, IDLE
        goal, heading = self._edge_goal(robot, tangent)
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

    def _preposition(self, chain: list[int], exclude: set[int], slot: int = 0) -> tuple[int, int] | None:
        """Send the nearest free robot to queue slot ``slot`` behind the root (no climbing).

        Slot 0 is the joiner's staging point; every further slot is one robot
        length plus a 0.3 margin further back.  Returns (robot id, action), with
        IDLE once the robot waits in place, or None when no robot is free.
        """
        root = self.env.robots[chain[0]]
        length = self.env.config.robot.length
        staging = root.position - root.heading * (length + 0.02 + self.staging_distance + slot * (length + 0.3))
        candidates = self._free_ground_robots(exclude)
        if not candidates:
            return None
        robot = min(candidates, key=lambda item: float(np.linalg.norm(staging - item.position)))
        move = self._drive_to(robot, staging, 0.1)
        if move is None:
            heading_error = _wrap(root.theta - robot.theta)
            move = self._rotate(robot, heading_error) if abs(heading_error) > 0.06 else IDLE
        return robot.robot_id, move

    def _join(self, chain: list[int], members: set[int] | None = None) -> tuple[int | None, int]:
        root = self.env.robots[chain[0]]
        length = self.env.config.robot.length
        goal = root.position - root.heading * (length + 0.02)

        def cost(robot: RobotState) -> float:
            return float(np.linalg.norm(goal - robot.position))

        robot = self._pick(self._free_ground_robots(members or set(chain)), cost)
        if robot is None:
            return None, IDLE
        move = self._approach(robot, goal, root.theta)
        if move is not None:
            return robot.robot_id, move
        support = self.env._climb_support(robot)  # noqa: SLF001 - privileged teacher
        if support is not None and support.robot_id in chain:
            return robot.robot_id, int(DiscreteAction.CLIMB)
        return robot.robot_id, int(DiscreteAction.FORWARD)
