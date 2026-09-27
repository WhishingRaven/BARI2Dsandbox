"""Quasi-static rigid-body equilibrium for the opt-in ``static`` physics model.

Every robot is a rigid rectangular body with weight ``W`` acting at its center.
Gravity is out of the 2D plane, so each body has three equilibrium equations:
vertical force balance and the two first-moment (tipping) balances about its
center.  Bodies interact only through three kinds of interface:

``SupportPatch``
    Compression-only contact over a convex patch (a robot resting on a bank,
    or an upper-layer robot resting on a lower-layer robot).  The patch can
    push up anywhere inside its convex hull but can never pull, so a body whose
    weight falls outside every patch tips over.

``Joint``
    An anchor.  It transmits a vertical force, a bending couple about the axis
    perpendicular to the connection and a torsion couple about the connection
    axis, each bounded by a strength limit.

Same-layer side contact between robots transmits no vertical load at all:
without an anchor two touching robots cannot hold each other up.

The admissible set of interface forces is linear, so both questions the
environment asks are linear programs (the lower-bound theorem of plasticity,
as used for rigid-block assemblies):

* ``stabilize``: can the current structure carry its own weight?  If not,
  which joint breaks or which body tips first?
* ``max_load``: the largest external load the spanning structure can carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.optimize import linprog

from bari2d.env.field import LEFT_BANK, RIGHT_BANK

Node = int | str

# Relative cost of the three kinds of LP variables used by ``stabilize``.
_WITHIN_LIMIT_COST = 1.0e-3  # small regulariser: prefer the least-effort force distribution
_VIOLATION_COST = 1.0  # per unit of joint overload, normalised by the joint limit
_DEFICIT_COST = 10.0  # per unit of missing support, normalised by body weight: joints give way first
_TOLERANCE = 1.0e-6


# --------------------------------------------------------------------------------------
# Convex polygon helpers
# --------------------------------------------------------------------------------------


def polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_halfplane(polygon: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Keep the part of a convex polygon with ``point @ normal <= offset``."""
    if len(polygon) == 0:
        return polygon
    result: list[np.ndarray] = []
    count = len(polygon)
    for index in range(count):
        current = polygon[index]
        following = polygon[(index + 1) % count]
        current_value = float(current @ normal) - offset
        following_value = float(following @ normal) - offset
        if current_value <= 0.0:
            result.append(current)
        if (current_value < 0.0 < following_value) or (following_value < 0.0 < current_value):
            fraction = current_value / (current_value - following_value)
            result.append(current + fraction * (following - current))
    return np.asarray(result, dtype=float).reshape(-1, 2)


def clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Intersect two convex polygons (Sutherland-Hodgman)."""
    if polygon_area(clip) < 0.0:
        clip = clip[::-1]
    result = np.asarray(subject, dtype=float)
    count = len(clip)
    for index in range(count):
        if len(result) == 0:
            break
        start = clip[index]
        end = clip[(index + 1) % count]
        edge = end - start
        outward = np.array([edge[1], -edge[0]], dtype=float)
        result = clip_halfplane(result, outward, float(start @ outward))
    return result


def point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray, tolerance: float = 1.0e-9) -> bool:
    if len(polygon) < 3 or abs(polygon_area(polygon)) < 1.0e-12:
        return False
    if polygon_area(polygon) < 0.0:
        polygon = polygon[::-1]
    for index in range(len(polygon)):
        start = polygon[index]
        end = polygon[(index + 1) % len(polygon)]
        edge = end - start
        relative = point - start
        if edge[0] * relative[1] - edge[1] * relative[0] < -tolerance * max(np.linalg.norm(edge), 1.0):
            return False
    return True


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain; returns counter-clockwise hull vertices."""
    unique = np.unique(np.round(np.asarray(points, dtype=float), 12), axis=0)
    if len(unique) < 3:
        return unique
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]

    def cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
        return float((first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


# --------------------------------------------------------------------------------------
# Structure description
# --------------------------------------------------------------------------------------


@dataclass
class SupportPatch:
    """Compression-only contact: ``lower`` pushes ``upper`` up inside ``vertices``."""

    upper: int
    lower: Node
    vertices: np.ndarray


@dataclass
class Joint:
    """Anchor between ``owner`` and ``partner`` (a robot or a bank)."""

    owner: int
    partner: Node
    point: np.ndarray
    direction: np.ndarray
    vertical_limit: float
    bending_limit: float
    torsion_limit: float

    @property
    def key(self) -> tuple[int, Node]:
        return (self.owner, self.partner)


@dataclass
class Body:
    robot_id: int
    center: np.ndarray
    weight: float
    length: float
    over_gap: bool
    fully_on_bank: bool
    progress: float


@dataclass
class StaticStructure:
    bodies: dict[int, Body]
    patches: list[SupportPatch]
    joints: list[Joint]

    def body_joints(self, robot_id: int) -> list[Joint]:
        return [joint for joint in self.joints if joint.owner == robot_id or joint.partner == robot_id]

    def remove_bodies(self, robot_ids: Iterable[int]) -> None:
        removed = set(robot_ids)
        for robot_id in removed:
            self.bodies.pop(robot_id, None)
        self.patches = [patch for patch in self.patches if patch.upper not in removed and patch.lower not in removed]
        self.joints = [joint for joint in self.joints if joint.owner not in removed and joint.partner not in removed]

    def components(self, joints_only: bool = False) -> list[set[int]]:
        """Bodies coupled by robot-robot joints (and stack patches unless ``joints_only``)."""
        adjacency: dict[int, set[int]] = {robot_id: set() for robot_id in self.bodies}
        links: list[tuple[Node, Node]] = [(joint.owner, joint.partner) for joint in self.joints]
        if not joints_only:
            links += [(patch.upper, patch.lower) for patch in self.patches]
        for first, second in links:
            if isinstance(first, int) and isinstance(second, int) and first in adjacency and second in adjacency:
                adjacency[first].add(second)
                adjacency[second].add(first)
        remaining = set(adjacency)
        result: list[set[int]] = []
        while remaining:
            start = remaining.pop()
            component = {start}
            frontier = [start]
            while frontier:
                node = frontier.pop()
                for neighbor in adjacency[node]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        frontier.append(neighbor)
            result.append(component)
        return result

    def reachable_from(self, bank: str) -> set[int]:
        """Bodies connected to ``bank`` through load-bearing interfaces (supports and anchors)."""
        adjacency: dict[Node, set[Node]] = {bank: set()}
        links = [(patch.upper, patch.lower) for patch in self.patches]
        links += [(joint.owner, joint.partner) for joint in self.joints]
        for first, second in links:
            adjacency.setdefault(first, set()).add(second)
            adjacency.setdefault(second, set()).add(first)
        reached: set[Node] = {bank}
        frontier: list[Node] = [bank]
        while frontier:
            node = frontier.pop()
            for neighbor in adjacency.get(node, ()):
                if neighbor not in reached and (isinstance(neighbor, int) and neighbor in self.bodies):
                    reached.add(neighbor)
                    frontier.append(neighbor)
        return {node for node in reached if isinstance(node, int)}

    def anchored_from(self, bank: str) -> set[int]:
        """Bodies held by a structure that grips ``bank``.

        Starts from robots anchored to the bank, then follows robot-robot anchors
        (both ways) and support patches upward (robots standing on a held body).
        Robots merely resting on the bank do not count.
        """
        reached = {joint.owner for joint in self.joints if joint.partner == bank and joint.owner in self.bodies}
        frontier = list(reached)
        while frontier:
            node = frontier.pop()
            neighbors = [joint.partner for joint in self.joints if joint.owner == node]
            neighbors += [joint.owner for joint in self.joints if joint.partner == node]
            neighbors += [patch.upper for patch in self.patches if patch.lower == node]
            for neighbor in neighbors:
                if isinstance(neighbor, int) and neighbor in self.bodies and neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        return reached

    def banks_reached(self, component: set[int]) -> set[str]:
        banks = {patch.lower for patch in self.patches if patch.upper in component and isinstance(patch.lower, str)}
        banks |= {joint.partner for joint in self.joints if joint.owner in component and isinstance(joint.partner, str)}
        return banks  # type: ignore[return-value]

    def load_path(self, component: set[int]) -> list[Node]:
        """Breadth-first bank-to-bank path through load-bearing interfaces."""
        adjacency: dict[Node, set[Node]] = {LEFT_BANK: set(), RIGHT_BANK: set()}
        adjacency.update({robot_id: set() for robot_id in component})
        links = [(patch.upper, patch.lower) for patch in self.patches if patch.upper in component]
        links += [(joint.owner, joint.partner) for joint in self.joints if joint.owner in component]
        for first, second in links:
            if first in adjacency and second in adjacency:
                adjacency[first].add(second)
                adjacency[second].add(first)
        parents: dict[Node, Node | None] = {LEFT_BANK: None}
        queue: list[Node] = [LEFT_BANK]
        while queue:
            node = queue.pop(0)
            if node == RIGHT_BANK:
                break
            for neighbor in sorted(adjacency[node], key=str):
                if neighbor not in parents:
                    parents[neighbor] = node
                    queue.append(neighbor)
        if RIGHT_BANK not in parents:
            return []
        path: list[Node] = [RIGHT_BANK]
        while parents[path[-1]] is not None:
            path.append(parents[path[-1]])  # type: ignore[arg-type]
        return path[::-1]


# --------------------------------------------------------------------------------------
# Linear program assembly
# --------------------------------------------------------------------------------------


@dataclass
class _Column:
    cost: float
    upper: float | None
    entries: dict[int, float]


@dataclass
class _Program:
    columns: list[_Column] = field(default_factory=list)
    joint_columns: dict[int, dict[str, list[int]]] = field(default_factory=dict)
    deficit_columns: dict[int, list[int]] = field(default_factory=dict)
    load_column: int | None = None

    def add(self, cost: float, upper: float | None, entries: dict[int, float]) -> int:
        self.columns.append(_Column(cost, upper, entries))
        return len(self.columns) - 1


def _body_rows(bodies: list[int]) -> dict[int, int]:
    return {robot_id: 3 * index for index, robot_id in enumerate(bodies)}


def _force_entries(rows: dict[int, int], structure: StaticStructure, robot_id: Node, point: np.ndarray, sign: float) -> dict[int, float]:
    """Row coefficients of an upward force ``sign`` applied to ``robot_id`` at ``point``."""
    if not isinstance(robot_id, int) or robot_id not in rows:
        return {}
    row = rows[robot_id]
    offset = point - structure.bodies[robot_id].center
    return {row: sign, row + 1: sign * float(offset[0]), row + 2: sign * float(offset[1])}


def _couple_entries(rows: dict[int, int], robot_id: Node, vector: np.ndarray, sign: float) -> dict[int, float]:
    if not isinstance(robot_id, int) or robot_id not in rows:
        return {}
    row = rows[robot_id]
    return {row + 1: sign * float(vector[0]), row + 2: sign * float(vector[1])}


def _merge(*parts: dict[int, float]) -> dict[int, float]:
    merged: dict[int, float] = {}
    for part in parts:
        for row, value in part.items():
            merged[row] = merged.get(row, 0.0) + value
    return merged


def _build_program(
    structure: StaticStructure,
    bodies: list[int],
    allow_violation: bool,
    load_weights: dict[int, float] | None,
    maximum_load: float,
) -> tuple[_Program, np.ndarray]:
    rows = _body_rows(bodies)
    body_set = set(bodies)
    program = _Program()
    total_weight = sum(structure.bodies[robot_id].weight for robot_id in bodies)
    force_scale = max(total_weight, 1.0e-9)

    for patch in structure.patches:
        if patch.upper not in body_set:
            continue
        for vertex in patch.vertices:
            entries = _merge(
                _force_entries(rows, structure, patch.upper, vertex, 1.0),
                _force_entries(rows, structure, patch.lower, vertex, -1.0),
            )
            program.add(_WITHIN_LIMIT_COST / force_scale if allow_violation else 0.0, None, entries)

    for joint_index, joint in enumerate(structure.joints):
        if joint.owner not in body_set:
            continue
        perpendicular = np.array([-joint.direction[1], joint.direction[0]])
        generators = {
            "vertical": (
                joint.vertical_limit,
                _merge(
                    _force_entries(rows, structure, joint.owner, joint.point, 1.0),
                    _force_entries(rows, structure, joint.partner, joint.point, -1.0),
                ),
            ),
            "bending": (
                joint.bending_limit,
                _merge(
                    _couple_entries(rows, joint.owner, joint.direction, 1.0),
                    _couple_entries(rows, joint.partner, joint.direction, -1.0),
                ),
            ),
            "torsion": (
                joint.torsion_limit,
                _merge(
                    _couple_entries(rows, joint.owner, perpendicular, 1.0),
                    _couple_entries(rows, joint.partner, perpendicular, -1.0),
                ),
            ),
        }
        columns: dict[str, list[int]] = {}
        for name, (limit, entries) in generators.items():
            limit = max(float(limit), 1.0e-9)
            negated = {row: -value for row, value in entries.items()}
            within = [
                program.add(_WITHIN_LIMIT_COST / limit if allow_violation else 0.0, limit, entries),
                program.add(_WITHIN_LIMIT_COST / limit if allow_violation else 0.0, limit, negated),
            ]
            excess = []
            if allow_violation:
                excess = [
                    program.add(_VIOLATION_COST / limit, None, entries),
                    program.add(_VIOLATION_COST / limit, None, negated),
                ]
            columns[name] = within + excess
        program.joint_columns[joint_index] = columns

    if allow_violation:
        for robot_id in bodies:
            body = structure.bodies[robot_id]
            row = rows[robot_id]
            weight = max(body.weight, 1.0e-9)
            moment_scale = weight * max(body.length, 1.0e-9)
            program.deficit_columns[robot_id] = [
                program.add(_DEFICIT_COST / weight, None, {row: 1.0}),
                program.add(_DEFICIT_COST / moment_scale, None, {row + 1: 1.0}),
                program.add(_DEFICIT_COST / moment_scale, None, {row + 1: -1.0}),
                program.add(_DEFICIT_COST / moment_scale, None, {row + 2: 1.0}),
                program.add(_DEFICIT_COST / moment_scale, None, {row + 2: -1.0}),
            ]

    right_hand_side = np.zeros(3 * len(bodies))
    for robot_id in bodies:
        right_hand_side[rows[robot_id]] = structure.bodies[robot_id].weight
    if load_weights:
        load_entries: dict[int, float] = {}
        for robot_id, share in load_weights.items():
            if robot_id in rows:
                row = rows[robot_id]
                load_entries[row] = load_entries.get(row, 0.0) - share
        program.load_column = program.add(-1.0, maximum_load, load_entries)
    return program, right_hand_side


def _solve(program: _Program, right_hand_side: np.ndarray) -> np.ndarray | None:
    count = len(program.columns)
    if count == 0:
        return None if np.any(np.abs(right_hand_side) > _TOLERANCE) else np.zeros(0)
    matrix = np.zeros((len(right_hand_side), count))
    for column_index, column in enumerate(program.columns):
        for row, value in column.entries.items():
            matrix[row, column_index] += value
    costs = np.array([column.cost for column in program.columns])
    bounds = [(0.0, column.upper) for column in program.columns]
    result = linprog(costs, A_eq=matrix, b_eq=right_hand_side, bounds=bounds, method="highs")
    if result.status != 0:
        return None
    return np.asarray(result.x)


def _joint_utilisation(structure: StaticStructure, program: _Program, solution: np.ndarray) -> dict[int, dict[str, float]]:
    """Per joint and component, |value| / limit (values above 1 are overloads)."""
    usage: dict[int, dict[str, float]] = {}
    for joint_index, columns in program.joint_columns.items():
        joint = structure.joints[joint_index]
        limits = {"vertical": joint.vertical_limit, "bending": joint.bending_limit, "torsion": joint.torsion_limit}
        usage[joint_index] = {}
        for name, indices in columns.items():
            within = solution[indices[0]] - solution[indices[1]]
            excess = solution[indices[2]] - solution[indices[3]] if len(indices) == 4 else 0.0
            usage[joint_index][name] = abs(within + excess) / max(limits[name], 1.0e-9)
    return usage


# --------------------------------------------------------------------------------------
# Public queries
# --------------------------------------------------------------------------------------


@dataclass
class StabilityOutcome:
    broken_joints: list[tuple[int, Node]]  # failed by overload
    released_joints: list[tuple[int, Node]]  # let go of a cluster that tipped away
    unsupported: set[int]
    utilisation: dict[int, float]
    iterations: int


@dataclass
class CapacityOutcome:
    capacity: float
    spans: bool
    failure_mode: str
    critical_joints: list[tuple[int, Node]]
    load_path: list[Node]
    load_bodies: list[int]


def _single_body_is_stable(structure: StaticStructure, robot_id: int) -> bool:
    vertices = [patch.vertices for patch in structure.patches if patch.upper == robot_id]
    if not vertices:
        return False
    hull = convex_hull(np.vstack(vertices))
    return point_in_convex_polygon(structure.bodies[robot_id].center, hull, tolerance=1.0e-9)


def stabilize(structure: StaticStructure, max_iterations: int | None = None) -> StabilityOutcome:
    """Remove failed joints and unsupported bodies until the remaining structure stands.

    Each iteration solves a minimum-violation equilibrium problem.  If a joint
    must be overloaded it breaks (largest normalised overload first).  If no
    joint overload can restore equilibrium, the rigid cluster containing the
    body with the largest support deficit tips over; bodies lying entirely on a
    bank stay put and simply let go of that cluster.
    """
    broken: list[tuple[int, Node]] = []
    released: list[tuple[int, Node]] = []
    unsupported: set[int] = set()
    utilisation: dict[int, float] = {robot_id: 0.0 for robot_id in structure.bodies}
    limit = max_iterations if max_iterations is not None else 3 * max(len(structure.bodies), 1) + 3
    iterations = 0
    while iterations < limit:
        iterations += 1
        changed = False
        for component in structure.components():
            if not structure.banks_reached(component):
                unsupported |= component
                structure.remove_bodies(component)
                changed = True
                break
            has_joint = any(joint.owner in component for joint in structure.joints)
            if len(component) == 1 and not has_joint:
                (robot_id,) = tuple(component)
                if not _single_body_is_stable(structure, robot_id):
                    unsupported.add(robot_id)
                    structure.remove_bodies([robot_id])
                    changed = True
                    break
                continue
            bodies = sorted(component)
            program, right_hand_side = _build_program(structure, bodies, True, None, 0.0)
            solution = _solve(program, right_hand_side)
            if solution is None:  # pragma: no cover - the relaxed problem is always feasible
                unsupported |= component
                structure.remove_bodies(component)
                changed = True
                break
            usage = _joint_utilisation(structure, program, solution)
            overloads = {index: max(values.values()) for index, values in usage.items()}
            worst_joint = max(overloads, key=lambda index: overloads[index], default=None)
            if worst_joint is not None and overloads[worst_joint] > 1.0 + 1.0e-6:
                broken.append(structure.joints[worst_joint].key)
                del structure.joints[worst_joint]
                changed = True
                break
            deficits = {
                robot_id: float(np.sum(solution[indices])) for robot_id, indices in program.deficit_columns.items()
            }
            lacking = [robot_id for robot_id, value in deficits.items() if value > _TOLERANCE]
            if lacking:
                movable = [robot_id for robot_id in lacking if not structure.bodies[robot_id].fully_on_bank]
                worst_body = max(movable or lacking, key=lambda robot_id: deficits[robot_id])
                cluster = next(item for item in structure.components(joints_only=True) if worst_body in item)
                falling = {robot_id for robot_id in cluster if not structure.bodies[robot_id].fully_on_bank}
                cluster_joints = [
                    index
                    for index, joint in enumerate(structure.joints)
                    if joint.owner in cluster or joint.partner in cluster
                ]
                if not falling and cluster_joints:
                    # A body lying flat on a bank cannot be tipped; the grip pulling on it gives way.
                    most_used = max(cluster_joints, key=lambda index: overloads.get(index, 0.0))
                    released.append(structure.joints[most_used].key)
                    del structure.joints[most_used]
                else:
                    if not falling:
                        # Riders resting on an immovable body slide off instead.
                        falling = {patch.upper for patch in structure.patches if patch.lower == worst_body} or {worst_body}
                    for joint in structure.joints:
                        if (joint.owner in falling) != (joint.partner in falling) and isinstance(joint.partner, int):
                            released.append(joint.key)
                    unsupported |= falling
                    structure.remove_bodies(falling)
                changed = True
                break
            for index, values in usage.items():
                joint = structure.joints[index]
                value = max(values.values())
                for robot_id in (joint.owner, joint.partner):
                    if isinstance(robot_id, int) and robot_id in utilisation:
                        utilisation[robot_id] = max(utilisation[robot_id], value)
        if not changed:
            break
    return StabilityOutcome(broken, released, unsupported, utilisation, iterations)


def max_load(structure: StaticStructure, protocol: str = "uniform", maximum_load: float = 15.0) -> CapacityOutcome:
    """Largest external load the bank-to-bank structure carries in equilibrium."""
    spanning = [
        component
        for component in structure.components()
        if {LEFT_BANK, RIGHT_BANK} <= structure.banks_reached(component)
    ]
    if not spanning:
        return CapacityOutcome(0.0, False, "loss_of_connectivity", [], [], [])
    bodies = sorted(set().union(*spanning))

    def midline_distance(robot_id: int) -> float:
        return abs(structure.bodies[robot_id].progress - 0.5)

    if protocol == "center_point":
        load_bodies = [min(bodies, key=midline_distance)]
    else:
        load_bodies = [robot_id for robot_id in bodies if structure.bodies[robot_id].over_gap]
        if not load_bodies:
            load_bodies = [min(component, key=midline_distance) for component in spanning]
    shares = {robot_id: 1.0 / len(load_bodies) for robot_id in load_bodies}
    program, right_hand_side = _build_program(structure, bodies, False, shares, maximum_load)
    path = structure.load_path(set(bodies))
    solution = _solve(program, right_hand_side)
    if solution is None or program.load_column is None:
        return CapacityOutcome(0.0, True, "structural_collapse", [], path, load_bodies)
    capacity = float(solution[program.load_column])
    usage = _joint_utilisation(structure, program, solution)
    critical: list[tuple[int, Node]] = []
    kinds: set[str] = set()
    for index, values in usage.items():
        for name, value in values.items():
            if value >= 1.0 - 1.0e-6:
                critical.append(structure.joints[index].key)
                kinds.add(name)
    if capacity >= maximum_load - 1.0e-6:
        mode = "none"
    elif "bending" in kinds:
        mode = "anchor_bending"
    elif "vertical" in kinds:
        mode = "anchor_shear"
    elif "torsion" in kinds:
        mode = "anchor_torsion"
    else:
        mode = "support_tipping"
    return CapacityOutcome(capacity, True, mode, sorted(set(critical), key=str), path, load_bodies)
