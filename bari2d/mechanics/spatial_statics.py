from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Iterable, Mapping, Sequence

import numpy as np

from bari2d.env.contact_model import ContactGraph, MechanicalEdge, Node
from bari2d.env.field import LEFT_BANK, RIGHT_BANK, GapField
from bari2d.env.robot import RobotState
from bari2d.utils.config import RobotConfig


EdgeKey = frozenset[Node]


@dataclass(frozen=True)
class SpatialWrench:
    """Six-dimensional wrench expressed in world axes.

    Forces are in N and moments are free couples at the contact point in N m.
    The moment of the contact force about a robot COM is handled separately by
    the rigid-body equilibrium transformation.
    """

    fx: float
    fy: float
    fz: float
    mx: float
    my: float
    mz: float

    def as_array(self) -> np.ndarray:
        return np.asarray([self.fx, self.fy, self.fz, self.mx, self.my, self.mz], dtype=float)

    @classmethod
    def from_array(cls, values: np.ndarray) -> "SpatialWrench":
        values = np.asarray(values, dtype=float)
        if values.shape != (6,):
            raise ValueError(f"Expected a 6-vector, got shape {values.shape}")
        return cls(*(float(value) for value in values))


@dataclass(frozen=True)
class SpatialWrenchLimit:
    """Component-wise absolute wrench limits expressed in world axes."""

    fx: float = inf
    fy: float = inf
    fz: float = inf
    mx: float = inf
    my: float = inf
    mz: float = inf

    def as_array(self) -> np.ndarray:
        return np.asarray([self.fx, self.fy, self.fz, self.mx, self.my, self.mz], dtype=float)


@dataclass(frozen=True)
class ContactFrame:
    """Right-handed orthonormal frame attached to a contact patch.

    The local x axis is the contact normal pointing from ``edge.source`` toward
    ``edge.target``. Local y/z are tangent axes. Columns of
    ``rotation_world_from_local`` are therefore ``(normal, tangent_u,
    tangent_v)`` expressed in world coordinates.
    """

    normal: tuple[float, float, float]
    tangent_u: tuple[float, float, float]
    tangent_v: tuple[float, float, float]

    @property
    def rotation_world_from_local(self) -> np.ndarray:
        return np.column_stack(
            (
                np.asarray(self.normal, dtype=float),
                np.asarray(self.tangent_u, dtype=float),
                np.asarray(self.tangent_v, dtype=float),
            )
        )

    def vector_to_local(self, vector_world: np.ndarray) -> np.ndarray:
        return self.rotation_world_from_local.T @ np.asarray(vector_world, dtype=float)

    def vector_to_world(self, vector_local: np.ndarray) -> np.ndarray:
        return self.rotation_world_from_local @ np.asarray(vector_local, dtype=float)


@dataclass(frozen=True)
class LocalContactWrench:
    """Contact wrench resolved in the connector's own frame.

    ``axial`` is force normal to the contact patch. ``shear_u`` and
    ``shear_v`` are in-plane shear forces. ``torsion`` is moment about the
    contact normal, while ``bending_u`` and ``bending_v`` are bending moments
    about the two tangent axes.

    Array order is intentionally force-like components first followed by
    moment-like components::

        [axial, shear_u, shear_v, torsion, bending_u, bending_v]
    """

    axial: float
    shear_u: float
    shear_v: float
    torsion: float
    bending_u: float
    bending_v: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.axial,
                self.shear_u,
                self.shear_v,
                self.torsion,
                self.bending_u,
                self.bending_v,
            ],
            dtype=float,
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> "LocalContactWrench":
        values = np.asarray(values, dtype=float)
        if values.shape != (6,):
            raise ValueError(f"Expected a 6-vector, got shape {values.shape}")
        return cls(*(float(value) for value in values))

    @classmethod
    def from_world(cls, wrench: SpatialWrench, frame: ContactFrame) -> "LocalContactWrench":
        force_local = frame.vector_to_local(wrench.as_array()[:3])
        moment_local = frame.vector_to_local(wrench.as_array()[3:])
        return cls(
            axial=float(force_local[0]),
            shear_u=float(force_local[1]),
            shear_v=float(force_local[2]),
            torsion=float(moment_local[0]),
            bending_u=float(moment_local[1]),
            bending_v=float(moment_local[2]),
        )


@dataclass(frozen=True)
class LocalWrenchLimit:
    """Symmetric connector strength limits in the local contact frame.

    Force limits are in N and moment limits in N m. This is the preferred
    strength representation for physical connectors because the allowable
    axial, shear, bending and torsional loads are properties of the connector,
    not of the global simulation axes.
    """

    axial: float = inf
    shear_u: float = inf
    shear_v: float = inf
    torsion: float = inf
    bending_u: float = inf
    bending_v: float = inf

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.axial,
                self.shear_u,
                self.shear_v,
                self.torsion,
                self.bending_u,
                self.bending_v,
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class SpatialContactStiffness:
    """Diagonal 6-DOF contact stiffness in world axes.

    kx, ky, kz have units N/m. krx, kry, krz have units N m/rad.
    """

    kx: float
    ky: float
    kz: float
    krx: float
    kry: float
    krz: float

    def as_matrix(self) -> np.ndarray:
        values = np.asarray([self.kx, self.ky, self.kz, self.krx, self.kry, self.krz], dtype=float)
        if np.any(values <= 0.0):
            raise ValueError("All spatial contact stiffness components must be positive")
        return np.diag(values)


@dataclass(frozen=True)
class SpatialLoad:
    """External 3-D load applied to a robot.

    `point` is the world-coordinate force application point. If omitted, force
    is applied at the robot COM. `moment` is an additional free couple.
    """

    robot_id: int
    force: tuple[float, float, float]
    point: tuple[float, float, float] | None = None
    moment: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SpatialContactResponse:
    source: Node
    target: Node
    kind: str
    point: tuple[float, float, float]
    wrench_on_source: SpatialWrench
    relative_displacement: tuple[float, float, float, float, float, float]
    frame: ContactFrame

    @property
    def key(self) -> EdgeKey:
        return frozenset((self.source, self.target))

    @property
    def local_wrench_on_source(self) -> LocalContactWrench:
        return LocalContactWrench.from_world(self.wrench_on_source, self.frame)

    @property
    def wrench_on_target(self) -> SpatialWrench:
        """Equal-and-opposite world-frame wrench on the target endpoint."""
        return SpatialWrench.from_array(-self.wrench_on_source.as_array())

    @property
    def local_wrench_on_target(self) -> LocalContactWrench:
        """Equal-and-opposite wrench resolved in the same canonical contact frame."""
        return LocalContactWrench.from_array(-self.local_wrench_on_source.as_array())


@dataclass
class SpatialStaticResult:
    stable: bool
    displacements: dict[int, tuple[float, float, float, float, float, float]]
    contacts: list[SpatialContactResponse]
    residual_norm: float
    rank: int
    dof_count: int
    failure_mode: str = "none"

    def contact_by_key(self) -> dict[EdgeKey, SpatialContactResponse]:
        return {contact.key: contact for contact in self.contacts}


@dataclass
class SpatialCapacityResult:
    max_payload_kg: float
    failure_mode: str
    critical_contact: tuple[Node, Node] | None
    critical_component: str | None
    critical_limit: float | None
    critical_value: float | None
    base_response: SpatialStaticResult
    unit_payload_response: SpatialStaticResult


@dataclass(frozen=True)
class _ContactElement:
    edge: MechanicalEdge
    point: np.ndarray
    B: np.ndarray
    stiffness: np.ndarray
    frame: ContactFrame


_COMPONENT_NAMES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
_LOCAL_COMPONENT_NAMES = ("Axial", "ShearU", "ShearV", "Torsion", "BendingU", "BendingV")


def _robot_component(graph: ContactGraph, start: int) -> set[int]:
    """Robot-only connected component; fixed supports are terminal nodes."""

    visited: set[int] = {start}
    frontier = [start]
    while frontier:
        node = frontier.pop()
        for neighbor in graph.neighbors(node):
            if isinstance(neighbor, int) and neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return visited


def _fixed_supports_for(graph: ContactGraph, robot_ids: Iterable[int]) -> set[Node]:
    supports: set[Node] = set()
    for robot_id in robot_ids:
        for neighbor in graph.neighbors(robot_id):
            if not isinstance(neighbor, int):
                supports.add(neighbor)
    return supports


def _spanning_robot_components(graph: ContactGraph, robot_count: int) -> list[set[int]]:
    remaining = set(range(robot_count))
    components: list[set[int]] = []
    while remaining:
        start = next(iter(remaining))
        component = _robot_component(graph, start)
        remaining.difference_update(component)
        supports = _fixed_supports_for(graph, component)
        if LEFT_BANK in supports and RIGHT_BANK in supports:
            components.append(component)
    return components


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _yaw_rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def robot_height(config: RobotConfig) -> float:
    """Return the vertical box height used by the 2.5-D -> 3-D adapter.

    BARI2D stores discrete layers but no independent physical height. The layer
    increment (`climb_height`) is therefore the only geometry-consistent choice
    for the extruded rigid-body height.
    """

    return max(float(config.climb_height), 1.0e-9)


def robot_center_3d(robot: RobotState, config: RobotConfig) -> np.ndarray:
    height = robot_height(config)
    return np.asarray(
        [float(robot.position[0]), float(robot.position[1]), (float(robot.layer) + 0.5) * height],
        dtype=float,
    )


def robot_face_centers_3d(robot: RobotState, config: RobotConfig) -> np.ndarray:
    """Centers of +x,-x,+y,-y,+z,-z faces of the extruded robot box."""

    height = robot_height(config)
    local = np.asarray(
        [
            [config.length / 2.0, 0.0, 0.0],
            [-config.length / 2.0, 0.0, 0.0],
            [0.0, config.width / 2.0, 0.0],
            [0.0, -config.width / 2.0, 0.0],
            [0.0, 0.0, height / 2.0],
            [0.0, 0.0, -height / 2.0],
        ],
        dtype=float,
    )
    return local @ _yaw_rotation(robot.theta).T + robot_center_3d(robot, config)


def _lateral_face_toward(robot: RobotState, direction_xy: np.ndarray, config: RobotConfig) -> np.ndarray:
    direction_xy = np.asarray(direction_xy, dtype=float)
    norm = float(np.linalg.norm(direction_xy))
    center = robot_center_3d(robot, config)
    if norm <= 1.0e-12:
        return center.copy()
    direction_xy /= norm
    faces = robot_face_centers_3d(robot, config)[:4]
    scores = (faces[:, :2] - center[:2]) @ direction_xy
    return faces[int(np.argmax(scores))]


def _polygon_signed_cross(a: np.ndarray, b: np.ndarray, point: np.ndarray) -> float:
    edge = b - a
    rel = point - a
    return float(edge[0] * rel[1] - edge[1] * rel[0])


def _line_intersection(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    r = p2 - p1
    s = q2 - q1
    denominator = float(r[0] * s[1] - r[1] * s[0])
    if abs(denominator) <= 1.0e-14:
        return 0.5 * (p1 + p2)
    qp = q1 - p1
    t = float((qp[0] * s[1] - qp[1] * s[0]) / denominator)
    return p1 + t * r


def _convex_polygon_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping for two CCW convex polygons."""

    output = [np.asarray(point, dtype=float) for point in subject]
    if not output:
        return np.empty((0, 2), dtype=float)
    for index in range(len(clip)):
        a = np.asarray(clip[index], dtype=float)
        b = np.asarray(clip[(index + 1) % len(clip)], dtype=float)
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        previous_inside = _polygon_signed_cross(a, b, previous) >= -1.0e-10
        for current in input_points:
            current_inside = _polygon_signed_cross(a, b, current) >= -1.0e-10
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, a, b))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, a, b))
            previous = current
            previous_inside = current_inside
    if not output:
        return np.empty((0, 2), dtype=float)
    return np.asarray(output, dtype=float)


def _polygon_centroid(polygon: np.ndarray) -> np.ndarray | None:
    if len(polygon) < 3:
        return None
    twice_area = 0.0
    centroid = np.zeros(2, dtype=float)
    for index in range(len(polygon)):
        first = polygon[index]
        second = polygon[(index + 1) % len(polygon)]
        cross = float(first[0] * second[1] - second[0] * first[1])
        twice_area += cross
        centroid += (first + second) * cross
    if abs(twice_area) <= 1.0e-12:
        return None
    # centroid denominator = 3 * twice_area
    return centroid / (3.0 * twice_area)


def _overlap_center_xy(first: RobotState, second: RobotState, config: RobotConfig) -> np.ndarray:
    intersection = _convex_polygon_intersection(first.corners(config), second.corners(config))
    centroid = _polygon_centroid(intersection)
    if centroid is not None and np.all(np.isfinite(centroid)):
        return centroid
    if len(intersection):
        return np.mean(intersection, axis=0)
    # The legacy contact graph has a small contact tolerance. If two projected
    # rectangles are only tolerance-touching, use the midpoint as a deterministic
    # fallback rather than inventing an overlap polygon.
    return 0.5 * (np.asarray(first.position, dtype=float) + np.asarray(second.position, dtype=float))


def contact_point_for_edge_3d(
    edge: MechanicalEdge,
    robots: Sequence[RobotState],
    field: GapField,
    robot_config: RobotConfig,
) -> np.ndarray:
    """Return the assumed center of the physical contact/anchor patch.

    - same-layer robot pairs: midpoint of opposing lateral face centers;
    - adjacent-layer robot pairs: centroid of the projected overlap patch at
      the horizontal interface between the two box layers;
    - bank anchors: lateral face center directed toward the bank;
    - non-anchor bank contacts: bottom-face center, representing ground support
      on the solid bank surface.
    """

    if isinstance(edge.source, int) and isinstance(edge.target, int):
        first = robots[edge.source]
        second = robots[edge.target]
        if first.layer != second.layer:
            xy = _overlap_center_xy(first, second, robot_config)
            first_center = robot_center_3d(first, robot_config)
            second_center = robot_center_3d(second, robot_config)
            z = 0.5 * (first_center[2] + second_center[2])
            return np.asarray([xy[0], xy[1], z], dtype=float)

        direction = np.asarray(second.position - first.position, dtype=float)
        if float(np.linalg.norm(direction)) <= 1.0e-12:
            direction = first.heading
        first_face = _lateral_face_toward(first, direction, robot_config)
        second_face = _lateral_face_toward(second, -direction, robot_config)
        return 0.5 * (first_face + second_face)

    robot_id = edge.source if isinstance(edge.source, int) else edge.target
    bank = edge.target if isinstance(edge.source, int) else edge.source
    if not isinstance(robot_id, int):
        raise ValueError(f"Mechanical edge has no robot endpoint: {edge}")
    robot = robots[robot_id]

    if edge.kind != "anchor":
        # Ordinary bank contact means the robot is standing on the bank/ground.
        faces = robot_face_centers_3d(robot, robot_config)
        return faces[5].copy()

    if bank == LEFT_BANK:
        direction = -field.normal
    elif bank == RIGHT_BANK:
        direction = field.normal
    else:
        raise ValueError(f"Unsupported fixed endpoint: {bank!r}")
    return _lateral_face_toward(robot, direction, robot_config)


def _robot_lateral_face_normal_toward(robot: RobotState, direction_xy: np.ndarray) -> np.ndarray:
    """Outward normal of the robot side face that most directly faces a direction."""

    direction_xy = np.asarray(direction_xy, dtype=float)
    norm = float(np.linalg.norm(direction_xy))
    if norm <= 1.0e-12:
        direction_xy = np.asarray(robot.heading, dtype=float)
        norm = float(np.linalg.norm(direction_xy))
    direction_xy = direction_xy / max(norm, 1.0e-12)
    heading = np.asarray([robot.heading[0], robot.heading[1], 0.0], dtype=float)
    lateral = np.asarray([-robot.heading[1], robot.heading[0], 0.0], dtype=float)
    candidates = (heading, -heading, lateral, -lateral)
    direction3 = np.asarray([direction_xy[0], direction_xy[1], 0.0], dtype=float)
    return max(candidates, key=lambda candidate: float(candidate @ direction3)).copy()


def _orthonormal_contact_frame(normal: np.ndarray, preferred_tangent: np.ndarray) -> ContactFrame:
    normal = np.asarray(normal, dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1.0e-12:
        raise ValueError("Contact normal must be non-zero")
    normal = normal / normal_norm

    preferred_tangent = np.asarray(preferred_tangent, dtype=float)
    tangent_u = preferred_tangent - normal * float(preferred_tangent @ normal)
    tangent_norm = float(np.linalg.norm(tangent_u))
    if tangent_norm <= 1.0e-10:
        fallbacks = (
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
        )
        for fallback in fallbacks:
            tangent_u = fallback - normal * float(fallback @ normal)
            tangent_norm = float(np.linalg.norm(tangent_u))
            if tangent_norm > 1.0e-10:
                break
    tangent_u /= max(tangent_norm, 1.0e-12)
    tangent_v = np.cross(normal, tangent_u)
    tangent_v /= max(float(np.linalg.norm(tangent_v)), 1.0e-12)
    # Recompute tangent_u to suppress accumulated roundoff and guarantee a
    # right-handed basis: n x t_u = t_v.
    tangent_u = np.cross(tangent_v, normal)
    tangent_u /= max(float(np.linalg.norm(tangent_u)), 1.0e-12)
    return ContactFrame(
        normal=tuple(float(value) for value in normal),
        tangent_u=tuple(float(value) for value in tangent_u),
        tangent_v=tuple(float(value) for value in tangent_v),
    )


def contact_frame_for_edge_3d(
    edge: MechanicalEdge,
    robots: Sequence[RobotState],
    field: GapField,
    robot_config: RobotConfig,
) -> ContactFrame:
    """Construct the connector frame used to resolve local wrench components.

    The normal always points from ``edge.source`` toward ``edge.target``.
    For robot side contacts it follows the actual selected robot face normal;
    for stacked contacts it is vertical; for ordinary bank/ground contacts it
    is normal to the ground plane. A robot heading projected into the contact
    plane is preferred as the first tangent axis.
    """

    source = edge.source
    target = edge.target

    if isinstance(source, int) and isinstance(target, int):
        source_robot = robots[source]
        target_robot = robots[target]
        if source_robot.layer != target_robot.layer:
            sign = 1.0 if target_robot.layer > source_robot.layer else -1.0
            normal = np.asarray([0.0, 0.0, sign], dtype=float)
        else:
            direction = np.asarray(target_robot.position - source_robot.position, dtype=float)
            normal = _robot_lateral_face_normal_toward(source_robot, direction)
        preferred = np.asarray([source_robot.heading[0], source_robot.heading[1], 0.0], dtype=float)
        if abs(float(preferred @ normal)) > 0.95:
            preferred = np.asarray([-source_robot.heading[1], source_robot.heading[0], 0.0], dtype=float)
        return _orthonormal_contact_frame(normal, preferred)

    robot_id = source if isinstance(source, int) else target
    bank = target if isinstance(source, int) else source
    if not isinstance(robot_id, int):
        raise ValueError(f"Mechanical edge has no robot endpoint: {edge}")
    robot = robots[robot_id]
    source_is_robot = isinstance(source, int)

    if edge.kind != "anchor":
        robot_to_support = np.asarray([0.0, 0.0, -1.0], dtype=float)
    else:
        if bank == LEFT_BANK:
            bank_direction = -field.normal
        elif bank == RIGHT_BANK:
            bank_direction = field.normal
        else:
            raise ValueError(f"Unsupported fixed endpoint: {bank!r}")
        robot_to_support = _robot_lateral_face_normal_toward(robot, bank_direction)

    normal = robot_to_support if source_is_robot else -robot_to_support
    preferred = np.asarray([robot.heading[0], robot.heading[1], 0.0], dtype=float)
    if abs(float(preferred @ normal)) > 0.95:
        preferred = np.asarray([-robot.heading[1], robot.heading[0], 0.0], dtype=float)
    return _orthonormal_contact_frame(normal, preferred)


def _default_contact_stiffness(
    edge: MechanicalEdge,
    robot_config: RobotConfig,
    rotational_stiffness_scale: float,
) -> SpatialContactStiffness:
    k = max(float(edge.stiffness), 1.0e-9)
    length = max(float(robot_config.length), 1.0e-9)
    width = max(float(robot_config.width), 1.0e-9)
    height = robot_height(robot_config)
    scale = max(float(rotational_stiffness_scale), 1.0e-12)
    return SpatialContactStiffness(
        kx=k,
        ky=k,
        kz=k,
        krx=k * (width**2 + height**2) / 12.0 * scale,
        kry=k * (length**2 + height**2) / 12.0 * scale,
        krz=k * (length**2 + width**2) / 12.0 * scale,
    )


class SpatialStaticSolver:
    """6-DOF linear-elastic rigid-body statics for a BARI2D morphology.

    BARI2D itself stores (x, y, yaw, layer), not a full 3-D pose. This solver
    extrudes each robot footprint into a box whose height equals `climb_height`,
    places layer N at z=(N+1/2)*height, and then solves *small-displacement* 3-D
    statics with six incremental DOFs per robot:

        (ux, uy, uz, rx, ry, rz)

    Every graph edge is treated as a welded six-component spring at the contact
    patch center. This deliberately preserves the simulator's current assumption
    that an existing contact does not detach merely because the wrench is large.
    Strength limits are evaluated separately by `capacity()`.
    """

    def __init__(
        self,
        robots: Sequence[RobotState],
        graph: ContactGraph,
        field: GapField,
        robot_config: RobotConfig,
        *,
        stiffness: SpatialContactStiffness | Mapping[EdgeKey, SpatialContactStiffness] | None = None,
        rotational_stiffness_scale: float = 1.0,
        rank_tolerance: float = 1.0e-10,
        verify_rank: bool = True,
    ):
        self.robots = list(robots)
        self.graph = graph
        self.field = field
        self.robot_config = robot_config
        self.stiffness = stiffness
        self.rotational_stiffness_scale = float(rotational_stiffness_scale)
        self.rank_tolerance = float(rank_tolerance)
        self.verify_rank = bool(verify_rank)

    def _component_for_loads(self, loads: Sequence[SpatialLoad]) -> set[int]:
        active: set[int] = set()
        seen_components: set[frozenset[int]] = set()
        for load in loads:
            robot_id = load.robot_id
            if robot_id < 0 or robot_id >= len(self.robots):
                raise IndexError(f"Invalid robot_id {robot_id}")
            if self.robots[robot_id].fallen:
                raise ValueError(f"Cannot apply a static load to fallen robot {robot_id}")
            component = frozenset(_robot_component(self.graph, robot_id))
            if component not in seen_components:
                seen_components.add(component)
                active.update(component)
        return active

    def _all_loaded_components_supported(self, loads: Sequence[SpatialLoad]) -> bool:
        checked: set[frozenset[int]] = set()
        for robot_id in {load.robot_id for load in loads}:
            component = frozenset(_robot_component(self.graph, robot_id))
            if component in checked:
                continue
            checked.add(component)
            supports = _fixed_supports_for(self.graph, component)
            if not supports:
                return False
        return True

    def _stiffness_for(self, edge: MechanicalEdge) -> np.ndarray:
        if isinstance(self.stiffness, SpatialContactStiffness):
            return self.stiffness.as_matrix()
        if self.stiffness is not None:
            selected = self.stiffness.get(edge.key)
            if selected is not None:
                return selected.as_matrix()
        return _default_contact_stiffness(
            edge,
            self.robot_config,
            self.rotational_stiffness_scale,
        ).as_matrix()

    def _element(
        self,
        edge: MechanicalEdge,
        index: Mapping[int, int],
        dof_count: int,
    ) -> _ContactElement | None:
        robot_endpoints = [node for node in (edge.source, edge.target) if isinstance(node, int)]
        if not any(node in index for node in robot_endpoints):
            return None

        point = contact_point_for_edge_3d(edge, self.robots, self.field, self.robot_config)
        B = np.zeros((6, dof_count), dtype=float)

        def add_endpoint(node: Node, sign: float) -> None:
            if not isinstance(node, int) or node not in index:
                return
            base = 6 * index[node]
            r = point - robot_center_3d(self.robots[node], self.robot_config)
            point_motion = np.zeros((6, 6), dtype=float)
            point_motion[:3, :3] = np.eye(3)
            # Contact-point displacement is u + phi x r = u - [r]x phi.
            point_motion[:3, 3:] = -_skew(r)
            point_motion[3:, 3:] = np.eye(3)
            B[:, base : base + 6] += sign * point_motion

        # Relative deformation = target contact frame motion - source motion.
        add_endpoint(edge.source, -1.0)
        add_endpoint(edge.target, +1.0)

        return _ContactElement(
            edge=edge,
            point=point,
            B=B,
            stiffness=self._stiffness_for(edge),
            frame=contact_frame_for_edge_3d(edge, self.robots, self.field, self.robot_config),
        )

    def _assemble_component(
        self,
        active: set[int],
    ) -> tuple[list[int], dict[int, int], np.ndarray, list[_ContactElement], int, int]:
        ordered = sorted(active)
        index = {robot_id: i for i, robot_id in enumerate(ordered)}
        dof_count = 6 * len(ordered)
        K = np.zeros((dof_count, dof_count), dtype=float)
        elements: list[_ContactElement] = []
        for edge in self.graph.edges.values():
            robot_nodes = [node for node in (edge.source, edge.target) if isinstance(node, int)]
            if any(node not in active for node in robot_nodes):
                continue
            element = self._element(edge, index, dof_count)
            if element is None:
                continue
            K += element.B.T @ element.stiffness @ element.B
            elements.append(element)

        if self.verify_rank:
            singular_values = np.linalg.svd(K, compute_uv=False)
            largest = float(singular_values[0]) if singular_values.size else 0.0
            threshold = self.rank_tolerance * max(largest, 1.0)
            rank = int(np.count_nonzero(singular_values > threshold))
        else:
            # Procedurally generated training bridges are constructed from welded
            # 6-DOF connections with fixed bank supports, so they are full rank by
            # construction. The direct solve below still falls back to SVD/lstsq
            # if this assumption is violated numerically.
            rank = dof_count
        return ordered, index, K, elements, rank, dof_count

    def _force_vector(
        self,
        loads: Sequence[SpatialLoad],
        index: Mapping[int, int],
        dof_count: int,
    ) -> np.ndarray:
        force = np.zeros(dof_count, dtype=float)
        for load in loads:
            if load.robot_id not in index:
                continue
            base = 6 * index[load.robot_id]
            applied_force = np.asarray(load.force, dtype=float)
            applied_moment = np.asarray(load.moment, dtype=float)
            if applied_force.shape != (3,) or applied_moment.shape != (3,):
                raise ValueError("SpatialLoad force and moment must both be 3-vectors")
            point = (
                np.asarray(load.point, dtype=float)
                if load.point is not None
                else robot_center_3d(self.robots[load.robot_id], self.robot_config)
            )
            if point.shape != (3,):
                raise ValueError("SpatialLoad point must be a 3-vector")
            r = point - robot_center_3d(self.robots[load.robot_id], self.robot_config)
            force[base : base + 3] += applied_force
            force[base + 3 : base + 6] += np.cross(r, applied_force) + applied_moment
        return force

    def _result_from_solution(
        self,
        ordered: Sequence[int],
        K: np.ndarray,
        elements: Sequence[_ContactElement],
        rank: int,
        force: np.ndarray,
        q: np.ndarray,
    ) -> SpatialStaticResult:
        dof_count = len(q)
        residual = K @ q - force
        residual_norm = float(np.linalg.norm(residual))
        force_scale = max(float(np.linalg.norm(force)), 1.0)
        stable = rank == dof_count and residual_norm <= 1.0e-8 * force_scale
        failure_mode = "none" if stable else "mechanism"
        displacements = {
            robot_id: tuple(float(value) for value in q[6 * i : 6 * i + 6])
            for i, robot_id in enumerate(ordered)
        }
        contacts: list[SpatialContactResponse] = []
        for element in elements:
            relative = element.B @ q
            wrench = element.stiffness @ relative
            contacts.append(
                SpatialContactResponse(
                    source=element.edge.source,
                    target=element.edge.target,
                    kind=element.edge.kind,
                    point=tuple(float(value) for value in element.point),
                    wrench_on_source=SpatialWrench.from_array(wrench),
                    relative_displacement=tuple(float(value) for value in relative),
                    frame=element.frame,
                )
            )
        return SpatialStaticResult(
            stable=stable,
            displacements=displacements,
            contacts=contacts,
            residual_norm=residual_norm,
            rank=rank,
            dof_count=dof_count,
            failure_mode=failure_mode,
        )

    def _solve_same_component_cases(
        self,
        load_cases: Sequence[Sequence[SpatialLoad]],
        active: set[int],
    ) -> list[SpatialStaticResult]:
        if not active:
            return [SpatialStaticResult(False, {}, [], inf, 0, 0, "unsupported_component") for _ in load_cases]
        ordered, index, K, elements, rank, dof_count = self._assemble_component(active)
        forces = [self._force_vector(case, index, dof_count) for case in load_cases]
        rhs = np.column_stack(forces)
        if rank == dof_count:
            try:
                solutions = np.linalg.solve(K, rhs)
            except np.linalg.LinAlgError:
                singular_values = np.linalg.svd(K, compute_uv=False)
                largest = float(singular_values[0]) if singular_values.size else 0.0
                threshold = self.rank_tolerance * max(largest, 1.0)
                rank = int(np.count_nonzero(singular_values > threshold))
                solutions, _, _, _ = np.linalg.lstsq(K, rhs, rcond=self.rank_tolerance)
        else:
            solutions, _, _, _ = np.linalg.lstsq(K, rhs, rcond=self.rank_tolerance)
        return [
            self._result_from_solution(ordered, K, elements, rank, force, solutions[:, case_index])
            for case_index, force in enumerate(forces)
        ]

    def solve(self, loads: Iterable[SpatialLoad]) -> SpatialStaticResult:
        loads = tuple(loads)
        if not loads:
            return SpatialStaticResult(True, {}, [], 0.0, 0, 0)

        active = self._component_for_loads(loads)
        if not active or not self._all_loaded_components_supported(loads):
            return SpatialStaticResult(False, {}, [], inf, 0, 6 * len(active), "unsupported_component")
        return self._solve_same_component_cases((loads,), active)[0]

    def self_weight_loads(
        self,
        robot_ids: Iterable[int],
        *,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        mass_per_robot: float | None = None,
    ) -> tuple[SpatialLoad, ...]:
        gravity_vector = np.asarray(gravity, dtype=float)
        if gravity_vector.shape != (3,):
            raise ValueError("gravity must be a 3-vector")
        mass = float(self.robot_config.mass if mass_per_robot is None else mass_per_robot)
        if mass < 0.0:
            raise ValueError("Robot mass must be non-negative")
        force = mass * gravity_vector
        return tuple(
            SpatialLoad(
                robot_id=robot_id,
                force=(float(force[0]), float(force[1]), float(force[2])),
                point=tuple(float(value) for value in robot_center_3d(self.robots[robot_id], self.robot_config)),
            )
            for robot_id in robot_ids
            if not self.robots[robot_id].fallen
        )

    def capacity(
        self,
        payload_per_kg: SpatialLoad,
        limits: SpatialWrenchLimit | Mapping[EdgeKey, SpatialWrenchLimit],
        *,
        base_loads: Iterable[SpatialLoad] = (),
        include_self_weight: bool = True,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        mass_per_robot: float | None = None,
    ) -> SpatialCapacityResult:
        """Return the exact payload mass at the first component-wise limit.

        For the linear small-displacement model, every contact wrench is affine
        in payload mass:

            w(m) = w_base + m * w_per_kg

        `w_base` includes robot self-weight by default. The critical mass is
        therefore obtained analytically, without incrementing payload in a loop.
        """

        component_robot_ids = sorted(_robot_component(self.graph, payload_per_kg.robot_id))

        loads = list(base_loads)
        if include_self_weight:
            loads.extend(
                self.self_weight_loads(
                    component_robot_ids,
                    gravity=gravity,
                    mass_per_robot=mass_per_robot,
                )
            )
        # A zero selector ensures the bridge component is assembled even if
        # self-weight is disabled and no explicit base load was supplied.
        loads.append(
            SpatialLoad(
                payload_per_kg.robot_id,
                (0.0, 0.0, 0.0),
                payload_per_kg.point,
                (0.0, 0.0, 0.0),
            )
        )

        active = set(component_robot_ids)
        supports = _fixed_supports_for(self.graph, active)
        if not supports:
            base_response = SpatialStaticResult(False, {}, [], inf, 0, 6 * len(active), "unsupported_component")
            unit_response = base_response
        else:
            base_response, unit_response = self._solve_same_component_cases(
                (tuple(loads), (payload_per_kg,)),
                active,
            )

        if not base_response.stable or not unit_response.stable:
            return SpatialCapacityResult(
                0.0,
                "mechanism",
                None,
                None,
                None,
                None,
                base_response,
                unit_response,
            )

        base_by_key = base_response.contact_by_key()
        unit_by_key = unit_response.contact_by_key()

        def limit_for(key: EdgeKey) -> SpatialWrenchLimit:
            if isinstance(limits, SpatialWrenchLimit):
                return limits
            return limits.get(key, SpatialWrenchLimit())

        best_mass = inf
        best_contact: SpatialContactResponse | None = None
        best_component: str | None = None
        best_limit: float | None = None
        best_value: float | None = None

        for key in set(base_by_key) | set(unit_by_key):
            base_contact = base_by_key.get(key)
            unit_contact = unit_by_key.get(key)
            representative = base_contact or unit_contact
            if representative is None:
                continue
            a = base_contact.wrench_on_source.as_array() if base_contact is not None else np.zeros(6)
            b = unit_contact.wrench_on_source.as_array() if unit_contact is not None else np.zeros(6)
            maxima = limit_for(key).as_array()

            for component_index, (a_value, b_value, maximum) in enumerate(zip(a, b, maxima)):
                if maximum < 0.0:
                    raise ValueError("Wrench limits must be non-negative")
                if not np.isfinite(maximum):
                    continue
                if abs(a_value) > maximum + 1.0e-12:
                    candidate = 0.0
                elif abs(b_value) <= 1.0e-12:
                    candidate = inf
                elif b_value > 0.0:
                    candidate = (maximum - a_value) / b_value
                else:
                    candidate = (-maximum - a_value) / b_value
                candidate = max(float(candidate), 0.0)
                if candidate < best_mass:
                    best_mass = candidate
                    best_contact = representative
                    best_component = _COMPONENT_NAMES[component_index]
                    best_limit = float(maximum)
                    best_value = float(a_value + candidate * b_value)

        if best_contact is None:
            return SpatialCapacityResult(
                inf,
                "none",
                None,
                None,
                None,
                None,
                base_response,
                unit_response,
            )

        return SpatialCapacityResult(
            max_payload_kg=float(best_mass),
            failure_mode="contact_limit",
            critical_contact=(best_contact.source, best_contact.target),
            critical_component=best_component,
            critical_limit=best_limit,
            critical_value=best_value,
            base_response=base_response,
            unit_payload_response=unit_response,
        )

    def capacity_local(
        self,
        payload_per_kg: SpatialLoad,
        limits: LocalWrenchLimit | Mapping[EdgeKey, LocalWrenchLimit],
        *,
        base_loads: Iterable[SpatialLoad] = (),
        include_self_weight: bool = True,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        mass_per_robot: float | None = None,
    ) -> SpatialCapacityResult:
        """Return payload capacity using connector-local strength components.

        The statics solve itself remains in world coordinates, but every
        contact wrench is rotated into the contact frame before evaluating
        strength. Thus limits correspond to physically meaningful connector
        modes: axial, two shears, torsion and two bending axes.
        """

        component_robot_ids = sorted(_robot_component(self.graph, payload_per_kg.robot_id))
        loads = list(base_loads)
        if include_self_weight:
            loads.extend(
                self.self_weight_loads(
                    component_robot_ids,
                    gravity=gravity,
                    mass_per_robot=mass_per_robot,
                )
            )
        loads.append(
            SpatialLoad(
                payload_per_kg.robot_id,
                (0.0, 0.0, 0.0),
                payload_per_kg.point,
                (0.0, 0.0, 0.0),
            )
        )

        active = set(component_robot_ids)
        supports = _fixed_supports_for(self.graph, active)
        if not supports:
            base_response = SpatialStaticResult(False, {}, [], inf, 0, 6 * len(active), "unsupported_component")
            unit_response = base_response
        else:
            base_response, unit_response = self._solve_same_component_cases(
                (tuple(loads), (payload_per_kg,)),
                active,
            )

        if not base_response.stable or not unit_response.stable:
            return SpatialCapacityResult(
                0.0,
                "mechanism",
                None,
                None,
                None,
                None,
                base_response,
                unit_response,
            )

        base_by_key = base_response.contact_by_key()
        unit_by_key = unit_response.contact_by_key()

        def limit_for(key: EdgeKey) -> LocalWrenchLimit:
            if isinstance(limits, LocalWrenchLimit):
                return limits
            return limits.get(key, LocalWrenchLimit())

        best_mass = inf
        best_contact: SpatialContactResponse | None = None
        best_component: str | None = None
        best_limit: float | None = None
        best_value: float | None = None

        for key in set(base_by_key) | set(unit_by_key):
            base_contact = base_by_key.get(key)
            unit_contact = unit_by_key.get(key)
            representative = base_contact or unit_contact
            if representative is None:
                continue
            a = base_contact.local_wrench_on_source.as_array() if base_contact is not None else np.zeros(6)
            b = unit_contact.local_wrench_on_source.as_array() if unit_contact is not None else np.zeros(6)
            maxima = limit_for(key).as_array()

            for component_index, (a_value, b_value, maximum) in enumerate(zip(a, b, maxima)):
                if maximum < 0.0:
                    raise ValueError("Wrench limits must be non-negative")
                if not np.isfinite(maximum):
                    continue
                if abs(a_value) > maximum + 1.0e-12:
                    candidate = 0.0
                elif abs(b_value) <= 1.0e-12:
                    candidate = inf
                elif b_value > 0.0:
                    candidate = (maximum - a_value) / b_value
                else:
                    candidate = (-maximum - a_value) / b_value
                candidate = max(float(candidate), 0.0)
                if candidate < best_mass:
                    best_mass = candidate
                    best_contact = representative
                    best_component = _LOCAL_COMPONENT_NAMES[component_index]
                    best_limit = float(maximum)
                    best_value = float(a_value + candidate * b_value)

        if best_contact is None:
            return SpatialCapacityResult(
                inf,
                "none",
                None,
                None,
                None,
                None,
                base_response,
                unit_response,
            )

        return SpatialCapacityResult(
            max_payload_kg=float(best_mass),
            failure_mode="contact_limit",
            critical_contact=(best_contact.source, best_contact.target),
            critical_component=best_component,
            critical_limit=best_limit,
            critical_value=best_value,
            base_response=base_response,
            unit_payload_response=unit_response,
        )


def bridge_center_payload_3d(
    robots: Sequence[RobotState],
    graph: ContactGraph,
    field: GapField,
    robot_config: RobotConfig,
    *,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> SpatialLoad:
    """Create a 1 kg gravity payload on the top face nearest gap center."""

    spanning_components = _spanning_robot_components(graph, len(robots))
    if not spanning_components:
        raise ValueError("Bridge does not span both banks through a robot-to-robot structure")

    target = np.asarray(field.center, dtype=float)
    candidates = [
        node
        for component in spanning_components
        for node in component
        if not robots[node].fallen
    ]
    if not candidates:
        raise ValueError("Spanning structure contains no active robots")
    robot_id = min(candidates, key=lambda node: float(np.linalg.norm(robots[node].position - target)))
    faces = robot_face_centers_3d(robots[robot_id], robot_config)
    top_center = faces[4]
    gravity_vector = np.asarray(gravity, dtype=float)
    if gravity_vector.shape != (3,) or float(np.linalg.norm(gravity_vector)) <= 1.0e-12:
        raise ValueError("gravity must be a non-zero 3-vector")
    return SpatialLoad(
        robot_id=robot_id,
        force=tuple(float(value) for value in gravity_vector),
        point=tuple(float(value) for value in top_center),
    )
