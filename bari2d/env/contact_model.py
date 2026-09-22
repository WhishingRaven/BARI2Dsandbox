from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from bari2d.env.field import LEFT_BANK, RIGHT_BANK, GapField
from bari2d.env.robot import RobotState
from bari2d.utils.config import ContactConfig, RobotConfig


Node = int | str


@dataclass
class MechanicalEdge:
    source: Node
    target: Node
    kind: str
    capacity: float
    stiffness: float
    force: float = 0.0
    relative_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def key(self) -> frozenset[Node]:
        return frozenset((self.source, self.target))


class ContactGraph:
    def __init__(self, robot_count: int):
        self.nodes: set[Node] = set(range(robot_count)) | {LEFT_BANK, RIGHT_BANK}
        self.edges: dict[frozenset[Node], MechanicalEdge] = {}
        self.node_progress: dict[int, float] = {}
        self.node_tangent: dict[int, float] = {}

    def add_edge(self, edge: MechanicalEdge) -> None:
        existing = self.edges.get(edge.key)
        if existing is None or edge.kind == "anchor" or existing.kind != "anchor":
            self.edges[edge.key] = edge

    def remove_edge(self, source: Node, target: Node) -> None:
        self.edges.pop(frozenset((source, target)), None)

    def neighbors(self, node: Node) -> list[Node]:
        result: list[Node] = []
        for edge in self.edges.values():
            if edge.source == node:
                result.append(edge.target)
            elif edge.target == node:
                result.append(edge.source)
        return result

    def connected_component(self, start: Node) -> set[Node]:
        visited: set[Node] = {start}
        frontier = [start]
        while frontier:
            node = frontier.pop()
            for neighbor in self.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        return visited

    @property
    def spans(self) -> bool:
        return RIGHT_BANK in self.connected_component(LEFT_BANK)

    def spanning_progress(self, robots: list[RobotState], field: GapField) -> float:
        connected = self.connected_component(LEFT_BANK)
        progress = [field.normalized_progress(robots[node].position) for node in connected if isinstance(node, int)]
        return max(progress, default=0.0)

    def serializable_edges(self) -> list[dict[str, object]]:
        return [
            {
                "source": edge.source,
                "target": edge.target,
                "kind": edge.kind,
                "capacity": edge.capacity,
                "force": edge.force,
            }
            for edge in self.edges.values()
        ]


@dataclass
class AnchorConnection:
    owner: int
    partner: Node
    rest_distance: float
    rest_angle: float
    failed: bool = False


def _axes(corners: np.ndarray) -> list[np.ndarray]:
    edges = [corners[1] - corners[0], corners[3] - corners[0]]
    result = []
    for edge in edges:
        axis = np.array([-edge[1], edge[0]], dtype=float)
        result.append(axis / max(np.linalg.norm(axis), 1.0e-9))
    return result


def oriented_box_contact(
    first: RobotState, second: RobotState, robot_config: RobotConfig, tolerance: float = 0.025
) -> tuple[bool, np.ndarray, float]:
    first_corners = first.corners(robot_config)
    second_corners = second.corners(robot_config)
    minimum_overlap = float("inf")
    minimum_axis = np.array([1.0, 0.0])
    for axis in _axes(first_corners) + _axes(second_corners):
        first_projection = first_corners @ axis
        second_projection = second_corners @ axis
        overlap = min(first_projection.max(), second_projection.max()) - max(
            first_projection.min(), second_projection.min()
        )
        if overlap < -tolerance:
            return False, minimum_axis, 0.0
        if overlap < minimum_overlap:
            minimum_overlap = overlap
            minimum_axis = axis
    direction = second.position - first.position
    if direction @ minimum_axis < 0.0:
        minimum_axis = -minimum_axis
    return True, minimum_axis, max(float(minimum_overlap), 0.0)


def oriented_boxes_overlap(
    first: RobotState, second: RobotState, robot_config: RobotConfig
) -> bool:
    """Return whether two robot footprints have a positive-area overlap."""
    touching, _, penetration = oriented_box_contact(first, second, robot_config, tolerance=0.0)
    return touching and penetration > 1.0e-9


class ContactModel:
    def __init__(self, config: ContactConfig, robot_config: RobotConfig):
        self.config = config
        self.robot_config = robot_config
        self.anchors: dict[int, AnchorConnection] = {}
        self.anchor_failures = 0

    @property
    def anchor_capacity(self) -> float:
        config = self.config
        axial = min(config.anchor_tension_limit, config.anchor_compression_limit)
        return float(np.hypot(axial, config.anchor_shear_limit))

    def reset(self) -> None:
        self.anchors.clear()
        self.anchor_failures = 0

    def candidate_anchor(self, robot: RobotState, robots: list[RobotState], field: GapField) -> Node | None:
        bank_contacts = field.bank_contact(robot.corners(self.robot_config))
        if bank_contacts:
            return sorted(bank_contacts)[0]
        candidates: list[tuple[float, int]] = []
        for other in robots:
            if other.robot_id == robot.robot_id or other.fallen:
                continue
            distance = float(np.linalg.norm(other.position - robot.position))
            if distance <= self.robot_config.anchor_range and abs(other.layer - robot.layer) <= 1:
                candidates.append((distance, other.robot_id))
        return min(candidates, default=(0.0, -1))[1] if candidates else None

    def can_anchor(self, robot: RobotState, robots: list[RobotState], field: GapField) -> bool:
        return not robot.anchored and self.candidate_anchor(robot, robots, field) is not None

    def anchor(self, robot: RobotState, robots: list[RobotState], field: GapField) -> bool:
        partner = self.candidate_anchor(robot, robots, field)
        if partner is None:
            return False
        if isinstance(partner, int):
            relative = robots[partner].position - robot.position
            distance = float(np.linalg.norm(relative))
            angle = float(np.arctan2(relative[1], relative[0]) - robot.theta)
        else:
            distance = 0.0
            angle = 0.0
        self.anchors[robot.robot_id] = AnchorConnection(robot.robot_id, partner, distance, angle)
        robot.anchored = True
        robot.anchor_partner = partner
        return True

    def release(self, robot: RobotState) -> bool:
        if not robot.anchored:
            return False
        self.anchors.pop(robot.robot_id, None)
        robot.anchored = False
        robot.anchor_partner = None
        return True

    def _anchor_edge(
        self, anchor: AnchorConnection, robots: list[RobotState], rng: np.random.Generator
    ) -> MechanicalEdge | None:
        owner = robots[anchor.owner]
        if isinstance(anchor.partner, int):
            partner = robots[anchor.partner]
            relative = partner.position - owner.position
            distance = float(np.linalg.norm(relative))
            direction = relative / max(distance, 1.0e-9)
            extension = distance - anchor.rest_distance
            axial_force = self.config.contact_stiffness * extension
            transverse = abs(float(owner.heading[0] * direction[1] - owner.heading[1] * direction[0]))
            shear_force = self.config.contact_stiffness * 0.1 * transverse
            angular_error = abs(float((np.arctan2(relative[1], relative[0]) - owner.theta - anchor.rest_angle + np.pi) % (2 * np.pi) - np.pi))
            rotational_force = self.config.anchor_rotational_stiffness * angular_error
            failed = (
                axial_force > self.config.anchor_tension_limit
                or -axial_force > self.config.anchor_compression_limit
                or shear_force > self.config.anchor_shear_limit
                or rng.random() < self.config.anchor_failure_probability
            )
            force = abs(axial_force) + shear_force + rotational_force
            relative_pose = (float(relative[0]), float(relative[1]), angular_error)
        else:
            failed = rng.random() < self.config.anchor_failure_probability
            force = 0.0
            relative_pose = (0.0, 0.0, 0.0)
        if failed:
            anchor.failed = True
            owner.anchored = False
            owner.anchor_partner = None
            self.anchor_failures += 1
            return None
        owner.strain += force
        if isinstance(anchor.partner, int):
            robots[anchor.partner].strain += force
        return MechanicalEdge(
            anchor.owner,
            anchor.partner,
            "anchor",
            self.anchor_capacity,
            max(self.config.contact_stiffness, 1.0e-6),
            force,
            relative_pose,
        )

    def build_graph(
        self, robots: list[RobotState], field: GapField, rng: np.random.Generator
    ) -> ContactGraph:
        graph = ContactGraph(len(robots))
        for robot in robots:
            _, tangent = field.coordinates(robot.position)
            graph.node_progress[robot.robot_id] = field.normalized_progress(robot.position)
            graph.node_tangent[robot.robot_id] = tangent
            robot.strain = 0.0
            if robot.fallen:
                continue
            for bank in field.bank_contact(robot.corners(self.robot_config)):
                graph.add_edge(
                    MechanicalEdge(
                        robot.robot_id,
                        bank,
                        "bank_contact",
                        self.config.bank_capacity * self.config.friction,
                        self.config.contact_stiffness,
                    )
                )
        for index, first in enumerate(robots):
            if first.fallen:
                continue
            for second in robots[index + 1 :]:
                if second.fallen or abs(first.layer - second.layer) > 1:
                    continue
                touching, normal, penetration = oriented_box_contact(first, second, self.robot_config)
                if not touching:
                    continue
                relative_velocity = abs(first.velocity - second.velocity)
                force = max(
                    0.0,
                    self.config.contact_stiffness * penetration
                    + self.config.contact_damping * relative_velocity,
                )
                first.strain += force
                second.strain += force
                relative = second.position - first.position
                graph.add_edge(
                    MechanicalEdge(
                        first.robot_id,
                        second.robot_id,
                        "contact",
                        self.config.contact_capacity * self.config.friction,
                        self.config.contact_stiffness,
                        force,
                        (float(relative[0]), float(relative[1]), float(np.arctan2(normal[1], normal[0]))),
                    )
                )
        failed_owners = []
        for owner, anchor in self.anchors.items():
            edge = self._anchor_edge(anchor, robots, rng)
            if edge is None:
                failed_owners.append(owner)
            else:
                graph.add_edge(edge)
        for owner in failed_owners:
            self.anchors.pop(owner, None)
        return graph

    def contact_counts(self, graph: ContactGraph, robot_count: int) -> np.ndarray:
        return np.array(
            [sum(isinstance(node, int) for node in graph.neighbors(robot_id)) for robot_id in range(robot_count)],
            dtype=np.float32,
        )

    def resolve_contacts(self, robots: list[RobotState]) -> None:
        """Apply a positional normal response and Coulomb-like velocity damping."""
        for index, first in enumerate(robots):
            if first.fallen:
                continue
            for second in robots[index + 1 :]:
                if second.fallen or first.layer != second.layer:
                    continue
                touching, normal, penetration = oriented_box_contact(
                    first, second, self.robot_config, tolerance=0.0
                )
                correction = max(penetration - 0.005, 0.0)
                if not touching or correction <= 0.0:
                    continue
                first_mobile = not first.anchored
                second_mobile = not second.anchored
                if first_mobile and second_mobile:
                    first.position -= normal * correction * 0.5
                    second.position += normal * correction * 0.5
                elif first_mobile:
                    first.position -= normal * correction
                elif second_mobile:
                    second.position += normal * correction
                damping = max(0.0, 1.0 - self.config.friction)
                first.velocity *= damping
                second.velocity *= damping
