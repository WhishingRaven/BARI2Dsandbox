"""Glue between ``BridgeEnv`` and the rigid-body equilibrium model in ``statics``.

Only used when ``environment.physics.model == "static"``; the legacy physics
path does not import anything from here.
"""

from __future__ import annotations

import numpy as np

from bari2d.env import statics
from bari2d.env.contact_model import AnchorConnection, ContactModel, MechanicalEdge, Node, oriented_boxes_overlap
from bari2d.env.field import LEFT_BANK, RIGHT_BANK, GapField
from bari2d.env.robot import RobotState
from bari2d.utils.config import ContactConfig, PhysicsConfig, RobotConfig


PHYSICS_MODELS = {"legacy", "static"}


def bank_polygon(field: GapField, robot: RobotState, corners: np.ndarray, bank: str) -> np.ndarray:
    """Part of a robot footprint resting on ``bank`` (bank edge taken at the robot's tangent)."""
    _, tangent = field.coordinates(robot.position)
    left, right = field.boundaries(tangent)
    normal = field.normal
    center_offset = float(field.center @ normal)
    if bank == LEFT_BANK:
        return statics.clip_halfplane(corners, normal, left + center_offset)
    return statics.clip_halfplane(corners, -normal, -(right + center_offset))


def polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    area = statics.polygon_area(polygon)
    if abs(area) < 1.0e-12:
        return np.mean(polygon, axis=0)
    x, y = polygon[:, 0], polygon[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    return np.array(
        [np.sum((x + np.roll(x, -1)) * cross), np.sum((y + np.roll(y, -1)) * cross)]
    ) / (6.0 * area)


class StaticContactModel(ContactModel):
    """Anchor semantics under gravity.

    * A robot on an upper layer can only grip the robot it stands on (it cannot
      reach the ground).
    * Gripping while standing on the front half of an aligned robot is a
      grip-and-extend manoeuvre: the robot steps off the front end while
      holding on and becomes the new end-to-end tip of the chain, one layer
      down.  This is how a gap is crossed at deck level (walk over the
      structure, then attach at its frontier).
    * In-plane anchor failure only follows from real stretching or compression
      of the joint; vertical load and bending are handled by the equilibrium
      solver.
    """

    def __init__(self, config: ContactConfig, robot_config: RobotConfig, physics: PhysicsConfig):
        super().__init__(config, robot_config)
        self.physics = physics

    def support_partner(self, robot: RobotState, robots: list[RobotState]) -> int | None:
        if robot.layer < 1:
            return None
        corners = robot.corners(self.robot_config)
        best: int | None = None
        best_area = 1.0e-9
        for other in robots:
            if other.robot_id == robot.robot_id or other.fallen or other.layer != robot.layer - 1:
                continue
            area = abs(statics.polygon_area(statics.clip_convex(corners, other.corners(self.robot_config))))
            if area > best_area:
                best, best_area = other.robot_id, area
        return best

    def candidate_anchor(self, robot: RobotState, robots: list[RobotState], field: GapField) -> Node | None:
        if robot.layer >= 1:
            return self.support_partner(robot, robots)
        return super().candidate_anchor(robot, robots, field)

    def extension_pose(
        self, robot: RobotState, partner: RobotState, robots: list[RobotState], field: GapField
    ) -> np.ndarray | None:
        if not self.physics.extend_on_anchor or robot.layer < 1 or partner.layer != robot.layer - 1:
            return None
        if float(robot.heading @ partner.heading) < self.physics.extend_alignment:
            return None
        if float((robot.position - partner.position) @ partner.heading) <= 0.0:
            return None
        target = partner.position + partner.heading * self.robot_config.length
        probe = RobotState(robot.robot_id, target.copy(), partner.theta, layer=partner.layer)
        if not all(field.inside(corner) for corner in probe.corners(self.robot_config)):
            return None
        for other in robots:
            if other.robot_id in (robot.robot_id, partner.robot_id) or other.fallen or other.layer != partner.layer:
                continue
            if oriented_boxes_overlap(probe, other, self.robot_config):
                return None
        return target

    def _attach(self, robot: RobotState, partner: Node, robots: list[RobotState]) -> bool:
        if isinstance(partner, int):
            relative = robots[partner].position - robot.position
            distance = float(np.linalg.norm(relative))
            angle = float(np.arctan2(relative[1], relative[0]) - robot.theta)
        else:
            distance, angle = 0.0, 0.0
        self.anchors[robot.robot_id] = AnchorConnection(robot.robot_id, partner, distance, angle)
        robot.anchored = True
        robot.anchor_partner = partner
        return True

    def anchor(self, robot: RobotState, robots: list[RobotState], field: GapField) -> bool:
        partner = self.candidate_anchor(robot, robots, field)
        if partner is None:
            return False
        if isinstance(partner, int):
            target = self.extension_pose(robot, robots[partner], robots, field)
            if target is not None:
                support = robots[partner]
                robot.position = target
                robot.theta = support.theta
                robot.layer = support.layer
                robot.velocity = 0.0
                robot.steering = 0.0
        return self._attach(robot, partner, robots)

    def _anchor_edge(
        self, anchor: AnchorConnection, robots: list[RobotState], rng: np.random.Generator
    ) -> MechanicalEdge | None:
        owner = robots[anchor.owner]
        if isinstance(anchor.partner, int):
            partner = robots[anchor.partner]
            relative = partner.position - owner.position
            distance = float(np.linalg.norm(relative))
            axial_force = self.config.contact_stiffness * (distance - anchor.rest_distance)
            failed = (
                partner.fallen
                or axial_force > self.config.anchor_tension_limit
                or -axial_force > self.config.anchor_compression_limit
                or rng.random() < self.config.anchor_failure_probability
            )
            force = abs(axial_force)
            angle = float(np.arctan2(relative[1], relative[0]) - owner.theta - anchor.rest_angle)
            relative_pose = (float(relative[0]), float(relative[1]), abs((angle + np.pi) % (2 * np.pi) - np.pi))
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


def build_structure(
    robots: list[RobotState],
    field: GapField,
    contact_model: ContactModel,
    robot_config: RobotConfig,
    physics: PhysicsConfig,
    robot_weight: float,
    strength_scale: float = 1.0,
) -> statics.StaticStructure:
    """Translate robot poses, layers and anchors into equilibrium interfaces."""
    active = [robot for robot in robots if not robot.fallen]
    corners = {robot.robot_id: robot.corners(robot_config) for robot in active}
    bodies: dict[int, statics.Body] = {}
    patches: list[statics.SupportPatch] = []
    for robot in active:
        footprint = corners[robot.robot_id]
        fully_on_bank = robot.layer == 0 and all(field.bank_at(corner) is not None for corner in footprint)
        bodies[robot.robot_id] = statics.Body(
            robot.robot_id,
            np.asarray(robot.position, dtype=float).copy(),
            robot_weight,
            robot_config.length,
            field.is_gap(robot.position),
            fully_on_bank,
            field.normalized_progress(robot.position),
        )
        if robot.layer <= physics.ground_contact_max_layer:
            for bank in (LEFT_BANK, RIGHT_BANK):
                polygon = bank_polygon(field, robot, footprint, bank)
                if abs(statics.polygon_area(polygon)) > 1.0e-9:
                    patches.append(statics.SupportPatch(robot.robot_id, bank, polygon))
        if robot.layer >= 1:
            for other in active:
                if other.layer != robot.layer - 1:
                    continue
                polygon = statics.clip_convex(footprint, corners[other.robot_id])
                if abs(statics.polygon_area(polygon)) > 1.0e-9:
                    patches.append(statics.SupportPatch(robot.robot_id, other.robot_id, polygon))

    if physics.anchor_vertical_limit is None:
        vertical_limit = contact_model.config.anchor_shear_limit  # already carries episode strength scaling
    else:
        vertical_limit = physics.anchor_vertical_limit * strength_scale
    bending_limit = physics.anchor_bending_limit * strength_scale
    torsion_limit = physics.anchor_torsion_limit * strength_scale
    joints: list[statics.Joint] = []
    for owner_id, anchor in contact_model.anchors.items():
        if owner_id not in bodies:
            continue
        owner = robots[owner_id]
        partner = anchor.partner
        if isinstance(partner, int):
            if partner not in bodies:
                continue
            offset = robots[partner].position - owner.position
            distance = float(np.linalg.norm(offset))
            point = 0.5 * (owner.position + robots[partner].position)
            direction = offset / distance if distance > 1.0e-9 else owner.heading
        else:
            polygon = bank_polygon(field, owner, corners[owner_id], partner)
            point = polygon_centroid(polygon) if abs(statics.polygon_area(polygon)) > 1.0e-9 else owner.position
            direction = field.normal if partner == LEFT_BANK else -field.normal
        joints.append(
            statics.Joint(
                owner_id,
                partner,
                np.asarray(point, dtype=float),
                np.asarray(direction, dtype=float),
                vertical_limit,
                bending_limit,
                torsion_limit,
            )
        )
    if physics.splice_grip:
        footprint_area = robot_config.length * robot_config.width
        for owner_id, anchor in contact_model.anchors.items():
            owner = robots[owner_id]
            if owner_id not in bodies or owner.layer < 1 or not isinstance(anchor.partner, int):
                continue
            for other in active:
                if other.robot_id in (owner_id, anchor.partner) or other.layer != owner.layer - 1:
                    continue
                overlap = statics.clip_convex(corners[owner_id], corners[other.robot_id])
                area = abs(statics.polygon_area(overlap))
                if area < physics.splice_min_overlap * footprint_area:
                    continue
                joints.append(
                    statics.Joint(
                        owner_id,
                        other.robot_id,
                        np.asarray(polygon_centroid(overlap), dtype=float),
                        np.asarray(other.heading, dtype=float),
                        vertical_limit,
                        bending_limit,
                        torsion_limit,
                    )
                )
    return statics.StaticStructure(bodies, patches, joints)
