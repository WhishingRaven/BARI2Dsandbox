from __future__ import annotations

import numpy as np

from bari2d.env.contact_model import ContactGraph, MechanicalEdge
from bari2d.env.field import LEFT_BANK, GapField
from bari2d.env.robot import RobotState
from bari2d.mechanics.spatial_statics import (
    LocalWrenchLimit,
    SpatialContactStiffness,
    SpatialLoad,
    SpatialStaticSolver,
    SpatialWrenchLimit,
    contact_frame_for_edge_3d,
    contact_point_for_edge_3d,
    robot_center_3d,
)
from bari2d.utils.config import RobotConfig


def _field() -> GapField:
    return GapField(
        length=10.0,
        width=6.0,
        gap_width=4.0,
        center=np.asarray([5.0, 3.0]),
        orientation=0.0,
    )


def _robot(robot_id: int, x: float, y: float = 3.0, *, layer: int = 0) -> RobotState:
    return RobotState(robot_id, np.asarray([x, y], dtype=float), 0.0, layer=layer)


def _rigid_stiffness() -> SpatialContactStiffness:
    return SpatialContactStiffness(1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6)


def test_single_anchored_robot_balances_arbitrary_3d_wrench() -> None:
    config = RobotConfig(count=1, length=1.0, width=0.4, climb_height=0.3)
    robots = [_robot(0, 3.0)]
    graph = ContactGraph(1)
    graph.add_edge(MechanicalEdge(0, LEFT_BANK, "anchor", 1.0, 1000.0))
    solver = SpatialStaticSolver(robots, graph, _field(), config, stiffness=_rigid_stiffness())

    center = robot_center_3d(robots[0], config)
    point = center + np.asarray([0.2, -0.1, 0.15])
    applied_force = np.asarray([10.0, -20.0, -30.0])
    applied_moment = np.asarray([2.0, 3.0, -4.0])
    result = solver.solve(
        [
            SpatialLoad(
                0,
                tuple(applied_force),
                point=tuple(point),
                moment=tuple(applied_moment),
            )
        ]
    )

    assert result.stable
    assert result.rank == result.dof_count == 6
    assert result.residual_norm < 1.0e-9
    assert len(result.contacts) == 1

    contact = result.contacts[0]
    reaction_force = contact.wrench_on_source.as_array()[:3]
    reaction_moment = contact.wrench_on_source.as_array()[3:]
    np.testing.assert_allclose(reaction_force + applied_force, np.zeros(3), atol=1.0e-8)

    rc = np.asarray(contact.point) - center
    total_moment = np.cross(point - center, applied_force) + applied_moment
    total_moment += np.cross(rc, reaction_force) + reaction_moment
    np.testing.assert_allclose(total_moment, np.zeros(3), atol=1.0e-8)


def test_adjacent_layers_contact_at_overlap_interface_center() -> None:
    config = RobotConfig(count=2, length=1.0, width=0.6, climb_height=0.4)
    robots = [_robot(0, 4.0, layer=0), _robot(1, 4.1, layer=1)]
    edge = MechanicalEdge(0, 1, "contact", 1.0, 1000.0)
    point = contact_point_for_edge_3d(edge, robots, _field(), config)

    np.testing.assert_allclose(point[:2], [4.05, 3.0], atol=1.0e-8)
    np.testing.assert_allclose(point[2], 0.4, atol=1.0e-8)


def test_stacked_contact_frame_resolves_vertical_force_as_axial() -> None:
    config = RobotConfig(count=2, length=1.0, width=0.6, climb_height=0.4)
    robots = [_robot(0, 4.0, layer=0), _robot(1, 4.0, layer=1)]
    edge = MechanicalEdge(0, 1, "contact", 1.0, 1000.0)
    frame = contact_frame_for_edge_3d(edge, robots, _field(), config)

    np.testing.assert_allclose(frame.normal, [0.0, 0.0, 1.0], atol=1.0e-10)
    rotation = frame.rotation_world_from_local
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1.0e-10)


def test_stacked_robot_transmits_vertical_payload_to_grounded_robot() -> None:
    config = RobotConfig(count=2, length=1.0, width=0.6, mass=1.0, climb_height=0.4)
    robots = [_robot(0, 4.0, layer=0), _robot(1, 4.0, layer=1)]
    graph = ContactGraph(2)
    graph.add_edge(MechanicalEdge(0, LEFT_BANK, "bank_contact", 1.0, 1000.0))
    graph.add_edge(MechanicalEdge(0, 1, "contact", 1.0, 1000.0))
    solver = SpatialStaticSolver(robots, graph, _field(), config, stiffness=_rigid_stiffness())

    payload = SpatialLoad(1, (0.0, 0.0, -9.81))
    base = solver.self_weight_loads([0, 1])
    result = solver.solve((*base, payload))
    assert result.stable

    contacts = {contact.key: contact for contact in result.contacts}
    robot_contact = contacts[frozenset((0, 1))].wrench_on_source
    ground_contact = contacts[frozenset((0, LEFT_BANK))].wrench_on_source
    local_robot_contact = contacts[frozenset((0, 1))].local_wrench_on_source

    # Top robot has its own 1 kg weight + 1 kg payload, so the inter-robot
    # contact carries 19.62 N vertically. Ground carries both robot weights and
    # the extra payload: 29.43 N total.
    np.testing.assert_allclose(abs(robot_contact.fz), 19.62, atol=1.0e-7)
    np.testing.assert_allclose(abs(ground_contact.fz), 29.43, atol=1.0e-7)
    np.testing.assert_allclose(abs(local_robot_contact.axial), 19.62, atol=1.0e-7)
    np.testing.assert_allclose(local_robot_contact.shear_u, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(local_robot_contact.shear_v, 0.0, atol=1.0e-7)


def test_capacity_includes_self_weight_and_solves_analytically() -> None:
    config = RobotConfig(count=1, length=1.0, width=0.4, mass=1.0, climb_height=0.3)
    robots = [_robot(0, 3.0)]
    graph = ContactGraph(1)
    graph.add_edge(MechanicalEdge(0, LEFT_BANK, "anchor", 1.0, 1000.0))
    solver = SpatialStaticSolver(robots, graph, _field(), config, stiffness=_rigid_stiffness())

    center = robot_center_3d(robots[0], config)
    payload = SpatialLoad(0, (0.0, 0.0, -9.81), point=tuple(center))
    result = solver.capacity(
        payload,
        SpatialWrenchLimit(fz=100.0),
        include_self_weight=True,
    )

    expected = (100.0 - 9.81) / 9.81
    assert result.failure_mode == "contact_limit"
    assert result.critical_component == "Fz"
    np.testing.assert_allclose(result.max_payload_kg, expected, atol=1.0e-9)
    np.testing.assert_allclose(abs(result.critical_value or 0.0), 100.0, atol=1.0e-8)


def test_local_capacity_uses_axial_limit_for_stacked_vertical_contact() -> None:
    config = RobotConfig(count=2, length=1.0, width=0.6, mass=1.0, climb_height=0.4)
    robots = [_robot(0, 4.0, layer=0), _robot(1, 4.0, layer=1)]
    graph = ContactGraph(2)
    graph.add_edge(MechanicalEdge(0, LEFT_BANK, "bank_contact", 1.0, 1000.0))
    stack_edge = MechanicalEdge(0, 1, "contact", 1.0, 1000.0)
    graph.add_edge(stack_edge)
    solver = SpatialStaticSolver(robots, graph, _field(), config, stiffness=_rigid_stiffness())

    payload = SpatialLoad(1, (0.0, 0.0, -9.81))
    limits = {
        stack_edge.key: LocalWrenchLimit(axial=40.0),
        frozenset((0, LEFT_BANK)): LocalWrenchLimit(),
    }
    result = solver.capacity_local(payload, limits, include_self_weight=True)

    # Inter-robot contact already carries the upper robot's 9.81 N self-weight;
    # each kg of payload adds another 9.81 N in the local axial direction.
    np.testing.assert_allclose(result.max_payload_kg, (40.0 - 9.81) / 9.81, atol=1.0e-8)
    assert result.critical_contact is not None
    assert frozenset(result.critical_contact) == stack_edge.key
    assert result.critical_component == "Axial"


def test_local_capacity_is_invariant_to_global_yaw_rotation() -> None:
    config = RobotConfig(count=1, length=1.0, width=0.4, mass=1.0, climb_height=0.3)
    limit = LocalWrenchLimit(
        axial=150.0,
        shear_u=120.0,
        shear_v=100.0,
        torsion=35.0,
        bending_u=45.0,
        bending_v=55.0,
    )

    capacities = []
    for angle in (0.0, 0.73):
        field = GapField(
            length=10.0,
            width=6.0,
            gap_width=4.0,
            center=np.asarray([5.0, 3.0]),
            orientation=angle,
        )
        position = field.center - 1.8 * field.normal
        robot = RobotState(0, np.asarray(position, dtype=float), angle, layer=0)
        graph = ContactGraph(1)
        graph.add_edge(MechanicalEdge(0, LEFT_BANK, "anchor", 1.0, 1000.0))
        solver = SpatialStaticSolver([robot], graph, field, config, stiffness=_rigid_stiffness())

        center = robot_center_3d(robot, config)
        point = center + np.asarray([field.tangent[0], field.tangent[1], 0.0]) * 0.15
        point[2] += 0.10
        payload = SpatialLoad(0, (0.0, 0.0, -9.81), point=tuple(point))
        result = solver.capacity_local(payload, limit, include_self_weight=True)
        capacities.append(result.max_payload_kg)

    np.testing.assert_allclose(capacities[0], capacities[1], rtol=1.0e-10, atol=1.0e-10)


def test_unsupported_3d_component_is_reported() -> None:
    config = RobotConfig(count=2, length=1.0, width=0.4, climb_height=0.3)
    robots = [_robot(0, 4.0), _robot(1, 4.9)]
    graph = ContactGraph(2)
    graph.add_edge(MechanicalEdge(0, 1, "contact", 1.0, 1000.0))
    solver = SpatialStaticSolver(robots, graph, _field(), config)

    result = solver.solve([SpatialLoad(0, (0.0, 0.0, -10.0))])
    assert not result.stable
    assert result.failure_mode == "unsupported_component"
