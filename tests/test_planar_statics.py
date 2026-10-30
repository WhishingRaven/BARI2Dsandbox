from __future__ import annotations

import numpy as np

from bari2d.env.contact_model import ContactGraph, MechanicalEdge
from bari2d.env.field import LEFT_BANK, RIGHT_BANK, GapField
from bari2d.env.robot import RobotState
from bari2d.mechanics import PlanarLoad, PlanarStaticSolver, PlanarWrenchLimit
from bari2d.utils.config import RobotConfig


def _field() -> GapField:
    return GapField(
        length=10.0,
        width=6.0,
        gap_width=4.0,
        center=np.asarray([5.0, 3.0]),
        orientation=0.0,
    )


def _robot(robot_id: int, x: float, y: float = 3.0) -> RobotState:
    return RobotState(robot_id, np.asarray([x, y], dtype=float), 0.0)


def test_single_robot_ground_reaction_balances_force_and_moment() -> None:
    config = RobotConfig(count=1, length=1.0, width=0.4)
    robots = [_robot(0, 3.0)]
    graph = ContactGraph(1)
    graph.add_edge(MechanicalEdge(0, LEFT_BANK, "anchor", 1.0, 1000.0))
    solver = PlanarStaticSolver(robots, graph, _field(), config)

    point = (3.4, 3.2)
    result = solver.solve([PlanarLoad(0, (10.0, -20.0), point=point, moment=3.0)])

    assert result.stable
    assert len(result.contacts) == 1
    reaction = result.contacts[0].wrench_on_source
    np.testing.assert_allclose([reaction.fx, reaction.fy], [-10.0, 20.0], atol=1.0e-8)

    r = np.asarray(point) - robots[0].position
    applied_torque = r[0] * -20.0 - r[1] * 10.0 + 3.0
    contact_point = np.asarray(result.contacts[0].point)
    rc = contact_point - robots[0].position
    reaction_torque = rc[0] * reaction.fy - rc[1] * reaction.fx + reaction.mz
    np.testing.assert_allclose(reaction_torque + applied_torque, 0.0, atol=1.0e-8)


def test_two_supports_share_symmetric_center_load() -> None:
    config = RobotConfig(count=1, length=1.0, width=0.4)
    robots = [_robot(0, 5.0)]
    graph = ContactGraph(1)
    graph.add_edge(MechanicalEdge(0, LEFT_BANK, "anchor", 1.0, 1000.0))
    graph.add_edge(MechanicalEdge(0, RIGHT_BANK, "anchor", 1.0, 1000.0))
    solver = PlanarStaticSolver(robots, graph, _field(), config)

    result = solver.solve([PlanarLoad(0, (0.0, -100.0))])
    assert result.stable
    fy = sorted(contact.wrench_on_source.fy for contact in result.contacts)
    np.testing.assert_allclose(fy, [50.0, 50.0], atol=1.0e-8)


def test_capacity_is_computed_directly_from_unit_payload_response() -> None:
    config = RobotConfig(count=1, length=1.0, width=0.4)
    robots = [_robot(0, 3.0)]
    graph = ContactGraph(1)
    graph.add_edge(MechanicalEdge(0, LEFT_BANK, "anchor", 1.0, 1000.0))
    solver = PlanarStaticSolver(robots, graph, _field(), config)

    # 1 kg produces 10 N. A 35 N |Fy| limit therefore fails at 3.5 kg.
    result = solver.capacity(
        PlanarLoad(0, (0.0, -10.0)),
        PlanarWrenchLimit(fx=np.inf, fy=35.0, mz=np.inf),
    )
    assert result.failure_mode == "contact_limit"
    assert result.critical_component == "Fy"
    np.testing.assert_allclose(result.max_payload_kg, 3.5, atol=1.0e-10)
    np.testing.assert_allclose(abs(result.critical_value or 0.0), 35.0, atol=1.0e-10)


def test_unsupported_component_is_reported_as_mechanism() -> None:
    config = RobotConfig(count=2, length=1.0, width=0.4)
    robots = [_robot(0, 4.0), _robot(1, 4.9)]
    graph = ContactGraph(2)
    graph.add_edge(MechanicalEdge(0, 1, "contact", 1.0, 1000.0))
    solver = PlanarStaticSolver(robots, graph, _field(), config)

    result = solver.solve([PlanarLoad(0, (0.0, -10.0))])
    assert not result.stable
    assert result.failure_mode == "unsupported_component"
