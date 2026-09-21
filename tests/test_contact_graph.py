from __future__ import annotations

import numpy as np

from bari2d.env.contact_model import ContactGraph, MechanicalEdge
from bari2d.env.field import LEFT_BANK, RIGHT_BANK
from bari2d.env.robot import RobotState


def test_graph_reports_path_only_when_banks_connected() -> None:
    graph = ContactGraph(3)
    graph.add_edge(MechanicalEdge(LEFT_BANK, 0, "bank_contact", 5.0, 20.0))
    graph.add_edge(MechanicalEdge(0, 1, "contact", 5.0, 20.0))
    graph.add_edge(MechanicalEdge(1, 2, "anchor", 8.0, 20.0))
    assert not graph.spans
    graph.add_edge(MechanicalEdge(2, RIGHT_BANK, "bank_contact", 5.0, 20.0))
    assert graph.spans
    assert graph.connected_component(LEFT_BANK) == {LEFT_BANK, 0, 1, 2, RIGHT_BANK}


def test_anchoring_and_release_modify_mechanical_graph(env) -> None:
    robot = env.robots[0]
    assert env.contact_model.anchor(robot, env.robots, env.field)
    graph = env.contact_model.build_graph(env.robots, env.field, env.rng)
    assert robot.anchored
    assert any(edge.kind == "anchor" and edge.source == robot.robot_id for edge in graph.edges.values())
    assert env.contact_model.release(robot)
    graph = env.contact_model.build_graph(env.robots, env.field, env.rng)
    assert not robot.anchored
    assert not any(edge.kind == "anchor" and edge.source == robot.robot_id for edge in graph.edges.values())


def test_anchor_fails_when_tension_limit_exceeded(env) -> None:
    first, second = env.robots[:2]
    first.position = env.field.center.copy()
    second.position = first.position + np.array([0.7, 0.0])
    assert env.contact_model.anchor(first, env.robots, env.field)
    assert isinstance(first.anchor_partner, int)
    env.robots[first.anchor_partner].position += np.array([2.0, 0.0])
    env.contact_model.build_graph(env.robots, env.field, env.rng)
    assert not first.anchored
    assert env.contact_model.anchor_failures == 1
