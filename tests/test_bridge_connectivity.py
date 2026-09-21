from __future__ import annotations

from bari2d.utils.scripted import assemble_parallel_bridge


def test_manual_bridge_connects_banks_and_reaches_full_progress(env) -> None:
    assemble_parallel_bridge(env, rows=1, anchor=False)
    assert env.graph.spans
    assert env.current_progress == 1.0


def test_disconnected_structure_does_not_span(env) -> None:
    assemble_parallel_bridge(env, rows=1, anchor=False)
    middle = len([robot for robot in env.robots if env.field.is_gap(robot.position)]) // 2
    env.robots[middle].position += env.field.tangent * 2.0
    env.set_robot_states(env.robots)
    assert not env.graph.spans

