from __future__ import annotations

from bari2d.env.load_evaluator import FastLoadEvaluator, IncrementalLoadEvaluator
from bari2d.utils.scripted import assemble_parallel_bridge


def test_spanning_bridge_has_meaningful_capacity(env) -> None:
    assemble_parallel_bridge(env, rows=1)
    fast = FastLoadEvaluator().evaluate(env.graph, env.config.load)
    accurate = IncrementalLoadEvaluator(env.config.load).evaluate(env.graph)
    assert fast.capacity > 0.0
    assert accurate.capacity > 0.0
    assert accurate.load_bearing_path


def test_parallel_structure_is_stronger(env) -> None:
    assemble_parallel_bridge(env, rows=1)
    weak = IncrementalLoadEvaluator(env.config.load).evaluate(env.graph).capacity
    env.reset(seed=13)
    assemble_parallel_bridge(env, rows=2)
    strong = IncrementalLoadEvaluator(env.config.load).evaluate(env.graph).capacity
    assert strong > weak


def test_disconnected_structure_has_zero_capacity(env) -> None:
    result = IncrementalLoadEvaluator(env.config.load).evaluate(env.graph)
    assert result.capacity == 0.0
    assert result.failure_mode == "loss_of_connectivity"

