from __future__ import annotations

from dataclasses import dataclass

from bari2d.env.contact_model import ContactGraph, MechanicalEdge, Node
from bari2d.env.field import LEFT_BANK, RIGHT_BANK
from bari2d.utils.config import LoadConfig


@dataclass
class LoadTestResult:
    capacity: float
    failure_mode: str
    failed_edges: list[tuple[Node, Node]]
    load_bearing_path: list[Node]
    estimated_displacement: float
    load_protocol: str = "uniform"


@dataclass
class StructuralResponse:
    capacity: float
    stiffness: float
    paths: list[list[Node]]


class LoadProtocol:
    """Strategy seam for mapping an external loading pattern onto graph demand."""

    name = "center_point"

    def response(self, graph: ContactGraph) -> StructuralResponse:
        responses = _spanning_component_responses(graph)
        if not responses:
            return StructuralResponse(0.0, 0.0, [])
        selected = min(
            responses,
            key=lambda item: abs(graph.node_tangent.get(item[0], 0.0)),
        )[1]
        return selected


class UniformLoadProtocol(LoadProtocol):
    name = "uniform"

    def response(self, graph: ContactGraph) -> StructuralResponse:
        responses = [response for _, response in _spanning_component_responses(graph)]
        return StructuralResponse(
            sum(response.capacity for response in responses),
            sum(response.stiffness for response in responses),
            [path for response in responses for path in response.paths],
        )


LOAD_PROTOCOLS: dict[str, type[LoadProtocol]] = {
    "center_point": LoadProtocol,
    "uniform": UniformLoadProtocol,
}


def load_protocol(name: str) -> LoadProtocol:
    try:
        return LOAD_PROTOCOLS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown load protocol: {name}") from exc


def _residual_network(graph: ContactGraph, blocked: set[Node] | None = None) -> dict[Node, dict[Node, float]]:
    blocked = blocked or set()
    residual: dict[Node, dict[Node, float]] = {node: {} for node in graph.nodes if node not in blocked}
    for edge in graph.edges.values():
        if edge.source in blocked or edge.target in blocked:
            continue
        residual[edge.source][edge.target] = residual[edge.source].get(edge.target, 0.0) + edge.capacity
        residual[edge.target][edge.source] = residual[edge.target].get(edge.source, 0.0) + edge.capacity
    return residual


def maximum_flow(
    graph: ContactGraph,
    source: Node = LEFT_BANK,
    sink: Node = RIGHT_BANK,
    blocked: set[Node] | None = None,
) -> tuple[float, list[Node]]:
    blocked = blocked or set()
    if source not in graph.nodes or sink not in graph.nodes or source in blocked or sink in blocked:
        return 0.0, []
    residual = _residual_network(graph, blocked)
    total = 0.0
    representative_path: list[Node] = []
    while True:
        parent: dict[Node, Node | None] = {source: None}
        queue = [source]
        while queue and sink not in parent:
            current = queue.pop(0)
            for neighbor, capacity in residual[current].items():
                if capacity > 1.0e-9 and neighbor not in parent:
                    parent[neighbor] = current
                    queue.append(neighbor)
        if sink not in parent:
            break
        path = [sink]
        while path[-1] != source:
            path.append(parent[path[-1]])  # type: ignore[arg-type]
        path.reverse()
        amount = min(residual[path[index]][path[index + 1]] for index in range(len(path) - 1))
        total += amount
        for index in range(len(path) - 1):
            first, second = path[index], path[index + 1]
            residual[first][second] -= amount
            residual[second][first] = residual[second].get(first, 0.0) + amount
        if not representative_path:
            representative_path = path
    return total, representative_path


def _path_stiffness(graph: ContactGraph, path: list[Node]) -> float:
    if len(path) < 2:
        return 0.0
    compliance = 0.0
    for first, second in zip(path, path[1:]):
        edge = graph.edges.get(frozenset((first, second)))
        if edge is None:
            return 0.0
        compliance += 1.0 / max(edge.stiffness, 1.0e-9)
    return 1.0 / max(compliance, 1.0e-9)


def _robot_components(graph: ContactGraph) -> list[set[int]]:
    remaining = {node for node in graph.nodes if isinstance(node, int)}
    components: list[set[int]] = []
    while remaining:
        start = remaining.pop()
        component = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for neighbor in graph.neighbors(current):
                if isinstance(neighbor, int) and neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        components.append(component)
    return components


def _component_response(graph: ContactGraph, component: set[int]) -> tuple[int, StructuralResponse] | None:
    left_supported = any(LEFT_BANK in graph.neighbors(node) for node in component)
    right_supported = any(RIGHT_BANK in graph.neighbors(node) for node in component)
    if not left_supported or not right_supported:
        return None
    load_node = min(component, key=lambda node: abs(graph.node_progress.get(node, 0.5) - 0.5))
    left_capacity, left_path = maximum_flow(graph, load_node, LEFT_BANK, {RIGHT_BANK})
    right_capacity, right_path = maximum_flow(graph, load_node, RIGHT_BANK, {LEFT_BANK})
    capacity = 2.0 * min(left_capacity, right_capacity)
    combined_path = list(reversed(left_path)) + right_path[1:]
    stiffness = _path_stiffness(graph, combined_path)
    return load_node, StructuralResponse(capacity, stiffness, [combined_path])


def _spanning_component_responses(graph: ContactGraph) -> list[tuple[int, StructuralResponse]]:
    responses = [_component_response(graph, component) for component in _robot_components(graph)]
    return [response for response in responses if response is not None]


class FastLoadEvaluator:
    """Replaceable graph-flow proxy used inside training steps."""

    def evaluate(self, graph: ContactGraph, config: LoadConfig | None = None) -> LoadTestResult:
        protocol = load_protocol(config.load_protocol if config else "uniform")
        response = protocol.response(graph)
        path = response.paths[0] if response.paths else []
        stiffness = response.stiffness
        displacement_limit = config.displacement_limit if config else 2.0
        displacement_capacity = stiffness * displacement_limit if stiffness > 0.0 else 0.0
        capacity = min(response.capacity, displacement_capacity) if response.capacity > 0.0 else 0.0
        displacement = capacity / stiffness if stiffness > 0.0 else float("inf")
        mode = "none" if graph.spans else "loss_of_connectivity"
        return LoadTestResult(capacity, mode, [], path, displacement, protocol.name)


class IncrementalLoadEvaluator:
    """Quasi-static terminal load test with edge failure and displacement checks."""

    def __init__(self, config: LoadConfig):
        self.config = config
        self.protocol = load_protocol(config.load_protocol)

    def evaluate(self, graph: ContactGraph, maximum_load: float | None = None) -> LoadTestResult:
        if not graph.spans:
            return LoadTestResult(0.0, "loss_of_connectivity", [], [], float("inf"), self.protocol.name)
        response = self.protocol.response(graph)
        theoretical = response.capacity
        path = response.paths[0] if response.paths else []
        stiffness = response.stiffness
        if stiffness <= 0.0:
            return LoadTestResult(0.0, "structural_collapse", [], path, float("inf"), self.protocol.name)
        upper = theoretical + self.config.load_step if maximum_load is None else min(theoretical + self.config.load_step, maximum_load)
        sustainable = 0.0
        failure_mode = "edge_capacity"
        failed_edges: list[tuple[Node, Node]] = []
        applied = self.config.load_step
        while applied <= upper + 1.0e-9:
            displacement = applied / stiffness
            if displacement > self.config.displacement_limit:
                failure_mode = "structural_collapse"
                break
            if applied > theoretical + 1.0e-9:
                failed_edges = self._bottlenecks(graph, path)
                failure_mode = self._classify_failure(graph, failed_edges)
                break
            sustainable = applied
            applied += self.config.load_step
        capacity = min(sustainable, theoretical, stiffness * self.config.displacement_limit)
        if maximum_load is not None and capacity >= maximum_load:
            failure_mode = "none"
        return LoadTestResult(
            float(capacity), failure_mode, failed_edges, path, float(capacity / stiffness), self.protocol.name
        )

    @staticmethod
    def _bottlenecks(graph: ContactGraph, path: list[Node]) -> list[tuple[Node, Node]]:
        path_edges: list[MechanicalEdge] = []
        for first, second in zip(path, path[1:]):
            edge = graph.edges.get(frozenset((first, second)))
            if edge is not None:
                path_edges.append(edge)
        if not path_edges:
            return []
        minimum = min(edge.capacity for edge in path_edges)
        return [(edge.source, edge.target) for edge in path_edges if abs(edge.capacity - minimum) < 1.0e-9]

    @staticmethod
    def _classify_failure(graph: ContactGraph, failed_edges: list[tuple[Node, Node]]) -> str:
        kinds = {
            graph.edges[frozenset((source, target))].kind
            for source, target in failed_edges
            if frozenset((source, target)) in graph.edges
        }
        if "bank_contact" in kinds:
            return "robot_slip"
        if "anchor" in kinds:
            return "anchor_failure"
        if "contact" in kinds:
            return "contact_loss"
        return "structural_collapse"
