from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from bari2d.bridge_generation.random_bridge import GeneratedBridge
from bari2d.capacity_learning.model import CapacityGNN
from bari2d.env.contact_model import ContactGraph, MechanicalEdge, Node
from bari2d.env.field import LEFT_BANK, RIGHT_BANK
from bari2d.mechanics import (
    LocalWrenchLimit,
    SpatialStaticSolver,
    bridge_center_payload_3d,
    contact_frame_for_edge_3d,
    contact_point_for_edge_3d,
    robot_center_3d,
)
from bari2d.utils.config import CapacityTeacherConfig, EnvironmentConfig


_COMPONENT_NAMES = ("Axial", "ShearU", "ShearV", "Torsion", "BendingU", "BendingV")


@dataclass(frozen=True)
class CapacityTeacherEvaluation:
    capacity_kg: float
    terminal_reward: float
    agent_credits: np.ndarray
    critical_component: str | None = None
    backend: str = "exact"


def _robot_component(graph: ContactGraph, start: int) -> set[int]:
    visited = {start}
    frontier = [start]
    while frontier:
        node = frontier.pop()
        for neighbor in graph.neighbors(node):
            if isinstance(neighbor, int) and neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return visited


def _fixed_supports(graph: ContactGraph, robot_ids: Iterable[int]) -> set[Node]:
    supports: set[Node] = set()
    for robot_id in robot_ids:
        for neighbor in graph.neighbors(robot_id):
            if not isinstance(neighbor, int):
                supports.add(neighbor)
    return supports


def _spanning_components(graph: ContactGraph, robot_count: int) -> list[set[int]]:
    remaining = set(range(robot_count))
    result: list[set[int]] = []
    while remaining:
        start = next(iter(remaining))
        component = _robot_component(graph, start)
        remaining.difference_update(component)
        supports = _fixed_supports(graph, component)
        if LEFT_BANK in supports and RIGHT_BANK in supports:
            result.append(component)
    return result


def _graph_without_robot(graph: ContactGraph, robot_id: int) -> ContactGraph:
    reduced = ContactGraph(len([node for node in graph.nodes if isinstance(node, int)]))
    reduced.node_progress = dict(graph.node_progress)
    reduced.node_tangent = dict(graph.node_tangent)
    for edge in graph.edges.values():
        if edge.source == robot_id or edge.target == robot_id:
            continue
        reduced.add_edge(
            MechanicalEdge(
                edge.source,
                edge.target,
                edge.kind,
                edge.capacity,
                edge.stiffness,
                edge.force,
                edge.relative_pose,
            )
        )
    return reduced


class CapacityTeacher:
    """Training-only bridge-capacity oracle for decentralized policy learning.

    The actor never receives this object or any global structural information.
    During CTDE training the teacher converts the final bridge into a global
    terminal reward and optional leave-one-robot-out difference rewards.

    `backend="exact"` uses the 6-DOF statics solver and is the ground-truth
    implementation. `backend="gnn"` loads a geometry-only CapacityGNN checkpoint
    and is intended for faster policy training once a trustworthy surrogate has
    been trained. Exact evaluation should still be used for validation.
    """

    def __init__(
        self,
        config: CapacityTeacherConfig,
        environment_config: EnvironmentConfig,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.environment_config = environment_config
        self.device = torch.device(device)
        self.limit = LocalWrenchLimit(
            axial=float(config.axial_limit),
            shear_u=float(config.shear_u_limit),
            shear_v=float(config.shear_v_limit),
            torsion=float(config.torsion_limit),
            bending_u=float(config.bending_u_limit),
            bending_v=float(config.bending_v_limit),
        )
        self.model: CapacityGNN | None = None
        if config.backend == "gnn":
            self.model = self._load_gnn(config.checkpoint)
        elif config.backend != "exact":
            raise ValueError(f"Unknown capacity teacher backend: {config.backend!r}")

    def _load_gnn(self, path: str | None) -> CapacityGNN:
        if not path:
            raise ValueError("capacity teacher backend='gnn' requires a checkpoint path")
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Capacity GNN checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if checkpoint.get("feature_mode") != "geometry_only":
            raise ValueError("Policy teacher requires a geometry_only CapacityGNN checkpoint")
        if int(checkpoint.get("schema_version", 0)) != 2 or checkpoint.get("wrench_frame") != "contact_local":
            raise ValueError("Capacity GNN checkpoint must use schema-v2 contact-local features")
        model = CapacityGNN(
            node_size=int(checkpoint["node_size"]),
            edge_size=int(checkpoint["edge_size"]),
            hidden_size=int(checkpoint.get("hidden_size", 128)),
            layers=int(checkpoint.get("layers", 4)),
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _score(self, capacity_kg: float) -> float:
        reference = max(float(self.config.capacity_reference_kg), 1.0e-9)
        capped = min(max(float(capacity_kg), 0.0), float(self.config.capacity_cap_kg))
        return log1p(capped / reference)

    def _selected_component(self, env, graph: ContactGraph) -> tuple[set[int], int] | None:
        components = _spanning_components(graph, len(env.robots))
        if not components:
            return None
        target = np.asarray(env.field.center, dtype=float)
        candidates = [
            robot_id
            for component in components
            for robot_id in component
            if not env.robots[robot_id].fallen
        ]
        if not candidates:
            return None
        payload_robot = min(
            candidates,
            key=lambda robot_id: float(np.linalg.norm(env.robots[robot_id].position - target)),
        )
        component = next(component for component in components if payload_robot in component)
        return component, payload_robot

    def _exact_capacity(self, env, graph: ContactGraph) -> tuple[float, str | None]:
        selected = self._selected_component(env, graph)
        if selected is None:
            return 0.0, None
        try:
            payload = bridge_center_payload_3d(env.robots, graph, env.field, env.config.robot)
            solver = SpatialStaticSolver(
                env.robots,
                graph,
                env.field,
                env.config.robot,
                verify_rank=bool(self.config.exact_verify_rank),
            )
            result = solver.capacity_local(payload, self.limit, include_self_weight=True)
        except (ValueError, np.linalg.LinAlgError):
            return 0.0, None
        capacity = float(result.max_payload_kg)
        if not np.isfinite(capacity):
            capacity = float(self.config.capacity_cap_kg)
        return max(capacity, 0.0), result.critical_component

    @staticmethod
    def _world_to_field_vector(vector: np.ndarray, field) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        normal3 = np.asarray([field.normal[0], field.normal[1], 0.0])
        tangent3 = np.asarray([field.tangent[0], field.tangent[1], 0.0])
        return np.asarray([vector @ normal3, vector @ tangent3, vector[2]], dtype=float)

    def _virtual_node_position(self, env, node: Node) -> np.ndarray:
        if isinstance(node, int):
            return robot_center_3d(env.robots[node], env.config.robot)
        longitudinal = -env.field.gap_width / 2.0 if node == LEFT_BANK else env.field.gap_width / 2.0
        xy = env.field.center + env.field.normal * longitudinal
        return np.asarray([xy[0], xy[1], 0.0], dtype=float)

    def _geometry_graph(self, env, graph: ContactGraph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        selected = self._selected_component(env, graph)
        if selected is None:
            return None
        active, payload_robot = selected
        supports = _fixed_supports(graph, active)
        node_order: list[Node] = sorted(active)
        if LEFT_BANK in supports:
            node_order.append(LEFT_BANK)
        if RIGHT_BANK in supports:
            node_order.append(RIGHT_BANK)
        node_to_index = {node: index for index, node in enumerate(node_order)}

        gap_scale = max(float(env.field.gap_width), 1.0e-6)
        height = max(float(env.config.robot.climb_height), 1.0e-6)
        max_layer = max(max((env.robots[node].layer for node in active), default=0), 1)
        node_features: list[list[float]] = []
        for node in node_order:
            if isinstance(node, int):
                robot = env.robots[node]
                longitudinal, tangent = env.field.coordinates(robot.position)
                z = robot_center_3d(robot, env.config.robot)[2]
                yaw = robot.theta - env.field.orientation
                node_features.append(
                    [
                        1.0,
                        0.0,
                        0.0,
                        longitudinal / gap_scale,
                        tangent / gap_scale,
                        z / height,
                        float(np.cos(yaw)),
                        float(np.sin(yaw)),
                        float(robot.layer) / max_layer,
                        1.0 if node == payload_robot else 0.0,
                        float(env.config.robot.mass),
                    ]
                )
            else:
                node_features.append(
                    [
                        0.0,
                        1.0 if node == LEFT_BANK else 0.0,
                        1.0 if node == RIGHT_BANK else 0.0,
                        -0.5 if node == LEFT_BANK else 0.5,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                )

        force_scale = 300.0
        moment_scale = 60.0
        length_scale = max(float(env.config.robot.length), 1.0e-6)
        height_scale = max(float(env.config.robot.climb_height), 1.0e-6)
        limit_array = self.limit.as_array()
        scaled_limits = np.concatenate((limit_array[:3] / force_scale, limit_array[3:] / moment_scale))
        edge_indices: list[tuple[int, int]] = []
        edge_features: list[list[float]] = []
        for edge in graph.edges.values():
            robot_nodes = [node for node in (edge.source, edge.target) if isinstance(node, int)]
            if not robot_nodes or any(node not in active for node in robot_nodes):
                continue
            if edge.source not in node_to_index or edge.target not in node_to_index:
                continue
            contact_point = contact_point_for_edge_3d(edge, env.robots, env.field, env.config.robot)
            frame = contact_frame_for_edge_3d(edge, env.robots, env.field, env.config.robot)
            source_pos = self._virtual_node_position(env, edge.source)
            target_pos = self._virtual_node_position(env, edge.target)
            relative = self._world_to_field_vector(target_pos - source_pos, env.field)
            source_offset = self._world_to_field_vector(contact_point - source_pos, env.field)
            target_offset = self._world_to_field_vector(contact_point - target_pos, env.field)
            if isinstance(edge.source, int) and isinstance(edge.target, int):
                relative_yaw = env.robots[edge.target].theta - env.robots[edge.source].theta
                cos_yaw, sin_yaw = float(np.cos(relative_yaw)), float(np.sin(relative_yaw))
            else:
                cos_yaw, sin_yaw = 1.0, 0.0
            kind = [
                1.0 if edge.kind == "contact" else 0.0,
                1.0 if edge.kind == "anchor" else 0.0,
                1.0 if edge.kind == "bank_contact" else 0.0,
            ]
            normal = self._world_to_field_vector(np.asarray(frame.normal), env.field)
            tangent_u = self._world_to_field_vector(np.asarray(frame.tangent_u), env.field)
            tangent_v = self._world_to_field_vector(np.asarray(frame.tangent_v), env.field)
            geometry = [
                *kind,
                relative[0] / length_scale,
                relative[1] / length_scale,
                relative[2] / height_scale,
                source_offset[0] / length_scale,
                source_offset[1] / length_scale,
                source_offset[2] / height_scale,
                target_offset[0] / length_scale,
                target_offset[1] / length_scale,
                target_offset[2] / height_scale,
                cos_yaw,
                sin_yaw,
                float(np.log10(max(float(edge.stiffness), 1.0e-12))),
                *(float(value) for value in normal),
                *(float(value) for value in tangent_u),
                *(float(value) for value in tangent_v),
                *(float(value) for value in scaled_limits),
            ]
            for direction, (source, target) in enumerate(((edge.source, edge.target), (edge.target, edge.source))):
                row = list(geometry)
                if direction == 1:
                    row[3:6] = [-row[3], -row[4], -row[5]]
                    original_source = row[6:9]
                    original_target = row[9:12]
                    row[6:9] = [-value for value in original_target]
                    row[9:12] = [-value for value in original_source]
                    row[13] *= -1.0
                edge_indices.append((node_to_index[source], node_to_index[target]))
                edge_features.append(row)
        if not edge_indices:
            return None
        nodes = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(edge_indices, dtype=torch.long, device=self.device).T.contiguous()
        edges = torch.tensor(edge_features, dtype=torch.float32, device=self.device)
        node_graph = torch.zeros(nodes.shape[0], dtype=torch.long, device=self.device)
        return nodes, edge_index, edges, node_graph

    def _gnn_capacity(self, env, graph: ContactGraph) -> tuple[float, str | None]:
        assert self.model is not None
        encoded = self._geometry_graph(env, graph)
        if encoded is None:
            return 0.0, None
        nodes, edge_index, edges, node_graph = encoded
        with torch.no_grad():
            output = self.model(nodes, edge_index, edges, node_graph)
        capacity = float(output.capacity_kg.reshape(-1)[0].item())
        component_index = int(output.critical_component_logits.reshape(-1, 6)[0].argmax().item())
        return max(capacity, 0.0), _COMPONENT_NAMES[component_index]

    def capacity(self, env, graph: ContactGraph | None = None) -> tuple[float, str | None]:
        graph = env.graph if graph is None else graph
        if self.config.backend == "gnn":
            return self._gnn_capacity(env, graph)
        return self._exact_capacity(env, graph)

    def evaluate(self, env) -> CapacityTeacherEvaluation:
        robot_count = len(env.robots)
        credits = np.zeros(robot_count, dtype=np.float32)
        capacity, critical_component = self.capacity(env)
        if capacity <= 0.0:
            return CapacityTeacherEvaluation(
                capacity_kg=0.0,
                terminal_reward=float(self.config.no_span_penalty),
                agent_credits=credits,
                critical_component=None,
                backend=self.config.backend,
            )

        base_score = self._score(capacity)
        terminal_reward = float(self.config.terminal_reward_scale) * base_score

        if self.config.counterfactual_credit:
            selected = self._selected_component(env, env.graph)
            active = set() if selected is None else selected[0]
            for robot_id in active:
                reduced = _graph_without_robot(env.graph, robot_id)
                counterfactual_capacity, _ = self.capacity(env, reduced)
                difference = base_score - self._score(counterfactual_capacity)
                difference *= float(self.config.counterfactual_credit_scale)
                clip = float(self.config.counterfactual_credit_clip)
                credits[robot_id] = float(np.clip(difference, -clip, clip))

        return CapacityTeacherEvaluation(
            capacity_kg=capacity,
            terminal_reward=terminal_reward,
            agent_credits=credits,
            critical_component=critical_component,
            backend=self.config.backend,
        )
