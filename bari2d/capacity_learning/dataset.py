from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from bari2d.bridge_generation import GeneratedBridge, RandomBridgeConfig, RandomBridgeGenerator
from bari2d.env.contact_model import Node
from bari2d.env.field import LEFT_BANK, RIGHT_BANK
from bari2d.mechanics.spatial_statics import (
    EdgeKey,
    LocalWrenchLimit,
    SpatialCapacityResult,
    SpatialLoad,
    SpatialStaticSolver,
    robot_center_3d,
    robot_face_centers_3d,
)
from bari2d.utils.config import ContactConfig, RobotConfig


FeatureMode = Literal["mechanics_informed", "geometry_only"]
PayloadMode = Literal["center", "random"]


@dataclass(frozen=True)
class WrenchLimitSampler:
    """Sample connector-local strength limits for every connection.

    ``axial``/``shear`` are forces in N; ``torsion``/``bending`` are moments in
    N m. Limits are sampled log-uniformly so a training set spans weak and
    strong connectors without being dominated by the top of the numeric range.
    """

    axial_range: tuple[float, float] = (80.0, 250.0)
    shear_range: tuple[float, float] = (80.0, 250.0)
    torsion_range: tuple[float, float] = (50.0, 500.0)
    bending_range: tuple[float, float] = (50.0, 500.0)
    edge_scale_range: tuple[float, float] = (0.80, 1.25)

    @staticmethod
    def _log_uniform(rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
        if low <= 0.0 or high < low:
            raise ValueError("Wrench limit ranges must be positive and ordered")
        return np.exp(rng.uniform(np.log(low), np.log(high), size=size))

    def sample(
        self,
        bridge: GeneratedBridge,
        rng: np.random.Generator,
    ) -> dict[EdgeKey, LocalWrenchLimit]:
        result: dict[EdgeKey, LocalWrenchLimit] = {}
        for edge in bridge.graph.edges.values():
            axial = self._log_uniform(rng, *self.axial_range, size=1)
            shear = self._log_uniform(rng, *self.shear_range, size=2)
            torsion = self._log_uniform(rng, *self.torsion_range, size=1)
            bending = self._log_uniform(rng, *self.bending_range, size=2)
            scale = float(rng.uniform(*self.edge_scale_range))
            values = np.concatenate((axial, shear, torsion, bending)) * scale
            result[edge.key] = LocalWrenchLimit(*(float(value) for value in values))
        return result


@dataclass
class CapacityGraphSample:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    target_capacity_kg: torch.Tensor
    critical_component: torch.Tensor
    critical_edge_mask: torch.Tensor
    edge_capacity_kg: torch.Tensor
    metadata: dict[str, object]

    @property
    def node_count(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.edge_features.shape[0])

    def to_dict(self) -> dict[str, object]:
        return {
            "node_features": self.node_features.cpu(),
            "edge_index": self.edge_index.cpu(),
            "edge_features": self.edge_features.cpu(),
            "target_capacity_kg": self.target_capacity_kg.cpu(),
            "critical_component": self.critical_component.cpu(),
            "critical_edge_mask": self.critical_edge_mask.cpu(),
            "edge_capacity_kg": self.edge_capacity_kg.cpu(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "CapacityGraphSample":
        return cls(
            node_features=torch.as_tensor(values["node_features"], dtype=torch.float32),
            edge_index=torch.as_tensor(values["edge_index"], dtype=torch.long),
            edge_features=torch.as_tensor(values["edge_features"], dtype=torch.float32),
            target_capacity_kg=torch.as_tensor(values["target_capacity_kg"], dtype=torch.float32),
            critical_component=torch.as_tensor(values["critical_component"], dtype=torch.long),
            critical_edge_mask=torch.as_tensor(values["critical_edge_mask"], dtype=torch.float32),
            edge_capacity_kg=torch.as_tensor(values["edge_capacity_kg"], dtype=torch.float32),
            metadata=dict(values.get("metadata", {})),
        )


_COMPONENT_TO_INDEX = {
    name: index
    for index, name in enumerate(("Axial", "ShearU", "ShearV", "Torsion", "BendingU", "BendingV"))
}


class CapacityDatasetBuilder:
    """Convert procedural bridges into supervised graph samples.

    `mechanics_informed` edge features include the base/self-weight wrench and
    the 1 kg payload wrench sensitivity. `geometry_only` omits those 12 values,
    allowing experiments that ask the GNN to infer mechanics from topology and
    geometry rather than receiving the solved contact response directly.
    """

    NODE_FEATURE_NAMES = (
        "is_robot",
        "is_left_bank",
        "is_right_bank",
        "longitudinal_over_gap",
        "tangent_over_gap",
        "z_over_robot_height",
        "cos_relative_yaw",
        "sin_relative_yaw",
        "layer_normalized",
        "is_payload_robot",
        "robot_mass_kg",
    )

    BASE_EDGE_FEATURE_NAMES = (
        "is_contact",
        "is_anchor",
        "is_bank_contact",
        "relative_longitudinal",
        "relative_tangent",
        "relative_z",
        "contact_offset_source_x",
        "contact_offset_source_y",
        "contact_offset_source_z",
        "contact_offset_target_x",
        "contact_offset_target_y",
        "contact_offset_target_z",
        "cos_relative_yaw",
        "sin_relative_yaw",
        "log10_scalar_stiffness",
        "contact_normal_longitudinal",
        "contact_normal_tangent",
        "contact_normal_z",
        "contact_tangent_u_longitudinal",
        "contact_tangent_u_tangent",
        "contact_tangent_u_z",
        "contact_tangent_v_longitudinal",
        "contact_tangent_v_tangent",
        "contact_tangent_v_z",
        "limit_axial_scaled",
        "limit_shear_u_scaled",
        "limit_shear_v_scaled",
        "limit_torsion_scaled",
        "limit_bending_u_scaled",
        "limit_bending_v_scaled",
    )

    MECHANICS_EDGE_FEATURE_NAMES = (
        "base_axial_over_limit",
        "base_shear_u_over_limit",
        "base_shear_v_over_limit",
        "base_torsion_over_limit",
        "base_bending_u_over_limit",
        "base_bending_v_over_limit",
        "unit_axial_over_limit",
        "unit_shear_u_over_limit",
        "unit_shear_v_over_limit",
        "unit_torsion_over_limit",
        "unit_bending_u_over_limit",
        "unit_bending_v_over_limit",
    )

    def __init__(
        self,
        generator: RandomBridgeGenerator | None = None,
        *,
        robot_config: RobotConfig | None = None,
        contact_config: ContactConfig | None = None,
        bridge_config: RandomBridgeConfig | None = None,
        limit_sampler: WrenchLimitSampler | None = None,
        feature_mode: FeatureMode = "mechanics_informed",
        payload_mode: PayloadMode = "random",
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        seed: int | None = None,
        min_capacity_kg: float = 0.01,
        max_capacity_kg: float = 1.0e5,
        edge_capacity_clip_kg: float = 100.0,
    ) -> None:
        if feature_mode not in ("mechanics_informed", "geometry_only"):
            raise ValueError(f"Unknown feature_mode {feature_mode!r}")
        if payload_mode not in ("center", "random"):
            raise ValueError(f"Unknown payload_mode {payload_mode!r}")
        self.robot_config = robot_config or (generator.robot_config if generator else RobotConfig())
        self.contact_config = contact_config or (generator.contact_config if generator else ContactConfig())
        self.generator = generator or RandomBridgeGenerator(
            self.robot_config,
            self.contact_config,
            bridge_config,
            seed=seed,
        )
        self.limit_sampler = limit_sampler or WrenchLimitSampler()
        self.feature_mode = feature_mode
        self.payload_mode = payload_mode
        self.gravity = tuple(float(value) for value in gravity)
        self.rng = np.random.default_rng(seed)
        self.min_capacity_kg = float(min_capacity_kg)
        self.max_capacity_kg = float(max_capacity_kg)
        self.edge_capacity_clip_kg = float(edge_capacity_clip_kg)
        if self.edge_capacity_clip_kg <= 0.0:
            raise ValueError("edge_capacity_clip_kg must be positive")

    @property
    def node_feature_names(self) -> tuple[str, ...]:
        return self.NODE_FEATURE_NAMES

    @property
    def edge_feature_names(self) -> tuple[str, ...]:
        if self.feature_mode == "mechanics_informed":
            return self.BASE_EDGE_FEATURE_NAMES + self.MECHANICS_EDGE_FEATURE_NAMES
        return self.BASE_EDGE_FEATURE_NAMES

    def _payload(self, bridge: GeneratedBridge) -> SpatialLoad:
        candidates = bridge.payload_candidates
        if not candidates:
            raise ValueError("Generated bridge has no payload candidates")
        if self.payload_mode == "center":
            robot_id = min(
                candidates,
                key=lambda node: float(np.linalg.norm(bridge.robots[node].position - bridge.field.center)),
            )
        else:
            # Avoid making endpoint supports dominate the random-load data when possible.
            interior = [
                node
                for node in candidates
                if 0.12 < bridge.field.normalized_progress(bridge.robots[node].position) < 0.88
            ]
            choices = interior or candidates
            robot_id = int(self.rng.choice(choices))
        robot = bridge.robots[robot_id]
        top_center = robot_face_centers_3d(robot, self.robot_config)[4]
        if self.payload_mode == "random":
            # A real object need not sit exactly over the robot COM. Sampling a
            # point on the top face introduces physically meaningful local Mx/My
            # variation while keeping the applied force equal to 1 kg of gravity.
            local_x = float(self.rng.uniform(-0.35, 0.35) * self.robot_config.length)
            local_y = float(self.rng.uniform(-0.35, 0.35) * self.robot_config.width)
            c, s = float(np.cos(robot.theta)), float(np.sin(robot.theta))
            top_center = top_center + np.asarray(
                [c * local_x - s * local_y, s * local_x + c * local_y, 0.0],
                dtype=float,
            )
        return SpatialLoad(
            robot_id=robot_id,
            force=self.gravity,
            point=tuple(float(value) for value in top_center),
        )

    @staticmethod
    def _world_to_field_vector(vector: np.ndarray, bridge: GeneratedBridge) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        normal3 = np.asarray([bridge.field.normal[0], bridge.field.normal[1], 0.0])
        tangent3 = np.asarray([bridge.field.tangent[0], bridge.field.tangent[1], 0.0])
        return np.asarray([vector @ normal3, vector @ tangent3, vector[2]], dtype=float)

    def _virtual_node_position(self, node: Node, bridge: GeneratedBridge) -> np.ndarray:
        if isinstance(node, int):
            return robot_center_3d(bridge.robots[node], self.robot_config)
        longitudinal = -bridge.field.gap_width / 2.0 if node == LEFT_BANK else bridge.field.gap_width / 2.0
        xy = bridge.field.center + bridge.field.normal * longitudinal
        return np.asarray([xy[0], xy[1], 0.0], dtype=float)

    def _node_features(self, bridge: GeneratedBridge, payload_robot: int) -> tuple[torch.Tensor, dict[Node, int]]:
        node_order: list[Node] = list(range(len(bridge.robots))) + [LEFT_BANK, RIGHT_BANK]
        node_to_index = {node: index for index, node in enumerate(node_order)}
        features: list[list[float]] = []
        gap_scale = max(float(bridge.field.gap_width), 1.0e-6)
        height = max(float(self.robot_config.climb_height), 1.0e-6)
        max_layer = max(max((robot.layer for robot in bridge.robots), default=0), 1)

        for node in node_order:
            if isinstance(node, int):
                robot = bridge.robots[node]
                longitudinal, tangent = bridge.field.coordinates(robot.position)
                z = robot_center_3d(robot, self.robot_config)[2]
                yaw = robot.theta - bridge.field.orientation
                row = [
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
                    float(self.robot_config.mass),
                ]
            else:
                row = [
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
            features.append(row)
        return torch.tensor(features, dtype=torch.float32), node_to_index

    def _edge_features(
        self,
        bridge: GeneratedBridge,
        capacity: SpatialCapacityResult,
        limits: dict[EdgeKey, LocalWrenchLimit],
        node_to_index: dict[Node, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base = capacity.base_response.contact_by_key()
        unit = capacity.unit_payload_response.contact_by_key()
        edge_indices: list[tuple[int, int]] = []
        features: list[list[float]] = []
        critical_mask: list[float] = []
        edge_capacities: list[float] = []
        force_scale = 300.0
        moment_scale = 60.0
        length_scale = max(float(self.robot_config.length), 1.0e-6)
        height_scale = max(float(self.robot_config.climb_height), 1.0e-6)
        critical_key = frozenset(capacity.critical_contact) if capacity.critical_contact is not None else None

        for edge in bridge.graph.edges.values():
            response = base.get(edge.key) or unit.get(edge.key)
            if response is None:
                continue
            contact_point = np.asarray(response.point, dtype=float)
            source_pos = self._virtual_node_position(edge.source, bridge)
            target_pos = self._virtual_node_position(edge.target, bridge)
            relative = self._world_to_field_vector(target_pos - source_pos, bridge)
            source_offset = self._world_to_field_vector(contact_point - source_pos, bridge)
            target_offset = self._world_to_field_vector(contact_point - target_pos, bridge)

            if isinstance(edge.source, int) and isinstance(edge.target, int):
                relative_yaw = bridge.robots[edge.target].theta - bridge.robots[edge.source].theta
                cos_yaw, sin_yaw = float(np.cos(relative_yaw)), float(np.sin(relative_yaw))
            else:
                cos_yaw, sin_yaw = 1.0, 0.0

            kind = [
                1.0 if edge.kind == "contact" else 0.0,
                1.0 if edge.kind == "anchor" else 0.0,
                1.0 if edge.kind == "bank_contact" else 0.0,
            ]
            limit_array = limits[edge.key].as_array()
            scaled_limits = np.concatenate((limit_array[:3] / force_scale, limit_array[3:] / moment_scale))
            base_wrench = base.get(edge.key).local_wrench_on_source.as_array() if edge.key in base else np.zeros(6)
            unit_wrench = unit.get(edge.key).local_wrench_on_source.as_array() if edge.key in unit else np.zeros(6)
            frame = response.frame
            frame_normal = self._world_to_field_vector(np.asarray(frame.normal), bridge)
            frame_tangent_u = self._world_to_field_vector(np.asarray(frame.tangent_u), bridge)
            frame_tangent_v = self._world_to_field_vector(np.asarray(frame.tangent_v), bridge)

            edge_capacity = float("inf")
            for a_value, b_value, maximum in zip(base_wrench, unit_wrench, limit_array):
                if abs(a_value) > maximum + 1.0e-12:
                    candidate = 0.0
                elif abs(b_value) <= 1.0e-12:
                    candidate = float("inf")
                elif b_value > 0.0:
                    candidate = (maximum - a_value) / b_value
                else:
                    candidate = (-maximum - a_value) / b_value
                edge_capacity = min(edge_capacity, max(float(candidate), 0.0))
            edge_capacity_supervision = min(edge_capacity, self.edge_capacity_clip_kg)

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
                *(float(value) for value in frame_normal),
                *(float(value) for value in frame_tangent_u),
                *(float(value) for value in frame_tangent_v),
                *(float(value) for value in scaled_limits),
            ]

            for direction, (source, target) in enumerate(((edge.source, edge.target), (edge.target, edge.source))):
                sign = 1.0 if direction == 0 else -1.0
                row = list(geometry)
                if direction == 1:
                    # Reverse direction: relative and source/target contact offsets swap signs/roles.
                    row[3:6] = [-row[3], -row[4], -row[5]]
                    original_source = row[6:9]
                    original_target = row[9:12]
                    row[6:9] = [-value for value in original_target]
                    row[9:12] = [-value for value in original_source]
                    row[13] *= -1.0
                if self.feature_mode == "mechanics_informed":
                    safe_limit = np.maximum(limit_array, 1.0e-9)
                    row.extend((sign * base_wrench / safe_limit).tolist())
                    row.extend((sign * unit_wrench / safe_limit).tolist())
                edge_indices.append((node_to_index[source], node_to_index[target]))
                features.append(row)
                critical_mask.append(1.0 if critical_key is not None and edge.key == critical_key else 0.0)
                edge_capacities.append(float(edge_capacity_supervision))

        if not edge_indices:
            raise ValueError("Bridge produced no statics-bearing edges")
        return (
            torch.tensor(edge_indices, dtype=torch.long).T.contiguous(),
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(critical_mask, dtype=torch.float32),
            torch.tensor(edge_capacities, dtype=torch.float32),
        )

    def build_sample(self, bridge: GeneratedBridge | None = None) -> CapacityGraphSample:
        bridge = bridge or self.generator.generate()
        limits = self.limit_sampler.sample(bridge, self.rng)
        payload = self._payload(bridge)
        solver = SpatialStaticSolver(
            bridge.robots,
            bridge.graph,
            bridge.field,
            self.robot_config,
            verify_rank=False,
        )
        capacity = solver.capacity_local(payload, limits, include_self_weight=True, gravity=self.gravity)

        if not np.isfinite(capacity.max_payload_kg):
            raise ValueError("Capacity is not finite")
        if not (self.min_capacity_kg <= capacity.max_payload_kg <= self.max_capacity_kg):
            raise ValueError(f"Capacity {capacity.max_payload_kg:.6g} kg outside configured range")
        if capacity.failure_mode != "contact_limit" or capacity.critical_component is None:
            raise ValueError(f"Unexpected capacity failure mode: {capacity.failure_mode}")

        node_features, node_to_index = self._node_features(bridge, payload.robot_id)
        edge_index, edge_features, critical_edge_mask, edge_capacity_kg = self._edge_features(
            bridge, capacity, limits, node_to_index
        )
        component = _COMPONENT_TO_INDEX[capacity.critical_component]
        # Store a small geometric snapshot for reproducible physical top-down
        # renders, including the *true* payload offset and contact centers.
        # Optional metadata: existing schema-v2 datasets remain loadable.
        scale = max(float(bridge.field.gap_width), 1.0e-9)
        payload_xy = self._world_to_field_vector(np.asarray(payload.point) - np.array([
            bridge.field.center[0], bridge.field.center[1], 0.0
        ]), bridge)[:2] / scale
        contact_snapshots = []
        for edge in bridge.graph.edges.values():
            response = capacity.base_response.contact_by_key().get(edge.key)
            if response is None:
                continue
            point = self._world_to_field_vector(np.asarray(response.point) - np.array([
                bridge.field.center[0], bridge.field.center[1], 0.0
            ]), bridge)[:2] / scale
            contact_snapshots.append({
                "source": int(node_to_index[edge.source]),
                "target": int(node_to_index[edge.target]),
                "xy": [float(point[0]), float(point[1])],
                "kind": str(edge.kind),
            })
        view_snapshot = {
            "gap_width_m": float(bridge.field.gap_width),
            "robot_length_m": float(self.robot_config.length),
            "robot_width_m": float(self.robot_config.width),
            "payload_xy": [float(payload_xy[0]), float(payload_xy[1])],
            # Exact integer RobotState.layer, indexed by graph node ID.
            "robot_layers": {
                str(node_to_index[robot.robot_id]): int(robot.layer)
                for robot in bridge.robots
            },
            "contacts": contact_snapshots,
        }
        metadata: dict[str, object] = {
            "topdown_view": view_snapshot,
            **bridge.metadata,
            "schema_version": 2,
            "wrench_frame": "contact_local",
            "robot_count": bridge.robot_count,
            "undirected_edge_count": bridge.edge_count,
            "payload_robot": payload.robot_id,
            "payload_mode": self.payload_mode,
            "feature_mode": self.feature_mode,
            "critical_contact": tuple(capacity.critical_contact or ()),
            "critical_component": capacity.critical_component,
            "residual_norm": capacity.base_response.residual_norm,
        }
        return CapacityGraphSample(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            target_capacity_kg=torch.tensor(float(capacity.max_payload_kg), dtype=torch.float32),
            critical_component=torch.tensor(component, dtype=torch.long),
            critical_edge_mask=critical_edge_mask,
            edge_capacity_kg=edge_capacity_kg,
            metadata=metadata,
        )

    def generate_sample(self, max_attempts: int = 100) -> CapacityGraphSample:
        errors: list[str] = []
        for _ in range(max_attempts):
            try:
                return self.build_sample()
            except (ValueError, np.linalg.LinAlgError) as exc:
                errors.append(str(exc))
        raise RuntimeError(
            "Could not generate a valid capacity sample after "
            f"{max_attempts} attempts; last errors: {errors[-3:]}"
        )

    def generate(self, count: int) -> list[CapacityGraphSample]:
        if count < 0:
            raise ValueError("count must be non-negative")
        return [self.generate_sample() for _ in range(count)]


class InfiniteBridgeDataset(IterableDataset):
    """PyTorch IterableDataset that procedurally generates samples forever."""

    def __init__(
        self,
        *,
        seed: int = 0,
        feature_mode: FeatureMode = "mechanics_informed",
        payload_mode: PayloadMode = "random",
        robot_config: RobotConfig | None = None,
        contact_config: ContactConfig | None = None,
        bridge_config: RandomBridgeConfig | None = None,
        limit_sampler: WrenchLimitSampler | None = None,
    ) -> None:
        super().__init__()
        self.seed = int(seed)
        self.feature_mode = feature_mode
        self.payload_mode = payload_mode
        self.robot_config = robot_config or RobotConfig()
        self.contact_config = contact_config or ContactConfig()
        self.bridge_config = bridge_config or RandomBridgeConfig()
        self.limit_sampler = limit_sampler or WrenchLimitSampler()

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        seed = self.seed + 1_000_003 * worker_id
        generator = RandomBridgeGenerator(
            self.robot_config,
            self.contact_config,
            self.bridge_config,
            seed=seed,
        )
        builder = CapacityDatasetBuilder(
            generator,
            limit_sampler=self.limit_sampler,
            feature_mode=self.feature_mode,
            payload_mode=self.payload_mode,
            seed=seed + 17,
        )
        while True:
            yield builder.generate_sample()


def collate_capacity_graphs(samples: Sequence[CapacityGraphSample]) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    node_features: list[torch.Tensor] = []
    edge_features: list[torch.Tensor] = []
    edge_indices: list[torch.Tensor] = []
    node_graph: list[torch.Tensor] = []
    edge_graph: list[torch.Tensor] = []
    critical_edge_mask: list[torch.Tensor] = []
    edge_capacity_kg: list[torch.Tensor] = []
    node_offset = 0
    for graph_index, sample in enumerate(samples):
        node_features.append(sample.node_features)
        edge_features.append(sample.edge_features)
        edge_indices.append(sample.edge_index + node_offset)
        node_graph.append(torch.full((sample.node_count,), graph_index, dtype=torch.long))
        edge_graph.append(torch.full((sample.edge_count,), graph_index, dtype=torch.long))
        critical_edge_mask.append(sample.critical_edge_mask)
        edge_capacity_kg.append(sample.edge_capacity_kg)
        node_offset += sample.node_count
    return {
        "node_features": torch.cat(node_features, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_features": torch.cat(edge_features, dim=0),
        "node_graph": torch.cat(node_graph, dim=0),
        "edge_graph": torch.cat(edge_graph, dim=0),
        "target_capacity_kg": torch.stack([sample.target_capacity_kg for sample in samples]),
        "critical_component": torch.stack([sample.critical_component for sample in samples]),
        "critical_edge_mask": torch.cat(critical_edge_mask, dim=0),
        "edge_capacity_kg": torch.cat(edge_capacity_kg, dim=0),
    }


def save_capacity_dataset(
    path: str | Path,
    samples: Iterable[CapacityGraphSample],
    *,
    metadata: dict[str, object] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "samples": [sample.to_dict() for sample in samples],
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    return path


def load_capacity_dataset(path: str | Path) -> tuple[list[CapacityGraphSample], dict[str, object]]:
    # weights_only=False is required because the dataset intentionally stores a
    # Python metadata dictionary in addition to tensors. Only load trusted files.
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported capacity dataset version: {payload.get('format_version')}")
    samples = [CapacityGraphSample.from_dict(item) for item in payload["samples"]]
    return samples, dict(payload.get("metadata", {}))
