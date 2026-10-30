from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import numpy as np

from bari2d.env.contact_model import ContactGraph, MechanicalEdge
from bari2d.env.field import LEFT_BANK, RIGHT_BANK, GapField
from bari2d.env.robot import RobotState
from bari2d.utils.config import ContactConfig, RobotConfig


@dataclass(frozen=True)
class RandomBridgeConfig:
    """Geometry/topology ranges for procedural bridge generation.

    The generator deliberately creates a guaranteed spanning backbone first and
    then adds rows, braces, branches, and stacked robots. Connections are welded
    graph edges because the current simulator treats anchored contacts as
    non-detaching. The spatial statics solver determines the six-component
    wrench carried by each connection.
    """

    gap_width_range: tuple[float, float] = (3.0, 6.0)
    field_length: float = 18.0
    field_width: float = 10.0
    orientation_range_deg: tuple[float, float] = (-18.0, 18.0)
    rows_range: tuple[int, int] = (1, 3)
    spacing_fraction_range: tuple[float, float] = (0.72, 0.90)
    row_spacing_fraction_range: tuple[float, float] = (0.90, 1.25)
    yaw_jitter_deg: float = 9.0
    position_jitter_fraction: float = 0.06
    cross_brace_probability: float = 0.55
    diagonal_brace_probability: float = 0.30
    branch_probability: float = 0.18
    max_branch_length: int = 2
    stack_probability: float = 0.16
    max_stack_layer: int = 2

    def __post_init__(self) -> None:
        if self.gap_width_range[0] <= 0.0 or self.gap_width_range[1] < self.gap_width_range[0]:
            raise ValueError("Invalid gap_width_range")
        if self.rows_range[0] < 1 or self.rows_range[1] < self.rows_range[0]:
            raise ValueError("Invalid rows_range")
        if not 0.0 <= self.cross_brace_probability <= 1.0:
            raise ValueError("cross_brace_probability must be in [0, 1]")
        if not 0.0 <= self.diagonal_brace_probability <= 1.0:
            raise ValueError("diagonal_brace_probability must be in [0, 1]")
        if not 0.0 <= self.branch_probability <= 1.0:
            raise ValueError("branch_probability must be in [0, 1]")
        if not 0.0 <= self.stack_probability <= 1.0:
            raise ValueError("stack_probability must be in [0, 1]")


@dataclass
class GeneratedBridge:
    robots: list[RobotState]
    graph: ContactGraph
    field: GapField
    backbone_rows: list[list[int]]
    payload_candidates: list[int]
    metadata: dict[str, float | int] = field(default_factory=dict)

    @property
    def robot_count(self) -> int:
        return len(self.robots)

    @property
    def edge_count(self) -> int:
        return len(self.graph.edges)


class RandomBridgeGenerator:
    """Generate an unbounded stream of diverse, guaranteed-spanning bridges."""

    def __init__(
        self,
        robot_config: RobotConfig | None = None,
        contact_config: ContactConfig | None = None,
        config: RandomBridgeConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.robot_config = robot_config or RobotConfig()
        self.contact_config = contact_config or ContactConfig()
        self.config = config or RandomBridgeConfig()
        self.rng = np.random.default_rng(seed)

    def _field(self) -> GapField:
        cfg = self.config
        gap_width = float(self.rng.uniform(*cfg.gap_width_range))
        orientation = float(np.deg2rad(self.rng.uniform(*cfg.orientation_range_deg)))
        return GapField(
            length=cfg.field_length,
            width=cfg.field_width,
            gap_width=gap_width,
            center=np.asarray([cfg.field_length / 2.0, cfg.field_width / 2.0], dtype=float),
            orientation=orientation,
        )

    def _edge(self, source: int | str, target: int | str, kind: str = "anchor") -> MechanicalEdge:
        stiffness = max(float(self.contact_config.contact_stiffness), 1.0e-6)
        if kind == "anchor":
            axial = min(
                float(self.contact_config.anchor_tension_limit),
                float(self.contact_config.anchor_compression_limit),
            )
            capacity = float(np.hypot(axial, self.contact_config.anchor_shear_limit))
        elif isinstance(source, str) or isinstance(target, str):
            capacity = float(self.contact_config.bank_capacity * self.contact_config.friction)
        else:
            capacity = float(self.contact_config.contact_capacity * self.contact_config.friction)
        return MechanicalEdge(source, target, kind, capacity, stiffness)

    def generate(self) -> GeneratedBridge:
        field = self._field()
        robot_cfg = self.robot_config
        cfg = self.config

        rows = int(self.rng.integers(cfg.rows_range[0], cfg.rows_range[1] + 1))
        spacing_fraction = float(self.rng.uniform(*cfg.spacing_fraction_range))
        nominal_spacing = robot_cfg.length * spacing_fraction
        bank_overlap = 0.32 * robot_cfg.length
        span_length = field.gap_width + 2.0 * bank_overlap
        chain_length = max(3, int(ceil(span_length / max(nominal_spacing, 1.0e-9))) + 1)
        longitudinal_positions = np.linspace(
            -field.gap_width / 2.0 - bank_overlap,
            field.gap_width / 2.0 + bank_overlap,
            chain_length,
        )

        row_spacing = robot_cfg.width * float(self.rng.uniform(*cfg.row_spacing_fraction_range))
        robots: list[RobotState] = []
        backbone_rows: list[list[int]] = []

        for row_index in range(rows):
            tangent_base = (row_index - (rows - 1) / 2.0) * row_spacing
            row_ids: list[int] = []
            for index, longitudinal in enumerate(longitudinal_positions):
                endpoint = index in (0, chain_length - 1)
                tangent_jitter = 0.0 if endpoint else float(
                    self.rng.normal(0.0, cfg.position_jitter_fraction * robot_cfg.width)
                )
                longitudinal_jitter = 0.0 if endpoint else float(
                    self.rng.normal(0.0, cfg.position_jitter_fraction * robot_cfg.length)
                )
                position = (
                    field.center
                    + field.normal * (longitudinal + longitudinal_jitter)
                    + field.tangent * (tangent_base + tangent_jitter)
                )
                theta = field.orientation + float(np.deg2rad(self.rng.normal(0.0, cfg.yaw_jitter_deg)))
                robot_id = len(robots)
                robots.append(RobotState(robot_id, position.astype(float), theta, layer=0))
                row_ids.append(robot_id)
            backbone_rows.append(row_ids)

        graph = ContactGraph(len(robots))

        # Guaranteed spanning row chains and bank supports.
        for row_ids in backbone_rows:
            graph.add_edge(self._edge(row_ids[0], LEFT_BANK, "anchor"))
            graph.add_edge(self._edge(row_ids[-1], RIGHT_BANK, "anchor"))
            for first, second in zip(row_ids[:-1], row_ids[1:]):
                graph.add_edge(self._edge(first, second, "anchor"))

        # Parallel-row braces turn simple chains into ladder/truss-like networks.
        for row_index in range(rows - 1):
            upper = backbone_rows[row_index]
            lower = backbone_rows[row_index + 1]
            # At least one brace is mandatory so all rows belong to the same
            # robot-only structural component. Bank nodes are fixed supports,
            # not physical couplers between otherwise independent rows.
            mandatory_cross = int(self.rng.integers(0, chain_length))
            graph.add_edge(self._edge(upper[mandatory_cross], lower[mandatory_cross], "anchor"))
            for index in range(chain_length):
                if index != mandatory_cross and self.rng.random() < cfg.cross_brace_probability:
                    graph.add_edge(self._edge(upper[index], lower[index], "anchor"))
                if index + 1 < chain_length and self.rng.random() < cfg.diagonal_brace_probability:
                    if self.rng.random() < 0.5:
                        graph.add_edge(self._edge(upper[index], lower[index + 1], "anchor"))
                    else:
                        graph.add_edge(self._edge(lower[index], upper[index + 1], "anchor"))

        # Add short side branches. They contribute self-weight and create local
        # stress paths without changing the guaranteed bank-to-bank backbone.
        backbone_ids = [node for row in backbone_rows for node in row]
        for base_id in list(backbone_ids):
            if self.rng.random() >= cfg.branch_probability:
                continue
            branch_length = int(self.rng.integers(1, cfg.max_branch_length + 1))
            previous = base_id
            side = -1.0 if self.rng.random() < 0.5 else 1.0
            base = robots[base_id]
            for step in range(branch_length):
                offset = field.tangent * side * robot_cfg.width * (0.82 + 0.78 * step)
                offset += field.normal * float(self.rng.normal(0.0, 0.10 * robot_cfg.length))
                position = base.position + offset
                robot_id = len(robots)
                theta = field.orientation + float(np.deg2rad(self.rng.normal(0.0, cfg.yaw_jitter_deg * 1.5)))
                robots.append(RobotState(robot_id, position.astype(float), theta, layer=0))
                graph.nodes.add(robot_id)
                graph.add_edge(self._edge(previous, robot_id, "anchor"))
                previous = robot_id

        # Stack some robots to introduce genuine z offsets and nontrivial Mx/My.
        stacked_by_base: dict[int, list[int]] = {}
        for base_id in list(backbone_ids):
            if self.rng.random() >= cfg.stack_probability:
                continue
            current_base = base_id
            stack_ids: list[int] = []
            stack_height = int(self.rng.integers(1, cfg.max_stack_layer + 1))
            for layer in range(1, stack_height + 1):
                base = robots[base_id]
                xy_jitter = (
                    field.normal * float(self.rng.normal(0.0, 0.035 * robot_cfg.length))
                    + field.tangent * float(self.rng.normal(0.0, 0.035 * robot_cfg.width))
                )
                robot_id = len(robots)
                robots.append(
                    RobotState(
                        robot_id,
                        (base.position + xy_jitter).astype(float),
                        base.theta + float(np.deg2rad(self.rng.normal(0.0, cfg.yaw_jitter_deg * 0.5))),
                        layer=layer,
                    )
                )
                graph.nodes.add(robot_id)
                graph.add_edge(self._edge(current_base, robot_id, "contact"))
                current_base = robot_id
                stack_ids.append(robot_id)
            stacked_by_base[base_id] = stack_ids

        # Connect neighboring top robots when both columns were stacked to the
        # same layer. This creates 3-D load-sharing loops.
        for row_ids in backbone_rows:
            for first, second in zip(row_ids[:-1], row_ids[1:]):
                first_stack = stacked_by_base.get(first, [])
                second_stack = stacked_by_base.get(second, [])
                for first_top, second_top in zip(first_stack, second_stack):
                    graph.add_edge(self._edge(first_top, second_top, "anchor"))

        # Fill graph coordinate caches used elsewhere in the simulator.
        for robot in robots:
            _, tangent = field.coordinates(robot.position)
            graph.node_progress[robot.robot_id] = field.normalized_progress(robot.position)
            graph.node_tangent[robot.robot_id] = tangent

        # Payload candidates are backbone/stack robots near the gap rather than
        # dead-end branches. This keeps the task "load on the bridge deck".
        payload_candidates = list(backbone_ids)
        payload_candidates.extend(node for stack in stacked_by_base.values() for node in stack)

        return GeneratedBridge(
            robots=robots,
            graph=graph,
            field=field,
            backbone_rows=backbone_rows,
            payload_candidates=payload_candidates,
            metadata={
                "rows": rows,
                "chain_length": chain_length,
                "gap_width": field.gap_width,
                "orientation": field.orientation,
            },
        )

    def stream(self):
        while True:
            yield self.generate()
