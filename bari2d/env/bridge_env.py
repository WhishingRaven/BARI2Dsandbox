from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from bari2d.env.contact_model import ContactGraph, ContactModel
from bari2d.env.field import LEFT_BANK, RIGHT_BANK, GapField, GapGenerator
from bari2d.env.load_evaluator import FastLoadEvaluator, IncrementalLoadEvaluator, LoadTestResult
from bari2d.env.robot import ACTION_COUNT, DiscreteAction, RobotState
from bari2d.env.sensors import ir_distances
from bari2d.utils.config import EnvironmentConfig


@dataclass(frozen=True)
class ObservationLayout:
    ir: slice
    strain: slice
    action_history: slice
    internal: slice
    goal: slice
    latent: slice
    size: int
    history: int
    ray_count: int


def make_observation_layout(config: EnvironmentConfig) -> ObservationLayout:
    history = config.sensor.history
    ray_count = len(config.sensor.ir_angles_deg)
    start = 0
    ir = slice(start, start + history * ray_count)
    start = ir.stop
    strain = slice(start, start + history)
    start = strain.stop
    action_history = slice(start, start + history)
    start = action_history.stop
    internal = slice(start, start + 6)
    start = internal.stop
    goal = slice(start, start + 1)
    start = goal.stop
    latent = slice(start, start + config.latent_dim)
    start = latent.stop
    return ObservationLayout(ir, strain, action_history, internal, goal, latent, start, history, ray_count)


class BridgeEnv:
    """Small deterministic-first 2.5D multi-robot construction environment.

    Actors receive only locally sensed histories, internal state, mission target,
    and optional episode-persistent noise. Global state is exposed separately for
    centralized training and never included in local observations.
    """

    def __init__(self, config: EnvironmentConfig | None = None):
        self.config = deepcopy(config or EnvironmentConfig())
        self.layout = make_observation_layout(self.config)
        self.rng = np.random.default_rng(self.config.seed)
        self.gap_generator = GapGenerator(self.config.gap)
        self.contact_model = ContactModel(deepcopy(self.config.contact), self.config.robot)
        self.fast_evaluator = FastLoadEvaluator()
        self.accurate_evaluator = IncrementalLoadEvaluator(self.config.load)
        self.field: GapField
        self.robots: list[RobotState] = []
        self.graph = ContactGraph(self.config.robot.count)
        self.target_load = self.config.load.target_min
        self.step_count = 0
        self.phase = "construction"
        self.current_capacity = 0.0
        self.current_progress = 0.0
        self.max_progress = 0.0
        self.last_load_test: LoadTestResult | None = None
        self._episode_success = False
        self._ir_history = np.zeros((self.config.robot.count, self.config.sensor.history, len(self.config.sensor.ir_angles_deg)), dtype=np.float32)
        self._strain_history = np.zeros((self.config.robot.count, self.config.sensor.history), dtype=np.float32)
        self._action_history = np.full((self.config.robot.count, self.config.sensor.history), int(DiscreteAction.IDLE), dtype=np.int64)
        self._previous_contacts: set[frozenset[int | str]] = set()
        self._total_motion = 0.0
        self._fallen_count = 0
        self._used_robots: set[int] = set()
        self._episode_randomization: dict[str, float] = {}

    @property
    def observation_size(self) -> int:
        return self.layout.size

    @property
    def global_state_size(self) -> int:
        return self.config.robot.count * 9 + 7

    @property
    def action_count(self) -> int:
        return ACTION_COUNT

    @property
    def graph_node_size(self) -> int:
        return 13

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._configure_episode_randomization()
        self.field = self.gap_generator.generate(self.rng, self.config.curriculum_stage)
        self.target_load = self._sample_target_load()
        self.robots = self._initial_robots()
        self.contact_model.reset()
        self.graph = self.contact_model.build_graph(self.robots, self.field, self.rng)
        self.step_count = 0
        self.phase = "construction"
        self.current_capacity = self.fast_evaluator.evaluate(self.graph, self.config.load).capacity
        self.current_progress = self.graph.spanning_progress(self.robots, self.field)
        self.max_progress = self.current_progress
        self.last_load_test = None
        self._episode_success = False
        self._total_motion = 0.0
        self._fallen_count = 0
        self._used_robots.clear()
        self._previous_contacts = set(self.graph.edges)
        self._ir_history.fill(1.0)
        self._strain_history.fill(0.0)
        self._action_history.fill(int(DiscreteAction.IDLE))
        self._update_histories()
        return self.observations(), self.info()

    def _configure_episode_randomization(self) -> None:
        self.contact_model.config = deepcopy(self.config.contact)
        self._episode_randomization = {
            "friction": self.config.contact.friction,
            "robot_mass": self.config.robot.mass,
            "anchor_strength_scale": 1.0,
            "sensor_noise": self.config.sensor.sensor_noise,
            "actuator_noise": self.config.actuator_noise,
        }
        if self.config.curriculum_stage >= 4:
            friction_scale = float(self.rng.uniform(0.8, 1.2))
            anchor_scale = float(self.rng.uniform(0.8, 1.2))
            self.contact_model.config.friction *= friction_scale
            self.contact_model.config.anchor_tension_limit *= anchor_scale
            self.contact_model.config.anchor_compression_limit *= anchor_scale
            self.contact_model.config.anchor_shear_limit *= anchor_scale
            self._episode_randomization.update(
                friction=self.contact_model.config.friction,
                robot_mass=self.config.robot.mass * float(self.rng.uniform(0.85, 1.15)),
                anchor_strength_scale=anchor_scale,
                sensor_noise=max(self.config.sensor.sensor_noise, 0.01),
                actuator_noise=max(self.config.actuator_noise, 0.01),
            )

    def _sample_target_load(self) -> float:
        if self.config.curriculum_stage <= 1:
            return self.config.load.target_min
        return float(self.rng.uniform(self.config.load.target_min, self.config.load.target_max))

    def _initial_robots(self) -> list[RobotState]:
        count = self.config.robot.count
        robot_config = self.config.robot
        longitudinal_left, _ = self.field.boundaries(0.0)
        edge_center = self.field.center + self.field.normal * longitudinal_left
        rows = max(1, int(np.floor(self.field.width / (robot_config.width * 1.5))))
        rows = min(rows, count)
        robots: list[RobotState] = []
        for robot_id in range(count):
            row = robot_id % rows
            column = robot_id // rows
            lateral = (row - (rows - 1) / 2.0) * robot_config.width * 1.35
            backward = 0.7 + column * robot_config.length * 1.15
            position = edge_center - self.field.normal * backward + self.field.tangent * lateral
            if self.config.curriculum_stage >= 2:
                position += self.rng.normal(0.0, 0.08, size=2)
                theta = self.field.orientation + float(self.rng.normal(0.0, 0.12))
            else:
                theta = self.field.orientation
            position[0] = np.clip(position[0], robot_config.length / 2.0, self.field.length - robot_config.length / 2.0)
            position[1] = np.clip(position[1], robot_config.width / 2.0, self.field.width - robot_config.width / 2.0)
            latent = self.rng.normal(0.0, self.config.latent_sigma, size=self.config.latent_dim).astype(np.float32)
            robots.append(RobotState(robot_id, position.astype(float), float(theta), latent=latent))
        return robots

    def set_robot_states(self, robots: Iterable[RobotState]) -> None:
        """Install a scripted morphology for deterministic tests and research probes."""
        states = list(robots)
        if len(states) != self.config.robot.count:
            raise ValueError(f"Expected {self.config.robot.count} robots, got {len(states)}")
        for expected_id, robot in enumerate(states):
            if robot.robot_id != expected_id:
                raise ValueError("Robot IDs must be contiguous and match list order")
            if robot.latent.size == 0:
                robot.latent = np.zeros(self.config.latent_dim, dtype=np.float32)
            elif robot.latent.shape != (self.config.latent_dim,):
                raise ValueError(f"Robot {robot.robot_id} latent must have shape {(self.config.latent_dim,)}")
        self.robots = states
        self.graph = self.contact_model.build_graph(self.robots, self.field, self.rng)
        self.current_progress = self.graph.spanning_progress(self.robots, self.field)
        self.current_capacity = self.fast_evaluator.evaluate(self.graph, self.config.load).capacity

    def action_masks(self) -> np.ndarray:
        masks = np.ones((len(self.robots), ACTION_COUNT), dtype=bool)
        for robot in self.robots:
            if robot.fallen:
                masks[robot.robot_id] = False
                masks[robot.robot_id, DiscreteAction.IDLE] = True
                continue
            masks[robot.robot_id, DiscreteAction.RELEASE] = robot.anchored
            masks[robot.robot_id, DiscreteAction.ANCHOR] = self.contact_model.can_anchor(robot, self.robots, self.field)
            masks[robot.robot_id, DiscreteAction.CLIMB] = self._climb_support(robot) is not None
            if robot.anchored:
                masks[robot.robot_id, :7] = False
        return masks

    def _climb_support(self, robot: RobotState) -> RobotState | None:
        candidates: list[tuple[float, RobotState]] = []
        for other in self.robots:
            if other.robot_id == robot.robot_id or other.fallen or other.layer >= self.config.robot.max_layer:
                continue
            relative = other.position - robot.position
            distance = float(np.linalg.norm(relative))
            if distance <= self.config.robot.length * 1.25 and relative @ robot.heading > 0.0:
                candidates.append((distance, other))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def step(self, actions: np.ndarray | list[int]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(actions, dtype=np.int64).copy()
        if action_array.shape != (len(self.robots),):
            raise ValueError(f"Expected action shape {(len(self.robots),)}, got {action_array.shape}")
        masks = self.action_masks()
        if np.any(action_array < 0) or np.any(action_array >= ACTION_COUNT):
            raise ValueError("Action outside discrete action space")
        invalid = ~masks[np.arange(len(self.robots)), action_array]
        action_array[invalid] = int(DiscreteAction.IDLE)

        old_progress = self.current_progress
        old_quality = min(self.current_capacity / max(self.target_load, 1.0e-9), 1.0)
        old_energy = sum(robot.energy for robot in self.robots)
        old_fallen = sum(robot.fallen for robot in self.robots)
        old_failures = self.contact_model.anchor_failures
        previously_used = len(self._used_robots)
        newly_anchored = 0
        for robot, action_value in zip(self.robots, action_array):
            action = DiscreteAction(int(action_value))
            if action != DiscreteAction.IDLE:
                self._used_robots.add(robot.robot_id)
            robot.previous_action = int(action)
            robot.head_lifted = action == DiscreteAction.CLIMB
            if action == DiscreteAction.RELEASE:
                self.contact_model.release(robot)
            elif action == DiscreteAction.ANCHOR:
                newly_anchored += int(self.contact_model.anchor(robot, self.robots, self.field))
            elif action == DiscreteAction.CLIMB:
                support = self._climb_support(robot)
                if support is not None:
                    robot.layer = min(support.layer + 1, self.config.robot.max_layer)
                    robot.position = robot.position + robot.heading * self.config.robot.length * 0.3
                    robot.energy += self._episode_randomization["robot_mass"] * self.config.robot.climb_height
                robot.previous_action = int(action)
            elif not robot.anchored and not robot.fallen:
                robot.step_kinematics(
                    int(action),
                    self.config.robot,
                    self.config.time_step,
                    self.rng,
                    self._episode_randomization["actuator_noise"],
                )
            else:
                robot.previous_action = int(action)
            self._constrain_to_field(robot)

        self.contact_model.resolve_contacts(self.robots)
        for robot in self.robots:
            self._constrain_to_field(robot)
        self.graph = self.contact_model.build_graph(self.robots, self.field, self.rng)
        self._mark_fallen_robots()
        if sum(robot.fallen for robot in self.robots) > old_fallen:
            self.graph = self.contact_model.build_graph(self.robots, self.field, self.rng)
        self.current_progress = self.graph.spanning_progress(self.robots, self.field)
        self.max_progress = max(self.max_progress, self.current_progress)
        fast_result = self.fast_evaluator.evaluate(self.graph, self.config.load)
        self.current_capacity = fast_result.capacity
        self.step_count += 1
        self._update_histories()

        accurate_capacity: float | None = None
        success = False
        time_limit = self.step_count >= self.config.max_steps
        if (self.graph.spans and self.current_capacity >= self.target_load) or time_limit:
            self.phase = "load_test"
            self.last_load_test = self.accurate_evaluator.evaluate(self.graph)
            accurate_capacity = self.last_load_test.capacity
            self.current_capacity = accurate_capacity
            success = self.graph.spans and accurate_capacity >= self.target_load
            self.phase = "complete" if success or time_limit else "construction"

        terminated = success
        self._episode_success = success
        truncated = time_limit and not terminated

        new_quality = min(self.current_capacity / max(self.target_load, 1.0e-9), 1.0)
        energy_delta = sum(robot.energy for robot in self.robots) - old_energy
        collapsed = sum(robot.fallen for robot in self.robots) - old_fallen
        reward_config = self.config.reward
        reward_components = {
            "span": reward_config.span_delta * (self.current_progress - old_progress),
            "mechanical": reward_config.mechanical_delta * (new_quality - old_quality),
            "time": -reward_config.time_penalty,
            "energy": -reward_config.energy_penalty * energy_delta,
            "anchor": -reward_config.anchor_penalty * newly_anchored,
            "robot_use": -reward_config.robot_use_penalty * (len(self._used_robots) - previously_used),
            "collapse": -reward_config.collapse_penalty * collapsed,
            "success": reward_config.success_reward if success else 0.0,
        }
        self._total_motion += energy_delta
        self._fallen_count = sum(robot.fallen for robot in self.robots)
        info = self.info()
        info.update(
            reward_components=reward_components,
            accurate_capacity=accurate_capacity,
            new_anchor_failures=self.contact_model.anchor_failures - old_failures,
            auxiliary_targets=self.auxiliary_targets(),
        )
        return self.observations(), float(sum(reward_components.values())), terminated, truncated, info

    def _constrain_to_field(self, robot: RobotState) -> None:
        margin_x = self.config.robot.length / 2.0
        margin_y = self.config.robot.width / 2.0
        robot.position[0] = float(np.clip(robot.position[0], margin_x, self.field.length - margin_x))
        robot.position[1] = float(np.clip(robot.position[1], margin_y, self.field.width - margin_y))

    def _mark_fallen_robots(self) -> None:
        supported = self.graph.connected_component(LEFT_BANK) | self.graph.connected_component(RIGHT_BANK)
        for robot in self.robots:
            if robot.fallen or robot.robot_id in supported:
                continue
            if self.field.is_gap(robot.position) and not self.field.bank_contact(robot.corners(self.config.robot)):
                robot.fallen = True
                robot.velocity = 0.0
                self.contact_model.release(robot)

    def _update_histories(self) -> None:
        self._ir_history = np.roll(self._ir_history, -1, axis=1)
        self._strain_history = np.roll(self._strain_history, -1, axis=1)
        self._action_history = np.roll(self._action_history, -1, axis=1)
        sensor_config = deepcopy(self.config.sensor)
        sensor_config.sensor_noise = self._episode_randomization.get("sensor_noise", sensor_config.sensor_noise)
        strain_scale = max(self.contact_model.anchor_capacity, self.config.contact.contact_capacity, 1.0)
        for robot in self.robots:
            self._ir_history[robot.robot_id, -1] = ir_distances(
                robot, self.robots, self.field, self.config.robot, sensor_config, self.rng
            )
            noisy_strain = robot.strain + float(self.rng.normal(0.0, sensor_config.sensor_noise))
            self._strain_history[robot.robot_id, -1] = np.clip(noisy_strain / strain_scale, 0.0, 2.0)
            self._action_history[robot.robot_id, -1] = robot.previous_action

    def observations(self) -> np.ndarray:
        values = np.zeros((len(self.robots), self.layout.size), dtype=np.float32)
        maximum_steering = max(np.deg2rad(self.config.robot.max_steering_deg), 1.0e-9)
        for robot in self.robots:
            index = robot.robot_id
            values[index, self.layout.ir] = self._ir_history[index].reshape(-1)
            values[index, self.layout.strain] = self._strain_history[index]
            values[index, self.layout.action_history] = self._action_history[index] / max(ACTION_COUNT - 1, 1)
            values[index, self.layout.internal] = np.array(
                [
                    float(robot.anchored),
                    float(robot.head_lifted),
                    robot.velocity / max(self.config.robot.max_speed, 1.0e-9),
                    robot.steering / maximum_steering,
                    robot.layer / max(self.config.robot.max_layer, 1),
                    float(robot.fallen),
                ],
                dtype=np.float32,
            )
            values[index, self.layout.goal] = self.target_load / max(self.config.load.max_target, 1.0e-9)
            values[index, self.layout.latent] = robot.latent
        return values

    def auxiliary_targets(self) -> dict[str, np.ndarray]:
        count = len(self.robots)
        neighbor_count = self.contact_model.contact_counts(self.graph, count)
        traffic = np.zeros(count, dtype=np.float32)
        persistence = np.zeros(count, dtype=np.float32)
        movement_actions = {
            int(DiscreteAction.FORWARD), int(DiscreteAction.BACKWARD), int(DiscreteAction.FORWARD_LEFT),
            int(DiscreteAction.FORWARD_RIGHT), int(DiscreteAction.BACKWARD_LEFT), int(DiscreteAction.BACKWARD_RIGHT),
        }
        for robot in self.robots:
            robot_neighbors = [node for node in self.graph.neighbors(robot.robot_id) if isinstance(node, int)]
            traffic[robot.robot_id] = sum(self.robots[node].previous_action in movement_actions for node in robot_neighbors)
            persistence[robot.robot_id] = sum(
                frozenset((robot.robot_id, node)) in self._previous_contacts for node in robot_neighbors
            )
        force_trend = self._strain_history[:, -1] - self._strain_history[:, -2]
        self._previous_contacts = set(self.graph.edges)
        return {
            "connectivity": neighbor_count,
            "traffic": traffic,
            "contact_persistence": persistence,
            "force_trend": force_trend.astype(np.float32),
        }

    def global_state(self) -> np.ndarray:
        features: list[float] = []
        strain_scale = max(self.contact_model.anchor_capacity, 1.0)
        for robot in self.robots:
            features.extend(
                [
                    robot.position[0] / self.field.length,
                    robot.position[1] / self.field.width,
                    np.sin(robot.theta),
                    np.cos(robot.theta),
                    robot.strain / strain_scale,
                    float(robot.anchored),
                    robot.velocity / max(self.config.robot.max_speed, 1.0e-9),
                    robot.layer / max(self.config.robot.max_layer, 1),
                    float(robot.fallen),
                ]
            )
        features.extend(
            [
                self.field.gap_width / self.field.length,
                np.sin(self.field.orientation),
                np.cos(self.field.orientation),
                self.target_load / max(self.config.load.max_target, 1.0e-9),
                self.current_progress,
                self.current_capacity / max(self.config.load.max_target, 1.0e-9),
                float(self.graph.spans),
            ]
        )
        return np.asarray(features, dtype=np.float32)

    def graph_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        node_features = np.array(
            [
                [
                    robot.position[0] / self.field.length,
                    robot.position[1] / self.field.width,
                    np.sin(robot.theta),
                    np.cos(robot.theta),
                    robot.strain / max(self.contact_model.anchor_capacity, 1.0),
                    float(robot.anchored),
                    robot.velocity / max(self.config.robot.max_speed, 1.0e-9),
                    robot.layer / max(self.config.robot.max_layer, 1),
                    float(robot.fallen),
                    float(LEFT_BANK in self.field.bank_contact(robot.corners(self.config.robot))),
                    float(RIGHT_BANK in self.field.bank_contact(robot.corners(self.config.robot))),
                    self.target_load / max(self.config.load.max_target, 1.0e-9),
                    self.field.gap_width / self.field.length,
                ]
                for robot in self.robots
            ],
            dtype=np.float32,
        )
        indices: list[list[int]] = []
        edge_features: list[list[float]] = []
        kind_values = {"contact": 0.0, "anchor": 0.5, "bank_contact": 1.0}
        for edge in self.graph.edges.values():
            if not isinstance(edge.source, int) or not isinstance(edge.target, int):
                continue
            indices.extend([[edge.source, edge.target], [edge.target, edge.source]])
            feature = [
                kind_values[edge.kind],
                edge.force / max(edge.capacity, 1.0e-9),
                edge.capacity / max(self.config.load.max_target, 1.0e-9),
                edge.relative_pose[0] / self.config.robot.length,
                edge.relative_pose[1] / self.config.robot.length,
                edge.relative_pose[2] / np.pi,
            ]
            edge_features.extend([feature, feature])
        return (
            node_features,
            np.asarray(indices, dtype=np.int64).reshape(-1, 2),
            np.asarray(edge_features, dtype=np.float32).reshape(-1, 6),
        )

    def dense_graph_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nodes, edge_indices, edge_values = self.graph_state()
        count = len(self.robots)
        adjacency = np.zeros((count, count), dtype=np.float32)
        edges = np.zeros((count, count, 6), dtype=np.float32)
        for (source, target), values in zip(edge_indices, edge_values):
            adjacency[source, target] = 1.0
            edges[source, target] = values
        return nodes, adjacency, edges

    def info(self) -> dict[str, Any]:
        return {
            "success": self._episode_success,
            "phase": self.phase,
            "step": self.step_count,
            "gap_width": self.field.gap_width,
            "gap_orientation": self.field.orientation,
            "gap_irregularity": self.field.irregularity,
            "gap_irregularity_frequency": self.field.irregularity_frequency,
            "gap_phase": self.field.phase,
            "target_load": self.target_load,
            "capacity": self.current_capacity,
            "capacity_ratio": self.current_capacity / max(self.target_load, 1.0e-9),
            "span": self.graph.spans,
            "span_progress": self.current_progress,
            "max_span_progress": self.max_progress,
            "anchored_robots": sum(robot.anchored for robot in self.robots),
            "robots_used": len(self._used_robots),
            "energy": self._total_motion,
            "anchor_failures": self.contact_model.anchor_failures,
            "fallen_robots": sum(robot.fallen for robot in self.robots),
            "contact_graph": self.graph.serializable_edges(),
            "morphology": [
                {
                    "id": robot.robot_id,
                    "x": float(robot.position[0]),
                    "y": float(robot.position[1]),
                    "theta": robot.theta,
                    "layer": robot.layer,
                    "anchored": robot.anchored,
                    "strain": robot.strain,
                    "fallen": robot.fallen,
                }
                for robot in self.robots
            ],
            "randomization": dict(self._episode_randomization),
            "load_test": None if self.last_load_test is None else {
                "capacity": self.last_load_test.capacity,
                "failure_mode": self.last_load_test.failure_mode,
                "failed_edges": self.last_load_test.failed_edges,
                "load_bearing_path": self.last_load_test.load_bearing_path,
                "estimated_displacement": self.last_load_test.estimated_displacement,
                "load_protocol": self.last_load_test.load_protocol,
            },
            "construction_time": self.step_count * self.config.time_step,
        }
