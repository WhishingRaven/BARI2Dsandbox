from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from bari2d.env.contact_model import ContactGraph, ContactModel, oriented_boxes_overlap
from bari2d.env.field import LEFT_BANK, RIGHT_BANK, GapField, GapGenerator
from bari2d.env.load_evaluator import FastLoadEvaluator, IncrementalLoadEvaluator, LoadTestResult
from bari2d.env.robot import ACTION_COUNT, DiscreteAction, RobotState
from bari2d.env.sensors import cliff_sensor_direction, front_over_cliff, ir_distances, ir_ray_count
from bari2d.env.static_physics import PHYSICS_MODELS, StaticContactModel, build_structure
from bari2d.env.statics import max_load, stabilize
from bari2d.utils.config import BEACON_FEATURE_SIZE, EnvironmentConfig


START_SETUPS = ("none", "root_only", "walker_behind_root", "walker_on_root", "row_spanning")
ANCHOR_RULES = ("edge_band", "front_downward_ir")
CUE_SIZE = 2


@dataclass(frozen=True)
class ObservationLayout:
    ir: slice
    strain: slice
    action_history: slice
    internal: slice
    beacon: slice
    cues: slice
    goal: slice
    latent: slice
    size: int
    history: int
    ray_count: int


def make_observation_layout(config: EnvironmentConfig) -> ObservationLayout:
    history = config.sensor.history
    ray_count = ir_ray_count(config.sensor)
    start = 0
    ir = slice(start, start + history * ray_count)
    start = ir.stop
    strain = slice(start, start + history)
    start = strain.stop
    action_history = slice(start, start + history)
    start = action_history.stop
    internal = slice(start, start + 6)
    start = internal.stop
    beacon = slice(start, start + (BEACON_FEATURE_SIZE if config.beacon.enabled else 0))
    start = beacon.stop
    cues = slice(start, start + (CUE_SIZE if config.sensor.structure_cues else 0))
    start = cues.stop
    goal = slice(start, start + 1)
    start = goal.stop
    latent = slice(start, start + config.latent_dim)
    start = latent.stop
    return ObservationLayout(ir, strain, action_history, internal, beacon, cues, goal, latent, start, history, ray_count)


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
        if self.config.physics.model not in PHYSICS_MODELS:
            raise ValueError(f"Unknown physics model: {self.config.physics.model}")
        self.static_physics = self.config.physics.model == "static"
        if self.static_physics:
            self.contact_model: ContactModel = StaticContactModel(
                deepcopy(self.config.contact), self.config.robot, self.config.physics
            )
        else:
            self.contact_model = ContactModel(deepcopy(self.config.contact), self.config.robot)
        if self.config.contact.anchor_rule not in ANCHOR_RULES:
            raise ValueError(f"Unknown contact.anchor_rule: {self.config.contact.anchor_rule}")
        if self.config.contact.anchor_rule == "front_downward_ir" and (
            not self.config.sensor.downward_ir_enabled or self.config.sensor.downward_ir_offset <= 0.0
        ):
            raise ValueError("anchor_rule front_downward_ir needs the downward IR ahead of the centre (sensor.downward_ir_offset > 0)")
        if not 0 <= self.config.robot.active_min <= self.config.robot.count:
            raise ValueError("robot.active_min must be between 0 and robot.count")
        self._static_spans = False
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
        self._ir_history = np.zeros((self.config.robot.count, self.config.sensor.history, self.layout.ray_count), dtype=np.float32)
        self._strain_history = np.zeros((self.config.robot.count, self.config.sensor.history), dtype=np.float32)
        self._action_history = np.full((self.config.robot.count, self.config.sensor.history), int(DiscreteAction.IDLE), dtype=np.int64)
        self._previous_contacts: set[frozenset[int | str]] = set()
        self._beacon_source_id: int | None = None
        self._beacon_edge_direction = np.zeros(2, dtype=float)
        self._beacon_age = 0
        self._beacon_features = np.zeros((self.config.robot.count, BEACON_FEATURE_SIZE), dtype=np.float32)
        self._idle_streak = np.zeros(self.config.robot.count, dtype=np.int64)
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
        setup = self._start_setup()
        if setup != "none":
            self.contact_model.anchor(self.robots[0], self.robots, self.field)
        if setup == "row_spanning":
            # Grip every further link of the prepared row to the one behind it.
            attach = getattr(self.contact_model, "_attach", None)
            for robot in self.robots[1 : self._prepared_row_length]:
                if attach is not None:
                    attach(robot, robot.robot_id - 1, self.robots)
                else:
                    self.contact_model.anchor(robot, self.robots, self.field)
        self.graph = self.contact_model.build_graph(self.robots, self.field, self.rng)
        self.step_count = 0
        self.phase = "construction"
        self.current_capacity = self._evaluate_load().capacity
        self.current_progress = self._structure_progress()
        self.max_progress = self.current_progress
        self.last_load_test = None
        self._episode_success = False
        self._total_motion = 0.0
        self._fallen_count = 0
        self._used_robots.clear()
        self._previous_contacts = set(self.graph.edges)
        self._beacon_source_id = None
        self._beacon_edge_direction.fill(0.0)
        self._beacon_age = 0
        self._beacon_features.fill(0.0)
        self._idle_streak.fill(0)
        self._ir_history.fill(1.0)
        self._strain_history.fill(0.0)
        self._action_history.fill(int(DiscreteAction.IDLE))
        self._update_histories()
        self._update_beacon()
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
        if self.config.curriculum_stage >= self.config.randomization_stage:
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
        staged_targets = self.config.load.curriculum_targets
        if staged_targets:
            index = min(max(self.config.curriculum_stage - 1, 0), len(staged_targets) - 1)
            return float(staged_targets[index])
        if self.config.curriculum_stage <= 1:
            return self.config.load.target_min
        return float(self.rng.uniform(self.config.load.target_min, self.config.load.target_max))

    def _initial_robots(self) -> list[RobotState]:
        """Sample non-overlapping robot poses over the left starting bank.

        The construction task still begins on one bank of the gap, but robots
        no longer receive a pre-aligned formation.  Both positions and heading
        are randomized on every reset, subject to the full footprint remaining
        on the left bank and not overlapping another robot.
        """
        count = self.config.robot.count
        robot_config = self.config.robot
        robots: list[RobotState] = self._prepared_robots(self._start_setup())[:count]
        for robot in robots:
            robot.latent = self.rng.normal(0.0, self.config.latent_sigma, size=self.config.latent_dim).astype(np.float32)
        prepared = len(robots)
        for robot_id in range(len(robots), count):
            candidate = self._sample_initial_robot(robot_id, robots)
            candidate.latent = self.rng.normal(
                0.0, self.config.latent_sigma, size=self.config.latent_dim
            ).astype(np.float32)
            robots.append(candidate)
        self._remove_absent_robots(robots, prepared)
        return robots

    def _remove_absent_robots(self, robots: list[RobotState], prepared: int) -> None:
        """Team-size randomisation: only a random number of robots takes part.

        Absent robots are marked fallen from the start, so they never act, are
        not sensed and carry nothing; robots of a prepared start setup always
        take part.  Draws nothing from the RNG when ``robot.active_min`` is 0.
        """
        count, active_min = len(robots), self.config.robot.active_min
        if active_min <= 0 or active_min >= count:
            return
        active = int(self.rng.integers(max(active_min, prepared), count + 1))
        absent = self.rng.choice(np.arange(prepared, count), size=count - active, replace=False)
        for robot_id in absent:
            robots[int(robot_id)].absent = True
            robots[int(robot_id)].fallen = True

    def _start_setup(self) -> str:
        setups = self.config.start_setups
        if not setups:
            return "none"
        setup = setups[min(max(self.config.curriculum_stage - 1, 0), len(setups) - 1)]
        if setup not in START_SETUPS:
            raise ValueError(f"Unknown start setup: {setup}")
        return setup

    def _prepared_robots(self, setup: str) -> list[RobotState]:
        """Robots placed for a start-state curriculum stage (anchoring happens in reset)."""
        self._prepared_row_length = 0
        if setup == "none":
            return []
        length = self.config.robot.length
        heading = float(np.arctan2(self.field.normal[1], self.field.normal[0]))
        half_span = max(self.field.width / 2.0 - 1.5, 0.0)
        tangent = float(self.rng.uniform(-half_span, half_span))
        left, _ = self.field.boundaries(tangent)
        root_position = self.field.center + self.field.normal * (left - 0.05) + self.field.tangent * tangent
        robots = [RobotState(0, root_position.astype(float), heading)]
        if setup == "row_spanning":
            # A straight single row from the root across the gap, resting at
            # least 0.05 on the far bank (as scripted.assemble_static_chain).
            _, right = self.field.boundaries(tangent)
            root_front = left - 0.05 + length / 2.0
            links = int(np.ceil(max(right + 0.05 - root_front, 0.0) / length))
            for link in range(1, links + 1):
                robots.append(RobotState(link, (root_position + self.field.normal * length * link).astype(float), heading))
            self._prepared_row_length = len(robots)
        if setup == "walker_behind_root":
            robots.append(RobotState(1, (root_position - self.field.normal * (length + 0.02)).astype(float), heading))
        elif setup == "walker_on_root":
            robots.append(RobotState(1, (root_position - self.field.normal * 0.25).astype(float), heading, layer=1))
        return robots

    def _sample_initial_robot(self, robot_id: int, existing: list[RobotState]) -> RobotState:
        """Draw one valid left-bank pose, rejecting gap, boundary, and overlap cases."""
        robot_config = self.config.robot
        # The circumscribed radius keeps every rotated footprint in the field
        # before the more exact left-bank test below.
        margin = 0.5 * np.hypot(robot_config.length, robot_config.width) + 1.0e-3
        low = np.array([margin, margin], dtype=float)
        high = np.array([self.field.length - margin, self.field.width - margin], dtype=float)
        edge_limit = self._spawn_edge_limit()
        attempts = max(2_000, self.config.robot.count * 250) * (10 if edge_limit > 0.0 else 1)
        for _ in range(attempts):
            position = self.rng.uniform(low, high)
            theta = float(self.rng.uniform(-np.pi, np.pi))
            candidate = RobotState(robot_id, position.astype(float), theta)
            if any(self.field.bank_at(corner) != LEFT_BANK for corner in candidate.corners(robot_config)):
                continue
            if edge_limit > 0.0:
                longitudinal, tangent = self.field.coordinates(position)
                left_boundary, _ = self.field.boundaries(tangent)
                if left_boundary - longitudinal > edge_limit:
                    continue
            if any(oriented_boxes_overlap(candidate, other, robot_config) for other in existing):
                continue
            return candidate
        raise RuntimeError(
            "Could not place every robot on the left bank without overlap. "
            "Increase the starting-bank area or reduce the robot count."
        )

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
        self.current_progress = self._structure_progress()
        self.current_capacity = self._evaluate_load().capacity

    def action_masks(self) -> np.ndarray:
        masks = np.ones((len(self.robots), ACTION_COUNT), dtype=bool)
        for robot in self.robots:
            if robot.fallen:
                masks[robot.robot_id] = False
                masks[robot.robot_id, DiscreteAction.IDLE] = True
                continue
            masks[robot.robot_id, DiscreteAction.RELEASE] = robot.anchored
            masks[robot.robot_id, DiscreteAction.ANCHOR] = self._anchor_action_available(robot)
            masks[robot.robot_id, DiscreteAction.CLIMB] = self._climb_support(robot) is not None
            if robot.anchored:
                masks[robot.robot_id, :7] = False
        return masks

    def _anchor_action_available(self, robot: RobotState) -> bool:
        """Return whether a physically meaningful anchor can be made locally."""
        if not self.contact_model.can_anchor(robot, self.robots, self.field):
            return False
        front_rule = self.config.contact.anchor_rule == "front_downward_ir"
        if not self.config.contact.anchor_requires_edge_or_contact and not front_rule:
            return True
        partner = self.contact_model.candidate_anchor(robot, self.robots, self.field)
        if isinstance(partner, int):
            if not self.config.contact.anchor_requires_edge_or_contact:
                return True
            return oriented_boxes_overlap(robot, self.robots[partner], self.config.robot)
        if partner is None:
            return False
        if front_rule:
            # Design rule: the front feet hook the lip, i.e. the belly IR (ahead
            # of the centre) sees no floor while the centre is still on the bank.
            return front_over_cliff(robot, self.robots, self.field, self.config.robot, self.config.sensor)
        reading = cliff_sensor_direction(robot, self.robots, self.field, self.config.robot, self.config.sensor)
        return reading is not None and reading[1] <= self.config.contact.anchor_edge_band

    def _climb_support(self, robot: RobotState) -> RobotState | None:
        if robot.layer >= self.config.robot.max_layer:
            return None
        candidates: list[tuple[float, RobotState]] = []
        for other in self.robots:
            if other.robot_id == robot.robot_id or other.fallen or other.layer != robot.layer:
                continue
            relative = other.position - robot.position
            distance = float(np.linalg.norm(relative))
            if distance <= self.config.robot.length * 1.25 and relative @ robot.heading > 0.0:
                candidates.append((distance, other))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _highest_overlapping_support_layer(self, robot: RobotState) -> int | None:
        """Return the highest lower layer whose footprint overlaps the robot."""
        layers: list[int] = []
        for other in self.robots:
            if other.robot_id == robot.robot_id or other.fallen or other.layer >= robot.layer:
                continue
            if oriented_boxes_overlap(robot, other, self.config.robot):
                layers.append(other.layer)
        return max(layers) if layers else None

    def _auto_descend_unsupported(self) -> None:
        """Drop elevated robots onto the highest overlapping lower footprint."""
        while True:
            next_layers: dict[int, int] = {}
            for robot in self.robots:
                if robot.fallen or robot.layer == 0:
                    continue
                support_layer = self._highest_overlapping_support_layer(robot)
                next_layer = support_layer + 1 if support_layer is not None else 0
                if next_layer < robot.layer:
                    next_layers[robot.robot_id] = next_layer
            if not next_layers:
                return
            for robot_id, layer in next_layers.items():
                self.robots[robot_id].layer = layer

    def step(self, actions: np.ndarray | list[int]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(actions, dtype=np.int64).copy()
        if action_array.shape != (len(self.robots),):
            raise ValueError(f"Expected action shape {(len(self.robots),)}, got {action_array.shape}")
        masks = self.action_masks()
        if np.any(action_array < 0) or np.any(action_array >= ACTION_COUNT):
            raise ValueError("Action outside discrete action space")
        invalid = ~masks[np.arange(len(self.robots)), action_array]
        action_array[invalid] = int(DiscreteAction.IDLE)
        idle_violators = self._update_idle_streaks(action_array)

        old_progress = self.current_progress
        old_quality = min(self.current_capacity / max(self.target_load, 1.0e-9), 1.0)
        old_energy = sum(robot.energy for robot in self.robots)
        old_fallen = sum(robot.fallen for robot in self.robots)
        old_failures = self.contact_model.anchor_failures
        previously_used = len(self._used_robots)
        old_beacon_source = self._beacon_source_id
        old_beacon_distance = self._mean_beacon_distance()
        old_climb = self._climb_potential()
        newly_anchored = 0
        climb_layers = {
            robot.robot_id: min(robot.layer + 1, self.config.robot.max_layer)
            for robot, action_value in zip(self.robots, action_array)
            if DiscreteAction(int(action_value)) == DiscreteAction.CLIMB
            and self._climb_support(robot) is not None
        }
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
                climb_layer = climb_layers.get(robot.robot_id)
                if climb_layer is not None:
                    robot.layer = climb_layer
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

        self._auto_descend_unsupported()
        self.contact_model.resolve_contacts(self.robots)
        for robot in self.robots:
            self._constrain_to_field(robot)
        self._auto_descend_unsupported()
        self.graph = self.contact_model.build_graph(self.robots, self.field, self.rng)
        self._mark_fallen_robots()
        if not self.static_physics and sum(robot.fallen for robot in self.robots) > old_fallen:
            self.graph = self.contact_model.build_graph(self.robots, self.field, self.rng)
        self.current_progress = self._structure_progress()
        self.max_progress = max(self.max_progress, self.current_progress)
        fast_result = self._evaluate_load()
        self.current_capacity = fast_result.capacity
        self.step_count += 1
        self._update_histories()
        beacon_activated = self._update_beacon()

        accurate_capacity: float | None = None
        success = False
        time_limit = self.step_count >= self.config.max_steps
        if (self._structure_spans() and self.current_capacity >= self.target_load) or time_limit:
            self.phase = "load_test"
            self.last_load_test = self._evaluate_load(accurate=True)
            accurate_capacity = self.last_load_test.capacity
            self.current_capacity = accurate_capacity
            success = self._structure_spans() and accurate_capacity >= self.target_load
            self.phase = "complete" if success or time_limit else "construction"

        terminated = success
        self._episode_success = success
        truncated = time_limit and not terminated

        new_quality = min(self.current_capacity / max(self.target_load, 1.0e-9), 1.0)
        energy_delta = sum(robot.energy for robot in self.robots) - old_energy
        collapsed = sum(robot.fallen for robot in self.robots) - old_fallen
        new_beacon_distance = self._mean_beacon_distance()
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
            "beacon_discovery": reward_config.beacon_discovery_reward if beacon_activated else 0.0,
            "beacon_gather": reward_config.beacon_gather_delta
            * (
                0.0
                if old_beacon_source != self._beacon_source_id
                or old_beacon_distance is None
                or new_beacon_distance is None
                else old_beacon_distance - new_beacon_distance
            ),
            "idle": -reward_config.idle_penalty * idle_violators,
            "climb": reward_config.climb_delta * (self._climb_potential() - old_climb),
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

    def _update_idle_streaks(self, actions: np.ndarray) -> int:
        """Track local non-anchored waiting after the beacon has reached a robot."""
        violators = 0
        grace = max(self.config.reward.idle_grace_steps, 0)
        for robot, action in zip(self.robots, actions):
            robot_id = robot.robot_id
            signal_available = self._beacon_features[robot_id, 0] > 0.0
            if robot.fallen or robot.anchored or not signal_available or action != int(DiscreteAction.IDLE):
                self._idle_streak[robot_id] = 0
                continue
            self._idle_streak[robot_id] += 1
            violators += int(self._idle_streak[robot_id] > grace)
        return violators

    def _constrain_to_field(self, robot: RobotState) -> None:
        margin_x = self.config.robot.length / 2.0
        margin_y = self.config.robot.width / 2.0
        robot.position[0] = float(np.clip(robot.position[0], margin_x, self.field.length - margin_x))
        robot.position[1] = float(np.clip(robot.position[1], margin_y, self.field.width - margin_y))

    def _robot_below_anchored(self, robot: RobotState) -> bool:
        """Local cue: this robot stands on an anchored robot (felt through its underside)."""
        if robot.fallen or robot.layer < 1:
            return False
        return any(
            other.anchored and not other.fallen and other.layer == robot.layer - 1
            and oriented_boxes_overlap(robot, other, self.config.robot)
            for other in self.robots
            if other.robot_id != robot.robot_id
        )

    def _structure_cues(self, robot: RobotState) -> np.ndarray:
        """Local contact cues: [robot directly ahead is anchored, robot beneath is anchored]."""
        if robot.fallen:
            return np.zeros(CUE_SIZE, dtype=np.float32)
        ahead = self._climb_support(robot)
        return np.array(
            [float(ahead is not None and ahead.anchored), float(self._robot_below_anchored(robot))],
            dtype=np.float32,
        )

    def _climb_potential(self) -> float:
        reward_config = self.config.reward
        cap = max(reward_config.climb_cap, 0)
        standing = sum(self._robot_below_anchored(robot) for robot in self.robots)
        if reward_config.climb_counts_extended:
            extended = sum(self._extended(robot) for robot in self.robots)
            if reward_config.climb_standing_cap >= 0:
                return float(min(standing, reward_config.climb_standing_cap) + min(extended, cap))
            standing += extended
        return float(min(standing, cap))

    def _extended(self, robot: RobotState) -> bool:
        """A ground-level robot gripping another robot end-to-end over the gap (a chain link)."""
        return bool(
            robot.anchored and isinstance(robot.anchor_partner, int) and robot.layer == 0 and not robot.fallen
            and self.field.is_gap(robot.position)
        )

    def structure_metrics(self) -> dict[str, int]:
        """Counts for logging the construction steps."""
        return {
            "edge_grips": sum(r.anchored and r.anchor_partner == LEFT_BANK and not r.fallen for r in self.robots),
            "on_structure": sum(self._robot_below_anchored(r) for r in self.robots),
            "extended": sum(self._extended(r) for r in self.robots),
        }

    def _structure_spans(self) -> bool:
        """Bank-to-bank connection that can carry load under the active physics model."""
        return self._static_spans if self.static_physics else self.graph.spans

    def _structure_progress(self) -> float:
        """Span progress in [0, 1] used by the ``span`` reward (see reward.progress_measure)."""
        measure = self.config.reward.progress_measure
        if measure == "center":
            return self.graph.spanning_progress(self.robots, self.field)
        if measure == "frontier":
            if self.static_physics:
                supported = self._static_structure().reachable_from(LEFT_BANK)
            else:
                supported = {node for node in self.graph.connected_component(LEFT_BANK) if isinstance(node, int)}
        elif measure == "anchored_frontier":
            if self.static_physics:
                supported = self._static_structure().anchored_from(LEFT_BANK)
            else:
                supported = self._anchored_component(LEFT_BANK)
        else:
            raise ValueError(f"Unknown progress measure: {measure}")
        baseline = max(self.config.reward.progress_baseline, 0.0)
        reach = 0.0
        for robot_id in supported:
            robot = self.robots[robot_id]
            if robot.fallen:
                continue
            for corner in robot.corners(self.config.robot):
                longitudinal, tangent = self.field.coordinates(corner)
                left, right = self.field.boundaries(tangent)
                start = left - baseline
                reach = max(reach, float(np.clip((longitudinal - start) / max(right - start, 1.0e-6), 0.0, 1.0)))
        return reach

    def _anchored_component(self, bank: str) -> set[int]:
        """Legacy-physics counterpart of StaticStructure.anchored_from: anchor edges only."""
        reached: set[int] = set()
        frontier: list[int | str] = [bank]
        visited: set[int | str] = {bank}
        while frontier:
            node = frontier.pop()
            for edge in self.graph.edges.values():
                if edge.kind != "anchor" or node not in (edge.source, edge.target):
                    continue
                other = edge.target if edge.source == node else edge.source
                if other not in visited and isinstance(other, int):
                    visited.add(other)
                    reached.add(other)
                    frontier.append(other)
        return reached

    def _spawn_edge_limit(self) -> float:
        distances = self.config.spawn_edge_distance
        if not distances:
            return 0.0
        return float(distances[min(max(self.config.curriculum_stage - 1, 0), len(distances) - 1)])

    def _robot_weight(self) -> float:
        mass = self._episode_randomization.get("robot_mass", self.config.robot.mass)
        return float(mass) * self.config.physics.gravity

    def _static_structure(self):
        return build_structure(
            self.robots,
            self.field,
            self.contact_model,
            self.config.robot,
            self.config.physics,
            self._robot_weight(),
            float(self._episode_randomization.get("anchor_strength_scale", 1.0)),
        )

    def _evaluate_load(self, accurate: bool = False) -> LoadTestResult:
        if not self.static_physics:
            if accurate:
                return self.accurate_evaluator.evaluate(self.graph)
            return self.fast_evaluator.evaluate(self.graph, self.config.load)
        outcome = max_load(self._static_structure(), self.config.load.load_protocol, self.config.load.max_target)
        self._static_spans = outcome.spans
        return LoadTestResult(
            outcome.capacity,
            outcome.failure_mode,
            [(owner, partner) for owner, partner in outcome.critical_joints],
            outcome.load_path,
            0.0,
            self.config.load.load_protocol,
        )

    def _stabilize_static(self) -> None:
        """Break overloaded anchors and drop robots whose weight is not supported."""
        outcome = stabilize(self._static_structure())
        changed = False
        overloaded = set(outcome.broken_joints)
        for owner, partner in outcome.broken_joints + outcome.released_joints:
            anchor = self.contact_model.anchors.get(owner)
            if anchor is None:
                continue
            if anchor.partner != partner and not self.config.physics.splice_grip:
                continue
            # With splice grips a joint to another robot below belongs to the
            # same grip: when it gives way the robot lets go entirely.
            self.contact_model.release(self.robots[owner])
            if (owner, partner) in overloaded:
                self.contact_model.anchor_failures += 1
            changed = True
        for robot_id in outcome.unsupported:
            robot = self.robots[robot_id]
            self.contact_model.release(robot)
            if robot.layer > 0 and self.field.bank_at(robot.position) is not None:
                robot.layer = 0  # tipped off a lower robot onto the ground
            else:
                robot.fallen = True
                robot.velocity = 0.0
            changed = True
        for owner_id, anchor in list(self.contact_model.anchors.items()):
            if isinstance(anchor.partner, int) and self.robots[anchor.partner].fallen:
                self.contact_model.release(self.robots[owner_id])
                changed = True
        if changed:
            self.graph = self.contact_model.build_graph(self.robots, self.field, self.rng)
        strain_scale = max(self.contact_model.anchor_capacity, self.config.contact.contact_capacity, 1.0)
        for robot_id, utilisation in outcome.utilisation.items():
            robot = self.robots[robot_id]
            if not robot.fallen:
                robot.strain += strain_scale * utilisation

    def _mark_fallen_robots(self) -> None:
        if self.static_physics:
            self._stabilize_static()
            return
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

    def _update_beacon(self) -> bool:
        """Latch the first cliff detector, then distribute a bounded-hop local beacon."""
        if not self.config.beacon.enabled:
            self._beacon_features.fill(0.0)
            return False
        if self._beacon_source_id is not None and self.robots[self._beacon_source_id].fallen:
            self._beacon_source_id = None
            self._beacon_age = 0
            self._beacon_edge_direction.fill(0.0)

        activated = False
        if self._beacon_source_id is None:
            detections = [
                (robot.robot_id, reading)
                for robot in self.robots
                if not robot.fallen
                if (
                    reading := cliff_sensor_direction(
                        robot, self.robots, self.field, self.config.robot, self.config.sensor
                    )
                ) is not None
            ]
            if detections:
                source_id, (direction, _) = min(detections, key=lambda item: (item[1][1], item[0]))
                self._beacon_source_id = source_id
                self._beacon_edge_direction = direction
                self._beacon_age = 0
                activated = True

        self._beacon_features.fill(0.0)
        if self._beacon_source_id is None:
            return activated

        self._beacon_age += 1
        source_id = self._beacon_source_id
        hops, parents = self._beacon_relay_paths(source_id)
        for robot in self.robots:
            hop_count = hops.get(robot.robot_id)
            if robot.fallen or hop_count is None:
                continue
            if robot.robot_id == source_id:
                direction = self._beacon_edge_direction
            else:
                parent = self.robots[parents[robot.robot_id]]
                offset = parent.position - robot.position
                direction = offset / max(float(np.linalg.norm(offset)), 1.0e-9)
            left = np.array([-robot.heading[1], robot.heading[0]])
            self._beacon_features[robot.robot_id] = np.array(
                [
                    1.0,
                    float(direction @ robot.heading),
                    float(direction @ left),
                    1.0 - hop_count / (self.config.beacon.max_hops + 1.0),
                    hop_count / max(self.config.beacon.max_hops, 1),
                    float(robot.robot_id == source_id),
                ],
                dtype=np.float32,
            )
        return activated

    def _beacon_relay_paths(self, source_id: int) -> tuple[dict[int, int], dict[int, int]]:
        """Compute the same bounded-hop paths produced by local radio relays."""
        max_hops = self.config.beacon.max_hops
        communication_range = self.config.beacon.communication_range
        hops = {source_id: 0}
        parents: dict[int, int] = {}
        frontier = [source_id]
        while frontier:
            sender_id = frontier.pop(0)
            if hops[sender_id] >= max_hops:
                continue
            sender = self.robots[sender_id]
            for receiver in self.robots:
                if receiver.fallen or receiver.robot_id in hops:
                    continue
                if np.linalg.norm(receiver.position - sender.position) > communication_range:
                    continue
                hops[receiver.robot_id] = hops[sender_id] + 1
                parents[receiver.robot_id] = sender_id
                frontier.append(receiver.robot_id)
        return hops, parents

    def _mean_beacon_distance(self) -> float | None:
        if self._beacon_source_id is None:
            return None
        source = self.robots[self._beacon_source_id]
        receivers = [robot for robot in self.robots if not robot.fallen and robot.robot_id != source.robot_id]
        if not receivers:
            return None
        return float(np.mean([np.linalg.norm(robot.position - source.position) for robot in receivers]))

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
            if self.layout.beacon.stop > self.layout.beacon.start:
                values[index, self.layout.beacon] = self._beacon_features[index]
            if self.layout.cues.stop > self.layout.cues.start:
                values[index, self.layout.cues] = self._structure_cues(robot)
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
            "span": self._structure_spans(),
            "span_progress": self.current_progress,
            "max_span_progress": self.max_progress,
            "anchored_robots": sum(robot.anchored for robot in self.robots),
            "robots_used": len(self._used_robots),
            "energy": self._total_motion,
            "anchor_failures": self.contact_model.anchor_failures,
            "fallen_robots": sum(robot.fallen and not robot.absent for robot in self.robots),
            "active_robots": sum(not robot.absent for robot in self.robots),
            "structure": self.structure_metrics(),
            "beacon": {
                "active": self._beacon_source_id is not None,
                "source_robot": self._beacon_source_id,
                "age_steps": self._beacon_age,
                "reachable_robots": int(np.count_nonzero(self._beacon_features[:, 0])),
            },
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
                    "absent": robot.absent,
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
