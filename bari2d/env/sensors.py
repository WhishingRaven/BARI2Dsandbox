from __future__ import annotations

import numpy as np

from bari2d.env.contact_model import oriented_boxes_overlap
from bari2d.env.field import LEFT_BANK, RIGHT_BANK, GapField
from bari2d.env.robot import RobotState
from bari2d.utils.config import RobotConfig, SensorConfig


def ir_ray_count(sensor_config: SensorConfig) -> int:
    """Return all planar rays plus the optional vertical, downward ray."""
    return len(sensor_config.ir_angles_deg) + int(sensor_config.downward_ir_enabled)


def _point_in_robot(point: np.ndarray, robot: RobotState, config: RobotConfig) -> bool:
    relative = point - robot.position
    cosine, sine = np.cos(robot.theta), np.sin(robot.theta)
    local = np.array([cosine * relative[0] + sine * relative[1], -sine * relative[0] + cosine * relative[1]])
    return abs(local[0]) <= config.length / 2.0 and abs(local[1]) <= config.width / 2.0


def downward_ir_point(robot: RobotState, sensor_config: SensorConfig) -> np.ndarray:
    """Planar position of the downward (belly) IR: sensor.downward_ir_offset ahead of the centre."""
    return robot.position + robot.heading * float(sensor_config.downward_ir_offset)


def _downward_ir_distance(
    robot: RobotState,
    robots: list[RobotState],
    field: GapField,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
) -> float:
    """Measure the nearest surface below the robot in the 2.5D approximation.

    Centre sensor (downward_ir_offset 0): a bank under the robot center is
    ground and a lower-layer robot is a surface where its footprint overlaps
    the sensing robot.  Offset sensor: the reading is taken at the sensor point
    only - a bank or a lower-layer robot directly under that point.  No surface
    is a cliff return at the maximum sensor range.
    """
    support_distances: list[float] = []
    if sensor_config.downward_ir_offset == 0.0:
        if field.bank_at(robot.position) is not None:
            support_distances.append(robot.layer * robot_config.climb_height)
        support_distances.extend(
            (robot.layer - other.layer) * robot_config.climb_height
            for other in robots
            if other.robot_id != robot.robot_id
            and not other.fallen
            and other.layer < robot.layer
            and oriented_boxes_overlap(robot, other, robot_config)
        )
    else:
        point = downward_ir_point(robot, sensor_config)
        if field.bank_at(point) is not None:
            support_distances.append(robot.layer * robot_config.climb_height)
        support_distances.extend(
            (robot.layer - other.layer) * robot_config.climb_height
            for other in robots
            if other.robot_id != robot.robot_id
            and not other.fallen
            and other.layer < robot.layer
            and _point_in_robot(point, other, robot_config)
        )
    measured = min(support_distances, default=sensor_config.ir_range)
    return float(np.clip(measured / sensor_config.ir_range, 0.0, 1.0))


def front_over_cliff(
    robot: RobotState,
    robots: list[RobotState],
    field: GapField,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
) -> bool:
    """The design's bank-grip cue: the belly IR sees nothing below while the body centre is still on a bank."""
    if robot.fallen or robot.layer != 0 or field.bank_at(robot.position) is None:
        return False
    return _downward_ir_distance(robot, robots, field, robot_config, sensor_config) >= 1.0


def _ray_points(origin: np.ndarray, direction: np.ndarray, sensor_config: SensorConfig) -> tuple[np.ndarray, np.ndarray]:
    """The ray-march sample distances and points (same steps as a scalar march)."""
    distances = np.arange(sensor_config.ir_step, sensor_config.ir_range + sensor_config.ir_step, sensor_config.ir_step)
    return distances, origin + direction * distances[:, None]


def _bank_codes(field: GapField, points: np.ndarray) -> np.ndarray:
    """Vectorised GapField.bank_at: 0 = no bank (gap or outside), 1 = left bank, 2 = right bank."""
    x, y = points[:, 0], points[:, 1]
    outside = (x < 0.0) | (x > field.length) | (y < 0.0) | (y > field.width)
    relative = points - field.center
    normal, tangent = field.normal, field.tangent
    longitudinal = relative[:, 0] * normal[0] + relative[:, 1] * normal[1]
    lateral = relative[:, 0] * tangent[0] + relative[:, 1] * tangent[1]
    perturbation = field.irregularity * np.sin(field.irregularity_frequency * lateral + field.phase)
    left = -field.gap_width / 2.0 + perturbation
    right = field.gap_width / 2.0 + perturbation
    codes = np.where(longitudinal <= left, 1, np.where(longitudinal >= right, 2, 0))
    return np.where(outside, 0, codes)


def _outside(field: GapField, points: np.ndarray) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    return ~((0.0 <= x) & (x <= field.length) & (0.0 <= y) & (y <= field.width))


def _blocked(points: np.ndarray, robot: RobotState, robots: list[RobotState], field: GapField, robot_config: RobotConfig) -> np.ndarray:
    """Points hitting an obstacle or the footprint of another robot within one layer."""
    blocked = np.zeros(len(points), dtype=bool)
    for x, y, radius in field.obstacles:
        blocked |= np.hypot(points[:, 0] - x, points[:, 1] - y) <= radius
    half_length, half_width = robot_config.length / 2.0, robot_config.width / 2.0
    for other in robots:
        if other.robot_id == robot.robot_id or other.fallen or abs(other.layer - robot.layer) > 1:
            continue
        relative_x = points[:, 0] - other.position[0]
        relative_y = points[:, 1] - other.position[1]
        cosine, sine = np.cos(other.theta), np.sin(other.theta)
        local_x = cosine * relative_x + sine * relative_y
        local_y = -sine * relative_x + cosine * relative_y
        blocked |= (np.abs(local_x) <= half_length) & (np.abs(local_y) <= half_width)
    return blocked


def ir_distances(
    robot: RobotState,
    robots: list[RobotState],
    field: GapField,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Planar IR rays (marched in ir_step steps, evaluated for all steps at once) plus the downward ray."""
    readings = []
    origin_bank = field.bank_at(robot.position)
    origin_code = {None: 0, LEFT_BANK: 1, RIGHT_BANK: 2}[origin_bank]
    for angle_deg in sensor_config.ir_angles_deg:
        angle = robot.theta + np.deg2rad(angle_deg)
        direction = np.array([np.cos(angle), np.sin(angle)])
        distances, points = _ray_points(robot.position, direction, sensor_config)
        codes = _bank_codes(field, points)
        stop = _outside(field, points)
        stop |= (codes != origin_code) & ((codes == 0) | (origin_code == 0))
        stop |= _blocked(points, robot, robots, field, robot_config)
        measured = float(distances[int(np.argmax(stop))]) if stop.any() else sensor_config.ir_range
        measured += float(rng.normal(0.0, sensor_config.sensor_noise))
        readings.append(float(np.clip(measured / sensor_config.ir_range, 0.0, 1.0)))
    if sensor_config.downward_ir_enabled:
        downward = _downward_ir_distance(robot, robots, field, robot_config, sensor_config)
        downward += float(rng.normal(0.0, sensor_config.sensor_noise))
        readings.append(float(np.clip(downward, 0.0, 1.0)))
    return np.asarray(readings, dtype=np.float32)


def cliff_sensor_direction(
    robot: RobotState,
    robots: list[RobotState],
    field: GapField,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
) -> tuple[np.ndarray, float] | None:
    """Return the nearest locally visible bank-to-gap direction, if any.

    This is a planar IR-style ray test. It only asks whether a ray originating
    at this robot crosses from its current bank into the gap.  It uses the
    same obstacle and robot occlusion rules as the planar IR sensor; other
    robots are only local line-of-sight blockers, never global observations.
    """
    origin_bank = field.bank_at(robot.position)
    if origin_bank is None:
        return None
    nearest: tuple[np.ndarray, float] | None = None
    for angle_deg in sensor_config.ir_angles_deg:
        angle = robot.theta + np.deg2rad(angle_deg)
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        distances, points = _ray_points(robot.position, direction, sensor_config)
        outside = _outside(field, points)
        gap = ~outside & (_bank_codes(field, points) == 0)
        blocked = ~outside & ~gap & _blocked(points, robot, robots, field, robot_config)
        stop = outside | gap | blocked
        if not stop.any():
            continue
        first = int(np.argmax(stop))
        if gap[first]:
            distance = float(distances[first])
            if nearest is None or distance < nearest[1]:
                nearest = (direction, distance)
    return nearest
