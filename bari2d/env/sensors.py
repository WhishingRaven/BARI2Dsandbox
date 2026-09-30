from __future__ import annotations

import numpy as np

from bari2d.env.contact_model import oriented_boxes_overlap
from bari2d.env.field import GapField
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


def ir_distances(
    robot: RobotState,
    robots: list[RobotState],
    field: GapField,
    robot_config: RobotConfig,
    sensor_config: SensorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    readings = []
    origin_bank = field.bank_at(robot.position)
    for angle_deg in sensor_config.ir_angles_deg:
        angle = robot.theta + np.deg2rad(angle_deg)
        direction = np.array([np.cos(angle), np.sin(angle)])
        measured = sensor_config.ir_range
        for distance in np.arange(sensor_config.ir_step, sensor_config.ir_range + sensor_config.ir_step, sensor_config.ir_step):
            point = robot.position + direction * distance
            if not field.inside(point):
                measured = distance
                break
            bank = field.bank_at(point)
            if bank != origin_bank and (bank is None or origin_bank is None):
                measured = distance
                break
            obstacle_hit = any(
                np.linalg.norm(point - np.array([x, y])) <= radius for x, y, radius in field.obstacles
            )
            robot_hit = any(
                other.robot_id != robot.robot_id
                and not other.fallen
                and abs(other.layer - robot.layer) <= 1
                and _point_in_robot(point, other, robot_config)
                for other in robots
            )
            if obstacle_hit or robot_hit:
                measured = distance
                break
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
        for distance in np.arange(sensor_config.ir_step, sensor_config.ir_range + sensor_config.ir_step, sensor_config.ir_step):
            point = robot.position + direction * distance
            if not field.inside(point):
                break
            if field.bank_at(point) is None:
                if nearest is None or distance < nearest[1]:
                    nearest = (direction, float(distance))
                break
            obstacle_hit = any(
                np.linalg.norm(point - np.array([x, y])) <= radius for x, y, radius in field.obstacles
            )
            robot_hit = any(
                other.robot_id != robot.robot_id
                and not other.fallen
                and abs(other.layer - robot.layer) <= 1
                and _point_in_robot(point, other, robot_config)
                for other in robots
            )
            if obstacle_hit or robot_hit:
                break
    return nearest
