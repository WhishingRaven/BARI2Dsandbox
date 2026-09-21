from __future__ import annotations

import numpy as np

from bari2d.env.field import GapField
from bari2d.env.robot import RobotState
from bari2d.utils.config import RobotConfig, SensorConfig


def _point_in_robot(point: np.ndarray, robot: RobotState, config: RobotConfig) -> bool:
    relative = point - robot.position
    cosine, sine = np.cos(robot.theta), np.sin(robot.theta)
    local = np.array([cosine * relative[0] + sine * relative[1], -sine * relative[0] + cosine * relative[1]])
    return abs(local[0]) <= config.length / 2.0 and abs(local[1]) <= config.width / 2.0


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
    return np.asarray(readings, dtype=np.float32)
