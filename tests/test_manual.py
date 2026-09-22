from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from bari2d.env.robot import DiscreteAction
from bari2d.utils.config import EnvironmentConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.manual import ManualSession
from bari2d.utils.visualization import robot_layer_color


def test_default_max_layer_is_ten() -> None:
    assert EnvironmentConfig().robot.max_layer == 10


def test_layer_colors_distinguish_the_lowest_and_highest_layers() -> None:
    assert robot_layer_color(0, 10) != robot_layer_color(10, 10)


def test_manual_session_only_commands_the_selected_robot() -> None:
    config = EnvironmentConfig()
    config.robot.count = 4
    config.max_steps = 10
    session = ManualSession(config, seed=4)
    session.select_robot(2)

    result = session.step(DiscreteAction.FORWARD)

    assert result.action == DiscreteAction.FORWARD
    assert session.env.robots[2].previous_action == DiscreteAction.FORWARD
    assert all(robot.previous_action == DiscreteAction.IDLE for robot in session.env.robots[:2])
    assert session.env.robots[3].previous_action == DiscreteAction.IDLE
    assert np.linalg.norm(session.env.robots[2].position - session.env.robots[0].position) > 0.0
