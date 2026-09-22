from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from bari2d.env.robot import DiscreteAction
from bari2d.utils.config import EnvironmentConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.manual import ManualSession, ManualViewer, configure_manual_keymap, normalize_control_key
from bari2d.utils.visualization import robot_layer_color


def test_default_max_layer_is_ten() -> None:
    assert EnvironmentConfig().robot.max_layer == 10


def test_layer_colors_distinguish_the_lowest_and_highest_layers() -> None:
    assert robot_layer_color(0, 10) != robot_layer_color(10, 10)


def test_manual_keymap_reserves_robot_control_keys() -> None:
    configure_manual_keymap()

    assert "s" not in plt.rcParams["keymap.save"]
    assert "q" not in plt.rcParams["keymap.quit"]
    assert plt.rcParams["toolbar"] == "None"


def test_manual_control_keys_accept_korean_input_source() -> None:
    assert normalize_control_key("W") == "w"
    assert normalize_control_key("ㅈ") == "w"
    assert normalize_control_key("ㄴ") == "s"
    assert normalize_control_key("ㅂ") == "q"
    assert normalize_control_key("cmd+w") == "w"


def test_manual_keyboard_controls_do_not_save_or_close_the_figure(tmp_path: Path) -> None:
    config = EnvironmentConfig()
    config.robot.count = 2
    session = ManualSession(config, seed=2)
    snapshot = tmp_path / "manual.png"
    viewer = ManualViewer(session, save_path=snapshot)

    viewer._on_key(SimpleNamespace(key="s"))
    assert session.env.step_count == 1
    assert not snapshot.exists()

    viewer._on_key(SimpleNamespace(key="q"))
    assert session.env.step_count == 2
    assert plt.fignum_exists(viewer.figure.number)

    viewer._save()
    assert snapshot.is_file()
    viewer._close()


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
