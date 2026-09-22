from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib import colors as matplotlib_colors, patches
import numpy as np
import pytest

from bari2d.env.robot import DiscreteAction
from bari2d.utils.config import EnvironmentConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.manual import ManualSession, ManualViewer, configure_manual_keymap, normalize_control_key
from bari2d.utils.visualization import draw_environment, robot_layer_color


def test_default_max_layer_is_ten() -> None:
    assert EnvironmentConfig().robot.max_layer == 10


def test_layer_colors_follow_the_visible_spectrum_and_are_opaque() -> None:
    layer_colors = [robot_layer_color(layer, 10) for layer in range(11)]
    hues = [matplotlib_colors.rgb_to_hsv(color[:3])[0] for color in layer_colors]

    assert hues == pytest.approx(np.linspace(0.0, 0.75, 11))
    assert all(color[3] == 1.0 for color in layer_colors)


def test_higher_layer_robot_is_drawn_over_every_lower_layer_detail() -> None:
    config = EnvironmentConfig()
    config.robot.count = 2
    session = ManualSession(config, seed=9)
    higher, lower = session.env.robots
    higher.position = lower.position.copy()
    higher.layer = 1
    lower.layer = 0
    session.env.set_robot_states(session.env.robots)
    figure, axis = plt.subplots()

    draw_environment(session.env, axis)

    robot_rectangles = [artist for artist in axis.patches if isinstance(artist, patches.Rectangle)]
    assert len(robot_rectangles) == 2
    assert robot_rectangles[0].get_facecolor() == pytest.approx(robot_layer_color(0, 10))
    assert robot_rectangles[1].get_facecolor() == pytest.approx(robot_layer_color(1, 10))
    assert all(rectangle.get_alpha() == 1.0 for rectangle in robot_rectangles)
    lower_detail_zorders = [artist.get_zorder() for artist in axis.lines[-10:-5]]
    lower_detail_zorders.append(axis.collections[-2].get_zorder())
    assert robot_rectangles[1].get_zorder() > max(lower_detail_zorders)
    plt.close(figure)


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
