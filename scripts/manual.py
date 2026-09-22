#!/usr/bin/env python3
"""Manually control one BARI2D robot per simulation step.

The selected robot receives the requested discrete action; every other robot
receives IDLE. This keeps the tool faithful to BridgeEnv's multi-agent action
interface while making individual construction experiments easy to inspect.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import matplotlib

# The native macOS canvas can leave keyboard focus with the window chrome or a
# toolbar.  The Tk canvas has an explicit focus API, which makes the manual
# controller reliable when it is launched as a program.  Do not override an
# explicitly requested backend: ``MPLBACKEND=Agg`` remains the headless path.
if __name__ == "__main__" and sys.platform == "darwin" and not os.environ.get("MPLBACKEND"):
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, patches
from matplotlib.widgets import Button

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.robot import DiscreteAction
from bari2d.utils.config import EnvironmentConfig, load_config
from bari2d.utils.visualization import draw_environment


ACTION_LABELS = {
    DiscreteAction.FORWARD: "전진",
    DiscreteAction.BACKWARD: "후진",
    DiscreteAction.FORWARD_LEFT: "전진 좌",
    DiscreteAction.FORWARD_RIGHT: "전진 우",
    DiscreteAction.BACKWARD_LEFT: "후진 좌",
    DiscreteAction.BACKWARD_RIGHT: "후진 우",
    DiscreteAction.CLIMB: "등반",
    DiscreteAction.ANCHOR: "앵커",
    DiscreteAction.RELEASE: "해제",
    DiscreteAction.IDLE: "대기",
}

KEY_ACTIONS = {
    "w": DiscreteAction.FORWARD,
    "s": DiscreteAction.BACKWARD,
    "a": DiscreteAction.FORWARD_LEFT,
    "d": DiscreteAction.FORWARD_RIGHT,
    "z": DiscreteAction.BACKWARD_LEFT,
    "c": DiscreteAction.BACKWARD_RIGHT,
    "e": DiscreteAction.CLIMB,
    "q": DiscreteAction.ANCHOR,
    "r": DiscreteAction.RELEASE,
    " ": DiscreteAction.IDLE,
}

# Matplotlib reports the currently active input character.  Accepting the
# corresponding two-beolsik jamo means the controller still works when the
# operating-system input source is Korean instead of English.
KOREAN_KEY_ALIASES = {
    "ㅈ": "w",
    "ㄴ": "s",
    "ㅁ": "a",
    "ㅇ": "d",
    "ㅋ": "z",
    "ㅊ": "c",
    "ㄷ": "e",
    "ㅂ": "q",
    "ㄱ": "r",
}

BUTTON_ACTIONS = (
    DiscreteAction.FORWARD,
    DiscreteAction.BACKWARD,
    DiscreteAction.FORWARD_LEFT,
    DiscreteAction.FORWARD_RIGHT,
    DiscreteAction.BACKWARD_LEFT,
    DiscreteAction.BACKWARD_RIGHT,
    DiscreteAction.CLIMB,
    DiscreteAction.ANCHOR,
    DiscreteAction.RELEASE,
    DiscreteAction.IDLE,
)


def configure_plot_fonts() -> None:
    """Use an installed Korean-capable font when one is available."""
    for family in ("Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.family"] = family
        break
    plt.rcParams["axes.unicode_minus"] = False


def configure_manual_keymap() -> None:
    """Reserve S and Q for robot control instead of Matplotlib shortcuts."""
    for keymap in ("keymap.save", "keymap.quit", "keymap.quit_all", "keymap.home"):
        plt.rcParams[keymap] = []


def normalize_control_key(key: str | None) -> str | None:
    """Map case and Korean two-beolsik input to a controller key."""
    if key is None:
        return None
    normalized = key.lower()
    return KOREAN_KEY_ALIASES.get(normalized, normalized)


@dataclass
class ManualStep:
    action: DiscreteAction
    reward: float
    done: bool


class ManualSession:
    """UI-independent human-control state for one BridgeEnv episode."""

    def __init__(
        self,
        environment: EnvironmentConfig,
        seed: int,
        stage: int | None = None,
        target_load: float | None = None,
    ):
        self.env = BridgeEnv(environment)
        self.seed = seed
        self.stage = stage
        self.target_load = target_load
        self.selected_robot = 0
        self.last_step = ManualStep(DiscreteAction.IDLE, 0.0, False)
        self.observation: np.ndarray
        self.info: dict[str, object]
        self.reset()

    @property
    def done(self) -> bool:
        return self.last_step.done

    def reset(self) -> None:
        if self.stage is not None:
            self.env.config.curriculum_stage = int(np.clip(self.stage, 1, 4))
        self.observation, self.info = self.env.reset(seed=self.seed)
        if self.target_load is not None:
            limits = self.env.config.load
            self.env.target_load = float(np.clip(self.target_load, limits.target_min, limits.max_target))
            self.observation = self.env.observations()
            self.info = self.env.info()
        self.selected_robot = min(self.selected_robot, len(self.env.robots) - 1)
        self.last_step = ManualStep(DiscreteAction.IDLE, 0.0, False)

    def select_robot(self, robot_id: int) -> None:
        if not 0 <= robot_id < len(self.env.robots):
            raise ValueError(f"Robot ID must be in [0, {len(self.env.robots) - 1}], got {robot_id}")
        self.selected_robot = robot_id

    def select_offset(self, offset: int) -> None:
        self.selected_robot = (self.selected_robot + offset) % len(self.env.robots)

    def step(self, action: DiscreteAction | int) -> ManualStep:
        requested = DiscreteAction(int(action))
        if self.done:
            return self.last_step
        actions = np.full(len(self.env.robots), int(DiscreteAction.IDLE), dtype=np.int64)
        actions[self.selected_robot] = int(requested)
        self.observation, reward, terminated, truncated, self.info = self.env.step(actions)
        executed = DiscreteAction(self.env.robots[self.selected_robot].previous_action)
        self.last_step = ManualStep(executed, reward, bool(terminated or truncated))
        return self.last_step


class ManualViewer:
    """Matplotlib keyboard and mouse controls over a ManualSession."""

    def __init__(self, session: ManualSession, save_path: Path = Path("manual.png")):
        configure_plot_fonts()
        configure_manual_keymap()
        self.session = session
        self.save_path = save_path
        self.status_message = ""
        self.figure = plt.figure(figsize=(15, 8.5))
        self.world_axis = self.figure.add_axes((0.05, 0.08, 0.64, 0.86))
        self.status_axis = self.figure.add_axes((0.72, 0.48, 0.25, 0.46))
        self.status_axis.axis("off")
        self.buttons: list[Button] = []
        self._make_buttons()
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.redraw()
        self._focus_keyboard()

    def _make_buttons(self) -> None:
        for index, action in enumerate(BUTTON_ACTIONS):
            column, row = divmod(index, 5)
            axis = self.figure.add_axes((0.72 + column * 0.125, 0.33 - row * 0.055, 0.115, 0.044))
            key = next((name for name, value in KEY_ACTIONS.items() if value == action), "")
            key_label = "Space" if key == " " else key
            label = f"{key_label or 'space'}  {ACTION_LABELS[action]}"
            button = Button(axis, label)
            button.on_clicked(lambda _event, selected=action: self._execute(selected))
            self.buttons.append(button)
        reset_axis = self.figure.add_axes((0.72, 0.045, 0.075, 0.05))
        reset = Button(reset_axis, "다시 시작")
        reset.on_clicked(lambda _event: self._reset())
        self.buttons.append(reset)
        save_axis = self.figure.add_axes((0.803, 0.045, 0.075, 0.05))
        save = Button(save_axis, "저장")
        save.on_clicked(lambda _event: self._save())
        self.buttons.append(save)
        close_axis = self.figure.add_axes((0.886, 0.045, 0.075, 0.05))
        close = Button(close_axis, "종료")
        close.on_clicked(lambda _event: self._close())
        self.buttons.append(close)

    def _on_key(self, event) -> None:
        key = normalize_control_key(event.key)
        if key in ("left", "["):
            self.session.select_offset(-1)
        elif key in ("right", "]"):
            self.session.select_offset(1)
        elif key == "home":
            self._reset()
            return
        elif key == "escape":
            self._close()
            return
        elif key in KEY_ACTIONS:
            self._execute(KEY_ACTIONS[key])
            return
        else:
            return
        self.redraw()

    def _on_click(self, event) -> None:
        if event.inaxes is not self.world_axis or event.xdata is None or event.ydata is None:
            return
        self._focus_keyboard()
        point = np.array([event.xdata, event.ydata])
        positions = np.array([robot.position for robot in self.session.env.robots])
        distances = np.linalg.norm(positions - point, axis=1)
        selected = int(np.argmin(distances))
        if distances[selected] <= self.session.env.config.robot.length:
            self.session.select_robot(selected)
            self.redraw()

    def _execute(self, action: DiscreteAction) -> None:
        self.session.step(action)
        self.redraw()
        self._focus_keyboard()

    def _reset(self) -> None:
        self.session.reset()
        self.status_message = "에피소드를 다시 시작했습니다."
        self.redraw()
        self._focus_keyboard()

    def _save(self) -> None:
        self.save(self.save_path)
        self.status_message = f"저장됨: {self.save_path}"
        self.redraw()
        self._focus_keyboard()

    def _close(self) -> None:
        plt.close(self.figure)

    def _focus_keyboard(self) -> None:
        """Return focus to the interactive canvas after a pointer action.

        ``FigureCanvasTkAgg`` exposes the actual Tk widget.  Other backends
        (including the Agg backend used for headless rendering) deliberately
        have no focus operation, so this is safely a no-op there.
        """
        get_widget = getattr(self.figure.canvas, "get_tk_widget", None)
        if get_widget is None:
            return
        widget = get_widget()
        widget.focus_set()
        widget.after_idle(widget.focus_set)

    def redraw(self) -> None:
        env = self.session.env
        draw_environment(env, self.world_axis)
        selected = env.robots[self.session.selected_robot]
        selected_outline = patches.Circle(
            selected.position,
            radius=env.config.robot.length * 0.72,
            fill=False,
            edgecolor="#ff7f00",
            linewidth=2.7,
            zorder=10,
        )
        self.world_axis.add_patch(selected_outline)
        for robot in env.robots:
            self.world_axis.text(
                *robot.position,
                f"{robot.robot_id}\nL{robot.layer}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if robot.fallen else "black",
                zorder=11,
            )
        self.world_axis.set_title(
            f"수동 조종 · 선택 로봇 {selected.robot_id} · step {env.step_count}/{env.config.max_steps}"
        )

        mask = env.action_masks()[self.session.selected_robot]
        allowed = ", ".join(ACTION_LABELS[DiscreteAction(index)] for index, enabled in enumerate(mask) if enabled)
        state = "종료됨" if self.session.done else "진행 중"
        self.status_axis.clear()
        self.status_axis.axis("off")
        self.status_axis.text(
            0.0,
            1.0,
            "\n".join(
                [
                    f"상태: {state}",
                    f"선택: 로봇 {selected.robot_id}",
                    f"위치: ({selected.position[0]:.2f}, {selected.position[1]:.2f})",
                    f"방위: {np.rad2deg(selected.theta):.1f}°",
                    f"층: {selected.layer} / {env.config.robot.max_layer}",
                    f"앵커: {selected.anchored}",
                    f"낙하: {selected.fallen}",
                    f"strain: {selected.strain:.3f}",
                    f"직전 action: {ACTION_LABELS[self.session.last_step.action]}",
                    f"직전 보상: {self.session.last_step.reward:.3f}",
                    "",
                    "로봇 선택: 클릭 또는 ←/→, [ / ]",
                    "운동: W/S, A/D, Z/C",
                    "구조: E 등반, Q 앵커, R 해제",
                    "대기: Space · 재시작: Home",
                    "저장·종료: 화면 하단 버튼",
                    "키보드: 지도 영역을 클릭하면 조종 입력에 포커스됩니다.",
                    "",
                    f"현재 허용: {allowed}",
                    "지지 범위를 벗어나면 자동 하강합니다.",
                    self.status_message,
                ]
            ),
            va="top",
            fontsize=9,
            wrap=True,
        )
        self.figure.canvas.draw_idle()

    def save(self, path: Path) -> None:
        self.figure.savefig(path, dpi=160)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively control BARI2D robots.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", type=int, choices=range(1, 5))
    parser.add_argument("--target-load", type=float)
    parser.add_argument("--output", type=Path, help="Save the initial or current control view to this PNG path.")
    parser.add_argument("--no-show", action="store_true", help="Render once and exit; useful for headless smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    session = ManualSession(
        config.environment,
        seed=args.seed,
        stage=args.stage,
        target_load=args.target_load,
    )
    viewer = ManualViewer(session, save_path=args.output or Path("manual.png"))
    if args.output is not None:
        viewer.save(args.output)
        print(f"Saved manual control view to {args.output}")
    if not args.no_show:
        plt.show()
    plt.close(viewer.figure)


if __name__ == "__main__":
    main()
