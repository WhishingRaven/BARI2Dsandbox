#!/usr/bin/env python3
"""Interactively inspect BARI2D policy inference.

The viewer discovers checkpoints, lets the user switch between them with radio
buttons, and animates the selected shared policy in a randomized bridge
environment. It displays executed local actions and the selected robot's next
action distribution; no centralized critic state is used for policy inference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager, patches
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.robot import DiscreteAction
from bari2d.models.actor import SharedRecurrentActor
from bari2d.utils.config import ExperimentConfig, config_from_dict, load_config
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

ACTION_SHORT_LABELS = {
    DiscreteAction.FORWARD: "F",
    DiscreteAction.BACKWARD: "B",
    DiscreteAction.FORWARD_LEFT: "FL",
    DiscreteAction.FORWARD_RIGHT: "FR",
    DiscreteAction.BACKWARD_LEFT: "BL",
    DiscreteAction.BACKWARD_RIGHT: "BR",
    DiscreteAction.CLIMB: "C",
    DiscreteAction.ANCHOR: "A",
    DiscreteAction.RELEASE: "R",
    DiscreteAction.IDLE: "I",
}


def configure_plot_fonts() -> None:
    """Prefer a locally installed Korean-capable font without adding a dependency."""
    for family in ("Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.family"] = family
        break
    plt.rcParams["axes.unicode_minus"] = False


@dataclass
class LoadedModel:
    path: Path
    config: ExperimentConfig
    actor: SharedRecurrentActor


@dataclass
class PolicyPrediction:
    actions: np.ndarray
    probabilities: np.ndarray
    entropy: np.ndarray
    latents: dict[str, np.ndarray]


def resolved_device(name: str) -> torch.device:
    requested = torch.device(name)
    if requested.type == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def discover_checkpoints(explicit: list[Path], models_dir: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    candidates = [path.expanduser().resolve() for path in explicit]
    latest_by_run: dict[Path, Path] = {}
    if models_dir.exists():
        for path in models_dir.rglob("*.pt"):
            resolved = path.resolve()
            current = latest_by_run.get(resolved.parent)
            if current is None or resolved.stat().st_mtime > current.stat().st_mtime:
                latest_by_run[resolved.parent] = resolved
    candidates.extend(latest_by_run[parent] for parent in sorted(latest_by_run))
    for path in candidates:
        if path.is_file() and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def load_inference_model(
    checkpoint_path: Path,
    fallback_config: Path | None,
    device: torch.device,
) -> LoadedModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint payload: {checkpoint_path}")
    config_values = checkpoint.get("experiment_config")
    if isinstance(config_values, dict):
        config = config_from_dict(config_values)
    else:
        config = load_config(fallback_config)

    env = BridgeEnv(config.environment)
    actor = SharedRecurrentActor(env.layout, env.action_count, config.model).to(device)
    algorithm_state = checkpoint.get("algorithm", checkpoint)
    if not isinstance(algorithm_state, dict) or "actor" not in algorithm_state:
        raise KeyError(f"Checkpoint has no actor state: {checkpoint_path}")
    actor.load_state_dict(algorithm_state["actor"])
    actor.eval()
    return LoadedModel(checkpoint_path, config, actor)


class InferenceSession:
    """Policy-only episode state, deliberately separate from viewer controls."""

    def __init__(
        self,
        model: LoadedModel,
        device: torch.device,
        seed: int,
        deterministic: bool = True,
        stage: int | None = None,
        target_load: float | None = None,
    ):
        configure_plot_fonts()
        self.device = device
        self.seed = seed
        self.deterministic = deterministic
        self.model = model
        self.env: BridgeEnv
        self.observation: np.ndarray
        self.hidden: torch.Tensor
        self.info: dict[str, Any] = {}
        self.last_prediction: PolicyPrediction | None = None
        self.preview_prediction: PolicyPrediction | None = None
        self.last_reward = 0.0
        self.done = False
        self.load(model, stage=stage, target_load=target_load)

    def load(
        self,
        model: LoadedModel,
        stage: int | None = None,
        target_load: float | None = None,
    ) -> None:
        self.model = model
        self.env = BridgeEnv(model.config.environment)
        self.reset(stage=stage, target_load=target_load)

    def reset(self, stage: int | None = None, target_load: float | None = None) -> None:
        if stage is not None:
            self.env.config.curriculum_stage = int(np.clip(stage, 1, 4))
        self.observation, self.info = self.env.reset(seed=self.seed)
        if target_load is not None:
            self.set_target_load(target_load, refresh=False)
        self.hidden = self.model.actor.initial_hidden(self.env.config.robot.count, self.device)
        self.last_prediction = None
        self.last_reward = 0.0
        self.done = False
        self.preview_prediction = self.predict()

    def set_target_load(self, target_load: float, refresh: bool = True) -> None:
        limits = self.env.config.load
        self.env.target_load = float(np.clip(target_load, limits.target_min, limits.max_target))
        self.observation = self.env.observations()
        self.info = self.env.info()
        if refresh and hasattr(self, "hidden"):
            self.preview_prediction = self.predict()

    @torch.no_grad()
    def predict(self) -> PolicyPrediction:
        observations = torch.as_tensor(self.observation, dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(self.env.action_masks(), dtype=torch.bool, device=self.device)
        actions, _, entropy, output = self.model.actor.act(
            observations,
            self.hidden,
            masks,
            deterministic=self.deterministic,
        )
        probabilities = torch.softmax(output.logits, dim=-1)
        return PolicyPrediction(
            actions=actions.detach().cpu().numpy(),
            probabilities=probabilities.detach().cpu().numpy(),
            entropy=entropy.detach().cpu().numpy(),
            latents={name: value.detach().cpu().numpy() for name, value in output.latents.items()},
        )

    def step(self) -> bool:
        if self.done:
            return False
        prediction = self.predict()
        self.last_prediction = prediction
        observations = torch.as_tensor(self.observation, dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(self.env.action_masks(), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            output = self.model.actor.forward_step(
                observations,
                self.hidden,
                masks,
            )
        self.hidden = output.hidden
        self.observation, self.last_reward, terminated, truncated, self.info = self.env.step(prediction.actions)
        self.done = bool(terminated or truncated)
        self.preview_prediction = self.predict() if not self.done else None
        return not self.done


def checkpoint_labels(paths: list[Path]) -> dict[str, Path]:
    labels: dict[str, Path] = {}
    for index, path in enumerate(paths, start=1):
        try:
            display = str(path.relative_to(Path.cwd()))
        except ValueError:
            display = path.name
        if len(display) > 34:
            display = f"…{display[-33:]}"
        labels[f"{index}. {display}"] = path
    return labels


class InferenceViewer:
    def __init__(
        self,
        checkpoints: list[Path],
        fallback_config: Path | None,
        device: torch.device,
        seed: int,
        stage: int | None,
        target_load: float | None,
        deterministic: bool,
        interval_ms: int,
    ):
        self.device = device
        self.fallback_config = fallback_config
        self.labels = checkpoint_labels(checkpoints)
        self.models: dict[Path, LoadedModel] = {}
        initial_path = checkpoints[0]
        initial_model = self._model_for(initial_path)
        self.session = InferenceSession(
            initial_model,
            device,
            seed,
            deterministic=deterministic,
            stage=stage,
            target_load=target_load,
        )
        self.playing = False
        self._syncing = False
        height = max(8.0, 6.8 + 0.24 * len(checkpoints))
        self.figure = plt.figure(figsize=(17.0, height))
        self.environment_axis = self.figure.add_axes([0.26, 0.16, 0.53, 0.78])
        self.probability_axis = self.figure.add_axes([0.82, 0.54, 0.16, 0.34])
        self.info_axis = self.figure.add_axes([0.82, 0.17, 0.16, 0.29])
        self.model_axis = self.figure.add_axes([0.02, 0.48, 0.21, 0.45])
        self.robot_axis = self.figure.add_axes([0.04, 0.36, 0.18, 0.035])
        self.target_axis = self.figure.add_axes([0.04, 0.28, 0.18, 0.035])
        self.stage_axis = self.figure.add_axes([0.04, 0.20, 0.18, 0.035])
        self.play_axis = self.figure.add_axes([0.03, 0.08, 0.09, 0.065])
        self.step_axis = self.figure.add_axes([0.135, 0.08, 0.09, 0.065])
        self.reset_axis = self.figure.add_axes([0.03, 0.015, 0.09, 0.05])
        self.mode_axis = self.figure.add_axes([0.135, 0.005, 0.10, 0.065])

        label_names = list(self.labels)
        self.model_selector = RadioButtons(self.model_axis, label_names, active=0, activecolor="#286d8e")
        for label in self.model_selector.labels:
            label.set_fontsize(8)
        self.robot_slider = Slider(
            self.robot_axis,
            "robot",
            0,
            max(self.session.env.config.robot.count - 1, 1),
            valinit=0,
            valstep=1,
        )
        load_settings = self.session.env.config.load
        self.target_slider = Slider(
            self.target_axis,
            "target",
            load_settings.target_min,
            load_settings.max_target,
            valinit=self.session.env.target_load,
        )
        self.stage_slider = Slider(self.stage_axis, "stage", 1, 4, valinit=self.session.env.config.curriculum_stage, valstep=1)
        self.play_button = Button(self.play_axis, "재생")
        self.step_button = Button(self.step_axis, "1 step")
        self.reset_button = Button(self.reset_axis, "재설정")
        self.mode_check = CheckButtons(self.mode_axis, ["stochastic"], [not deterministic])
        for label in self.mode_check.labels:
            label.set_fontsize(8)

        self.model_selector.on_clicked(self._on_model_selected)
        self.robot_slider.on_changed(lambda _value: self.refresh())
        self.target_slider.on_changed(self._on_target_changed)
        self.stage_slider.on_changed(self._on_stage_changed)
        self.play_button.on_clicked(self._toggle_play)
        self.step_button.on_clicked(self._on_step)
        self.reset_button.on_clicked(self._on_reset)
        self.mode_check.on_clicked(self._on_mode_changed)

        self.timer = self.figure.canvas.new_timer(interval=interval_ms)
        self.timer.add_callback(self._on_timer)
        self.refresh()

    def _model_for(self, path: Path) -> LoadedModel:
        if path not in self.models:
            self.models[path] = load_inference_model(path, self.fallback_config, self.device)
        return self.models[path]

    def _selected_robot(self) -> int:
        return int(np.clip(round(self.robot_slider.val), 0, self.session.env.config.robot.count - 1))

    def _sync_controls(self) -> None:
        self._syncing = True
        count = self.session.env.config.robot.count
        self.robot_slider.valmax = max(count - 1, 1)
        self.robot_slider.ax.set_xlim(0, self.robot_slider.valmax)
        if self.robot_slider.val > count - 1:
            self.robot_slider.set_val(count - 1)
        load_settings = self.session.env.config.load
        self.target_slider.valmin = load_settings.target_min
        self.target_slider.valmax = load_settings.max_target
        self.target_slider.ax.set_xlim(load_settings.target_min, load_settings.max_target)
        self.target_slider.set_val(self.session.env.target_load)
        self.stage_slider.set_val(self.session.env.config.curriculum_stage)
        self._syncing = False

    def _on_model_selected(self, label: str) -> None:
        self.playing = False
        self.timer.stop()
        path = self.labels[label]
        self.session.load(
            self._model_for(path),
            stage=int(round(self.stage_slider.val)),
            target_load=float(self.target_slider.val),
        )
        self._sync_controls()
        self.refresh()

    def _on_target_changed(self, value: float) -> None:
        if self._syncing:
            return
        self.session.set_target_load(float(value))
        self.refresh()

    def _on_stage_changed(self, value: float) -> None:
        if self._syncing:
            return
        self.playing = False
        self.timer.stop()
        self.session.reset(stage=int(round(value)), target_load=float(self.target_slider.val))
        self.refresh()

    def _on_mode_changed(self, _label: str) -> None:
        self.session.deterministic = not self.mode_check.get_status()[0]
        self.session.preview_prediction = self.session.predict()
        self.refresh()

    def _toggle_play(self, _event: Any) -> None:
        self.playing = not self.playing
        self.play_button.label.set_text("일시정지" if self.playing else "재생")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _on_step(self, _event: Any) -> None:
        self.playing = False
        self.timer.stop()
        self.session.step()
        self.refresh()

    def _on_reset(self, _event: Any) -> None:
        self.playing = False
        self.timer.stop()
        self.play_button.label.set_text("재생")
        self.session.reset(stage=int(round(self.stage_slider.val)), target_load=float(self.target_slider.val))
        self.refresh()

    def _on_timer(self) -> None:
        if not self.playing:
            return
        if not self.session.step():
            self.playing = False
            self.timer.stop()
            self.play_button.label.set_text("재생")
        self.refresh()

    def refresh(self) -> None:
        env = self.session.env
        draw_environment(env, self.environment_axis)
        selected = self._selected_robot()
        robot = env.robots[selected]
        marker = patches.Circle(
            robot.position,
            radius=max(env.config.robot.length, env.config.robot.width) * 0.68,
            fill=False,
            edgecolor="#ff7f00",
            linewidth=2.4,
            zorder=8,
        )
        self.environment_axis.add_patch(marker)
        if self.session.last_prediction is not None:
            for state, action_value in zip(env.robots, self.session.last_prediction.actions):
                label = ACTION_SHORT_LABELS[DiscreteAction(int(action_value))]
                self.environment_axis.text(
                    state.position[0],
                    state.position[1] + env.config.robot.width * 0.75,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#1f1f1f",
                    zorder=9,
                )
        self.environment_axis.set_title(
            f"{self.session.model.path.name} | step={env.step_count} | "
            f"target={env.target_load:.2f} | capacity={env.current_capacity:.2f}"
        )

        prediction = self.session.preview_prediction or self.session.last_prediction
        self.probability_axis.clear()
        self.info_axis.clear()
        if prediction is not None:
            probabilities = prediction.probabilities[selected]
            actions = list(DiscreteAction)
            colors = ["#286d8e"] * len(actions)
            colors[int(prediction.actions[selected])] = "#ff7f00"
            self.probability_axis.bar(range(len(actions)), probabilities, color=colors)
            self.probability_axis.set_xticks(range(len(actions)), [ACTION_SHORT_LABELS[action] for action in actions])
            self.probability_axis.set_ylim(0.0, 1.0)
            self.probability_axis.set_ylabel("P(action)")
            self.probability_axis.set_title(f"robot {selected}: 다음 행동")
            action = DiscreteAction(int(prediction.actions[selected]))
            latent_lines = [
                f"{name}: {values[selected].mean():+.2f}"
                for name, values in prediction.latents.items()
            ]
            info_lines = [
                f"모델: {self.session.model.config.model.architecture}",
                f"stage: {env.config.curriculum_stage}",
                f"robot {selected}: {ACTION_LABELS[action]}",
                f"P={probabilities[int(action)]:.3f}",
                f"entropy={prediction.entropy[selected]:.3f}",
                f"reward={self.session.last_reward:+.3f}",
                f"span={env.graph.spans}",
                f"progress={env.current_progress:.3f}",
                f"phase={env.phase}",
            ]
            info_lines.extend(latent_lines)
            self.info_axis.text(0.0, 1.0, "\n".join(info_lines), va="top", fontsize=9)
        self.info_axis.axis("off")
        self.figure.canvas.draw_idle()

    def save(self, output: Path) -> None:
        self.refresh()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(output, dpi=160, bbox_inches="tight")

    def show(self) -> None:
        plt.show()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select BARI2D checkpoints and visualize decentralized policy inference."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        default=[],
        help="Checkpoint path. Repeat to populate the model selector.",
    )
    parser.add_argument("--models-dir", type=Path, default=Path("runs"), help="Recursively search this directory for .pt files.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"), help="Fallback config for legacy checkpoints.")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", type=int, choices=range(1, 5))
    parser.add_argument("--target-load", type=float)
    parser.add_argument("--interval-ms", type=int, default=150)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of choosing the policy argmax.")
    parser.add_argument("--no-show", action="store_true", help="Render a static PNG instead of opening the interactive viewer.")
    parser.add_argument("--steps", type=int, default=0, help="Inference steps before --no-show rendering.")
    parser.add_argument("--output", type=Path, default=Path("inference.png"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    checkpoints = discover_checkpoints(arguments.checkpoint, arguments.models_dir)
    if not checkpoints:
        raise SystemExit(
            "No checkpoint found. Train first or pass --checkpoint path/to/checkpoint.pt."
        )
    viewer = InferenceViewer(
        checkpoints,
        arguments.config,
        resolved_device(arguments.device),
        arguments.seed,
        arguments.stage,
        arguments.target_load,
        deterministic=not arguments.stochastic,
        interval_ms=arguments.interval_ms,
    )
    if arguments.no_show:
        for _ in range(max(arguments.steps, 0)):
            if not viewer.session.step():
                break
        viewer.save(arguments.output)
        print(arguments.output)
        plt.close(viewer.figure)
        return
    viewer.show()


if __name__ == "__main__":
    main()
