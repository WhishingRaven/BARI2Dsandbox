#!/usr/bin/env python3
"""Self-recovery (resilience) evaluation: inject one fault at a random time into
random modules and count the episodes that still complete the task.

    python scripts/evaluate_resilience.py --checkpoint runs/beacon_structured_mappo_static_v14/checkpoint_007550.pt ^
        --stage 11 --seeds 0 50 --faults none power comms sensor decouple --json runs/resilience_stage11.json

Faults (one per episode).  The fault time is drawn uniformly between 10% and 90%
of the same seed's fault-free episode (--fault-timing baseline, the default), so
every fault lands while the task is still in progress; --fault-timing window draws
it from --fault-window instead.
  none      no fault (baseline with the same seeds)
  power     power cut: --fraction (10-20%) of the robots taking part stop for good
            (their actions are IDLE; --power-mode hold keeps their grip, release lets go)
  comms     communication delay: those robots receive the beacon signal --delay
            (50-100) steps late from the fault on (0.1 s per step -> >= 5 s)
  sensor    sensor failure: those robots' internal state channels (velocity, steering,
            layer: the IMU-like proprioception) read 0 from the fault on
  decouple  coupling release: one robot-to-robot joint of the structure is released
            (the first step at or after the fault time with such a joint; the bank grip
            if the structure has no robot-to-robot joint yet)
A seed fixes the start state, the sampled actions and the fault, so the faulted
episode follows its baseline exactly until the fault.  An episode that succeeds
before its fault was injected is not counted for that fault ("not injected").
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.robot import DiscreteAction
from bari2d.models.actor import SharedRecurrentActor
from bari2d.utils.config import config_from_dict, load_config

FAULTS = ("none", "power", "comms", "sensor", "decouple")
IDLE = int(DiscreteAction.IDLE)
# Internal state channels: anchored, head_lifted, velocity, steering, layer, fallen.
IMU_CHANNELS = (2, 3, 4)


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return float("nan"), float("nan")
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return max(0.0, centre - half), min(1.0, centre + half)


class FaultInjector:
    def __init__(self, env: BridgeEnv, kind: str, rng: np.random.Generator, arguments: argparse.Namespace,
                 window: tuple[int, int]):
        self.env, self.kind, self.rng = env, kind, rng
        low, high = window
        self.time = int(rng.integers(low, max(high, low) + 1))
        self.fraction = float(rng.uniform(*arguments.fraction))
        self.delay = int(rng.integers(arguments.delay[0], arguments.delay[1] + 1))
        self.power_mode = arguments.power_mode
        self.affected: list[int] = []
        self.injected_at: int | None = None
        self.detail = ""
        self.beacon_history: deque = deque(maxlen=self.delay + 1)

    def _pick_robots(self) -> list[int]:
        candidates = [robot.robot_id for robot in self.env.robots if not robot.absent and not robot.fallen]
        count = max(1, int(round(self.fraction * sum(not robot.absent for robot in self.env.robots))))
        count = min(count, len(candidates))
        return sorted(int(r) for r in self.rng.choice(candidates, size=count, replace=False)) if candidates else []

    def before_step(self) -> None:
        """Called before each env.step: inject the fault when its time has come."""
        env = self.env
        if self.kind == "none" or self.injected_at is not None or env.step_count < self.time:
            return
        if self.kind == "decouple":
            joints = [r for r in env.robots if r.anchored and isinstance(r.anchor_partner, int) and not r.fallen]
            if not joints:
                joints = [r for r in env.robots if r.anchored and not r.fallen]
            if not joints:
                return  # nothing to release yet; try again next step
            owner = joints[int(self.rng.integers(len(joints)))]
            self.detail = f"released robot {owner.robot_id} from {owner.anchor_partner}"
            env.contact_model.release(owner)
            self.affected = [owner.robot_id]
        else:
            self.affected = self._pick_robots()
            if self.kind == "power" and self.power_mode == "release":
                for robot_id in self.affected:
                    env.contact_model.release(env.robots[robot_id])
            self.detail = f"{len(self.affected)} robots"
        self.injected_at = env.step_count

    def actions(self, actions: np.ndarray) -> np.ndarray:
        if self.kind == "power" and self.injected_at is not None:
            actions = actions.copy()
            actions[self.affected] = IDLE
        return actions

    def observations(self, observation: np.ndarray) -> np.ndarray:
        """Called on every observation the policy is about to see."""
        env = self.env
        layout = env.layout
        if self.kind == "comms":
            self.beacon_history.append(observation[:, layout.beacon].copy())
        if self.injected_at is None or not self.affected:
            return observation
        observation = observation.copy()
        if self.kind == "comms" and layout.beacon.stop > layout.beacon.start:
            stale = self.beacon_history[0]  # the signal from `delay` steps ago
            observation[self.affected, layout.beacon] = stale[self.affected]
        elif self.kind == "sensor":
            for channel in IMU_CHANNELS:
                observation[self.affected, layout.internal.start + channel] = 0.0
        return observation


def run_episode(config, actor, seed: int, kind: str, arguments, window: tuple[int, int]) -> dict:
    env = BridgeEnv(config.environment)
    observation, _ = env.reset(seed=seed)
    generator = torch.Generator().manual_seed(seed)
    injector = FaultInjector(env, kind, np.random.default_rng([seed, FAULTS.index(kind), 7]), arguments, window)
    hidden = actor.initial_hidden(env.config.robot.count)
    info: dict = {}
    for _ in range(env._max_steps()):
        injector.before_step()
        seen = injector.observations(observation)
        with torch.no_grad():
            output = actor.forward_step(torch.as_tensor(seen), hidden, torch.as_tensor(env.action_masks()))
        hidden = output.hidden
        actions = torch.multinomial(torch.softmax(output.logits, dim=-1), 1, generator=generator).squeeze(-1).numpy()
        actions = injector.actions(actions)
        observation, _, terminated, truncated, info = env.step(actions)
        if terminated or truncated:
            break
    success = bool(info.get("success"))
    injected = kind == "none" or injector.injected_at is not None
    return {
        "seed": seed,
        "fault": kind,
        "success": success,
        "injected": injected,
        "fault_step": injector.injected_at,
        "fault_detail": injector.detail,
        "steps": env.step_count,
        "capacity": round(float(env.current_capacity), 2),
        "target": float(env.target_load),
        "robots": int(sum(not robot.absent for robot in env.robots)),
        "fallen": int(sum(robot.fallen and not robot.absent for robot in env.robots)),
        **env.structure_metrics(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="default: the config saved in the checkpoint")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs=2, default=[0, 50], metavar=("FIRST", "STOP"))
    parser.add_argument("--faults", nargs="+", default=list(FAULTS), choices=FAULTS)
    parser.add_argument("--fault-timing", choices=("baseline", "window"), default="baseline",
                        help="baseline: 10-90%% of the fault-free episode; window: --fault-window")
    parser.add_argument("--fault-window", type=int, nargs=2, default=[20, 300], metavar=("FIRST", "LAST"),
                        help="fault steps for --fault-timing window (0.1 s per step)")
    parser.add_argument("--fraction", type=float, nargs=2, default=[0.10, 0.20], help="share of robots hit (power, comms, sensor)")
    parser.add_argument("--delay", type=int, nargs=2, default=[50, 100], help="beacon delay in steps (comms)")
    parser.add_argument("--power-mode", choices=("hold", "release"), default="hold",
                        help="powered-off robots keep (hold) or lose (release) their grip")
    parser.add_argument("--max-steps", type=int, default=0, help="time limit (0 = the stage's limit; 6000 = 10 min)")
    parser.add_argument("--json", type=Path, help="write every episode result here")
    arguments = parser.parse_args()

    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    config = load_config(arguments.config) if arguments.config else config_from_dict(checkpoint["experiment_config"])
    config.environment.curriculum_stage = arguments.stage
    if arguments.max_steps:
        config.environment.max_steps = arguments.max_steps
        config.environment.curriculum_max_steps = []
    probe = BridgeEnv(config.environment)
    probe.reset(seed=arguments.seeds[0])
    actor = SharedRecurrentActor(probe.layout, probe.action_count, config.model)
    actor.load_state_dict(checkpoint.get("algorithm", checkpoint)["actor"])
    actor.eval()
    print(f"{arguments.checkpoint} | stage {arguments.stage} (gap {probe.field.gap_width:.1f}, target {probe.target_load:.0f}) "
          f"| seeds {arguments.seeds[0]}-{arguments.seeds[1] - 1} | faults {' '.join(arguments.faults)}", flush=True)

    results = []
    baseline: dict[int, dict] = {}
    if arguments.fault_timing == "baseline" or "none" in arguments.faults:
        for seed in range(*arguments.seeds):
            baseline[seed] = run_episode(config, actor, seed, "none", arguments, (0, 0))
            if "none" in arguments.faults:
                results.append(baseline[seed])
                print(json.dumps(baseline[seed]), flush=True)
    for kind in arguments.faults:
        if kind == "none":
            continue
        for seed in range(*arguments.seeds):
            if arguments.fault_timing == "baseline":
                length = baseline[seed]["steps"]
                window = (max(1, int(0.1 * length)), max(1, int(0.9 * length)))
            else:
                window = tuple(arguments.fault_window)
            result = run_episode(config, actor, seed, kind, arguments, window)
            result["baseline_success"] = baseline[seed]["success"] if baseline else None
            results.append(result)
            print(json.dumps(result), flush=True)
    print("\nfault      injected  success  rate   95% CI")
    for kind in arguments.faults:
        rows = [r for r in results if r["fault"] == kind and r["injected"]]
        wins = sum(r["success"] for r in rows)
        low, high = wilson(wins, len(rows))
        rate = wins / len(rows) if rows else float("nan")
        print(f"{kind:9s}  {len(rows):8d}  {wins:7d}  {rate:5.0%}  [{low:.0%}, {high:.0%}]")
    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
