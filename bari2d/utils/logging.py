from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class EpisodeLogger:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "episodes.jsonl"

    def log(self, metrics: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, default=_json_default, separators=(",", ":")) + "\n")


class PolicyStatistics:
    def __init__(self, action_count: int):
        self.action_counts = np.zeros(action_count, dtype=np.int64)
        self.latents: dict[str, list[np.ndarray]] = {}

    def add(self, actions: np.ndarray, latents: dict[str, Any]) -> None:
        self.action_counts += np.bincount(actions, minlength=len(self.action_counts))
        for name, values in latents.items():
            array = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
            self.latents.setdefault(name, []).append(array)

    def summarize(self) -> dict[str, Any]:
        total = max(int(self.action_counts.sum()), 1)
        result: dict[str, Any] = {"action_distribution": (self.action_counts / total).tolist()}
        for name, values in self.latents.items():
            combined = np.concatenate(values, axis=0)
            result[f"{name}_latent_mean"] = float(combined.mean())
            result[f"{name}_latent_std"] = float(combined.std())
        return result

