#!/usr/bin/env python3
"""Plot training progress of a run and flag common failure patterns.

Reads ``episodes.jsonl`` (one line per finished episode) and, when present,
``updates.jsonl`` (one line per PPO update) from a run directory, then writes a
PNG dashboard and prints a short text summary comparing the first and last
windows of episodes.

    python scripts/plot_training.py runs/beacon_structured_mappo_static
    python scripts/plot_training.py runs/beacon_structured_mappo_static --window 10 --output progress.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Validated categorical order (blue, orange, aqua, yellow); text never uses series colors.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
STAGE_BANDS = ["#f1f0ec", "#fcfcfb"]
MOVE_ACTIONS = range(0, 6)
CLIMB, ANCHOR, RELEASE, IDLE = 6, 7, 8, 9


def configure_fonts() -> None:
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in ("Malgun Gothic", "Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
        if family in available:
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def effective_history(rows: list[dict]) -> list[dict]:
    """Drop episodes a resumed run replaced (resuming from update k re-runs updates > k)."""
    kept: list[dict] = []
    for row in rows:
        while kept and kept[-1]["update"] > row["update"]:
            kept.pop()
        kept.append(row)
    return kept


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty(len(values), dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = float(np.nanmean(values[start : index + 1]))
    return result


def style(ax: plt.Axes, title: str, ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, fontweight="bold", pad=6)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
    ax.set_xlabel("업데이트", color=INK_2, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b8b1")
    ax.tick_params(colors=INK_2, labelsize=9)


def stage_bands(ax: plt.Axes, updates: np.ndarray, stages: np.ndarray) -> None:
    if len(updates) == 0:
        return
    start = 0
    for index in range(1, len(stages) + 1):
        if index == len(stages) or stages[index] != stages[start]:
            left = updates[start] if start == 0 else (updates[start - 1] + updates[start]) / 2
            right = updates[index - 1] if index == len(stages) else (updates[index - 1] + updates[index]) / 2
            ax.axvspan(left, right, color=STAGE_BANDS[int(stages[start]) % 2], lw=0, zorder=0)
            ax.text(right, 1.0, f"{int(stages[start])}단계 ", transform=ax.get_xaxis_transform(),
                    ha="right", va="bottom", fontsize=8.5, color=INK_2)
            start = index


def lines(ax: plt.Axes, x: np.ndarray, series: list[tuple[str, np.ndarray]], raw: bool = False) -> None:
    ends: list[tuple[float, str]] = []
    for index, (label, values) in enumerate(series):
        color = SERIES[index]
        if raw:
            ax.scatter(x, values, s=9, color=color, alpha=0.25, lw=0, zorder=2)
        smooth = rolling(values, WINDOW)
        ax.plot(x, smooth, color=color, lw=2, label=label, zorder=3)
        if len(x):
            ends.append((float(smooth[-1]), label))
    if len(series) > 1:
        ax.legend(loc="upper left", fontsize=8.5, frameon=False)
        place_end_labels(ax, x[-1], ends)


def place_end_labels(ax: plt.Axes, x_end: float, ends: list[tuple[float, str]]) -> None:
    """Direct labels at line ends, pushed apart so equal end values do not collide."""
    ax.figure.canvas.draw()
    low, high = ax.get_ylim()
    span = max(high - low, 1e-9)
    gap = 0.075
    order = sorted(ends, key=lambda item: item[0])
    positions = [min(max((value - low) / span, 0.0), 1.0) for value, _ in order]
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + gap)
    overflow = positions[-1] - 1.0 if positions else 0.0
    if overflow > 0:
        positions = [position - overflow for position in positions]
    for (value, label), position in zip(order, positions):
        ax.annotate(label, xy=(x_end, value), xycoords="data", xytext=(1.01, position), textcoords=ax.transAxes,
                    va="center", fontsize=8.5, color=INK_2, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color="#c9c8c1", lw=0.8, shrinkA=0, shrinkB=2))


def empty(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=INK_2, fontsize=9.5, wrap=True)


def summary(episodes: list[dict], updates: list[dict], window: int) -> list[str]:
    count = len(episodes)
    first, last = episodes[: min(window, count)], episodes[-min(window, count):]
    robots = len(episodes[-1].get("morphology", [])) or 1

    def mean(rows: list[dict], key: str) -> float:
        return float(np.mean([row.get(key, np.nan) for row in rows]))

    def share(rows: list[dict], indices) -> float:
        return float(np.mean([sum(row["action_distribution"][i] for i in indices) for row in rows]))

    lines_out = [
        f"에피소드 {count}개, 업데이트 {int(episodes[-1]['update'])}회까지, 현재 커리큘럼 {int(episodes[-1]['curriculum_stage'])}단계",
        f"{'':18}{'처음 ' + str(len(first)) + '개':>12}{'최근 ' + str(len(last)) + '개':>12}",
    ]
    rows = [
        ("성공률", mean(first, "success"), mean(last, "success"), "{:.0%}"),
        ("최대 진행도", mean(first, "max_span_progress"), mean(last, "max_span_progress"), "{:.2f}"),
        ("고정된 로봇 수", mean(first, "anchored_robots"), mean(last, "anchored_robots"), "{:.1f}"),
        ("추락한 로봇 수", mean(first, "fallen_robots"), mean(last, "fallen_robots"), "{:.1f}"),
        ("앵커 파손 수", mean(first, "anchor_failures"), mean(last, "anchor_failures"), "{:.1f}"),
        ("ANCHOR 비율", share(first, [ANCHOR]), share(last, [ANCHOR]), "{:.1%}"),
        ("CLIMB 비율", share(first, [CLIMB]), share(last, [CLIMB]), "{:.1%}"),
        ("IDLE 비율", share(first, [IDLE]), share(last, [IDLE]), "{:.1%}"),
    ]
    if all("episode_return" in row for row in episodes):
        rows.insert(1, ("에피소드 리턴", mean(first, "episode_return"), mean(last, "episode_return"), "{:.2f}"))
    for name, before, after, fmt in rows:
        lines_out.append(f"  {name:<16}{fmt.format(before):>12}{fmt.format(after):>12}")

    warnings = []
    shares = np.mean([row["action_distribution"] for row in last], axis=0)
    if shares.max() > 0.6:
        warnings.append(f"행동 붕괴: 최근 행동의 {shares.max():.0%}가 한 가지 행동(번호 {int(shares.argmax())})입니다.")
    if mean(last, "anchored_robots") >= 0.8 * robots and mean(last, "max_span_progress") < 0.05:
        warnings.append("'고정하고 멈추기' 패턴: 대부분 고정했지만 틈 위로 나아가지 않습니다 (v1에서 본 국소 최적).")
    if count >= 2 * window and mean(last, "success") <= mean(first, "success") and mean(last, "max_span_progress") <= mean(first, "max_span_progress"):
        warnings.append("정체: 최근 성공률과 진행도가 처음보다 나아지지 않았습니다.")
    if updates:
        entropy = [row["entropy"] for row in updates]
        if len(entropy) > 10 and entropy[-1] < 0.3 * entropy[0] and mean(last, "success") < 0.2:
            warnings.append(f"엔트로피 급락: {entropy[0]:.2f} → {entropy[-1]:.2f}. 성공 전에 탐색이 멈췄을 수 있습니다.")
    lines_out.append("")
    lines_out += (["확인할 점:"] + [f"  - {text}" for text in warnings]) if warnings else ["경고 없음."]
    return lines_out


WINDOW = 20


def main() -> None:
    global WINDOW
    parser = argparse.ArgumentParser(description="Plot training progress of a BARI run directory")
    parser.add_argument("run", help="run directory (or its episodes.jsonl)")
    parser.add_argument("--window", type=int, default=20, help="moving-average window in episodes")
    parser.add_argument("--output", help="PNG path (default: <run>/training_progress.png)")
    arguments = parser.parse_args()
    WINDOW = max(arguments.window, 1)
    run = Path(arguments.run)
    run_dir = run.parent if run.suffix == ".jsonl" else run
    episodes = effective_history(read_jsonl(run_dir / "episodes.jsonl"))
    updates = effective_history(read_jsonl(run_dir / "updates.jsonl"))
    if not episodes:
        raise SystemExit(f"No episodes found in {run_dir / 'episodes.jsonl'}")
    configure_fonts()

    x = np.array([row["update"] for row in episodes], dtype=float)
    stages = np.array([row["curriculum_stage"] for row in episodes])
    get = lambda key: np.array([float(row.get(key, np.nan)) for row in episodes])  # noqa: E731
    actions = np.array([row["action_distribution"] for row in episodes])

    fig, axes = plt.subplots(2, 4, figsize=(19, 8.8), facecolor=SURFACE)
    fig.subplots_adjust(left=0.04, right=0.95, top=0.88, bottom=0.08, wspace=0.34, hspace=0.42)
    fig.suptitle(f"학습 진행: {run_dir.name}  (이동평균 {WINDOW} 에피소드, 배경 = 커리큘럼 단계)",
                 x=0.04, ha="left", fontsize=14, fontweight="bold", color=INK)

    ax = axes[0, 0]; style(ax, "성공률", "비율"); stage_bands(ax, x, stages)
    lines(ax, x, [("성공률", get("success"))], raw=True); ax.set_ylim(-0.03, 1.03)

    ax = axes[0, 1]; style(ax, "다리 진행", "0~1"); stage_bands(ax, x, stages); ax.set_ylim(-0.03, 1.03)
    lines(ax, x, [("최대 진행도", get("max_span_progress")), ("용량/목표", np.minimum(get("capacity_ratio"), 1.0))])

    ax = axes[0, 2]; style(ax, "에피소드 리턴 (보상 합)"); stage_bands(ax, x, stages)
    if all("episode_return" in row for row in episodes):
        lines(ax, x, [("리턴", get("episode_return"))], raw=True)
    else:
        empty(ax, "이 로그에는 없음\n(수정된 trainer로 새로 학습하면 기록됨)")

    ax = axes[0, 3]; style(ax, "에피소드 길이", "스텝"); stage_bands(ax, x, stages)
    lines(ax, x, [("길이", get("step"))], raw=True); ax.set_ylim(bottom=0)

    ax = axes[1, 0]; style(ax, "구조 상태 (에피소드 끝)", "로봇 수 / 횟수"); stage_bands(ax, x, stages)
    lines(ax, x, [("고정됨", get("anchored_robots")), ("추락", get("fallen_robots")), ("앵커 파손", get("anchor_failures"))])
    ax.set_ylim(bottom=0)

    ax = axes[1, 1]; style(ax, "행동 비율", "비율"); stage_bands(ax, x, stages); ax.set_ylim(-0.02, 1.02)
    lines(ax, x, [("이동", actions[:, list(MOVE_ACTIONS)].sum(axis=1)), ("CLIMB", actions[:, CLIMB]),
                  ("ANCHOR", actions[:, ANCHOR]), ("IDLE", actions[:, IDLE])])

    ax = axes[1, 2]; style(ax, "정책 엔트로피", "nats")
    if updates:
        u = np.array([row["update"] for row in updates])
        ax.plot(u, [row["entropy"] for row in updates], color=SERIES[0], lw=2)
        ax.axhline(math.log(10), color=INK_2, lw=1, ls=(0, (4, 3)))
        ax.text(u[-1], math.log(10) - 0.04, "완전 무작위 (ln 10) ", ha="right", va="top", fontsize=8.5, color=INK_2)
        ax.set_ylim(bottom=0)
    else:
        empty(ax, "updates.jsonl 없음\n(수정된 trainer로 새로 학습하면 기록됨)")

    ax = axes[1, 3]; style(ax, "critic 손실", "MSE (로그 축)")
    if updates:
        ax.plot(u, [max(row["critic_loss"], 1e-8) for row in updates], color=SERIES[0], lw=2)
        ax.set_yscale("log")
    else:
        empty(ax, "updates.jsonl 없음\n(수정된 trainer로 새로 학습하면 기록됨)")

    output = Path(arguments.output) if arguments.output else run_dir / "training_progress.png"
    fig.savefig(output, dpi=120, facecolor=SURFACE)
    print("\n".join(summary(episodes, updates, WINDOW)))
    print(f"\n그래프: {output}")


if __name__ == "__main__":
    main()
