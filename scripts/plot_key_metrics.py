#!/usr/bin/env python3
"""Plot four key training metrics of a run in one figure.

    에피소드 리턴      (episodes.jsonl)   둑을 잡은 로봇 수 (episodes.jsonl)
    Critic loss        (updates.jsonl)    엔트로피          (updates.jsonl)

Each panel shows the raw values (thin, faint) and a moving average (bold); the
title row gives the first full-window average and the last average.  A dashed vertical line marks every
curriculum stage change.

    python scripts/plot_key_metrics.py runs/beacon_structured_mappo_static_v3
    python scripts/plot_key_metrics.py runs/beacon_structured_mappo_static_v4 --window 20 --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

SERIES = "#2a78d6"
SURFACE, INK, INK_2, GRID, AXIS, STAGE = "#ffffff", "#0b0b0b", "#52514e", "#e1e0d9", "#bdbcb5", "#9a9994"


# ---------------------------------------------------------------- data

def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def effective_history(rows: list[dict]) -> list[dict]:
    """Drop rows a resumed run replaced (resuming from update k re-runs updates > k)."""
    kept: list[dict] = []
    for row in rows:
        while kept and "update" in row and kept[-1].get("update", 0) > row["update"]:
            kept.pop()
        kept.append(row)
    return kept


def edge_grips(episode: dict) -> float:
    """Robots anchored to the left bank at the end of the episode."""
    structure = episode.get("structure")
    if isinstance(structure, dict) and "edge_grips" in structure:
        return float(structure["edge_grips"])
    return float(sum(
        1 for edge in episode.get("contact_graph", [])
        if edge.get("target") == "left_bank" and edge.get("kind") == "anchor"
    ))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty(len(values), dtype=float)
    for index in range(len(values)):
        result[index] = float(np.nanmean(values[max(0, index - window + 1): index + 1]))
    return result


# ---------------------------------------------------------------- drawing

def configure_fonts(plt) -> None:
    from matplotlib import font_manager

    for path in (
        "C:/Windows/Fonts/malgun.ttf",
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


def draw_panel(ax, x, raw, stages, window, title, xlabel, ylabel, fmt, zero_line=False, clip_spikes=False) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=11, color=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=10)

    if len(raw) == 0:
        ax.text(0.5, 0.5, "기록 없음", transform=ax.transAxes, ha="center", va="center", color=INK_2, fontsize=11)
        return

    smooth = moving_average(raw, window)
    if zero_line:
        ax.axhline(0, color=STAGE, lw=1)
    ax.plot(x, raw, color=SERIES, lw=0.8, alpha=0.25)
    ax.plot(x, smooth, color=SERIES, lw=2.2, label=f"{window}개 이동평균")

    # Curriculum stage changes.
    if stages is not None and len(stages) == len(x):
        for index in np.flatnonzero(np.diff(stages) != 0) + 1:
            ax.axvline(x[index], color=STAGE, lw=1, ls="--")
            ax.text(x[index], 1.0, f" {int(stages[index])}단계", transform=ax.get_xaxis_transform(),
                    ha="left", va="bottom", fontsize=9.5, color=INK_2)

    # Keep the view readable: a few spikes should not flatten the curve.
    low, high = float(np.nanmin(raw)), float(np.nanmax(raw))
    if clip_spikes:
        high = max(float(np.nanpercentile(raw, 99)), float(np.nanmax(smooth)))
    if zero_line:
        low, high = min(low, 0.0), max(high, 0.0)
    span = (high - low) or 1.0
    ax.set_ylim(low - 0.06 * span, high + 0.14 * span)
    ax.set_xlim(0, x[-1] * 1.12)

    box = dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none")
    ax.annotate(fmt.format(smooth[-1]), (x[-1], smooth[-1]), xytext=(8, 0), textcoords="offset points",
                ha="left", va="center", fontsize=10.5, fontweight="bold", color=INK, bbox=box)
    # First full-window average -> last average, in the title row.
    start = min(window, len(smooth)) - 1
    ax.set_title(f"{fmt.format(smooth[start])} → {fmt.format(smooth[-1])}", loc="right",
                 fontsize=10.5, color=INK_2, pad=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run directory containing episodes.jsonl and updates.jsonl")
    parser.add_argument("--window", type=int, default=50, help="moving-average window (default 50)")
    parser.add_argument("--output", help="PNG path (default: <run>/key_metrics.png)")
    parser.add_argument("--show", action="store_true", help="also open the figure in a window")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(args.run)
    if run.is_file():
        run = run.parent
    episodes = effective_history(read_jsonl(run / "episodes.jsonl"))
    updates = effective_history(read_jsonl(run / "updates.jsonl"))
    if not episodes and not updates:
        raise SystemExit(f"{run} 에 episodes.jsonl / updates.jsonl 이 없습니다.")

    episode_x = np.arange(1, len(episodes) + 1, dtype=float)
    episode_stage = np.array([row.get("curriculum_stage", 1) for row in episodes], dtype=float)
    returns = np.array([row.get("episode_return", np.nan) for row in episodes], dtype=float)
    grips = np.array([edge_grips(row) for row in episodes], dtype=float)

    update_x = np.array([row.get("update", index + 1) for index, row in enumerate(updates)], dtype=float)
    update_stage = np.array([row.get("curriculum_stage", 1) for row in updates], dtype=float)
    critic = np.array([row.get("critic_loss", np.nan) for row in updates], dtype=float)
    entropy = np.array([row.get("entropy", np.nan) for row in updates], dtype=float)

    configure_fonts(plt)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.2), facecolor=SURFACE)
    figure.subplots_adjust(left=0.07, right=0.96, top=0.86, bottom=0.08, hspace=0.55, wspace=0.25)
    successes = int(sum(bool(row.get("success")) for row in episodes))
    figure.text(0.07, 0.955, f"{run.name} 학습 지표", fontsize=17, fontweight="bold", color=INK)
    figure.text(0.07, 0.918, f"에피소드 {len(episodes):,}개 · 업데이트 {len(updates):,}회 · 성공 {successes:,}회 · "
                f"굵은 선 = {args.window}개 이동평균, 옅은 선 = 원래 값", fontsize=11.5, color=INK_2)

    draw_panel(axes[0, 0], episode_x, returns, episode_stage, args.window,
               "에피소드 리턴", "에피소드", "리턴 (보상 합)", "{:+.2f}", zero_line=True)
    draw_panel(axes[0, 1], episode_x, grips, episode_stage, args.window,
               "둑을 잡은 로봇 수", "에피소드", "대수 (에피소드 끝)", "{:.2f}")
    draw_panel(axes[1, 0], update_x, critic, update_stage, args.window,
               "Critic loss (MSE)", "업데이트", "MSE", "{:.3f}", clip_spikes=True)
    draw_panel(axes[1, 1], update_x, entropy, update_stage, args.window,
               "엔트로피", "업데이트", "nats", "{:.2f}")

    output = Path(args.output) if args.output else run / "key_metrics.png"
    figure.savefig(output, dpi=200, facecolor=SURFACE)
    print(f"저장: {output}")

    def first_last(name: str, values: np.ndarray, fmt: str) -> None:
        if len(values):
            w = min(args.window, len(values))
            print(f"  {name:<14} 처음 {w}개 평균 {fmt.format(np.nanmean(values[:w]))}  →  "
                  f"마지막 {w}개 평균 {fmt.format(np.nanmean(values[-w:]))}")

    first_last("에피소드 리턴", returns, "{:+.3f}")
    first_last("둑을 잡은 로봇", grips, "{:.2f}")
    first_last("critic loss", critic, "{:.4f}")
    first_last("엔트로피", entropy, "{:.3f}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
