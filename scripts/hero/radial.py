"""Hero figure 4: radial profile (circular bars per attack across 5 metrics)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    ACCENT,
    ATTACK_ORDER,
    BG,
    GRID,
    LABELS,
    PALETTE,
    PANEL,
    TEXT,
    TEXT_MUTED,
    add_panel,
    edits,
    step1_flip_rate,
)

_METRIC_NAMES = ["MEAN", "MAX", "FLIP", "P95", "Δ-LEN"]
_METRIC_HELP = [
    ("MEAN", "average normalised edit distance"),
    ("MAX", "worst-case sample"),
    ("FLIP", "fraction with first tool flipped"),
    ("P95", "95th-percentile edit distance"),
    ("Δ-LEN", "abs change in trajectory length"),
]


def _compute_metrics(recs: list[dict]) -> list[float]:
    eds = edits(recs)
    if eds.size == 0:
        return [0.0] * len(_METRIC_NAMES)
    bens = np.array([len(r.get("benign", {}).get("tool_sequence", []) or []) for r in recs])
    atts = np.array([len(r.get("attacked", {}).get("tool_sequence", []) or []) for r in recs])
    traj_d = np.abs(atts - bens).mean() if bens.size else 0.0
    return [
        float(eds.mean()),
        float(eds.max()),
        step1_flip_rate(recs),
        float(np.quantile(eds, 0.95)),
        float(traj_d),
    ]


def _draw_grid_circles(ax) -> None:
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(np.linspace(0, 2 * np.pi, 200), [r] * 200, color=GRID, linewidth=0.6, alpha=0.6)
        ax.text(np.pi / 2, r, f"{r:.2f}", color=TEXT_MUTED, fontsize=7,
                ha="center", va="center", alpha=0.55, zorder=2)


def _draw_radar_panel(ax, name: str, angles: np.ndarray, norm_row: np.ndarray, raw_row: list[float]) -> None:
    ax.set_facecolor(BG)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.15)
    _draw_grid_circles(ax)

    bar_w = 2 * np.pi / len(_METRIC_NAMES) * 0.72
    ax.bar(angles, norm_row, width=bar_w, color=PALETTE[name], alpha=0.92,
           edgecolor=BG, linewidth=1.6, zorder=3)
    for i_m, (ang, val_norm, val_raw) in enumerate(zip(angles, norm_row, raw_row, strict=True)):
        label = f"{val_raw:.0%}" if _METRIC_NAMES[i_m] == "FLIP" else f"{val_raw:.2f}"
        ax.text(ang, max(val_norm + 0.10, 0.18), label, ha="center", va="center",
                color=TEXT, fontsize=10, fontweight="bold",
                family="DejaVu Sans Mono", zorder=5)

    ax.set_xticks(angles)
    ax.set_xticklabels(_METRIC_NAMES, color=TEXT, fontsize=10.5, fontweight="bold")
    ax.tick_params(axis="x", pad=14)
    ax.set_yticks([])
    ax.spines["polar"].set_color(GRID)
    ax.spines["polar"].set_linewidth(1.0)
    ax.set_title(LABELS[name], color=PALETTE[name], fontsize=14, fontweight="bold", pad=22)
    ax.set_ylim(0, 1.30)


def _draw_legend_panel(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.axis("off")
    add_panel(ax, 0.02, 0.02, 0.96, 0.96, fc=PANEL, ec=GRID, radius=0.05)
    ax.text(0.06, 0.86, "READING THE RADAR", color=TEXT, fontsize=15, fontweight="bold")
    ax.text(0.06, 0.78, "Each spoke is one metric.", color=TEXT_MUTED, fontsize=10)
    ax.text(0.06, 0.71,
            "Bar length = column-normalised score across\nall attacks (longer = stronger attack).",
            color=TEXT_MUTED, fontsize=10)
    ax.text(0.06, 0.58, "METRICS", color=ACCENT, fontsize=11, fontweight="bold")
    for i, (k, v) in enumerate(_METRIC_HELP):
        ax.text(0.06, 0.49 - i * 0.075, k, color=TEXT, fontsize=9.5, fontweight="bold")
        ax.text(0.30, 0.49 - i * 0.075, v, color=TEXT_MUTED, fontsize=9.5)
    ax.text(0.06, 0.06,
            "Numbers shown above each bar are raw values\n(not normalised).",
            color=TEXT_MUTED, fontsize=9, alpha=0.85, style="italic")


def _add_titles(fig) -> None:
    fig.text(0.5, 0.965, "ATTACK PROFILES", color=TEXT, fontsize=24, fontweight="bold", ha="center")
    fig.text(0.5, 0.940,
             "Five-axis radar per attack · column-normalised across attacks",
             color=TEXT_MUTED, fontsize=11, ha="center")


def fig_radial(by_attack, out_path: Path) -> None:
    raw = {n: _compute_metrics(by_attack[n]) for n in ATTACK_ORDER}
    arr = np.array([raw[n] for n in ATTACK_ORDER])
    col_max = np.where(arr.max(axis=0) == 0, 1.0, arr.max(axis=0))
    norm = arr / col_max

    fig = plt.figure(figsize=(13, 13))
    fig.patch.set_facecolor(BG)
    cols, rows = 3, 2
    angles = np.linspace(0, 2 * np.pi, len(_METRIC_NAMES), endpoint=False)

    for idx, name in enumerate(ATTACK_ORDER):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="polar")
        _draw_radar_panel(ax, name, angles, norm[idx], raw[name])

    _draw_legend_panel(fig.add_subplot(rows, cols, 6))
    _add_titles(fig)

    fig.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.05, hspace=0.55, wspace=0.40)
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
