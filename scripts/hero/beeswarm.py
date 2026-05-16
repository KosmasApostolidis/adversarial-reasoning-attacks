"""Hero figure 1: beeswarm with stat-card inset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from ._common import (
    ACCENT,
    ATTACK_ORDER,
    BG,
    GRID,
    LABELS,
    PALETTE,
    PANEL,
    PANEL_LIGHT,
    TEXT,
    TEXT_MUTED,
    add_panel,
    beeswarm_y,
    bootstrap_ci,
    edits,
    step1_flip_rate,
)


@dataclass(frozen=True)
class _CardGeom:
    x: float
    width: float
    height: float
    y_top: float


def _draw_swarm_row(ax, i: int, name: str, vals: np.ndarray) -> None:
    jitter = beeswarm_y(vals, max_width=0.34)
    color = PALETTE[name]
    ax.scatter(
        vals,
        np.full_like(vals, i, dtype=float) + jitter,
        s=85,
        color=color,
        alpha=0.78,
        edgecolor=BG,
        linewidth=0.9,
        zorder=3,
    )
    med = float(np.median(vals))
    mn = float(np.mean(vals))
    ax.plot([med, med], [i - 0.42, i + 0.42], color=TEXT, linewidth=1.5, zorder=4, alpha=0.85)
    ax.scatter([mn], [i], marker="D", s=80, color=ACCENT, edgecolor=BG, linewidth=1.2, zorder=5)


def _decorate_swarm_axes(ax) -> None:
    ax.set_yticks(range(len(ATTACK_ORDER)))
    ax.set_yticklabels(
        [LABELS[a] for a in ATTACK_ORDER], color=TEXT, fontsize=12, fontweight="bold"
    )
    ax.invert_yaxis()
    ax.set_xlabel("Normalised trajectory edit distance", color=TEXT_MUTED, fontsize=11, labelpad=10)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(len(ATTACK_ORDER) - 0.5, -0.5)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xticks(np.linspace(0, 1, 5))
    for v in np.linspace(0, 1, 5):
        ax.axvline(v, color=GRID, linewidth=0.5, alpha=0.5, zorder=1)


def _add_headline(fig) -> None:
    fig.text(0.06, 0.945, "ATTACK LANDSCAPE", color=TEXT, fontsize=24, fontweight="bold")
    fig.text(
        0.06,
        0.918,
        "Trajectory edit distance per record · Qwen2.5-VL-7B medical agent · ProstateX val=5",
        color=TEXT_MUTED,
        fontsize=11,
    )


def _add_card_header(fig, card_x: float) -> None:
    fig.text(card_x, 0.945, "ATTACK STATISTICS", color=TEXT, fontsize=13, fontweight="bold")
    fig.text(
        card_x,
        0.918,
        "n samples · mean ed [95% CI] · step-1 flip rate",
        color=TEXT_MUTED,
        fontsize=10,
    )


def _draw_card_background(fig, geom: _CardGeom, color: str) -> None:
    add_panel(
        fig,
        geom.x,
        geom.y_top - geom.height,
        geom.width,
        geom.height,
        fc=PANEL,
        ec=GRID,
        radius=0.012,
    )
    stripe = Rectangle(
        (geom.x, geom.y_top - geom.height),
        0.005,
        geom.height,
        facecolor=color,
        edgecolor="none",
        transform=fig.transFigure,
    )
    fig.patches.append(stripe)


def _draw_card_stats_text(
    fig, geom: _CardGeom, name: str, eds: np.ndarray, lo: float, hi: float
) -> None:
    fig.text(
        geom.x + 0.018,
        geom.y_top - 0.022,
        LABELS[name],
        color=PALETTE[name],
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        geom.x + 0.018,
        geom.y_top - 0.062,
        f"μ = {eds.mean():.3f}",
        color=TEXT,
        fontsize=18,
        fontweight="bold",
        family="DejaVu Sans Mono",
    )
    fig.text(
        geom.x + 0.018,
        geom.y_top - 0.083,
        f"n={eds.size:>3d}   95% CI [{lo:.2f}, {hi:.2f}]",
        color=TEXT_MUTED,
        fontsize=9,
        family="DejaVu Sans Mono",
    )


def _draw_flip_bar(fig, geom: _CardGeom, flip: float, color: str) -> None:
    bar_x = geom.x + 0.018
    bar_w = geom.width - 0.036
    bar_y = geom.y_top - geom.height + 0.012
    fig.patches.append(
        Rectangle(
            (bar_x, bar_y),
            bar_w,
            0.008,
            facecolor=PANEL_LIGHT,
            edgecolor="none",
            transform=fig.transFigure,
        )
    )
    fig.patches.append(
        Rectangle(
            (bar_x, bar_y),
            bar_w * flip,
            0.008,
            facecolor=color,
            edgecolor="none",
            transform=fig.transFigure,
        )
    )
    fig.text(
        geom.x + 0.018, bar_y + 0.012, f"step-1 flip {flip:.0%}", color=TEXT_MUTED, fontsize=8.5
    )


def _draw_stat_card(fig, geom: _CardGeom, name: str, recs: list[dict]) -> None:
    _draw_card_background(fig, geom, PALETTE[name])
    eds = edits(recs)
    if eds.size == 0:
        fig.text(
            geom.x + 0.018, geom.y_top - geom.height / 2, "no data", color=TEXT_MUTED, fontsize=10
        )
        return
    lo, hi = bootstrap_ci(eds)
    flip = step1_flip_rate(recs)
    _draw_card_stats_text(fig, geom, name, eds, lo, hi)
    _draw_flip_bar(fig, geom, flip, PALETTE[name])


def _draw_legend(fig) -> None:
    leg_y = 0.065
    fig.text(0.06, leg_y, "MEDIAN", color=TEXT_MUTED, fontsize=8.5)
    fig.add_artist(
        plt.Line2D(
            [0.105, 0.125],
            [leg_y + 0.005, leg_y + 0.005],
            color=TEXT,
            linewidth=1.5,
            transform=fig.transFigure,
        )
    )
    fig.text(0.135, leg_y, "MEAN", color=TEXT_MUTED, fontsize=8.5)
    fig.add_artist(
        plt.Line2D(
            [0.165],
            [leg_y + 0.005],
            marker="D",
            markersize=8,
            color=ACCENT,
            markeredgecolor=BG,
            markeredgewidth=1.0,
            transform=fig.transFigure,
        )
    )
    fig.text(
        0.185, leg_y, "EACH DOT = ONE RECORD (sample × ε × seed)", color=TEXT_MUTED, fontsize=8.5
    )


def _add_footer(fig) -> None:
    fig.text(
        0.06,
        0.018,
        "adversarial-reasoning-attacks · 2026 · github.com/KosmasApostolidis/adversarial-reasoning-attacks",
        color=TEXT_MUTED,
        fontsize=8,
        alpha=0.6,
    )
    fig.text(
        0.96,
        0.018,
        "ε ∈ {2, 4, 8, 16}/255 · seeds {0, 1, 2}",
        color=TEXT_MUTED,
        fontsize=8,
        alpha=0.6,
        ha="right",
    )


def fig_beeswarm(by_attack, out_path: Path) -> None:
    fig = plt.figure(figsize=(15, 8.5))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.06, 0.17, 0.62, 0.72])
    ax.set_facecolor(BG)

    np.random.seed(0)
    for i, name in enumerate(ATTACK_ORDER):
        vals = edits(by_attack[name])
        if vals.size == 0:
            continue
        _draw_swarm_row(ax, i, name, vals)

    _decorate_swarm_axes(ax)
    _add_headline(fig)

    card_x, card_w, card_h, card_gap, card_y0 = 0.71, 0.25, 0.13, 0.018, 0.82
    _add_card_header(fig, card_x)
    for i, name in enumerate(ATTACK_ORDER):
        geom = _CardGeom(
            x=card_x, width=card_w, height=card_h, y_top=card_y0 - i * (card_h + card_gap)
        )
        _draw_stat_card(fig, geom, name, by_attack[name])

    _draw_legend(fig)
    _add_footer(fig)

    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
