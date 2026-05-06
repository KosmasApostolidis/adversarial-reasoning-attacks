"""Hero figure 2: ridgeline (joy plot) of edit-distance distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from ._common import (
    ACCENT,
    ATTACK_ORDER,
    BG,
    GRID,
    LABELS,
    PALETTE,
    TEXT,
    TEXT_MUTED,
    edits,
)


def fig_ridgeline(by_attack, out_path: Path) -> None:
    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.10, 0.10, 0.85, 0.78])
    ax.set_facecolor(BG)

    xs = np.linspace(-0.05, 1.05, 400)
    spacing = 1.0
    for i, name in enumerate(ATTACK_ORDER):
        vals = edits(by_attack[name])
        if vals.size == 0:
            continue
        bw = 0.06
        density = np.exp(-0.5 * ((xs[:, None] - vals[None, :]) / bw) ** 2).sum(axis=1)
        if density.max() > 0:
            density = density / density.max()
        y_base = (len(ATTACK_ORDER) - 1 - i) * spacing
        color = PALETTE[name]
        rgba_fill = to_rgba(color, alpha=0.55)
        rgba_edge = to_rgba(color, alpha=1.0)
        ax.fill_between(
            xs,
            y_base,
            y_base + density * 0.85,
            color=rgba_fill,
            edgecolor=rgba_edge,
            linewidth=2.0,
            zorder=3 - i * 0.05,
        )
        med = float(np.median(vals))
        mn = float(np.mean(vals))
        ax.plot([med, med], [y_base, y_base + 0.22], color=TEXT, linewidth=1.8, zorder=8)
        ax.scatter(
            [mn], [y_base], marker="D", s=70, color=ACCENT, edgecolor=BG, linewidth=1.0, zorder=9
        )
        ax.text(
            -0.05,
            y_base + 0.06,
            LABELS[name],
            color=color,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax.text(
            1.05,
            y_base + 0.06,
            f"n={vals.size}  μ={mn:.3f}",
            color=TEXT_MUTED,
            fontsize=10,
            ha="right",
            va="bottom",
            family="DejaVu Sans Mono",
        )

    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.4, len(ATTACK_ORDER) * spacing)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Normalised trajectory edit distance", color=TEXT_MUTED, fontsize=11)
    for x in np.linspace(0, 1, 6):
        ax.axvline(x, color=GRID, linewidth=0.4, alpha=0.5, zorder=1)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)

    fig.text(0.10, 0.945, "DISTRIBUTION SHAPES", color=TEXT, fontsize=24, fontweight="bold")
    fig.text(
        0.10,
        0.918,
        "Kernel density estimate of edit-distance per attack · vertical tick = median · diamond = mean",
        color=TEXT_MUTED,
        fontsize=11,
    )
    fig.text(
        0.10,
        0.04,
        "More mass near 1.0 = trajectory rewritten · more mass near 0 = agent unaffected",
        color=TEXT_MUTED,
        fontsize=9.5,
        alpha=0.85,
    )

    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
