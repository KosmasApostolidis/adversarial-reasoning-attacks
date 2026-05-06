"""Hero figure 3: attack × ε heatmap."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ._common import (
    ATTACK_ORDER,
    BG,
    GRID,
    LABELS,
    PALETTE,
    PANEL,
    PANEL_LIGHT,
    TEXT,
    TEXT_MUTED,
    fmt_eps,
)


def fig_heatmap(by_attack, out_path: Path) -> None:
    eps_vals = sorted({float(r["epsilon"]) for recs in by_attack.values() for r in recs})
    attacks_with_data = [a for a in ATTACK_ORDER if by_attack.get(a)]

    cell = np.full((len(attacks_with_data), len(eps_vals)), np.nan)
    counts = np.zeros_like(cell, dtype=int)
    for i, a in enumerate(attacks_with_data):
        groups = defaultdict(list)
        for r in by_attack[a]:
            groups[float(r["epsilon"])].append(r["edit_distance_norm"])
        for j, e in enumerate(eps_vals):
            if e in groups:
                cell[i, j] = float(np.mean(groups[e]))
                counts[i, j] = len(groups[e])

    cmap = LinearSegmentedColormap.from_list(
        "ed_dark", [PANEL, "#3A3F5C", "#7A4F8B", PALETTE["apgd"], "#FFD37A"]
    )
    fig = plt.figure(figsize=(12, 6.8))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.18, 0.18, 0.65, 0.66])
    ax.set_facecolor(BG)
    masked = np.ma.masked_invalid(cell)
    cmap.set_bad(PANEL_LIGHT)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=0.85, aspect="auto", origin="upper")

    for i in range(masked.shape[0]):
        for j in range(masked.shape[1]):
            if np.isnan(cell[i, j]):
                ax.text(
                    j,
                    i,
                    "n/a",
                    ha="center",
                    va="center",
                    color=TEXT_MUTED,
                    fontsize=10,
                    fontstyle="italic",
                )
            else:
                v = cell[i, j]
                color = "black" if v > 0.55 else TEXT
                ax.text(
                    j,
                    i - 0.10,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=14,
                    fontweight="bold",
                    family="DejaVu Sans Mono",
                )
                ax.text(
                    j,
                    i + 0.22,
                    f"n={counts[i, j]}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8.5,
                    alpha=0.85,
                )

    ax.set_xticks(range(len(eps_vals)))
    ax.set_xticklabels([fmt_eps(e) for e in eps_vals], color=TEXT_MUTED, fontsize=11)
    ax.set_yticks(range(len(attacks_with_data)))
    ax.set_yticklabels(
        [LABELS[a] for a in attacks_with_data], color=TEXT, fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Perturbation budget ε", color=TEXT_MUTED, fontsize=12)
    ax.tick_params(length=0)

    cbar_ax = fig.add_axes([0.86, 0.20, 0.018, 0.55])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.outline.set_edgecolor(GRID)
    cbar.outline.set_linewidth(0.7)
    cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    cbar.set_label("Mean edit distance", color=TEXT_MUTED, fontsize=10)

    fig.text(0.06, 0.93, "ATTACK × BUDGET HEATMAP", color=TEXT, fontsize=22, fontweight="bold")
    fig.text(
        0.06,
        0.895,
        "Mean normalised edit distance per (attack, ε) cell · brighter = more disruption",
        color=TEXT_MUTED,
        fontsize=11,
    )
    fig.text(
        0.06,
        0.05,
        "PGD evaluated only at smoke ε=8/255 (n=5); other attacks span full sweep (4 ε × 3 seeds × 5 samples = 60).",
        color=TEXT_MUTED,
        fontsize=9,
        alpha=0.85,
    )

    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
