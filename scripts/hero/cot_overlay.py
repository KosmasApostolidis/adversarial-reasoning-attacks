"""Hero figure 6: edit-distance vs CoT-drift comparison bars per attack.

For each attack: a paired bar group — left bar is mean normalised edit
distance (the original metric the paper reports), right bar is mean
cot_drift_score. Shows where attacks corrupt reasoning *beyond* what
they corrupt in the tool sequence ("silent CoT corruption" zones).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    ATTACK_ORDER,
    BG,
    GRID,
    LABELS,
    PALETTE,
    PANEL,
    TEXT,
    TEXT_MUTED,
    bootstrap_ci,
    cot_drifts,
    edits,
    has_cot,
)


@dataclass(frozen=True)
class _PairedStats:
    ed_means: list[float]
    ed_lo: list[float]
    ed_hi: list[float]
    dr_means: list[float]
    dr_lo: list[float]
    dr_hi: list[float]


def _compute_paired_stats(attacks: list[str], by_attack: dict) -> _PairedStats:
    ed_means, ed_lo, ed_hi, dr_means, dr_lo, dr_hi = [], [], [], [], [], []
    for a in attacks:
        recs = by_attack[a]
        e = edits(recs)
        d = cot_drifts(recs)
        d = d[~np.isnan(d)]
        ed_means.append(float(e.mean()) if e.size else 0.0)
        lo, hi = bootstrap_ci(e) if e.size else (0.0, 0.0)
        ed_lo.append(ed_means[-1] - lo)
        ed_hi.append(hi - ed_means[-1])
        dr_means.append(float(d.mean()) if d.size else 0.0)
        lo, hi = bootstrap_ci(d) if d.size else (0.0, 0.0)
        dr_lo.append(dr_means[-1] - lo)
        dr_hi.append(hi - dr_means[-1])
    return _PairedStats(ed_means, ed_lo, ed_hi, dr_means, dr_lo, dr_hi)


def _draw_paired_bars(ax, x, width: float, attacks: list[str], stats: _PairedStats):
    bars_ed = ax.bar(
        x - width / 2,
        stats.ed_means,
        width=width,
        color=[PALETTE[a] for a in attacks],
        edgecolor=GRID,
        linewidth=0.6,
        label="Edit distance (tools)",
    )
    bars_dr = ax.bar(
        x + width / 2,
        stats.dr_means,
        width=width,
        color=[PALETTE[a] for a in attacks],
        edgecolor=TEXT,
        linewidth=1.0,
        hatch="///",
        alpha=0.85,
        label="CoT drift (reasoning)",
    )
    ax.errorbar(
        x - width / 2,
        stats.ed_means,
        yerr=[stats.ed_lo, stats.ed_hi],
        fmt="none",
        ecolor=TEXT_MUTED,
        elinewidth=0.9,
        capsize=3,
    )
    ax.errorbar(
        x + width / 2,
        stats.dr_means,
        yerr=[stats.dr_lo, stats.dr_hi],
        fmt="none",
        ecolor=TEXT_MUTED,
        elinewidth=0.9,
        capsize=3,
    )
    return bars_ed, bars_dr


def _annotate_bars(ax, bars_ed, bars_dr, stats: _PairedStats) -> None:
    for rect, val in zip(bars_ed, stats.ed_means, strict=True):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            val + 0.015,
            f"{val:.2f}",
            ha="center",
            color=TEXT,
            fontsize=9,
        )
    for rect, val in zip(bars_dr, stats.dr_means, strict=True):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            val + 0.015,
            f"{val:.2f}",
            ha="center",
            color=TEXT,
            fontsize=9,
        )


def _decorate_axes_and_legend(ax, x, attacks: list[str]) -> None:
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in attacks], color=TEXT, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean disruption (95% bootstrap CI)", color=TEXT_MUTED, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    leg = ax.legend(
        loc="upper left",
        frameon=True,
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=10,
    )
    for t in leg.get_texts():
        t.set_color(TEXT)


def fig_cot_overlay(by_attack, out_path: Path) -> None:
    """Pair plot. No-op if no CoT data is present anywhere."""
    if not has_cot(by_attack):
        print("[cot_overlay] no cot_drift_score in records — skipping")
        return
    attacks = [a for a in ATTACK_ORDER if by_attack.get(a)]
    n = len(attacks)
    if n == 0:
        return

    fig = plt.figure(figsize=(12, 6.0))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.10, 0.20, 0.85, 0.62])
    ax.set_facecolor(PANEL)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)

    width = 0.36
    x = np.arange(n)
    stats = _compute_paired_stats(attacks, by_attack)
    bars_ed, bars_dr = _draw_paired_bars(ax, x, width, attacks, stats)
    _annotate_bars(ax, bars_ed, bars_dr, stats)
    _decorate_axes_and_legend(ax, x, attacks)

    fig.text(
        0.06, 0.91, "TOOLS vs REASONING DISRUPTION", color=TEXT, fontsize=22, fontweight="bold"
    )
    fig.text(
        0.06,
        0.875,
        "Tool-level edit distance (left, solid) vs CoT semantic drift (right, hatched). "
        "When the right bar is taller, the attack corrupts reasoning faster than it flips tools.",
        color=TEXT_MUTED,
        fontsize=10.5,
    )

    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
