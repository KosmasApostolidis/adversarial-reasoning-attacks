"""Hero figure 3: attack × ε heatmap (edit-distance, drift, faithfulness)."""

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


def _record_value(r: dict, field: str) -> float | None:
    if field == "edit_distance_norm":
        return r.get(field)
    return r.get(field)


def _faith_drop(r: dict) -> float | None:
    b = r.get("cot_faithfulness_benign")
    a = r.get("cot_faithfulness_attacked")
    if b is None or a is None:
        return None
    return float(b) - float(a)


def _build_cells(
    by_attack, value_fn, eps_vals: list[float], attacks: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    cell = np.full((len(attacks), len(eps_vals)), np.nan)
    counts = np.zeros_like(cell, dtype=int)
    for i, a in enumerate(attacks):
        groups: dict[float, list[float]] = defaultdict(list)
        for r in by_attack[a]:
            v = value_fn(r)
            if v is None:
                continue
            groups[float(r["epsilon"])].append(float(v))
        for j, e in enumerate(eps_vals):
            if e in groups:
                cell[i, j] = float(np.mean(groups[e]))
                counts[i, j] = len(groups[e])
    return cell, counts


def _annotate_cells(ax, cell: np.ndarray, counts: np.ndarray, vmin: float, vmax: float) -> None:
    for i in range(cell.shape[0]):
        for j in range(cell.shape[1]):
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
                continue
            v = cell[i, j]
            color = "black" if v > (vmin + 0.65 * (vmax - vmin)) else TEXT
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


def _decorate_heatmap_axes(ax, eps_vals: list[float], attacks: list[str]) -> None:
    ax.set_xticks(range(len(eps_vals)))
    ax.set_xticklabels([fmt_eps(e) for e in eps_vals], color=TEXT_MUTED, fontsize=11)
    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels([LABELS[a] for a in attacks], color=TEXT, fontsize=11, fontweight="bold")
    ax.set_xlabel("Perturbation budget ε", color=TEXT_MUTED, fontsize=12)
    ax.tick_params(length=0)


def _add_colorbar(fig, im, cbar_label: str) -> None:
    cbar_ax = fig.add_axes([0.86, 0.20, 0.018, 0.55])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.outline.set_edgecolor(GRID)
    cbar.outline.set_linewidth(0.7)
    cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    cbar.set_label(cbar_label, color=TEXT_MUTED, fontsize=10)


def _add_titles(fig, title: str, subtitle: str) -> None:
    fig.text(0.06, 0.93, title, color=TEXT, fontsize=22, fontweight="bold")
    fig.text(0.06, 0.895, subtitle, color=TEXT_MUTED, fontsize=11)
    fig.text(
        0.06,
        0.05,
        "PGD evaluated only at smoke ε=8/255 (n=5); other attacks span full sweep (4 ε × 3 seeds × 5 samples = 60).",
        color=TEXT_MUTED,
        fontsize=9,
        alpha=0.85,
    )


def _render_heatmap(
    by_attack,
    out_path: Path,
    *,
    value_fn,
    cbar_label: str,
    title: str,
    subtitle: str,
    vmin: float = 0.0,
    vmax: float = 0.85,
) -> None:
    eps_vals = sorted({float(r["epsilon"]) for recs in by_attack.values() for r in recs})
    attacks_with_data = [a for a in ATTACK_ORDER if by_attack.get(a)]
    cell, counts = _build_cells(by_attack, value_fn, eps_vals, attacks_with_data)

    if np.all(np.isnan(cell)):
        return

    cmap = LinearSegmentedColormap.from_list(
        "ed_dark", [PANEL, "#3A3F5C", "#7A4F8B", PALETTE["apgd"], "#FFD37A"]
    )
    fig = plt.figure(figsize=(12, 6.8))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.18, 0.18, 0.65, 0.66])
    ax.set_facecolor(BG)
    masked = np.ma.masked_invalid(cell)
    cmap.set_bad(PANEL_LIGHT)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="upper")

    _annotate_cells(ax, cell, counts, vmin, vmax)
    _decorate_heatmap_axes(ax, eps_vals, attacks_with_data)
    _add_colorbar(fig, im, cbar_label)
    _add_titles(fig, title, subtitle)

    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)


def fig_heatmap(by_attack, out_path: Path) -> None:
    _render_heatmap(
        by_attack,
        out_path,
        value_fn=lambda r: r.get("edit_distance_norm"),
        cbar_label="Mean edit distance",
        title="ATTACK × BUDGET HEATMAP",
        subtitle="Mean normalised edit distance per (attack, ε) cell · brighter = more disruption",
        vmax=0.85,
    )


def fig_heatmap_drift(by_attack, out_path: Path) -> None:
    """CoT drift × ε heatmap. Skips silently if no rows carry cot_drift_score."""
    _render_heatmap(
        by_attack,
        out_path,
        value_fn=lambda r: r.get("cot_drift_score"),
        cbar_label="Mean CoT drift",
        title="COT DRIFT × BUDGET HEATMAP",
        subtitle="Mean cot_drift_score per (attack, ε) cell · brighter = more reasoning corruption",
        vmax=1.0,
    )


def fig_heatmap_faith(by_attack, out_path: Path) -> None:
    """Faithfulness drop (benign − attacked) × ε heatmap."""
    _render_heatmap(
        by_attack,
        out_path,
        value_fn=_faith_drop,
        cbar_label="Mean Δ faithfulness (benign − attacked)",
        title="FAITHFULNESS DROP × BUDGET HEATMAP",
        subtitle="How much the agent's CoT stops matching its tool calls under attack",
        vmin=0.0,
        vmax=1.0,
    )
