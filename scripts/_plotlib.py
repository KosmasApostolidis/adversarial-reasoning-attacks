"""Shared plotting helpers for figure-generation scripts.

Pure functions only. Callers own ``plt.savefig``.

Replaces the per-script duplicates:
- ``_load`` / ``_load_records`` / ``load_records``  → :func:`load_records`
- ``_despine``                                     → :func:`despine`
- ``_panel`` / ``_panel_label``                    → :func:`panel_label`
- ``make_palette`` / ``_tool_palette``             → :func:`tool_palette`
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PALETTE20: tuple[tuple[float, float, float], ...] = tuple(
    plt.get_cmap("tab20").colors  # type: ignore[attr-defined]
)


def load_records(*paths: str | Path) -> list[dict[str, Any]]:
    """Load and concatenate JSONL records from one or more files.

    Missing files are silently skipped (matches existing
    ``make_attack_landscape`` / ``make_hero_figures`` behaviour). Empty lines
    are ignored.
    """
    out: list[dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out


def despine(ax: plt.Axes, *, top: bool = True, right: bool = True) -> None:
    """Hide top + right spines (canonical despine used across scripts)."""
    if top:
        ax.spines["top"].set_visible(False)
    if right:
        ax.spines["right"].set_visible(False)


def panel_label(
    ax: plt.Axes,
    letter: str,
    *,
    x: float = -0.13,
    y: float = 1.07,
    fontsize: int = 12,
    weight: str = "bold",
) -> None:
    """Add a panel-label letter at axes-relative ``(x, y)``."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=weight,
    )


def tool_palette(
    tools: Iterable[str],
    *,
    sort: bool = False,
    cmap_name: str = "tab20",
) -> dict[str, tuple[float, ...]]:
    """Return a stable color map ``{tool_name: rgba}``.

    ``sort=True`` reproduces ``make_paper_figures._tool_palette`` (alphabetical
    tools paired with PALETTE20). ``sort=False`` preserves caller-supplied
    order (matches ``make_figures.make_palette``).
    """
    items = sorted(tools) if sort else list(tools)
    cmap = plt.get_cmap(cmap_name)
    return {t: cmap(i % 20) for i, t in enumerate(items)}
