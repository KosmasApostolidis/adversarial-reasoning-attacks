"""Shared theme, palette, helpers, and IO for comprehensive figure modules."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _plotlib import despine, load_records  # noqa: E402
from _theme import DARK_AX, DARK_BG, DARK_FG, SHORT  # noqa: E402

STAT_OUT = Path("paper/figures/stats")
GRAPH_OUT = Path("paper/figures/graphs_v2")
STAT_OUT.mkdir(parents=True, exist_ok=True)
GRAPH_OUT.mkdir(parents=True, exist_ok=True)

C_BENIGN = "#2166ac"
C_NOISE = "#4dac26"
C_PGD = "#d73027"
C_LLAVA = "#f46d43"
C_ACCENT = "#762a83"
DARK_GRID = "#30363d"  # local: differs from reasoning_flow (#21262d)


def _s(t: str) -> str:
    return SHORT.get(t, t.replace("_", "\n"))


def _panel(ax, letter, x=-0.13, y=1.07):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")


def _dark_ax(ax):
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors=DARK_FG)
    for s in ax.spines.values():
        s.set_color(DARK_GRID)


__all__ = [
    "C_ACCENT",
    "C_BENIGN",
    "C_LLAVA",
    "C_NOISE",
    "C_PGD",
    "DARK_AX",
    "DARK_BG",
    "DARK_FG",
    "DARK_GRID",
    "GRAPH_OUT",
    "SHORT",
    "STAT_OUT",
    "_dark_ax",
    "_panel",
    "_s",
    "despine",
    "load_records",
]
