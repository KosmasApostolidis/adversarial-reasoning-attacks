"""Shared theme, palette, helpers, and IO for reasoning-flow figure modules."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _plotlib import load_records as _load  # noqa: E402
from _theme import DARK_AX, DARK_BG, DARK_FG, SHORT  # noqa: E402

OUT = Path("paper/figures/reasoning_flow")
OUT.mkdir(parents=True, exist_ok=True)

DARK_GRID = "#21262d"  # local: differs from comprehensive (#30363d)
C_BENIGN = "#58a6ff"  # blue
C_NOISE = "#3fb950"  # green
C_PGD = "#f85149"  # red
C_BOTH = "#d2a8ff"  # purple (shared transitions)
C_CHANGE = "#ffa657"  # orange (changed steps)


def _s(t):
    return SHORT.get(t, t.replace("_", " "))


def _dark_fig(w, h):
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    return fig


def _dark_ax(ax):
    ax.set_facecolor(DARK_AX)
    for s in ax.spines.values():
        s.set_color(DARK_GRID)
    ax.tick_params(colors=DARK_FG, which="both")
    ax.xaxis.label.set_color(DARK_FG)
    ax.yaxis.label.set_color(DARK_FG)
    ax.title.set_color(DARK_FG)


__all__ = [
    "C_BENIGN",
    "C_BOTH",
    "C_CHANGE",
    "C_NOISE",
    "C_PGD",
    "DARK_AX",
    "DARK_BG",
    "DARK_FG",
    "DARK_GRID",
    "OUT",
    "SHORT",
    "_dark_ax",
    "_dark_fig",
    "_load",
    "_s",
]
