"""Shared theme, palette, and IO for paper-grade figure modules."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _plotlib import despine, load_records, tool_palette  # noqa: E402

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
    }
)

C_BENIGN = "#2166ac"  # blue
C_NOISE = "#92c5de"  # light blue
C_PGD = "#d6604d"  # red
C_LLAVA = "#f4a582"  # orange
C_ACCENT = "#1a9850"  # green
PALETTE20 = plt.get_cmap("tab20").colors

OUT = Path("paper/figures/paper")
OUT.mkdir(parents=True, exist_ok=True)


def _panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )


def _tool_palette(all_tools: list[str]) -> dict[str, tuple]:
    return tool_palette(all_tools, sort=True)


__all__ = [
    "C_ACCENT",
    "C_BENIGN",
    "C_LLAVA",
    "C_NOISE",
    "C_PGD",
    "OUT",
    "PALETTE20",
    "_panel_label",
    "_tool_palette",
    "despine",
    "load_records",
]
