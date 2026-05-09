"""Shared theme, palette, helpers, and IO for attack-landscape figure modules."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _plotlib import (  # noqa: E402
    bootstrap_ci,
    cot_drifts,
    edits,
    has_cot,
    load_records,
    step1_flip_rate,
)

flip_rate = step1_flip_rate  # legacy alias for in-package callers

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

# Curated palette — distinct, perceptually ordered (cool→warm = weak→strong)
PALETTE = {
    "noise": "#9E9E9E",  # neutral grey
    "pgd": "#5E35B1",  # deep purple
    "apgd": "#C62828",  # red
    "targeted_tool": "#1565C0",  # blue
    "trajectory_drift": "#EF6C00",  # orange
}
LABELS = {
    "noise": "Uniform noise",
    "pgd": "PGD-L∞",
    "apgd": "APGD-L∞",
    "targeted_tool": "Targeted-Tool",
    "trajectory_drift": "Trajectory-Drift",
}
ATTACK_ORDER = ["noise", "pgd", "apgd", "targeted_tool", "trajectory_drift"]


def loss_finals(recs: list[dict], key_suffix: str) -> np.ndarray:
    out: list[float] = []
    for r in recs:
        meta = r.get("attacked", {}).get("metadata", {}) or {}
        v = meta.get(f"{key_suffix}_loss_final")
        if v is not None:
            out.append(float(v))
    return np.array(out, dtype=np.float64)


__all__ = [
    "ATTACK_ORDER",
    "LABELS",
    "PALETTE",
    "bootstrap_ci",
    "cot_drifts",
    "edits",
    "flip_rate",
    "has_cot",
    "load_records",
    "loss_finals",
]
