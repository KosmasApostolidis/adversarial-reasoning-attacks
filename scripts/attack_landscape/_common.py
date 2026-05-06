"""Shared theme, palette, helpers, and IO for attack-landscape figure modules."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _plotlib import load_records  # noqa: E402

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


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def edits(recs: Iterable[dict]) -> np.ndarray:
    return np.array([r["edit_distance_norm"] for r in recs], dtype=np.float64)


def flip_rate(recs: list[dict]) -> float:
    """Fraction of samples whose first attacked tool differs from first benign tool."""
    flips, total = 0, 0
    for r in recs:
        b = r.get("benign", {}).get("tool_sequence", []) or []
        a = r.get("attacked", {}).get("tool_sequence", []) or []
        if not b:
            continue
        total += 1
        if not a or a[0] != b[0]:
            flips += 1
    return flips / total if total else 0.0


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
    "edits",
    "flip_rate",
    "load_records",
    "loss_finals",
]
