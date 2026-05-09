"""Shared theme, palette, helpers for hero figures."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _plotlib import load_records  # noqa: E402

# ── Theme ──────────────────────────────────────────────────────────────────
BG = "#0B0F1A"
PANEL = "#141A29"
PANEL_LIGHT = "#1B2236"
TEXT = "#ECEFF4"
TEXT_MUTED = "#7A8499"
GRID = "#2A3147"
ACCENT = "#F2C94C"

PALETTE = {
    "noise": "#5C6B82",
    "pgd": "#B794F4",
    "apgd": "#FC8181",
    "targeted_tool": "#4FD1C5",
    "trajectory_drift": "#F6AD55",
}
LABELS = {
    "noise": "UNIFORM NOISE",
    "pgd": "PGD-L∞",
    "apgd": "APGD-L∞",
    "targeted_tool": "TARGETED-TOOL",
    "trajectory_drift": "TRAJECTORY-DRIFT",
}
ATTACK_ORDER = ["noise", "pgd", "apgd", "targeted_tool", "trajectory_drift"]

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.weight": "400",
        "axes.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.grid": False,
        "axes.titlecolor": TEXT,
        "figure.facecolor": BG,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.facecolor": BG,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.35,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
        "legend.labelcolor": TEXT,
        "text.color": TEXT,
    }
)


def edits(recs):
    return np.array([r["edit_distance_norm"] for r in recs], dtype=np.float64)


def cot_drifts(recs):
    """Per-record cot_drift_score; NaN where the field is absent."""
    out = []
    for r in recs:
        v = r.get("cot_drift_score")
        out.append(float(v) if v is not None else np.nan)
    return np.array(out, dtype=np.float64)


def faith_drops(recs):
    """Benign minus attacked faithfulness, per record. NaN if either is absent."""
    out = []
    for r in recs:
        b = r.get("cot_faithfulness_benign")
        a = r.get("cot_faithfulness_attacked")
        out.append(float(b) - float(a) if (b is not None and a is not None) else np.nan)
    return np.array(out, dtype=np.float64)


def hall_deltas(recs):
    """Attacked minus benign hallucination rate. NaN if either is absent."""
    out = []
    for r in recs:
        b = r.get("cot_hallucination_benign")
        a = r.get("cot_hallucination_attacked")
        out.append(float(a) - float(b) if (b is not None and a is not None) else np.nan)
    return np.array(out, dtype=np.float64)


def refusal_rates(recs):
    """(benign_rate, attacked_rate) over the record list. None if absent."""
    bs = [r.get("cot_refusal_benign") for r in recs if r.get("cot_refusal_benign") is not None]
    as_ = [r.get("cot_refusal_attacked") for r in recs if r.get("cot_refusal_attacked") is not None]
    if not bs and not as_:
        return (None, None)
    return (
        float(np.mean(bs)) if bs else None,
        float(np.mean(as_)) if as_ else None,
    )


def has_cot(by_attack: dict[str, list[dict]]) -> bool:
    """Returns True iff at least one record across all attacks carries
    cot_drift_score. Lets figure dispatchers skip CoT panels gracefully
    when records are pre-v0.4.0."""
    for recs in by_attack.values():
        if any("cot_drift_score" in r for r in recs):
            return True
    return False


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return tuple(np.quantile(boots, [alpha / 2, 1 - alpha / 2]))


def step1_flip_rate(recs):
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


def gather() -> dict[str, list[dict]]:
    root = Path(__file__).resolve().parents[2]
    return {
        "noise": load_records(root / "runs/main/noise/records.jsonl"),
        "pgd": load_records(root / "runs/main/pgd/records.jsonl"),
        "apgd": load_records(root / "runs/main/apgd/records.jsonl"),
        "targeted_tool": load_records(root / "runs/main/targeted_tool/records.jsonl"),
        "trajectory_drift": load_records(root / "runs/main/trajectory_drift/records.jsonl"),
    }


def add_panel(fig_or_ax, x, y, w, h, *, fc=PANEL, ec=GRID, alpha=1.0, radius=0.012):
    """Draw a rounded panel as a FancyBboxPatch in figure or axes coords."""
    is_fig = isinstance(fig_or_ax, plt.Figure)
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.001,rounding_size={radius}",
        linewidth=0.8,
        edgecolor=ec,
        facecolor=fc,
        alpha=alpha,
        transform=(fig_or_ax.transFigure if is_fig else fig_or_ax.transAxes),
        zorder=0,
    )
    fig_or_ax.patches.append(box) if is_fig else fig_or_ax.add_patch(box)


def beeswarm_y(values, max_width=0.32, sigma=0.018):
    """Compute y-jitter for a 1-D array, packing points without overlap."""
    n = len(values)
    if n == 0:
        return np.array([])
    bin_w = sigma
    bins = np.round(values / bin_w).astype(int)
    jitter = np.zeros(n)
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        k = len(idx)
        offsets = (np.arange(k) - (k - 1) / 2) * (max_width / max(k, 1)) * 0.85
        rng = np.random.default_rng(int(b) + 13)
        rng.shuffle(offsets)
        jitter[idx] = offsets
    return jitter


def fmt_eps(e: float) -> str:
    return f"{round(e * 255)}/255"
