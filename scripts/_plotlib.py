"""Shared plotting helpers for figure-generation scripts.

Pure functions only. Callers own ``plt.savefig``.

Replaces the per-script duplicates:
- ``_load`` / ``_load_records`` / ``load_records``  → :func:`load_records`
- ``_despine``                                     → :func:`despine`
- ``_panel`` / ``_panel_label``                    → :func:`panel_label`
- ``make_palette`` / ``_tool_palette``             → :func:`tool_palette`
- ``edits``                                        → :func:`edits`
- ``cot_drifts``                                   → :func:`cot_drifts`
- ``has_cot``                                      → :func:`has_cot`
- ``flip_rate`` / ``step1_flip_rate``              → :func:`step1_flip_rate`
- ``bootstrap_ci``                                 → :func:`bootstrap_ci`
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

PALETTE20: tuple[tuple[float, float, float], ...] = tuple(
    plt.get_cmap("tab20").colors  # type: ignore[attr-defined]
)


def load_records(*paths: str | Path, strict: bool = False) -> list[dict[str, Any]]:
    """Load and concatenate JSONL records from one or more files.

    Missing files are skipped with a ``logging.WARNING`` (previously
    silent — empty plots are easy to miss when a path drifts from the
    runner's output dir). Empty lines are ignored.

    If ``strict=True``, raises ``ValueError`` when zero records are
    collected across all paths — use this in figure scripts that should
    not produce empty PNGs silently.
    """
    out: list[dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            logger.warning("load_records: skipping missing file %s", path)
            continue
        with path.open("r", encoding="utf-8") as f:
            out.extend(json.loads(line) for line in f if line.strip())
    if strict and not out:
        raise ValueError(
            f"load_records(strict=True) collected 0 records from {len(paths)} "
            f"path(s); refusing to render an empty figure"
        )
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


def edits(recs: Iterable[dict]) -> np.ndarray:
    """Per-record ``edit_distance_norm`` as a float64 array.

    Hoisted from ``scripts/hero/_common.py`` and
    ``scripts/attack_landscape/_common.py`` (identical bodies).
    """
    return np.array([r["edit_distance_norm"] for r in recs], dtype=np.float64)


def cot_drifts(recs: Iterable[dict]) -> np.ndarray:
    """Per-record ``cot_drift_score``; NaN where the field is absent."""
    out: list[float] = []
    for r in recs:
        v = r.get("cot_drift_score")
        out.append(float(v) if v is not None else float("nan"))
    return np.array(out, dtype=np.float64)


def has_cot(by_attack: dict[str, list[dict]]) -> bool:
    """True iff at least one record across all attacks carries ``cot_drift_score``.

    Lets figure dispatchers skip CoT panels gracefully when records pre-date
    schema v0.4.0.
    """
    return any(any("cot_drift_score" in r for r in recs) for recs in by_attack.values())


def step1_flip_rate(recs: list[dict]) -> float:
    """Fraction of samples whose first attacked tool differs from first benign tool.

    Hoisted from ``hero._common.step1_flip_rate`` and
    ``attack_landscape._common.flip_rate`` (byte-identical bodies). Both legacy
    names re-export this symbol from their package's ``_common.py``.
    """
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


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``(0.0, 0.0)`` for an empty input. Output is always cast to
    Python ``float`` (canonicalizes the legacy ``hero._common`` variant which
    returned numpy scalars).
    """
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
