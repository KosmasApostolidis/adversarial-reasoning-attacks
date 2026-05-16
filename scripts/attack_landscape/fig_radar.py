"""Figure 3: radar chart of attacks across 5 metrics, column-normalised."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import ATTACK_ORDER, LABELS, PALETTE, edits, flip_rate

_METRIC_NAMES = [
    "Mean edit\ndistance",
    "Max edit\ndistance",
    "Step-1\nflip rate",
    "P95 edit\ndistance",
    "Trajectory\nlength Δ",
]


def _compute_metrics(name: str, by_attack: dict[str, list[dict]]) -> list[float]:
    recs = by_attack[name]
    eds = edits(recs)
    if eds.size == 0:
        return [0.0] * len(_METRIC_NAMES)
    bens_len = np.array([len(r.get("benign", {}).get("tool_sequence", []) or []) for r in recs])
    atts_len = np.array([len(r.get("attacked", {}).get("tool_sequence", []) or []) for r in recs])
    traj_delta = np.abs(atts_len - bens_len).mean() if bens_len.size else 0.0
    return [
        float(eds.mean()),
        float(eds.max()),
        flip_rate(recs),
        float(np.quantile(eds, 0.95)),
        float(traj_delta),
    ]


def _decorate_radar_axes(ax) -> None:
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8.5, color="#666666")
    ax.set_ylim(0, 1.12)
    ax.spines["polar"].set_color("#dddddd")
    ax.grid(color="#dddddd", linewidth=0.7)
    ax.set_title(
        "Attack profile across 5 metrics (column-normalised, larger = stronger attack)",
        fontsize=12,
        pad=22,
        fontweight="bold",
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), frameon=False, fontsize=10)


def fig_radar(by_attack: dict[str, list[dict]], out_path: Path) -> None:
    raw = {
        n: _compute_metrics(n, by_attack)
        for n in ATTACK_ORDER
        if n != "pgd" or len(by_attack[n]) > 0
    }
    arr = np.array([raw[n] for n in raw])
    col_max = np.where(arr.max(axis=0) == 0, 1.0, arr.max(axis=0))
    norm = arr / col_max

    angles = np.linspace(0, 2 * np.pi, len(_METRIC_NAMES), endpoint=False).tolist()
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for i, name in enumerate(raw):
        values = [*norm[i].tolist(), norm[i][0]]
        ax.plot(angles, values, color=PALETTE[name], linewidth=2.4, label=LABELS[name], zorder=3)
        ax.fill(angles, values, color=PALETTE[name], alpha=0.16, zorder=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(_METRIC_NAMES, fontsize=10)
    _decorate_radar_axes(ax)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
