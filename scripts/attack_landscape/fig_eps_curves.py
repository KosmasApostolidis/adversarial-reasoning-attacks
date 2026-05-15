"""Figure 2: per-attack epsilon vs edit-distance with 95% bootstrap CI bands."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import ATTACK_ORDER, LABELS, PALETTE, bootstrap_ci


def _plot_one_attack(ax, name: str, recs: list[dict]) -> None:
    groups: dict[float, list[float]] = defaultdict(list)
    for r in recs:
        groups[float(r["epsilon"])].append(r["edit_distance_norm"])
    xs = sorted(groups)
    if not xs:
        return
    ys_mean = np.array([np.mean(groups[e]) for e in xs])
    ci = np.array([bootstrap_ci(np.asarray(groups[e])) for e in xs])
    ys_lo, ys_hi = ci[:, 0], ci[:, 1]
    if len(xs) == 1:
        ax.errorbar(
            xs,
            ys_mean,
            yerr=[ys_mean - ys_lo, ys_hi - ys_mean],
            fmt="*",
            color=PALETTE[name],
            markersize=18,
            markeredgecolor="white",
            markeredgewidth=1.0,
            capsize=5,
            label=f"{LABELS[name]} (smoke only)",
        )
    else:
        ax.plot(
            xs,
            ys_mean,
            marker="o",
            color=PALETTE[name],
            linewidth=2.5,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=LABELS[name],
            zorder=3,
        )
        ax.fill_between(xs, ys_lo, ys_hi, color=PALETTE[name], alpha=0.20, zorder=2)


def fig_eps_curves(by_attack: dict[str, list[dict]], out_path: Path) -> None:
    eps_vals = sorted({float(r["epsilon"]) for recs in by_attack.values() for r in recs})
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name in ATTACK_ORDER:
        _plot_one_attack(ax, name, by_attack[name])
    ax.set_xlabel("Perturbation budget ε (normalised pixel domain)", fontsize=12)
    ax.set_ylabel("Mean normalised trajectory edit distance", fontsize=12)
    ax.set_title("Attack effectiveness vs ε (95% bootstrap CI bands)", fontsize=13, pad=12)
    ax.set_xticks(eps_vals)
    ax.set_xticklabels([f"{e:.4g}\n({round(e * 255)}/255)" for e in eps_vals], fontsize=10)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.95, frameon=True, edgecolor="#dddddd", fontsize=10)
    ax.set_ylim(0, max(0.85, ax.get_ylim()[1]))
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
