"""CoT-corruption axis: per-attack ε × cot_drift_score with bootstrap bands.

Mirrors fig_eps_curves but plots reasoning drift instead of edit distance.
Skips silently when no records carry cot_drift_score.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import ATTACK_ORDER, LABELS, PALETTE, bootstrap_ci, has_cot


def fig_cot_axis(by_attack: dict[str, list[dict]], out_path: Path) -> None:
    if not has_cot(by_attack):
        print("[fig_cot_axis] no cot_drift_score in records — skipping")
        return

    eps_vals = sorted({float(r["epsilon"]) for recs in by_attack.values() for r in recs})
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plotted_any = False
    for name in ATTACK_ORDER:
        recs = by_attack[name]
        groups: dict[float, list[float]] = defaultdict(list)
        for r in recs:
            v = r.get("cot_drift_score")
            if v is None:
                continue
            groups[float(r["epsilon"])].append(float(v))
        xs = sorted(groups)
        if not xs:
            continue
        plotted_any = True
        ys_mean = np.array([np.mean(groups[e]) for e in xs])
        ci = np.array([bootstrap_ci(np.asarray(groups[e])) for e in xs])
        ys_lo, ys_hi = ci[:, 0], ci[:, 1]
        if len(xs) == 1:
            ax.errorbar(
                xs, ys_mean,
                yerr=[ys_mean - ys_lo, ys_hi - ys_mean],
                fmt="*", color=PALETTE[name],
                markersize=18, markeredgecolor="white", markeredgewidth=1.0,
                capsize=5,
                label=f"{LABELS[name]} (smoke only)",
            )
        else:
            ax.plot(
                xs, ys_mean,
                marker="o", color=PALETTE[name],
                linewidth=2.5, markersize=8,
                markeredgecolor="white", markeredgewidth=0.8,
                label=LABELS[name], zorder=3,
            )
            ax.fill_between(xs, ys_lo, ys_hi, color=PALETTE[name], alpha=0.20, zorder=2)

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xlabel("Perturbation budget ε (normalised pixel domain)", fontsize=12)
    ax.set_ylabel("Mean cot_drift_score (NLI distance)", fontsize=12)
    ax.set_title("CoT corruption vs ε (95% bootstrap CI bands)", fontsize=13, pad=12)
    ax.set_xticks(eps_vals)
    ax.set_xticklabels([f"{e:.4g}\n({round(e * 255)}/255)" for e in eps_vals], fontsize=10)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.95, frameon=True, edgecolor="#dddddd", fontsize=10)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
