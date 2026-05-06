"""Figure 5: violin grid showing edit-distance distribution per (attack, eps) cell."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from ._common import ATTACK_ORDER, LABELS, PALETTE


def fig_violin_grid(by_attack: dict[str, list[dict]], out_path: Path) -> None:
    attacks_eps = [a for a in ATTACK_ORDER if any(by_attack.get(a, []))]
    eps_vals = sorted({float(r["epsilon"]) for a in attacks_eps for r in by_attack[a]})

    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_attacks = len(attacks_eps)
    width = 0.78 / n_attacks
    positions_list = []
    for ai, name in enumerate(attacks_eps):
        groups = defaultdict(list)
        for r in by_attack[name]:
            groups[float(r["epsilon"])].append(r["edit_distance_norm"])
        xs = [eps_vals.index(e) for e in eps_vals if e in groups]
        data = [groups[eps_vals[x]] for x in xs]
        if not data:
            continue
        offset = (ai - (n_attacks - 1) / 2) * width
        positions = [x + offset for x in xs]
        positions_list.append((name, positions))
        parts = ax.violinplot(
            data, positions=positions, widths=width * 0.9, showmeans=True, showextrema=False
        )
        for body in parts["bodies"]:
            body.set_facecolor(PALETTE[name])
            body.set_edgecolor(PALETTE[name])
            body.set_alpha(0.65)
        if "cmeans" in parts:
            parts["cmeans"].set_color("black")
            parts["cmeans"].set_linewidth(1.2)
    ax.set_xticks(range(len(eps_vals)))
    ax.set_xticklabels([f"{e:.4g}\n({round(e * 255)}/255)" for e in eps_vals], fontsize=10)
    ax.set_xlabel("Perturbation budget ε", fontsize=12)
    ax.set_ylabel("Normalised trajectory edit distance", fontsize=12)
    ax.set_title("Edit-distance distribution per (attack, ε) cell", fontsize=13, pad=12)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[a], alpha=0.65) for a, _ in positions_list]
    labels = [LABELS[a] for a, _ in positions_list]
    ax.legend(handles, labels, loc="upper left", frameon=True, edgecolor="#dddddd", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
