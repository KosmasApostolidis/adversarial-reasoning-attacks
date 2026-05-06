"""Figure 5: violin distributions across noise/PGD × Qwen/LLaVA."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    C_LLAVA,
    C_NOISE,
    C_PGD,
    OUT,
    despine,
    load_records,
)


def fig5_attack_landscape() -> None:
    all_noise = load_records("runs/main/noise/records.jsonl")
    all_pgd = load_records("runs/main/pgd/records.jsonl")
    noise_q = [r for r in all_noise if "qwen" in r.get("model_id", "").lower()]
    noise_l = [r for r in all_noise if "llava" in r.get("model_id", "").lower()]
    pgd_q = [r for r in all_pgd if "qwen" in r.get("model_id", "").lower()]

    groups = [
        ("Qwen\n(noise)", np.array([r["edit_distance_norm"] for r in noise_q]), C_NOISE),
        ("Qwen\n(PGD)", np.array([r["edit_distance_norm"] for r in pgd_q]), C_PGD),
        ("LLaVA\n(noise)", np.array([r["edit_distance_norm"] for r in noise_l]), C_LLAVA),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = [1, 2, 3]
    parts = ax.violinplot(
        [g[1] for g in groups],
        positions=positions,
        showmedians=True,
        showextrema=False,
        widths=0.5,
    )
    for pc, (_, _, c) in zip(parts["bodies"], groups, strict=False):
        pc.set_facecolor(c)
        pc.set_alpha(0.8)
    parts["cmedians"].set_colors(["black"] * len(groups))
    parts["cmedians"].set_linewidth(2.5)

    rng = np.random.default_rng(7)
    for xi, (_label, arr, c) in zip(positions, groups, strict=False):
        jitter = rng.uniform(-0.07, 0.07, len(arr))
        ax.scatter(
            np.full(len(arr), xi) + jitter,
            arr,
            s=55,
            zorder=5,
            color=c,
            edgecolors="white",
            linewidths=1.0,
        )
        ax.text(
            xi,
            arr.max() + 0.07,
            f"μ={arr.mean():.3f}",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color=c,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([g[0] for g in groups], fontsize=11)
    ax.set_ylabel("Normalised trajectory edit distance", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.set_title(
        "Attack landscape: trajectory drift by model and perturbation type",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    # Significance bracket (PGD vs noise, same model)
    y_bracket = max(r["edit_distance_norm"] for r in pgd_q) + 0.22
    ax.annotate(
        "",
        xy=(2, y_bracket),
        xytext=(1, y_bracket),
        arrowprops=dict(arrowstyle="-", color="black", lw=1.5),
    )
    ax.text(1.5, y_bracket + 0.03, "p < 0.05*", ha="center", fontsize=9, style="italic")

    despine(ax)
    fig.savefig(OUT / "fig5_attack_landscape.png")
    plt.close(fig)
    print("fig5 ✓")
