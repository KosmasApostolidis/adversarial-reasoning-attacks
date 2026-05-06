"""Figure 4: Qwen vs LLaVA under uniform noise."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    C_BENIGN,
    C_LLAVA,
    OUT,
    _panel_label,
    despine,
    load_records,
)


def fig4_cross_model() -> None:
    all_noise = load_records("runs/main/noise/records.jsonl")
    qwen_recs = [r for r in all_noise if "qwen" in r.get("model_id", "").lower()]
    llava_recs = [r for r in all_noise if "llava" in r.get("model_id", "").lower()]

    qd = np.array([r["edit_distance_norm"] for r in qwen_recs])
    ld = np.array([r["edit_distance_norm"] for r in llava_recs])
    eps = qwen_recs[0]["epsilon"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.36)

    # Violin + strip
    ax = axes[0]
    parts = ax.violinplot([qd, ld], positions=[1, 2], showmedians=True, showextrema=False)
    for pc, color in zip(parts["bodies"], [C_BENIGN, C_LLAVA], strict=False):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts["cmedians"].set_colors(["black", "black"])
    parts["cmedians"].set_linewidth(2)
    rng = np.random.default_rng(0)
    for xi, arr, c in [(1, qd, C_BENIGN), (2, ld, C_LLAVA)]:
        jitter = rng.uniform(-0.06, 0.06, len(arr))
        ax.scatter(
            np.full(len(arr), xi) + jitter,
            arr,
            s=50,
            zorder=5,
            color=c,
            edgecolors="white",
            linewidths=0.8,
        )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Qwen2.5-VL-7B", "LLaVA-v1.6-Mistral-7B"])
    ax.set_ylabel("Normalised edit distance")
    ax.set_title(f"Cross-model sensitivity at ε={eps:.4f}", pad=8)
    for xi, arr, c in [(1, qd, C_BENIGN), (2, ld, C_LLAVA)]:
        ax.text(
            xi,
            arr.max() + 0.07,
            f"μ={arr.mean():.3f}",
            ha="center",
            fontsize=9,
            color=c,
            fontweight="bold",
        )
    ax.set_ylim(bottom=0)
    despine(ax)
    _panel_label(ax, "A")

    # Per-patient grouped bars
    ax2 = axes[1]
    n = min(len(qwen_recs), len(llava_recs))
    x = np.arange(n)
    w = 0.35
    pids = [r["sample_id"].split("_p")[1].replace("_s", "·") for r in qwen_recs[:n]]
    ax2.bar(x - w / 2, qd[:n], w, color=C_BENIGN, label="Qwen2.5-VL", edgecolor="white")
    ax2.bar(x + w / 2, ld[:n], w, color=C_LLAVA, label="LLaVA-v1.6", edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(pids, fontsize=8)
    ax2.set_xlabel("Patient ID")
    ax2.set_ylabel("Normalised edit distance")
    ax2.set_title("Per-patient cross-model comparison", pad=8)
    ax2.legend()
    ax2.set_ylim(bottom=0)
    despine(ax2)
    _panel_label(ax2, "B")

    fig.suptitle(
        "LLaVA is more sensitive to uniform-noise perturbations than Qwen2.5-VL",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(OUT / "fig4_cross_model.png")
    plt.close(fig)
    print("fig4 ✓")
