"""Stat 3: trajectory length distributions, CDF, before/after scatter."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    C_BENIGN,
    C_LLAVA,
    C_NOISE,
    C_PGD,
    STAT_OUT,
    _panel,
    despine,
    load_records,
)


def _draw_histogram(ax, datasets, bins):
    for label, vals, c in datasets:
        ax.hist(vals, bins=bins, alpha=0.45, color=c, edgecolor=c, lw=1.2, label=label)
    ax.set_xlabel("Trajectory length (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Trajectory length distribution", pad=8)
    ax.legend(fontsize=8)
    despine(ax)
    _panel(ax, "A")


def _draw_cdf(ax, datasets):
    for label, vals, c in datasets:
        s = sorted(vals)
        y = np.arange(1, len(s) + 1) / len(s)
        ax.step(s, y, color=c, lw=2, label=label)
    ax.set_xlabel("Trajectory length (steps)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF of trajectory lengths", pad=8)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    despine(ax)
    _panel(ax, "B")


def _draw_before_after_scatter(ax, noise_r, pgd_r):
    bl_n = [len(r["benign"]["tool_sequence"]) for r in noise_r]
    al_n = [len(r["attacked"]["tool_sequence"]) for r in noise_r]
    bl_p = [len(r["benign"]["tool_sequence"]) for r in pgd_r]
    al_p = [len(r["attacked"]["tool_sequence"]) for r in pgd_r]

    ax.scatter(bl_n, al_n, s=90, color=C_NOISE, edgecolors="white", lw=0.8, label="Noise", zorder=4)
    ax.scatter(bl_p, al_p, s=90, color=C_PGD, edgecolors="white", lw=0.8, label="PGD", zorder=4)
    lim = max(max(bl_n + bl_p), max(al_n + al_p)) + 1
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, alpha=0.4, label="b_len = a_len")
    ax.set_xlabel("Benign trajectory length")
    ax.set_ylabel("Attacked trajectory length")
    ax.set_title("Length before vs after attack", pad=8)
    ax.legend(fontsize=9)
    despine(ax)
    _panel(ax, "C")


def stat3_trajectory_lengths():
    noise_r = load_records("runs/main/noise/records.jsonl")
    pgd_r = load_records("runs/main/pgd/records.jsonl")
    llava_r = [r for r in noise_r if "llava" in r.get("model_id", "").lower()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.35)

    datasets = [
        ("Benign (Qwen)", [len(r["benign"]["tool_sequence"]) for r in pgd_r], C_BENIGN),
        ("Noise-atk (Qwen)", [len(r["attacked"]["tool_sequence"]) for r in noise_r], C_NOISE),
        ("PGD-atk (Qwen)", [len(r["attacked"]["tool_sequence"]) for r in pgd_r], C_PGD),
        ("Noise-atk (LLaVA)", [len(r["attacked"]["tool_sequence"]) for r in llava_r], C_LLAVA),
    ]

    _draw_histogram(axes[0], datasets, np.arange(0, 16, 1))
    _draw_cdf(axes[1], datasets)
    _draw_before_after_scatter(axes[2], noise_r, pgd_r)

    fig.suptitle(
        "Trajectory length analysis: how attacks shorten/extend reasoning chains",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(STAT_OUT / "stat3_trajectory_lengths.png", bbox_inches="tight")
    plt.close(fig)
    print("stat3 ✓")
