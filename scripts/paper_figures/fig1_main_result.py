"""Figure 1: 3-panel main result — boxplot, per-sample bars, ε-sweep."""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from ._common import (
    C_NOISE,
    C_PGD,
    OUT,
    _panel_label,
    despine,
    load_records,
)


def _compute_eps_sweep(
    sweep_recs: list[dict],
) -> tuple[list[float], list[float], list[float]]:
    per_eps: dict[float, list[float]] = defaultdict(list)
    for r in sweep_recs:
        per_eps[r["epsilon"]].append(r["edit_distance_norm"])
    eps_sorted = sorted(per_eps)
    means = [np.mean(per_eps[e]) for e in eps_sorted]
    stds = [np.std(per_eps[e], ddof=1) if len(per_eps[e]) > 1 else 0 for e in eps_sorted]
    return eps_sorted, means, stds


def _draw_panel_a_box(ax, nd: np.ndarray, pd_: np.ndarray, eps: float) -> None:
    bp = ax.boxplot(
        [nd, pd_],
        tick_labels=["Uniform\nnoise", "PGD-L∞\n(20 steps)"],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=5, linestyle="none"),
    )
    bp["boxes"][0].set_facecolor(C_NOISE)
    bp["boxes"][1].set_facecolor(C_PGD)
    bp["boxes"][0].set_alpha(0.85)
    bp["boxes"][1].set_alpha(0.85)

    for i, arr in enumerate([nd, pd_], start=1):
        ax.scatter(
            np.full(len(arr), i) + np.random.default_rng(42).uniform(-0.1, 0.1, len(arr)),
            arr,
            s=40,
            zorder=5,
            color=[C_NOISE, C_PGD][i - 1],
            edgecolors="white",
            linewidths=0.6,
        )
        ax.text(i, arr.max() + 0.06, f"μ={arr.mean():.3f}", ha="center", fontsize=9, color="black")

    ax.set_ylabel("Normalised edit distance")
    ax.set_title(f"Attack effectiveness at ε = {eps:.4f}", pad=8)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    despine(ax)


def _draw_panel_b_bars(ax, pgd_recs: list[dict], nd: np.ndarray, pd_: np.ndarray) -> None:
    n = len(pgd_recs)
    x = np.arange(n)
    w = 0.35
    pids = [r["sample_id"].split("_p")[1].replace("_s", " s") for r in pgd_recs]
    ax.bar(x - w / 2, nd, w, color=C_NOISE, label="Noise", edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, pd_, w, color=C_PGD, label="PGD", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(pids, fontsize=8)
    ax.set_xlabel("Patient ID")
    ax.set_ylabel("Normalised edit distance")
    ax.set_title("Per-patient comparison", pad=8)
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    despine(ax)


def _draw_panel_c_eps(ax, eps_sorted: list[float], means: list[float], stds: list[float]) -> None:
    ax.errorbar(
        [e * 255 for e in eps_sorted],
        means,
        yerr=stds,
        marker="o",
        linewidth=2,
        markersize=7,
        capsize=4,
        color=C_NOISE,
        label="Uniform noise (Qwen)",
    )
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("ε (pixel units, ×255)")
    ax.set_ylabel("Mean normalised edit distance")
    ax.set_title("Dose–response (uniform noise)", pad=8)
    ax.set_xticks([e * 255 for e in eps_sorted])
    ax.set_xticklabels(["2", "4", "8", "16"])
    ax.legend()
    despine(ax)


def fig1_main_result() -> None:
    noise_recs = load_records("runs/main/noise/records.jsonl")
    pgd_recs = load_records("runs/main/pgd/records.jsonl")
    sweep_recs = [
        r
        for r in load_records("runs/main/noise/records.jsonl")
        if "qwen" in r.get("model_id", "").lower()
    ]

    nd = np.array([r["edit_distance_norm"] for r in noise_recs])
    pd_ = np.array([r["edit_distance_norm"] for r in pgd_recs])
    eps = pgd_recs[0]["epsilon"]
    eps_sorted, means, stds = _compute_eps_sweep(sweep_recs)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.subplots_adjust(wspace=0.38)

    _draw_panel_a_box(axes[0], nd, pd_, eps)
    _panel_label(axes[0], "A")

    _draw_panel_b_bars(axes[1], pgd_recs, nd, pd_)
    _panel_label(axes[1], "B")

    _draw_panel_c_eps(axes[2], eps_sorted, means, stds)
    _panel_label(axes[2], "C")

    fig.suptitle(
        "Adversarial perturbations alter VLM agent tool-call trajectories on prostate MRI",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(OUT / "fig1_main_result.png")
    plt.close(fig)
    print("fig1 ✓")
