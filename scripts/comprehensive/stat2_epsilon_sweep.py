"""Stat 2: ε sweep dose-response + attack mode comparison."""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from ._common import C_NOISE, C_PGD, STAT_OUT, _panel, despine, load_records


def _compute_sweep_stats(sweep):
    eps_vals = sorted(set(r["epsilon"] for r in sweep))
    per_eps: dict[float, list] = defaultdict(list)
    for r in sweep:
        per_eps[r["epsilon"]].append(r["edit_distance_norm"])
    means = [np.mean(per_eps[e]) for e in eps_vals]
    sems = [scipy_stats.sem(per_eps[e]) if len(per_eps[e]) > 1 else 0 for e in eps_vals]
    return eps_vals, per_eps, means, sems


def _draw_sweep_errorbar(ax, eps_vals, means, sems):
    ax.errorbar(
        [e * 255 for e in eps_vals],
        means,
        yerr=sems,
        fmt="-o",
        color=C_NOISE,
        lw=2.5,
        ms=9,
        capsize=5,
        label="mean ± SEM (noise, Qwen)",
    )
    ax.fill_between(
        [e * 255 for e in eps_vals],
        [m - s for m, s in zip(means, sems, strict=False)],
        [m + s for m, s in zip(means, sems, strict=False)],
        alpha=0.18,
        color=C_NOISE,
    )


def _draw_sweep_strip(ax, eps_vals, per_eps, rng):
    for e in eps_vals:
        ys = per_eps[e]
        xs = [e * 255] * len(ys) + rng.uniform(-0.2, 0.2, len(ys))
        ax.scatter(xs, ys, s=50, color=C_NOISE, alpha=0.6, edgecolors="white", lw=0.6, zorder=5)


def _draw_sweep_panel(ax, eps_vals, per_eps, means, sems, pgd_r, rng):
    _draw_sweep_errorbar(ax, eps_vals, means, sems)
    _draw_sweep_strip(ax, eps_vals, per_eps, rng)

    ax.axhline(
        pgd_r[0]["edit_distance_norm"]
        if False
        else np.mean([r["edit_distance_norm"] for r in pgd_r]),
        color=C_PGD,
        lw=2,
        linestyle="--",
        label=f"PGD mean={np.mean([r['edit_distance_norm'] for r in pgd_r]):.3f}",
    )

    ax.set_xlabel("ε (pixel units × 255)", fontsize=11)
    ax.set_ylabel("Normalised edit distance", fontsize=11)
    ax.set_title("Dose–response: uniform noise (ε sweep)", pad=8)
    ax.set_xticks([e * 255 for e in eps_vals])
    ax.set_xticklabels(["2/255", "4/255", "8/255", "16/255"])
    ax.legend(fontsize=9)
    despine(ax)
    _panel(ax, "A")


def _draw_bars_with_strip(ax, xs, groups, rng):
    for xi, (_label, vals, c) in zip(xs, groups, strict=False):
        ax.bar(
            xi,
            np.mean(vals),
            width=0.5,
            color=c,
            alpha=0.75,
            edgecolor="white",
            lw=1.5,
            yerr=scipy_stats.sem(vals),
            capsize=7,
            error_kw=dict(elinewidth=2, capthick=2, ecolor="#444"),
        )
        jit = rng.uniform(-0.07, 0.07, len(vals))
        ax.scatter(xi + jit, vals, s=70, color=c, edgecolors="white", lw=1, zorder=5, alpha=0.9)


def _draw_significance_bracket(ax, nd, pd_):
    y_max = max(max(nd), max(pd_)) + 0.12
    ax.annotate(
        "", xy=(2, y_max), xytext=(1, y_max), arrowprops=dict(arrowstyle="-", color="black", lw=1.8)
    )
    ax.text(
        1.5,
        y_max + 0.03,
        "★  3× more drift",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=C_PGD,
    )


def _draw_comparison_panel(ax, nd, pd_, rng):
    groups = [("Uniform\nNoise", nd, C_NOISE), ("PGD-L∞\n20 steps", pd_, C_PGD)]
    xs = [1, 2]
    _draw_bars_with_strip(ax, xs, groups, rng)
    _draw_significance_bracket(ax, nd, pd_)

    ax.set_xticks(xs)
    ax.set_xticklabels(["Uniform\nNoise", "PGD-L∞\n20 steps"], fontsize=11)
    ax.set_ylabel("Normalised edit distance", fontsize=11)
    ax.set_title("Attack effectiveness at ε=0.0314 (8/255)\nQwen2.5-VL-7B, n=5 patients", pad=8)
    ax.set_ylim(bottom=0)
    despine(ax)
    _panel(ax, "B")


def stat2_epsilon_sweep():
    sweep = load_records("runs/main/noise/records.jsonl")
    pgd_r = load_records("runs/main/pgd/records.jsonl")

    eps_vals, per_eps, means, sems = _compute_sweep_stats(sweep)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.35)

    rng = np.random.default_rng(9)
    _draw_sweep_panel(axes[0], eps_vals, per_eps, means, sems, pgd_r, rng)

    noise_r = load_records("runs/main/noise/records.jsonl")
    nd = [r["edit_distance_norm"] for r in noise_r]
    pd_ = [r["edit_distance_norm"] for r in pgd_r]
    _draw_comparison_panel(axes[1], nd, pd_, rng)

    fig.suptitle(
        "ε-sweep dose–response and attack mode comparison", fontsize=13, fontweight="bold", y=1.02
    )
    fig.savefig(STAT_OUT / "stat2_epsilon_sweep.png", bbox_inches="tight")
    plt.close(fig)
    print("stat2 ✓")
