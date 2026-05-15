"""Stat 1: 4-panel overview (violin/box, scatter, correlation, grouped bars)."""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as scipy_stats

from ._common import (
    C_BENIGN,
    C_LLAVA,
    C_NOISE,
    C_PGD,
    STAT_OUT,
    _panel,
    _s,
    despine,
    load_records,
)


def _draw_violin_box(ax, data, colors):
    vp = ax.violinplot(data, positions=[1, 2, 3], showmedians=False, showextrema=False, widths=0.55)
    for pc, c in zip(vp["bodies"], colors, strict=False):
        pc.set_facecolor(c)
        pc.set_alpha(0.35)
        pc.set_edgecolor(c)
        pc.set_linewidth(1.5)

    bp = ax.boxplot(
        data,
        positions=[1, 2, 3],
        widths=0.18,
        patch_artist=True,
        medianprops=dict(color="white", lw=2.5),
        whiskerprops=dict(lw=1.5),
        capprops=dict(lw=1.5),
        flierprops=dict(marker="d", ms=5, alpha=0.5),
    )
    for box, c in zip(bp["boxes"], colors, strict=False):
        box.set_facecolor(c)
        box.set_alpha(0.85)


def _draw_strip_with_means(ax, data, colors, rng):
    for xi, arr, c in zip([1, 2, 3], data, colors, strict=False):
        jit = rng.uniform(-0.08, 0.08, len(arr))
        ax.scatter(xi + jit, arr, s=55, color=c, edgecolors="white", lw=0.8, zorder=5)
        ax.text(
            xi,
            arr.max() + 0.07,
            f"μ={arr.mean():.3f}",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color=c,
        )


def _draw_panel_a(ax, nd, pd_, ld):
    data = [nd, pd_, ld]
    labels = ["Qwen\nNoise", "Qwen\nPGD", "LLaVA\nNoise"]
    colors = [C_NOISE, C_PGD, C_LLAVA]
    _draw_violin_box(ax, data, colors)

    rng = np.random.default_rng(42)
    _draw_strip_with_means(ax, data, colors, rng)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Normalised edit distance")
    ax.set_ylim(bottom=0)
    ax.set_title("Edit-distance distribution by condition", pad=8)
    despine(ax)
    _panel(ax, "A")


def _draw_scatter_points(ax, b_lens_n, b_lens_p, nd, pd_):
    ax.scatter(
        b_lens_n,
        nd,
        s=90,
        color=C_NOISE,
        edgecolors="white",
        lw=0.8,
        label="Noise",
        zorder=4,
        alpha=0.9,
    )
    ax.scatter(
        b_lens_p,
        pd_,
        s=90,
        color=C_PGD,
        edgecolors="white",
        lw=0.8,
        label="PGD",
        zorder=4,
        alpha=0.9,
    )


def _draw_regression_lines(ax, b_lens_n, b_lens_p, nd, pd_):
    for xs, ys, c in [(b_lens_n, nd, C_NOISE), (b_lens_p, pd_, C_PGD)]:
        if len(set(xs)) > 1:
            m, b, r, *_ = scipy_stats.linregress(xs, ys)
            xr = np.linspace(min(xs), max(xs), 50)
            ax.plot(xr, m * xr + b, color=c, lw=1.8, alpha=0.6, linestyle="--", label=f"r={r:.2f}")


def _draw_panel_b(ax, noise_r, pgd_r, nd, pd_):
    b_lens_n = [len(r["benign"]["tool_sequence"]) for r in noise_r]
    b_lens_p = [len(r["benign"]["tool_sequence"]) for r in pgd_r]
    _draw_scatter_points(ax, b_lens_n, b_lens_p, nd, pd_)
    _draw_regression_lines(ax, b_lens_n, b_lens_p, nd, pd_)
    ax.set_xlabel("Benign trajectory length (steps)")
    ax.set_ylabel("Normalised edit distance")
    ax.set_title("Benign length vs attack drift", pad=8)
    ax.legend(fontsize=9)
    despine(ax)
    _panel(ax, "B")


def _compute_corr_features(all_r):
    feats = {
        "ε": [r["epsilon"] for r in all_r],
        "b_len": [len(r["benign"]["tool_sequence"]) for r in all_r],
        "a_len": [len(r["attacked"]["tool_sequence"]) for r in all_r],
        "edit_dist": [r["edit_distance_norm"] for r in all_r],
        "pgd_loss": [r["attacked"]["metadata"]["pgd_loss_final"] for r in all_r],
        "elapsed_s": [r["elapsed_s"] for r in all_r],
    }
    feat_names = list(feats.keys())
    arr = np.array([feats[k] for k in feat_names])
    corr = np.corrcoef(arr)
    return feat_names, corr


def _annotate_corr_cells(ax, corr, feat_names):
    for i in range(len(feat_names)):
        for j in range(len(feat_names)):
            v = corr[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="black" if abs(v) < 0.5 else "white",
                fontweight="bold",
            )


def _draw_panel_c(fig, ax, pgd_r):
    feat_names, corr = _compute_corr_features(pgd_r)
    cmap = LinearSegmentedColormap.from_list("rwb", ["#2166ac", "white", "#d73027"], N=256)
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=9)
    _annotate_corr_cells(ax, corr, feat_names)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Pearson r", fontsize=9)
    ax.set_title("Feature correlation matrix (PGD records)", pad=8)
    _panel(ax, "C")


def _tool_counts(records, key, all_tools):
    c: Counter = Counter()
    for r in records:
        c.update(r[key]["tool_sequence"])
    return [c.get(t, 0) for t in all_tools]


def _draw_panel_d(ax, noise_r, pgd_r):
    all_tools = sorted(
        {
            t
            for r in pgd_r + noise_r
            for seq in [r["benign"]["tool_sequence"], r["attacked"]["tool_sequence"]]
            for t in seq
        }
    )
    n_tools = len(all_tools)
    x = np.arange(n_tools)
    w = 0.25

    b_cnt = _tool_counts(pgd_r, "benign", all_tools)
    n_cnt = _tool_counts(noise_r, "attacked", all_tools)
    p_cnt = _tool_counts(pgd_r, "attacked", all_tools)

    ax.bar(x - w, b_cnt, w, color=C_BENIGN, label="Benign", edgecolor="white", lw=0.5)
    ax.bar(x, n_cnt, w, color=C_NOISE, label="Noise atk", edgecolor="white", lw=0.5)
    ax.bar(x + w, p_cnt, w, color=C_PGD, label="PGD atk", edgecolor="white", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([_s(t) for t in all_tools], fontsize=8.5)
    ax.set_ylabel("Total invocations (all patients)")
    ax.set_title("Tool invocation frequency by condition", pad=8)
    ax.legend(fontsize=9)
    despine(ax)
    _panel(ax, "D")


def stat1_overview():
    noise_r = load_records("runs/main/noise/records.jsonl")
    pgd_r = load_records("runs/main/pgd/records.jsonl")
    llava_r = [r for r in noise_r if "llava" in r.get("model_id", "").lower()]

    nd = np.array([r["edit_distance_norm"] for r in noise_r])
    pd_ = np.array([r["edit_distance_norm"] for r in pgd_r])
    ld = np.array([r["edit_distance_norm"] for r in llava_r])

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.subplots_adjust(hspace=0.42, wspace=0.35)

    _draw_panel_a(axes[0, 0], nd, pd_, ld)
    _draw_panel_b(axes[0, 1], noise_r, pgd_r, nd, pd_)
    _draw_panel_c(fig, axes[1, 0], pgd_r)
    _draw_panel_d(axes[1, 1], noise_r, pgd_r)

    fig.suptitle(
        "Comprehensive statistical overview of adversarial attack effects",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.savefig(STAT_OUT / "stat1_overview.png", bbox_inches="tight")
    plt.close(fig)
    print("stat1 ✓")
