"""Figure 1: 2x2 landscape overview composite (box, bar, eps curve, flip rate)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import ATTACK_ORDER, LABELS, PALETTE, bootstrap_ci, edits, flip_rate


def _render_box_panel(ax_box, by_attack: dict[str, list[dict]]) -> None:
    data = [edits(by_attack[a]) for a in ATTACK_ORDER]
    parts = ax_box.violinplot(
        data,
        positions=range(len(ATTACK_ORDER)),
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, name in zip(parts["bodies"], ATTACK_ORDER, strict=True):
        body.set_facecolor(PALETTE[name])
        body.set_edgecolor(PALETTE[name])
        body.set_alpha(0.45)
    for i, (name, arr) in enumerate(zip(ATTACK_ORDER, data, strict=True)):
        if arr.size == 0:
            continue
        jitter = np.random.default_rng(7).uniform(-0.08, 0.08, arr.size)
        ax_box.scatter(
            np.full_like(arr, i, dtype=float) + jitter,
            arr,
            s=14,
            color=PALETTE[name],
            alpha=0.85,
            edgecolor="white",
            linewidths=0.4,
            zorder=3,
        )
        med = float(np.median(arr))
        ax_box.hlines(med, i - 0.22, i + 0.22, color="black", linewidth=1.6, zorder=4)
    ax_box.set_xticks(range(len(ATTACK_ORDER)))
    ax_box.set_xticklabels([LABELS[a] for a in ATTACK_ORDER], rotation=12, ha="right")
    ax_box.set_ylabel("Normalised trajectory edit distance")
    ax_box.set_title("(a)  Distribution per attack", loc="left", fontweight="bold", fontsize=12)
    ax_box.set_ylim(-0.05, 1.05)
    ax_box.axhline(0, color="#cccccc", linewidth=0.6)


def _render_bar_panel(ax_bar, by_attack: dict[str, list[dict]]) -> None:
    means = [edits(by_attack[a]).mean() if edits(by_attack[a]).size else 0.0 for a in ATTACK_ORDER]
    cis = [bootstrap_ci(edits(by_attack[a])) for a in ATTACK_ORDER]
    err_lo = np.array([m - lo for m, (lo, _) in zip(means, cis, strict=False)])
    err_hi = np.array([hi - m for m, (_, hi) in zip(means, cis, strict=False)])
    bars = ax_bar.bar(
        range(len(ATTACK_ORDER)),
        means,
        color=[PALETTE[a] for a in ATTACK_ORDER],
        alpha=0.92,
        yerr=[err_lo, err_hi],
        capsize=6,
        error_kw={"linewidth": 1.3, "ecolor": "#333333"},
        edgecolor="white",
        linewidth=1.0,
    )
    ax_bar.set_xticks(range(len(ATTACK_ORDER)))
    ax_bar.set_xticklabels([LABELS[a] for a in ATTACK_ORDER], rotation=12, ha="right")
    ax_bar.set_ylabel("Mean ± 95% bootstrap CI")
    ax_bar.set_title("(b)  Mean attack effectiveness", loc="left", fontweight="bold", fontsize=12)
    for b, m, n in zip(bars, means, [edits(by_attack[a]).size for a in ATTACK_ORDER], strict=True):
        ax_bar.text(
            b.get_x() + b.get_width() / 2,
            m + 0.025,
            f"{m:.3f}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#222222",
        )
    ax_bar.set_ylim(0, max(means + [hi for _, hi in cis]) * 1.20 + 0.05)


def _draw_eps_one_attack(ax_eps, name: str, recs: list[dict]) -> None:
    groups = defaultdict(list)
    for r in recs:
        groups[float(r["epsilon"])].append(r["edit_distance_norm"])
    if len(groups) < 2 and name != "pgd":
        pass
    xs = sorted(groups)
    if not xs:
        return
    ys_mean, ys_lo, ys_hi = [], [], []
    for e in xs:
        arr = np.asarray(groups[e])
        ys_mean.append(arr.mean())
        lo, hi = bootstrap_ci(arr)
        ys_lo.append(lo)
        ys_hi.append(hi)
    if len(xs) == 1:
        ax_eps.errorbar(
            xs,
            ys_mean,
            yerr=[[ys_mean[0] - ys_lo[0]], [ys_hi[0] - ys_mean[0]]],
            fmt="*",
            color=PALETTE[name],
            markersize=15,
            capsize=5,
            label=f"{LABELS[name]} (smoke only)",
        )
    else:
        ax_eps.plot(
            xs,
            ys_mean,
            marker="o",
            color=PALETTE[name],
            linewidth=2.2,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=LABELS[name],
        )
        ax_eps.fill_between(xs, ys_lo, ys_hi, color=PALETTE[name], alpha=0.18)


def _render_eps_panel(ax_eps, by_attack: dict[str, list[dict]], eps_vals: list[float]) -> None:
    for name in ATTACK_ORDER:
        _draw_eps_one_attack(ax_eps, name, by_attack[name])
    ax_eps.set_xlabel("ε (normalised pixel domain)")
    ax_eps.set_ylabel("Mean edit distance ± 95% CI")
    ax_eps.set_title(
        "(c)  Effectiveness vs perturbation budget", loc="left", fontweight="bold", fontsize=12
    )
    ax_eps.set_xticks(eps_vals)
    ax_eps.set_xticklabels([f"{e:.4g}\n({round(e * 255)}/255)" for e in eps_vals], fontsize=8.5)
    ax_eps.grid(linestyle=":", alpha=0.35)
    ax_eps.legend(loc="upper left", framealpha=0.9, frameon=True, edgecolor="#dddddd")


def _render_flip_panel(ax_flip, by_attack: dict[str, list[dict]]) -> None:
    flips = [flip_rate(by_attack[a]) for a in ATTACK_ORDER]
    bars = ax_flip.barh(
        range(len(ATTACK_ORDER)),
        flips,
        color=[PALETTE[a] for a in ATTACK_ORDER],
        alpha=0.92,
        edgecolor="white",
        linewidth=1.0,
    )
    ax_flip.set_yticks(range(len(ATTACK_ORDER)))
    ax_flip.set_yticklabels([LABELS[a] for a in ATTACK_ORDER])
    ax_flip.set_xlabel("Fraction of trajectories whose first tool flipped")
    ax_flip.set_title("(d)  Step-1 tool-flip rate", loc="left", fontweight="bold", fontsize=12)
    ax_flip.set_xlim(0, 1.05)
    for b, f in zip(bars, flips, strict=True):
        ax_flip.text(
            f + 0.015,
            b.get_y() + b.get_height() / 2,
            f"{f:.0%}",
            va="center",
            fontsize=10,
            color="#222222",
        )
    ax_flip.invert_yaxis()
    ax_flip.grid(axis="x", linestyle=":", alpha=0.35)


def fig_landscape_overview(by_attack: dict[str, list[dict]], out_path: Path) -> None:
    fig = plt.figure(figsize=(13.5, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.18)
    ax_box = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_eps = fig.add_subplot(gs[1, 0])
    ax_flip = fig.add_subplot(gs[1, 1])

    eps_vals = sorted({float(r["epsilon"]) for recs in by_attack.values() for r in recs})

    _render_box_panel(ax_box, by_attack)
    _render_bar_panel(ax_bar, by_attack)
    _render_eps_panel(ax_eps, by_attack, eps_vals)
    _render_flip_panel(ax_flip, by_attack)

    fig.suptitle(
        "Adversarial Attack Landscape on Qwen2.5-VL-7B Medical Agent (ProstateX val=5)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(out_path)
    plt.close(fig)
