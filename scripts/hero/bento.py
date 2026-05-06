"""Hero figure 5: bento composite (magazine layout)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    ACCENT,
    ATTACK_ORDER,
    BG,
    GRID,
    LABELS,
    PALETTE,
    PANEL,
    TEXT,
    TEXT_MUTED,
    add_panel,
    bootstrap_ci,
    edits,
    fmt_eps,
    step1_flip_rate,
)


def fig_bento(by_attack, out_path: Path) -> None:
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor(BG)

    fig.text(0.04, 0.955, "ADVERSARIAL ATTACK DOSSIER", color=TEXT, fontsize=26, fontweight="bold")
    fig.text(
        0.04,
        0.928,
        "Qwen2.5-VL-7B · ProstateX val=5 · ε ∈ {2,4,8,16}/255 · seeds {0,1,2}",
        color=TEXT_MUTED,
        fontsize=11,
    )

    add_panel(fig, 0.04, 0.66, 0.24, 0.24, fc=PANEL, ec=GRID, radius=0.018)
    strongest = max(
        ATTACK_ORDER, key=lambda a: edits(by_attack[a]).mean() if edits(by_attack[a]).size else 0
    )
    s_eds = edits(by_attack[strongest])
    s_lo, s_hi = bootstrap_ci(s_eds)
    fig.text(0.06, 0.870, "STRONGEST ATTACK", color=TEXT_MUTED, fontsize=10, fontweight="bold")
    fig.text(
        0.06, 0.835, LABELS[strongest], color=PALETTE[strongest], fontsize=20, fontweight="bold"
    )
    fig.text(
        0.06,
        0.795,
        f"{s_eds.mean():.3f}",
        color=TEXT,
        fontsize=44,
        fontweight="bold",
        family="DejaVu Sans Mono",
        va="top",
    )
    fig.text(0.06, 0.720, "mean normalised edit distance", color=TEXT_MUTED, fontsize=9.5)
    fig.text(
        0.06,
        0.685,
        f"95% CI [{s_lo:.3f}, {s_hi:.3f}]   ·   n = {s_eds.size}",
        color=TEXT_MUTED,
        fontsize=9,
        family="DejaVu Sans Mono",
    )

    add_panel(fig, 0.30, 0.66, 0.24, 0.24, fc=PANEL, ec=GRID, radius=0.018)
    flippers = max(ATTACK_ORDER, key=lambda a: step1_flip_rate(by_attack[a]))
    flip_v = step1_flip_rate(by_attack[flippers])
    fig.text(
        0.32, 0.870, "BIGGEST STEP-1 DISRUPTOR", color=TEXT_MUTED, fontsize=10, fontweight="bold"
    )
    fig.text(0.32, 0.835, LABELS[flippers], color=PALETTE[flippers], fontsize=20, fontweight="bold")
    fig.text(
        0.32,
        0.795,
        f"{flip_v:.0%}",
        color=TEXT,
        fontsize=44,
        fontweight="bold",
        family="DejaVu Sans Mono",
        va="top",
    )
    fig.text(
        0.32, 0.720, "of trajectories had the first tool flipped", color=TEXT_MUTED, fontsize=9.5
    )
    fig.text(
        0.32,
        0.685,
        f"n = {len(by_attack[flippers])} records  ·  ε ∈ [2,16]/255",
        color=TEXT_MUTED,
        fontsize=9,
        family="DejaVu Sans Mono",
    )

    add_panel(fig, 0.56, 0.66, 0.20, 0.24, fc=PANEL, ec=GRID, radius=0.018)
    total = sum(len(by_attack[a]) for a in ATTACK_ORDER)
    fig.text(0.58, 0.870, "TOTAL RECORDS", color=TEXT_MUTED, fontsize=10, fontweight="bold")
    fig.text(
        0.58,
        0.815,
        f"{total}",
        color=ACCENT,
        fontsize=54,
        fontweight="bold",
        family="DejaVu Sans Mono",
        va="top",
    )
    fig.text(
        0.58,
        0.720,
        f"across {sum(1 for a in ATTACK_ORDER if by_attack[a])} attack modes",
        color=TEXT_MUTED,
        fontsize=9.5,
    )
    fig.text(
        0.58,
        0.685,
        "5 ProstateX samples × 4 ε × 3 seeds",
        color=TEXT_MUTED,
        fontsize=9,
        style="italic",
    )

    add_panel(fig, 0.78, 0.66, 0.18, 0.24, fc=PANEL, ec=GRID, radius=0.018)
    fig.text(0.80, 0.870, "TOTAL GPU TIME", color=TEXT_MUTED, fontsize=10, fontweight="bold")
    fig.text(
        0.80,
        0.815,
        "46:00",
        color=PALETTE["targeted_tool"],
        fontsize=54,
        fontweight="bold",
        family="DejaVu Sans Mono",
        va="top",
    )
    fig.text(0.80, 0.720, "minutes : seconds (sequential)", color=TEXT_MUTED, fontsize=9.5)
    fig.text(
        0.80, 0.685, "single H200 NVL · 3 sweeps", color=TEXT_MUTED, fontsize=9, style="italic"
    )

    ax = fig.add_axes([0.04, 0.08, 0.42, 0.50])
    ax.set_facecolor(BG)
    means = [edits(by_attack[a]).mean() if edits(by_attack[a]).size else 0.0 for a in ATTACK_ORDER]
    cis = [bootstrap_ci(edits(by_attack[a])) for a in ATTACK_ORDER]
    err_lo = np.array([m - lo for m, (lo, _) in zip(means, cis, strict=True)])
    err_hi = np.array([hi - m for m, (_, hi) in zip(means, cis, strict=True)])
    ypos = np.arange(len(ATTACK_ORDER))
    ax.barh(
        ypos,
        means,
        color=[PALETTE[a] for a in ATTACK_ORDER],
        alpha=0.92,
        edgecolor=BG,
        linewidth=1.0,
        height=0.62,
    )
    ax.errorbar(
        means,
        ypos,
        xerr=[err_lo, err_hi],
        fmt="none",
        color=TEXT,
        capsize=5,
        linewidth=1.3,
        alpha=0.85,
    )
    for i, (m, n) in enumerate(
        zip(means, [edits(by_attack[a]).size for a in ATTACK_ORDER], strict=True)
    ):
        ax.text(
            m + 0.02,
            i,
            f"{m:.3f}  ",
            color=TEXT,
            fontsize=11,
            va="center",
            fontweight="bold",
            family="DejaVu Sans Mono",
        )
        ax.text(
            m + 0.02,
            i + 0.35,
            f"n={n}",
            color=TEXT_MUTED,
            fontsize=8.5,
            va="center",
            family="DejaVu Sans Mono",
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [LABELS[a] for a in ATTACK_ORDER], color=TEXT, fontsize=10.5, fontweight="bold"
    )
    ax.set_xticks(np.linspace(0, 0.8, 5))
    for v in np.linspace(0, 0.8, 5):
        ax.axvline(v, color=GRID, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_xlim(0, max(means) * 1.35)
    ax.set_xlabel("Mean ± 95% CI", color=TEXT_MUTED, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(
        "MEAN EDIT DISTANCE PER ATTACK",
        color=TEXT,
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=12,
    )

    ax = fig.add_axes([0.50, 0.30, 0.46, 0.28])
    ax.set_facecolor(BG)
    eps_vals = sorted({float(r["epsilon"]) for recs in by_attack.values() for r in recs})
    for name in ATTACK_ORDER:
        groups = defaultdict(list)
        for r in by_attack[name]:
            groups[float(r["epsilon"])].append(r["edit_distance_norm"])
        xs = sorted(groups)
        if not xs:
            continue
        ys_mean = np.array([np.mean(groups[e]) for e in xs])
        ci = np.array([bootstrap_ci(np.asarray(groups[e])) for e in xs])
        if len(xs) == 1:
            ax.scatter(
                xs,
                ys_mean,
                marker="*",
                s=240,
                color=PALETTE[name],
                edgecolor=BG,
                linewidth=1.2,
                label=f"{LABELS[name]}*",
                zorder=4,
            )
        else:
            ax.plot(
                xs,
                ys_mean,
                marker="o",
                color=PALETTE[name],
                linewidth=2.5,
                markersize=8,
                markeredgecolor=BG,
                markeredgewidth=0.8,
                label=LABELS[name],
                zorder=3,
            )
            ax.fill_between(xs, ci[:, 0], ci[:, 1], color=PALETTE[name], alpha=0.18, zorder=2)
    for e in eps_vals:
        ax.axvline(e, color=GRID, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_xticks(eps_vals)
    ax.set_xticklabels([fmt_eps(e) for e in eps_vals], color=TEXT_MUTED)
    ax.set_xlabel("Perturbation budget ε", color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel("Mean ed", color=TEXT_MUTED, fontsize=10)
    ax.set_title(
        "EFFECTIVENESS VS ε", color=TEXT, fontsize=12, fontweight="bold", loc="left", pad=12
    )
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    ax = fig.add_axes([0.50, 0.08, 0.46, 0.16])
    ax.set_facecolor(BG)
    flips = [step1_flip_rate(by_attack[a]) for a in ATTACK_ORDER]
    ypos = np.arange(len(ATTACK_ORDER))
    ax.barh(
        ypos,
        flips,
        color=[PALETTE[a] for a in ATTACK_ORDER],
        alpha=0.92,
        edgecolor=BG,
        linewidth=1.0,
        height=0.62,
    )
    for i, f in enumerate(flips):
        ax.text(
            f + 0.012,
            i,
            f"{f:.0%}",
            color=TEXT,
            fontsize=10,
            va="center",
            fontweight="bold",
            family="DejaVu Sans Mono",
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels([LABELS[a] for a in ATTACK_ORDER], color=TEXT, fontsize=9.5)
    ax.set_xlim(0, 1.05)
    ax.set_xticks(np.linspace(0, 1, 6))
    for v in np.linspace(0, 1, 6):
        ax.axvline(v, color=GRID, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_xlabel("Step-1 flip rate", color=TEXT_MUTED, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(
        "STEP-1 TOOL-FLIP RATE", color=TEXT, fontsize=12, fontweight="bold", loc="left", pad=10
    )

    fig.text(
        0.04,
        0.025,
        "github.com/KosmasApostolidis/adversarial-reasoning-attacks   · 2026",
        color=TEXT_MUTED,
        fontsize=8.5,
        alpha=0.7,
    )
    fig.text(
        0.96,
        0.025,
        "* PGD anchor: smoke ε=8/255 only (n=5)",
        color=TEXT_MUTED,
        fontsize=8.5,
        alpha=0.7,
        ha="right",
    )

    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
