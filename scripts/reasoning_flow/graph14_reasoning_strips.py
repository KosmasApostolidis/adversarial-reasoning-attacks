"""Graph 14: horizontal reasoning strips with divergence markers."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from ._common import (
    C_BENIGN,
    C_CHANGE,
    C_NOISE,
    C_PGD,
    DARK_BG,
    DARK_FG,
    DARK_GRID,
    OUT,
    _dark_fig,
    _load,
    _s,
)


def graph14_reasoning_strips():
    pgd_r = _load("runs/pgd_smoke/records.jsonl")
    noise_r = {r["sample_id"]: r for r in _load("runs/smoke/records.jsonl")}

    all_tools = sorted(
        {t for r in pgd_r for key in ["benign", "attacked"] for t in r[key]["tool_sequence"]}
    )
    cmap20 = plt.get_cmap("tab20")
    tool_color = {t: cmap20(i % 20) for i, t in enumerate(all_tools)}

    MAX_STEP = 10
    N_PAT = len(pgd_r)
    ROW_H = 2.2  # height per patient block

    fig_h = N_PAT * ROW_H + 1.2
    fig = _dark_fig(16, fig_h)
    ax = fig.add_axes([0.12, 0.06, 0.85, 0.86])
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    # Step grid lines
    for s in range(MAX_STEP + 1):
        ax.axvline(s, color=DARK_GRID, lw=0.6, alpha=0.6, zorder=1)

    BLOCK_W = 0.88
    BLOCK_H = 0.58

    for pi, rec in enumerate(pgd_r):
        pid = rec["sample_id"].split("_p")[1] if "_p" in rec["sample_id"] else str(pi)
        nr = noise_r.get(rec["sample_id"])
        y_base = (N_PAT - 1 - pi) * ROW_H

        seqs = {
            "Benign": (rec["benign"]["tool_sequence"], C_BENIGN, y_base + 1.35),
            "Noise": (nr["attacked"]["tool_sequence"] if nr else [], C_NOISE, y_base + 0.68),
            "PGD": (rec["attacked"]["tool_sequence"], C_PGD, y_base + 0.02),
        }

        # First divergence step vs benign
        b = rec["benign"]["tool_sequence"]
        a = rec["attacked"]["tool_sequence"]
        div_step = next(
            (i for i in range(min(len(b), len(a))) if b[i] != a[i]), min(len(b), len(a))
        )

        for cond_label, (seq, cond_c, y_row) in seqs.items():
            # Row label
            ax.text(
                -0.12,
                y_row + BLOCK_H / 2,
                cond_label,
                ha="right",
                va="center",
                fontsize=8.5,
                color=cond_c,
                fontweight="bold",
            )

            for step, tool in enumerate(seq[:MAX_STEP]):
                fc = tool_color[tool]
                # Brighter highlight if step is a divergence point
                is_changed = (
                    step >= div_step
                    and cond_label != "Benign"
                    and step < len(b)
                    and step < len(seq)
                    and b[step] != tool
                )
                edge_c = C_CHANGE if is_changed else "white"
                edge_w = 2.5 if is_changed else 0.8

                ax.add_patch(
                    mpatches.FancyBboxPatch(
                        (step + 0.06, y_row),
                        BLOCK_W,
                        BLOCK_H,
                        boxstyle="round,pad=0.04",
                        facecolor=fc,
                        edgecolor=edge_c,
                        linewidth=edge_w,
                        zorder=3,
                    )
                )
                ax.text(
                    step + 0.06 + BLOCK_W / 2,
                    y_row + BLOCK_H / 2,
                    _s(tool),
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color="white",
                    fontweight="bold",
                    zorder=4,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=DARK_BG)],
                )

        # Divergence vertical marker
        if div_step < MAX_STEP:
            ax.axvline(
                div_step + 0.02,
                color=C_CHANGE,
                lw=2,
                linestyle="--",
                alpha=0.7,
                zorder=2,
                ymin=(y_base) / (N_PAT * ROW_H),
                ymax=(y_base + ROW_H) / (N_PAT * ROW_H),
            )
            ax.text(
                div_step + 0.12,
                y_base + ROW_H - 0.12,
                f"⚡ div@{div_step + 1}",
                fontsize=8,
                color=C_CHANGE,
                fontweight="bold",
                zorder=5,
            )

        # Patient label on left
        ax.text(
            -0.12,
            y_base + ROW_H / 2 + 0.15,
            f"P{pid}",
            ha="right",
            va="center",
            fontsize=11,
            color=DARK_FG,
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground=DARK_BG)],
        )
        # Edit distance
        ax.text(
            MAX_STEP + 0.08,
            y_base + 0.69,
            f"ed={rec['edit_distance_norm']:.2f}",
            ha="left",
            va="center",
            fontsize=8.5,
            color=C_PGD,
            fontweight="bold",
        )

        # Separator
        if pi < N_PAT - 1:
            ax.axhline(y_base + ROW_H, color=DARK_GRID, lw=1, alpha=0.5, zorder=1)

    # Step labels at top
    for s in range(MAX_STEP):
        ax.text(
            s + 0.5,
            N_PAT * ROW_H + 0.12,
            f"step {s + 1}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#8b949e",
        )

    ax.set_xlim(-0.25, MAX_STEP + 0.4)
    ax.set_ylim(-0.2, N_PAT * ROW_H + 0.4)

    # Tool colour legend
    handles = [mpatches.Patch(color=tool_color[t], label=_s(t)) for t in all_tools]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(all_tools),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        labelcolor=DARK_FG,
    )

    fig.suptitle(
        "Reasoning strips — benign / noise / PGD trajectories with divergence markers",
        fontsize=13,
        fontweight="bold",
        color=DARK_FG,
        y=0.99,
    )
    fig.savefig(
        OUT / "graph14_reasoning_strips.png", bbox_inches="tight", facecolor=DARK_BG, dpi=200
    )
    plt.close(fig)
    print("graph14 ✓")
