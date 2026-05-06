"""Graph 11: per-patient reasoning path — benign vs PGD in tool space."""

from __future__ import annotations

from itertools import pairwise

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    C_BENIGN,
    C_CHANGE,
    C_PGD,
    DARK_BG,
    DARK_FG,
    DARK_GRID,
    OUT,
    _load,
    _s,
)


def graph11_reasoning_paths():
    pgd_r = _load("runs/pgd_smoke/records.jsonl")

    all_tools = sorted(
        {t for r in pgd_r for key in ["benign", "attacked"] for t in r[key]["tool_sequence"]}
    )
    n_tools = len(all_tools)

    # Fixed 2D positions in a circle
    angles = np.linspace(0, 2 * np.pi, n_tools, endpoint=False)
    tool_pos = {
        t: (np.cos(a) * 1.0, np.sin(a) * 1.0) for t, a in zip(all_tools, angles, strict=False)
    }

    n_patients = len(pgd_r)
    fig, axes = plt.subplots(2, n_patients, figsize=(4.2 * n_patients, 8))
    fig.patch.set_facecolor(DARK_BG)
    fig.subplots_adjust(hspace=0.08, wspace=0.06)

    for col, rec in enumerate(pgd_r):
        pid = rec["sample_id"].split("_p")[1] if "_p" in rec["sample_id"] else str(col)
        for row, (key, color, title) in enumerate(
            [
                ("benign", C_BENIGN, f"P{pid} — Benign"),
                ("attacked", C_PGD, f"P{pid} — PGD"),
            ]
        ):
            ax = axes[row, col]
            ax.set_facecolor(DARK_BG)

            seq = rec[key]["tool_sequence"]
            b_seq = rec["benign"]["tool_sequence"]

            # Background tool nodes (all tools, dim)
            for t in all_tools:
                x, y = tool_pos[t]
                ax.plot(
                    x,
                    y,
                    "o",
                    ms=18,
                    color="#21262d",
                    markeredgecolor=DARK_GRID,
                    markeredgewidth=1,
                    zorder=2,
                )
                ax.text(
                    x, y, _s(t), ha="center", va="center", fontsize=6, color="#8b949e", zorder=3
                )

            # Highlight visited tools
            visited = set(seq)
            for t in visited:
                x, y = tool_pos[t]
                in_both = t in set(b_seq)
                fc = color if in_both else C_CHANGE
                ax.plot(
                    x,
                    y,
                    "o",
                    ms=22,
                    color=fc,
                    alpha=0.85,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    zorder=4,
                )
                ax.text(
                    x,
                    y,
                    _s(t),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                    fontweight="bold",
                    zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=DARK_BG)],
                )

            # Draw path as gradient-colored arrows
            n = len(seq)
            cmap = mpl.cm.Blues if color == C_BENIGN else mpl.cm.Reds
            for step_i, (a_tool, b_tool) in enumerate(pairwise(seq)):
                x0, y0 = tool_pos[a_tool]
                x1, y1 = tool_pos[b_tool]
                frac = (step_i + 0.5) / max(n - 1, 1)
                c = cmap(0.4 + 0.55 * frac)
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=c,
                        lw=1.8,
                        connectionstyle="arc3,rad=0.18",
                        mutation_scale=12,
                    ),
                    alpha=0.85,
                    zorder=6,
                )
                # Step number near midpoint
                mx = (x0 + x1) / 2 + 0.05
                my = (y0 + y1) / 2 + 0.05
                ax.text(
                    mx,
                    my,
                    str(step_i + 1),
                    fontsize=6.5,
                    color=c,
                    fontweight="bold",
                    zorder=7,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=DARK_BG)],
                )

            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect("equal")
            ax.axis("off")

            # Edit distance annotation (bottom right)
            ed = rec["edit_distance_norm"]
            if row == 1:
                ax.text(
                    1.4,
                    -1.45,
                    f"ed={ed:.2f}",
                    ha="right",
                    va="bottom",
                    fontsize=8.5,
                    color=C_PGD,
                    fontweight="bold",
                )

            if col == 0:
                ax.text(
                    -1.55,
                    0,
                    title.split(" — ")[1],
                    ha="right",
                    va="center",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                    rotation=90,
                )

            # Column title above top row only
            if row == 0:
                ax.set_title(f"Patient {pid}", fontsize=10, color=DARK_FG, pad=6)

    fig.suptitle(
        "Reasoning paths through tool space — how PGD redirects the agent's decision flow",
        fontsize=13,
        fontweight="bold",
        color=DARK_FG,
        y=1.01,
    )
    fig.savefig(
        OUT / "graph11_reasoning_paths.png", bbox_inches="tight", facecolor=DARK_BG, dpi=200
    )
    plt.close(fig)
    print("graph11 ✓")
