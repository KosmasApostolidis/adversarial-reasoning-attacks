"""Graph 12: alluvial stream — tool flow at each step (benign vs PGD)."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ._common import C_BENIGN, C_PGD, DARK_BG, DARK_FG, DARK_GRID, OUT, _load, _s


def graph12_alluvial():
    pgd_r = _load("runs/pgd_smoke/records.jsonl")

    all_tools = sorted(
        {t for r in pgd_r for key in ["benign", "attacked"] for t in r[key]["tool_sequence"]}
    )
    tool_y = {t: i for i, t in enumerate(all_tools)}
    MAX_STEP = 8
    N_TOOLS = len(all_tools)
    N_PAT = len(pgd_r)

    # Colour per tool
    cmap20 = plt.get_cmap("tab20")
    tool_color = {t: cmap20(i % 20) for i, t in enumerate(all_tools)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(DARK_BG)
    fig.subplots_adjust(wspace=0.08)

    for ax, (key, cond_label) in zip(
        axes, [("benign", "Benign"), ("attacked", "PGD-attacked")], strict=False
    ):
        ax.set_facecolor(DARK_BG)

        # For each (step, tool) cell: count occupancy
        occ = np.zeros((N_TOOLS, MAX_STEP))
        for r in pgd_r:
            for step, tool in enumerate(r[key]["tool_sequence"][:MAX_STEP]):
                occ[tool_y[tool], step] += 1

        # Draw stream bands
        bar_w = 0.55
        for step in range(MAX_STEP):
            y_cursor = 0.0
            for t in all_tools:
                cnt = occ[tool_y[t], step]
                if cnt > 0:
                    height = cnt / N_PAT  # normalise to [0,1]
                    ax.add_patch(
                        mpatches.FancyBboxPatch(
                            (step - bar_w / 2, y_cursor),
                            bar_w,
                            height * (N_TOOLS * 0.7),
                            boxstyle="round,pad=0.02",
                            facecolor=tool_color[t],
                            alpha=0.88,
                            edgecolor="white",
                            linewidth=0.8,
                            zorder=3,
                        )
                    )
                    txt_y = y_cursor + height * N_TOOLS * 0.35
                    if height > 0.12:
                        ax.text(
                            step,
                            txt_y,
                            _s(t),
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white",
                            fontweight="bold",
                            zorder=4,
                        )
                    y_cursor += height * (N_TOOLS * 0.7) + 0.08

        # Flow ribbons between adjacent steps
        for step in range(MAX_STEP - 1):
            # Group by tool, draw connecting bezier-like fill
            for r in pgd_r:
                seq = r[key]["tool_sequence"]
                if step < len(seq) and step + 1 < len(seq):
                    t0, t1 = seq[step], seq[step + 1]
                    # Draw a thin arc
                    y0 = tool_y[t0] * 0.9 + occ[tool_y[t0], step] * 0.15
                    y1 = tool_y[t1] * 0.9 + occ[tool_y[t1], step + 1] * 0.15
                    ax.annotate(
                        "",
                        xy=(step + 0.6, y1),
                        xytext=(step + 0.4, y0),
                        arrowprops=dict(
                            arrowstyle="-",
                            color=tool_color[t0],
                            lw=1.2,
                            connectionstyle="arc3,rad=0.0",
                            alpha=0.35,
                        ),
                        zorder=2,
                    )

        ax.set_xticks(range(MAX_STEP))
        ax.set_xticklabels([f"Step {i + 1}" for i in range(MAX_STEP)], fontsize=9, color=DARK_FG)
        ax.set_yticks([])
        ax.set_xlim(-0.7, MAX_STEP - 0.3)
        ax.set_title(
            cond_label,
            fontsize=13,
            fontweight="bold",
            color=C_BENIGN if "Benign" in cond_label else C_PGD,
            pad=10,
        )
        for sp in ax.spines.values():
            sp.set_color(DARK_GRID)
        ax.tick_params(colors=DARK_FG)
        ax.axhline(0, color=DARK_GRID, lw=0.5)

    # Shared legend
    handles = [mpatches.Patch(color=tool_color[t], label=_s(t)) for t in all_tools]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(all_tools),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        labelcolor=DARK_FG,
    )
    fig.suptitle(
        "Alluvial stream — tool selection flow at each reasoning step: benign vs PGD",
        fontsize=13,
        fontweight="bold",
        color=DARK_FG,
        y=1.02,
    )
    fig.savefig(OUT / "graph12_alluvial.png", bbox_inches="tight", facecolor=DARK_BG, dpi=200)
    plt.close(fig)
    print("graph12 ✓")
