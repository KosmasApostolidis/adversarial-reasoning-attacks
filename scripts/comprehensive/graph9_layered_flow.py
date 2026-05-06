"""Graph 9: layered step graph — benign path vs PGD path."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from ._common import (
    C_BENIGN,
    C_PGD,
    DARK_AX,
    DARK_BG,
    DARK_FG,
    DARK_GRID,
    GRAPH_OUT,
    _s,
    load_records,
)


def graph9_layered_flow():
    pgd_r = load_records("runs/main/pgd/records.jsonl")

    all_tools = sorted(
        {t for r in pgd_r for key in ["benign", "attacked"] for t in r[key]["tool_sequence"]}
    )
    MAX_STEP = 7
    tool_y = {t: i for i, t in enumerate(all_tools)}

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    # For each patient and each condition, draw a path through step × tool space
    for ri, rec in enumerate(pgd_r):
        offset = (ri - 2) * 0.12  # vertical jitter per patient

        for seq, c, lw, alpha, ls in [
            (rec["benign"]["tool_sequence"], C_BENIGN, 2.0, 0.75, "-"),
            (rec["attacked"]["tool_sequence"], C_PGD, 2.0, 0.75, "--"),
        ]:
            xs = list(range(min(len(seq), MAX_STEP)))
            ys = [tool_y[seq[s]] + offset for s in xs]
            ax.plot(xs, ys, color=c, lw=lw, alpha=alpha, linestyle=ls, zorder=3)
            ax.scatter(xs, ys, s=60, color=c, edgecolors="white", lw=0.7, zorder=5, alpha=0.9)

    # Vertical grid lines per step
    for s in range(MAX_STEP):
        ax.axvline(s, color=DARK_GRID, lw=0.8, zorder=1)

    ax.set_xticks(range(MAX_STEP))
    ax.set_xticklabels([f"Step {i + 1}" for i in range(MAX_STEP)], fontsize=10, color=DARK_FG)
    ax.set_yticks(range(len(all_tools)))
    ax.set_yticklabels([_s(t) for t in all_tools], fontsize=9.5, color=DARK_FG)
    ax.set_xlabel("Trajectory step", fontsize=11, color=DARK_FG)
    ax.set_ylabel("Tool", fontsize=11, color=DARK_FG)
    ax.tick_params(colors=DARK_FG)
    for sp in ax.spines.values():
        sp.set_color(DARK_GRID)
    ax.set_xlim(-0.4, MAX_STEP - 0.6)
    ax.set_ylim(-0.8, len(all_tools) - 0.2)

    # Legend
    handles = [
        mpatches.Patch(color=C_BENIGN, label="Benign trajectory"),
        mpatches.Patch(color=C_PGD, label="PGD-attacked trajectory"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=10,
        framealpha=0.3,
        facecolor=DARK_AX,
        edgecolor=DARK_GRID,
        labelcolor=DARK_FG,
    )
    ax.set_title(
        "Layered flow: how PGD shifts tool selection at each reasoning step (n=5 patients)",
        fontsize=13,
        fontweight="bold",
        color=DARK_FG,
        pad=12,
    )
    fig.savefig(GRAPH_OUT / "graph9_layered_flow.png", bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("graph9 ✓")
