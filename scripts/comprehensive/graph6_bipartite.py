"""Graph 6: bipartite benign↔attacked alignment per patient."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from ._common import (
    C_BENIGN,
    C_NOISE,
    C_PGD,
    DARK_BG,
    DARK_FG,
    GRAPH_OUT,
    _s,
    load_records,
)


def _draw_column_nodes(ax, seq, xpos, max_len):
    for step, tool in enumerate(seq):
        y = 1 - (step + 0.5) / max_len
        fc = C_BENIGN if xpos < 0.5 else C_PGD
        circle = plt.Circle((xpos, y), 0.045, color=fc, zorder=4, linewidth=2)
        circle.set_edgecolor("white")
        ax.add_patch(circle)
        ax.text(
            xpos,
            y,
            _s(tool),
            ha="center",
            va="center",
            fontsize=6.5,
            color="white",
            fontweight="bold",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=1.5, foreground=DARK_BG)],
        )
    ax.text(
        xpos,
        1.04,
        "Benign" if xpos < 0.5 else "PGD",
        ha="center",
        fontsize=9,
        color=C_BENIGN if xpos < 0.5 else C_PGD,
        fontweight="bold",
    )


def _draw_alignment_edge(ax, by, ay, bt, at):
    color = C_NOISE if bt == at else C_PGD
    lw = 2.5 if bt == at else 1.8
    alpha = 0.85 if bt == at else 0.65
    ax.annotate(
        "",
        xy=(0.83, ay),
        xytext=(0.23, by),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=10),
        alpha=alpha,
    )


def _draw_alignment_edges(ax, b, a, max_len):
    for i in range(max_len):
        bt = b[i] if i < len(b) else None
        at = a[i] if i < len(a) else None
        if bt is None and at is None:
            continue
        by = 1 - (i + 0.5) / max_len if bt else None
        ay = 1 - (i + 0.5) / max_len if at else None

        if bt and at:
            _draw_alignment_edge(ax, by, ay, bt, at)
        elif bt and not at:
            ax.text(
                0.5, by, "✕", ha="center", va="center", fontsize=12, color="#8b949e", alpha=0.7
            )


def _draw_patient_panel(ax, rec):
    ax.set_facecolor(DARK_BG)
    b = rec["benign"]["tool_sequence"]
    a = rec["attacked"]["tool_sequence"]
    max_len = max(len(b), len(a), 1)

    for _col, seq, xpos in [(b, "left", 0.15), (a, "right", 0.85)]:
        _draw_column_nodes(ax, seq, xpos, max_len)

    _draw_alignment_edges(ax, b, a, max_len)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.12)
    ax.axis("off")
    pid = rec["sample_id"].split("_p")[1] if "_p" in rec["sample_id"] else rec["sample_id"]
    ax.set_title(
        f"P{pid}\ned={rec['edit_distance_norm']:.3f}", fontsize=9, color=DARK_FG, pad=4
    )


def _draw_legend_and_title(fig):
    handles = [
        mpatches.Patch(color=C_NOISE, label="Kept (same tool)"),
        mpatches.Patch(color=C_PGD, label="Substituted"),
        mpatches.Patch(color="#8b949e", label="Dropped (✕)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.04),
        labelcolor=DARK_FG,
    )
    fig.suptitle(
        "Bipartite trajectory alignment: benign ↔ PGD-attacked (per patient, per step)",
        fontsize=13,
        fontweight="bold",
        color=DARK_FG,
        y=1.01,
    )


def graph6_bipartite():
    pgd_r = load_records("runs/main/pgd/records.jsonl")
    n = len(pgd_r)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 7))
    fig.patch.set_facecolor(DARK_BG)

    for ax, rec in zip(axes, pgd_r, strict=False):
        _draw_patient_panel(ax, rec)

    _draw_legend_and_title(fig)

    fig.savefig(GRAPH_OUT / "graph6_bipartite.png", bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("graph6 ✓")
