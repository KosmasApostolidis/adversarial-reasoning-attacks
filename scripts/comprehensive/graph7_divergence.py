"""Graph 7: per-patient divergence tree (where reasoning splits)."""

from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from ._common import C_BENIGN, C_PGD, DARK_BG, DARK_FG, GRAPH_OUT, _s, load_records


def _draw_step_node(ax, draw_x, y, seq, step, merged, c, ec):
    fc = "#21262d" if merged else (C_BENIGN if c == C_BENIGN else C_PGD)
    circle = plt.Circle((draw_x, y), 0.055, color=fc, zorder=4)
    circle.set_edgecolor(ec)
    circle.set_linewidth(2)
    ax.add_patch(circle)
    if not merged or c == C_BENIGN:
        ax.text(
            draw_x,
            y,
            _s(seq[step]),
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            fontweight="bold",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=1.5, foreground=DARK_BG)],
        )


def _draw_step_pair(ax, step, y, b, a, div_step):
    for seq, xpos, c in [(b, 0.3, C_BENIGN), (a, 0.7, C_PGD)]:
        if step < len(seq):
            merged = step < div_step
            draw_x = 0.5 if merged else xpos
            ec = C_BENIGN if merged else c
            _draw_step_node(ax, draw_x, y, seq, step, merged, c, ec)


def _draw_divergence_marker(ax, step, y, max_len, div_step, b, a):
    if step == div_step and div_step < min(len(b), len(a)):
        ax.axhline(
            y + 0.5 / max_len,
            color=C_PGD,
            lw=1.5,
            linestyle="--",
            alpha=0.6,
            xmin=0.1,
            xmax=0.9,
        )
        ax.text(0.95, y + 0.5 / max_len, "⚡", fontsize=12, va="center", color=C_PGD, zorder=6)


def _draw_connection_lines(ax, step, y, max_len, b, a, div_step):
    if step > 0:
        py = 1 - (step - 0.5) / max_len
        for seq, xpos, c in [(b, 0.3, C_BENIGN), (a, 0.7, C_PGD)]:
            if step < len(seq) and step - 1 < len(seq):
                prev_merged = step - 1 < div_step
                curr_merged = step < div_step
                px = 0.5 if prev_merged else xpos
                cx = 0.5 if curr_merged else xpos
                if not (prev_merged and curr_merged and c == C_PGD):
                    ax.plot([px, cx], [py, y], color=c, lw=1.5, alpha=0.55, zorder=2)


def _draw_patient_panel(ax, rec):
    ax.set_facecolor(DARK_BG)
    b = rec["benign"]["tool_sequence"]
    a = rec["attacked"]["tool_sequence"]
    max_len = max(len(b), len(a))

    div_step = next((i for i in range(min(len(b), len(a))) if b[i] != a[i]), min(len(b), len(a)))

    for step in range(max_len):
        y = 1 - (step + 0.5) / max_len
        _draw_step_pair(ax, step, y, b, a, div_step)
        _draw_divergence_marker(ax, step, y, max_len, div_step, b, a)
        _draw_connection_lines(ax, step, y, max_len, b, a, div_step)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.1)
    ax.axis("off")
    pid = rec["sample_id"].split("_p")[1] if "_p" in rec["sample_id"] else rec["sample_id"]
    label = f"P{pid}  |  diverges at step {div_step + 1}"
    ax.set_title(label, fontsize=8.5, color=DARK_FG, pad=4)
    if div_step < max_len:
        ax.text(0.3, -0.05, "Benign", ha="center", fontsize=8, color=C_BENIGN)
        ax.text(0.7, -0.05, "PGD", ha="center", fontsize=8, color=C_PGD)


def graph7_divergence():
    pgd_r = load_records("runs/main/pgd/records.jsonl")

    fig, axes = plt.subplots(1, len(pgd_r), figsize=(4.5 * len(pgd_r), 7))
    fig.patch.set_facecolor(DARK_BG)

    for ax, rec in zip(axes, pgd_r, strict=False):
        _draw_patient_panel(ax, rec)

    fig.suptitle(
        "Divergence tree: where adversarial attack splits the reasoning chain",
        fontsize=13,
        fontweight="bold",
        color=DARK_FG,
        y=1.01,
    )
    fig.savefig(GRAPH_OUT / "graph7_divergence.png", bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("graph7 ✓")
