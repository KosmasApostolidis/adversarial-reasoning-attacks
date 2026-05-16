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

MAX_STEP = 10
ROW_H = 2.2  # height per patient block
BLOCK_W = 0.88
BLOCK_H = 0.58


def _draw_step_box(
    ax, step: int, y_row: float, tool: str, is_changed: bool, tool_color: dict
) -> None:
    fc = tool_color[tool]
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


def _draw_condition_row(
    ax,
    cond_label: str,
    seq: list[str],
    cond_c: str,
    y_row: float,
    b: list[str],
    div_step: int,
    tool_color: dict,
) -> None:
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
        # Brighter highlight if step is a divergence point
        is_changed = (
            step >= div_step
            and cond_label != "Benign"
            and step < len(b)
            and step < len(seq)
            and b[step] != tool
        )
        _draw_step_box(ax, step, y_row, tool, is_changed, tool_color)


def _draw_divergence_marker(ax, div_step: int, y_base: float, n_pat: int) -> None:
    if div_step < MAX_STEP:
        ax.axvline(
            div_step + 0.02,
            color=C_CHANGE,
            lw=2,
            linestyle="--",
            alpha=0.7,
            zorder=2,
            ymin=(y_base) / (n_pat * ROW_H),
            ymax=(y_base + ROW_H) / (n_pat * ROW_H),
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


def _draw_patient_labels(ax, pid: str, y_base: float, ed: float) -> None:
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
        f"ed={ed:.2f}",
        ha="left",
        va="center",
        fontsize=8.5,
        color=C_PGD,
        fontweight="bold",
    )


def _render_patient_block(
    ax, rec: dict, noise_r: dict, pi: int, n_pat: int, tool_color: dict
) -> None:
    pid = rec["sample_id"].split("_p")[1] if "_p" in rec["sample_id"] else str(pi)
    nr = noise_r.get(rec["sample_id"])
    y_base = (n_pat - 1 - pi) * ROW_H

    seqs = {
        "Benign": (rec["benign"]["tool_sequence"], C_BENIGN, y_base + 1.35),
        "Noise": (nr["attacked"]["tool_sequence"] if nr else [], C_NOISE, y_base + 0.68),
        "PGD": (rec["attacked"]["tool_sequence"], C_PGD, y_base + 0.02),
    }

    # First divergence step vs benign
    b = rec["benign"]["tool_sequence"]
    a = rec["attacked"]["tool_sequence"]
    div_step = next((i for i in range(min(len(b), len(a))) if b[i] != a[i]), min(len(b), len(a)))

    for cond_label, (seq, cond_c, y_row) in seqs.items():
        _draw_condition_row(ax, cond_label, seq, cond_c, y_row, b, div_step, tool_color)

    # Divergence vertical marker
    _draw_divergence_marker(ax, div_step, y_base, n_pat)

    _draw_patient_labels(ax, pid, y_base, rec["edit_distance_norm"])

    # Separator
    if pi < n_pat - 1:
        ax.axhline(y_base + ROW_H, color=DARK_GRID, lw=1, alpha=0.5, zorder=1)


def _draw_step_headers(ax, n_pat: int) -> None:
    # Step labels at top
    for s in range(MAX_STEP):
        ax.text(
            s + 0.5,
            n_pat * ROW_H + 0.12,
            f"step {s + 1}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#8b949e",
        )

    ax.set_xlim(-0.25, MAX_STEP + 0.4)
    ax.set_ylim(-0.2, n_pat * ROW_H + 0.4)


def _add_legend_and_suptitle(fig, all_tools: list[str], tool_color: dict) -> None:
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


def graph14_reasoning_strips():
    pgd_r = _load("runs/pgd_smoke/records.jsonl")
    noise_r = {r["sample_id"]: r for r in _load("runs/smoke/records.jsonl")}

    all_tools = sorted(
        {t for r in pgd_r for key in ["benign", "attacked"] for t in r[key]["tool_sequence"]}
    )
    cmap20 = plt.get_cmap("tab20")
    tool_color = {t: cmap20(i % 20) for i, t in enumerate(all_tools)}

    N_PAT = len(pgd_r)

    fig_h = N_PAT * ROW_H + 1.2
    fig = _dark_fig(16, fig_h)
    ax = fig.add_axes([0.12, 0.06, 0.85, 0.86])
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    # Step grid lines
    for s in range(MAX_STEP + 1):
        ax.axvline(s, color=DARK_GRID, lw=0.6, alpha=0.6, zorder=1)

    for pi, rec in enumerate(pgd_r):
        _render_patient_block(ax, rec, noise_r, pi, N_PAT, tool_color)

    _draw_step_headers(ax, N_PAT)
    _add_legend_and_suptitle(fig, all_tools, tool_color)

    fig.savefig(
        OUT / "graph14_reasoning_strips.png", bbox_inches="tight", facecolor=DARK_BG, dpi=200
    )
    plt.close(fig)
    print("graph14 ✓")
