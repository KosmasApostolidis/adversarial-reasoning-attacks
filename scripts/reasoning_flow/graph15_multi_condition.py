"""Graph 15: all 3 conditions (benign/noise/PGD) overlaid in step × tool grid."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    C_BENIGN,
    C_NOISE,
    C_PGD,
    DARK_AX,
    DARK_BG,
    DARK_FG,
    DARK_GRID,
    OUT,
    _load,
    _s,
)

MAX_STEP = 8


def _build_conditions(pgd_r: list[dict], noise_r: dict) -> list[tuple]:
    return [
        ("benign", pgd_r, C_BENIGN, 2.5, "-", 0.80),
        (
            "noise",
            [noise_r[r["sample_id"]] for r in pgd_r if r["sample_id"] in noise_r],
            C_NOISE,
            2.0,
            "--",
            0.65,
        ),
        ("pgd", pgd_r, C_PGD, 2.5, ":", 0.80),
    ]


def _draw_condition_traces(
    ax,
    conditions: list[tuple],
    tool_y: dict[str, int],
    rng: np.random.Generator,
) -> None:
    offset = {"benign": -0.12, "noise": 0.0, "pgd": 0.12}
    for cond_key, records, color, lw, ls, alpha in conditions:
        key = "attacked" if cond_key in ("noise", "pgd") else "benign"
        for _pi, rec in enumerate(records):
            seq = rec[key]["tool_sequence"]
            xs = list(range(min(len(seq), MAX_STEP)))
            jit = rng.uniform(-0.08, 0.08, len(xs))
            ys = [tool_y[seq[s]] + jit[s] + offset[cond_key] for s in xs]
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, linestyle=ls, zorder=3)
            ax.scatter(xs, ys, s=55, color=color, edgecolors="white", lw=0.6, zorder=5, alpha=0.9)


def _draw_grid_background(ax, all_tools: list[str]) -> None:
    # Vertical step lines
    for s in range(MAX_STEP):
        ax.axvline(s, color=DARK_GRID, lw=0.7, zorder=1)

    # Horizontal tool band shading (alternating)
    for i, _t in enumerate(all_tools):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#161b22", alpha=0.5, zorder=0)


def _decorate_axes(ax, all_tools: list[str]) -> None:
    ax.set_xticks(range(MAX_STEP))
    ax.set_xticklabels([f"Step {i + 1}" for i in range(MAX_STEP)], fontsize=11, color=DARK_FG)
    ax.set_yticks(range(len(all_tools)))
    ax.set_yticklabels([_s(t) for t in all_tools], fontsize=10.5, color=DARK_FG)
    ax.set_xlabel("Trajectory step", fontsize=12, color=DARK_FG)
    ax.set_ylabel("Tool", fontsize=12, color=DARK_FG)
    ax.tick_params(colors=DARK_FG)
    for sp in ax.spines.values():
        sp.set_color(DARK_GRID)
    ax.set_xlim(-0.5, MAX_STEP - 0.5)
    ax.set_ylim(-0.6, len(all_tools) - 0.4)


def _add_legend_and_title(ax) -> None:
    legend_handles = [
        mpatches.Patch(color=C_BENIGN, label="Benign  (solid)"),
        mpatches.Patch(color=C_NOISE, label="Noise-attacked  (dashed)"),
        mpatches.Patch(color=C_PGD, label="PGD-attacked  (dotted)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=11,
        framealpha=0.25,
        facecolor=DARK_AX,
        edgecolor=DARK_GRID,
        labelcolor=DARK_FG,
    )
    ax.set_title(
        "All 3 conditions overlaid — benign / noise / PGD reasoning paths (n=5 patients)",
        fontsize=13,
        fontweight="bold",
        color=DARK_FG,
        pad=12,
    )


def graph15_multi_condition():
    pgd_r = _load("runs/pgd_smoke/records.jsonl")
    noise_r = {r["sample_id"]: r for r in _load("runs/smoke/records.jsonl")}

    all_tools = sorted(
        {t for r in pgd_r for key in ["benign", "attacked"] for t in r[key]["tool_sequence"]}
    )
    tool_y = {t: i for i, t in enumerate(all_tools)}

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    conditions = _build_conditions(pgd_r, noise_r)
    rng = np.random.default_rng(21)

    _draw_condition_traces(ax, conditions, tool_y, rng)
    _draw_grid_background(ax, all_tools)
    _decorate_axes(ax, all_tools)
    _add_legend_and_title(ax)

    fig.savefig(
        OUT / "graph15_multi_condition.png", bbox_inches="tight", facecolor=DARK_BG, dpi=200
    )
    plt.close(fig)
    print("graph15 ✓")
