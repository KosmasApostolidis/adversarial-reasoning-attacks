"""Stat 4: tool occupancy per step position (benign vs noise vs PGD)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ._common import C_BENIGN, C_NOISE, C_PGD, STAT_OUT, _s, load_records


def _build_matrix(records, key, all_tools, max_step):
    mat = np.zeros((len(all_tools), max_step))
    tidx = {t: i for i, t in enumerate(all_tools)}
    for r in records:
        for step, tool in enumerate(r[key]["tool_sequence"][:max_step]):
            mat[tidx[tool], step] += 1
    return mat


def _annotate_cells(ax, mat, all_tools, max_step):
    for i in range(len(all_tools)):
        for j in range(max_step):
            v = mat[i, j]
            if v > 0:
                ax.text(
                    j,
                    i,
                    str(int(v)),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if v > mat.max() * 0.55 else "black",
                    fontweight="bold",
                )


def _draw_heatmap_panel(fig, ax, title, mat, c, all_tools, tool_labels, max_step):
    cmap = LinearSegmentedColormap.from_list(f"cm_{c}", ["#f7f7f7", c], N=256)
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(max_step))
    ax.set_xticklabels([f"s{i + 1}" for i in range(max_step)], fontsize=9)
    ax.set_yticks(range(len(all_tools)))
    ax.set_yticklabels(tool_labels, fontsize=9)
    ax.set_xlabel("Step position")
    ax.set_title(title, pad=8, fontsize=11)
    _annotate_cells(ax, mat, all_tools, max_step)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Count", fontsize=8)


def stat4_step_heatmap():
    noise_r = load_records("runs/main/noise/records.jsonl")
    pgd_r = load_records("runs/main/pgd/records.jsonl")

    all_tools = sorted(
        {
            t
            for r in pgd_r + noise_r
            for key in ["benign", "attacked"]
            for t in r[key]["tool_sequence"]
        }
    )
    MAX_STEP = 8

    conditions = [
        ("Benign", _build_matrix(pgd_r, "benign", all_tools, MAX_STEP), C_BENIGN),
        ("Noise-attacked", _build_matrix(noise_r, "attacked", all_tools, MAX_STEP), C_NOISE),
        ("PGD-attacked", _build_matrix(pgd_r, "attacked", all_tools, MAX_STEP), C_PGD),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.4)
    tool_labels = [_s(t) for t in all_tools]

    for ax, (title, mat, c) in zip(axes, conditions, strict=False):
        _draw_heatmap_panel(fig, ax, title, mat, c, all_tools, tool_labels, MAX_STEP)

    fig.suptitle(
        "Tool occupancy at each trajectory step: how PGD rewires step-by-step reasoning",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(STAT_OUT / "stat4_step_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("stat4 ✓")
