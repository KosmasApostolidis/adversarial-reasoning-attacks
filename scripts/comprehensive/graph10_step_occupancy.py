"""Graph 10: step-occupancy heatmap (which tools appear at each step under attack)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ._common import (
    C_BENIGN,
    C_NOISE,
    C_PGD,
    DARK_AX,
    DARK_BG,
    DARK_FG,
    DARK_GRID,
    GRAPH_OUT,
    _s,
    load_records,
)


def graph10_step_occupancy():
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
    tidx = {t: i for i, t in enumerate(all_tools)}
    MAX_STEP = 8

    def make_mat(records, key):
        m = np.zeros((len(all_tools), MAX_STEP))
        for r in records:
            for s, t in enumerate(r[key]["tool_sequence"][:MAX_STEP]):
                m[tidx[t], s] += 1
        return m

    b_mat = make_mat(pgd_r, "benign")
    n_mat = make_mat(noise_r, "attacked")
    p_mat = make_mat(pgd_r, "attacked")
    diff = p_mat - b_mat  # positive = PGD added, negative = PGD removed

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(DARK_BG)
    fig.subplots_adjust(wspace=0.35)

    titles = ["Benign", "Noise-attacked", "PGD − Benign  (Δ)"]
    mats = [b_mat, n_mat, diff]
    cmaps = [
        LinearSegmentedColormap.from_list("b", [DARK_AX, C_BENIGN], 256),
        LinearSegmentedColormap.from_list("n", [DARK_AX, C_NOISE], 256),
        LinearSegmentedColormap.from_list("d", [C_BENIGN, DARK_AX, C_PGD], 256),
    ]
    vmins = [0, 0, -diff.max()]
    vmaxs = [b_mat.max(), n_mat.max(), diff.max()]

    for ax, title, mat, cmap, vmin, vmax in zip(
        axes, titles, mats, cmaps, vmins, vmaxs, strict=False
    ):
        ax.set_facecolor(DARK_BG)
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(MAX_STEP))
        ax.set_xticklabels([f"s{i + 1}" for i in range(MAX_STEP)], fontsize=9, color=DARK_FG)
        ax.set_yticks(range(len(all_tools)))
        ax.set_yticklabels([_s(t) for t in all_tools], fontsize=9.5, color=DARK_FG)
        ax.set_xlabel("Step position", fontsize=10, color=DARK_FG)
        ax.set_title(title, fontsize=12, fontweight="bold", color=DARK_FG, pad=8)
        for sp in ax.spines.values():
            sp.set_color(DARK_GRID)
        ax.tick_params(colors=DARK_FG)

        for i in range(len(all_tools)):
            for j in range(MAX_STEP):
                v = mat[i, j]
                if v != 0:
                    txt = f"{v:+.0f}" if title.startswith("PGD") else str(int(v))
                    tcolor = "white" if abs(v) > (vmax - vmin) * 0.45 else "#cccccc"
                    ax.text(
                        j,
                        i,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=tcolor,
                        fontweight="bold",
                    )

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color=DARK_FG)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=DARK_FG)
        cb.set_label("Count / Δ", fontsize=8, color=DARK_FG)

    fig.suptitle(
        "Step-position occupancy heatmap — which tools PGD inserts, removes, or shifts",
        fontsize=13,
        fontweight="bold",
        color=DARK_FG,
        y=1.02,
    )
    fig.savefig(GRAPH_OUT / "graph10_step_occupancy.png", bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("graph10 ✓")
