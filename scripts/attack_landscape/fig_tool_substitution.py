"""Figure 4: tool-substitution heatmap matrix per attack."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ._common import ATTACK_ORDER, LABELS, PALETTE


def _collect_tool_universe(by_attack: dict[str, list[dict]]) -> list[str]:
    universe: set[str] = set()
    for recs in by_attack.values():
        for r in recs:
            universe.update(r.get("benign", {}).get("tool_sequence", []) or [])
            universe.update(r.get("attacked", {}).get("tool_sequence", []) or [])
    return sorted(universe)


def _compute_substitution_matrix(
    recs: list[dict], idx: dict[str, int], tools: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    mat = np.zeros((len(tools), len(tools)), dtype=np.float64)
    for r in recs:
        b = r.get("benign", {}).get("tool_sequence", []) or []
        a = r.get("attacked", {}).get("tool_sequence", []) or []
        for bi, ai in zip(b, a, strict=False):
            if bi in idx and ai in idx:
                mat[idx[bi], idx[ai]] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    norm = np.where(row_sums == 0, 0, mat / np.where(row_sums == 0, 1, row_sums))
    return mat, norm


def _render_one_substitution_panel(
    ax, name: str, mat: np.ndarray, norm: np.ndarray, tools: list[str], cmap
):
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(tools)):
        for j in range(len(tools)):
            if mat[i, j] > 0:
                color = "white" if norm[i, j] > 0.55 else "#333333"
                ax.text(
                    j,
                    i,
                    f"{int(mat[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=color,
                )
    ax.set_xticks(range(len(tools)))
    ax.set_yticks(range(len(tools)))
    ax.set_xticklabels(tools, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tools, fontsize=8)
    ax.set_xlabel("Attacked tool", fontsize=10)
    ax.set_ylabel("Benign tool", fontsize=10)
    ax.set_title(LABELS[name], fontsize=11, color=PALETTE[name], fontweight="bold")
    ax.plot(
        [-0.5, len(tools) - 0.5],
        [-0.5, len(tools) - 0.5],
        color="#888888",
        linewidth=0.6,
        linestyle=":",
        alpha=0.7,
    )
    return im


def fig_tool_substitution(by_attack: dict[str, list[dict]], out_path: Path) -> None:
    attacks_with_data = [a for a in ATTACK_ORDER if by_attack.get(a)]
    n_panels = len(attacks_with_data)
    cols = min(3, n_panels)
    rows = int(np.ceil(n_panels / cols))

    tools = _collect_tool_universe(by_attack)
    idx = {t: i for i, t in enumerate(tools)}

    cmap = LinearSegmentedColormap.from_list("flip", ["#f5f5f5", "#1565C0", "#5E35B1", "#C62828"])
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows), squeeze=False)

    im = None
    for ax_idx, name in enumerate(attacks_with_data):
        ax = axes[ax_idx // cols][ax_idx % cols]
        mat, norm = _compute_substitution_matrix(by_attack[name], idx, tools)
        im = _render_one_substitution_panel(ax, name, mat, norm, tools, cmap)

    for k in range(n_panels, rows * cols):
        axes[k // cols][k % cols].axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
        cbar.set_label("P(attacked tool | benign tool)", fontsize=10)
    fig.suptitle(
        "Tool-substitution matrices per attack (counts shown; diagonal = no flip)",
        fontsize=13,
        fontweight="bold",
        y=1.00,
    )
    fig.savefig(out_path)
    plt.close(fig)
