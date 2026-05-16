"""Graph 13: transition delta graph — what PGD added/removed."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx

from ._common import (
    C_BENIGN,
    C_PGD,
    DARK_AX,
    DARK_BG,
    DARK_FG,
    DARK_GRID,
    OUT,
    _load,
    _s,
)


def _compute_edge_counters(pgd_r: list[dict]) -> tuple[Counter, Counter]:
    benign_edges: Counter = Counter()
    attacked_edges: Counter = Counter()

    for r in pgd_r:
        for a, b in pairwise(r["benign"]["tool_sequence"]):
            benign_edges[(a, b)] += 1
        for a, b in pairwise(r["attacked"]["tool_sequence"]):
            attacked_edges[(a, b)] += 1
    return benign_edges, attacked_edges


def _build_delta_graph(
    all_tools: list[str], benign_edges: Counter, attacked_edges: Counter
) -> tuple[nx.DiGraph, dict]:
    G = nx.DiGraph()
    G.add_nodes_from(all_tools)
    all_edge_keys = set(benign_edges) | set(attacked_edges)
    for e in all_edge_keys:
        G.add_edge(*e)

    pos = nx.spring_layout(G, seed=13, k=2.5)
    return G, pos


def _draw_delta_nodes(ax, all_tools: list[str], pos: dict, highlight_tools=None) -> None:
    for t in all_tools:
        fc = "#21262d"
        ec = "#58a6ff" if (highlight_tools and t in highlight_tools) else DARK_GRID
        lw = 2 if (highlight_tools and t in highlight_tools) else 1
        circle = plt.Circle(pos[t], 0.08, color=fc, zorder=3)
        circle.set_edgecolor(ec)
        circle.set_linewidth(lw)
        ax.add_patch(circle)
        ax.text(
            pos[t][0],
            pos[t][1],
            _s(t),
            ha="center",
            va="center",
            fontsize=7,
            color=DARK_FG,
            fontweight="bold",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=1.5, foreground=DARK_BG)],
        )


def _draw_condition_panel(ax, primary: Counter, pos: dict, color: str) -> None:
    for (u, v), w in primary.items():
        ax.annotate(
            "",
            xy=pos[v],
            xytext=pos[u],
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=0.7 + w * 1.2,
                connectionstyle="arc3,rad=0.15",
                mutation_scale=12,
            ),
            alpha=0.8,
            zorder=2,
        )


def _draw_delta_arrows(ax, kept: dict, removed: dict, added: dict, pos: dict) -> None:
    for edges_dict, ec, rad, _label in [
        (kept, "#8b949e", 0.0, "kept"),
        (removed, C_BENIGN, 0.2, "removed (benign only)"),
        (added, C_PGD, -0.2, "added (PGD only)"),
    ]:
        for (u, v), w in edges_dict.items():
            ax.annotate(
                "",
                xy=pos[v],
                xytext=pos[u],
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=ec,
                    lw=0.7 + w * 1.4,
                    connectionstyle=f"arc3,rad={rad}",
                    mutation_scale=12,
                ),
                alpha=0.85,
                zorder=2,
            )


def _add_delta_legend(ax, kept: dict, removed: dict, added: dict) -> None:
    delta_handles = [
        mpatches.Patch(color="#8b949e", label=f"Kept ({len(kept)})"),
        mpatches.Patch(color=C_BENIGN, label=f"Removed by PGD ({len(removed)})"),
        mpatches.Patch(color=C_PGD, label=f"Added by PGD ({len(added)})"),
    ]
    ax.legend(
        handles=delta_handles,
        loc="lower right",
        fontsize=8.5,
        framealpha=0.25,
        facecolor=DARK_AX,
        edgecolor=DARK_GRID,
        labelcolor=DARK_FG,
    )


def _render_delta_panel(
    ax,
    all_tools: list[str],
    pos: dict,
    benign_edges: Counter,
    attacked_edges: Counter,
) -> None:
    # Added edges (in PGD, not in benign)
    added = {e: w for e, w in attacked_edges.items() if e not in benign_edges}
    removed = {e: w for e, w in benign_edges.items() if e not in attacked_edges}
    kept = {e: w for e, w in benign_edges.items() if e in attacked_edges}

    _draw_delta_arrows(ax, kept, removed, added, pos)

    _draw_delta_nodes(
        ax,
        all_tools,
        pos,
        highlight_tools=set(t for e in added for t in e) | set(t for e in removed for t in e),
    )
    # Delta legend
    _add_delta_legend(ax, kept, removed, added)


def _finalize_panel(ax, title: str, ax_i: int) -> None:
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    c_title = {0: C_BENIGN, 1: C_PGD, 2: DARK_FG}[ax_i]
    ax.set_title(title, fontsize=12, fontweight="bold", color=c_title, pad=10)


def graph13_transition_delta():
    pgd_r = _load("runs/pgd_smoke/records.jsonl")

    benign_edges, attacked_edges = _compute_edge_counters(pgd_r)

    all_tools = sorted(
        {t for r in pgd_r for key in ["benign", "attacked"] for t in r[key]["tool_sequence"]}
    )

    # Build graph
    _G, pos = _build_delta_graph(all_tools, benign_edges, attacked_edges)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor(DARK_BG)
    fig.subplots_adjust(wspace=0.06)

    panels = [
        ("Benign transitions", benign_edges, attacked_edges, C_BENIGN),
        ("PGD transitions", attacked_edges, benign_edges, C_PGD),
        ("Δ: PGD − Benign", None, None, None),
    ]

    for ax_i, (title, primary, _other, color) in enumerate(panels):
        ax = axes[ax_i]
        ax.set_facecolor(DARK_BG)

        if ax_i < 2:  # benign or PGD
            _draw_condition_panel(ax, primary, pos, color)
            _draw_delta_nodes(ax, all_tools, pos)
        else:  # Delta panel
            _render_delta_panel(ax, all_tools, pos, benign_edges, attacked_edges)

        _finalize_panel(ax, title, ax_i)

    fig.suptitle(
        "Transition delta graph — how PGD rewires the agent's tool-call graph",
        fontsize=14,
        fontweight="bold",
        color=DARK_FG,
        y=1.01,
    )
    fig.savefig(
        OUT / "graph13_transition_delta.png", bbox_inches="tight", facecolor=DARK_BG, dpi=200
    )
    plt.close(fig)
    print("graph13 ✓")
