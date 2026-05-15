"""Graph 8: tool-node influence graph (glow = PGD sensitivity)."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

from ._common import C_BENIGN, C_PGD, DARK_BG, DARK_FG, GRAPH_OUT, _s, load_records


def _compute_sensitivity(pgd_r, all_tools):
    sensitivity: dict[str, float] = {}
    benign_freq: Counter = Counter()
    attacked_freq: Counter = Counter()
    changed: Counter = Counter()

    for r in pgd_r:
        b, a = r["benign"]["tool_sequence"], r["attacked"]["tool_sequence"]
        for i, bt in enumerate(b):
            benign_freq[bt] += 1
            if i < len(a) and a[i] != bt:
                changed[bt] += 1
        attacked_freq.update(a)

    for t in all_tools:
        sensitivity[t] = changed.get(t, 0) / max(benign_freq.get(t, 1), 1)
    return sensitivity, benign_freq


def _build_transition_graph(pgd_r, all_tools):
    G = nx.DiGraph()
    for r in pgd_r:
        for seq in [r["benign"]["tool_sequence"], r["attacked"]["tool_sequence"]]:
            for a, b in pairwise(seq):
                if G.has_edge(a, b):
                    G[a][b]["w"] = G[a][b]["w"] + 1
                else:
                    G.add_edge(a, b, w=1)
    for t in all_tools:
        if t not in G:
            G.add_node(t)
    return G


def _draw_glow_rings(ax, all_tools, pos, sensitivity):
    for t in all_tools:
        s = sensitivity.get(t, 0)
        if s > 0:
            for radius_scale, alpha in [(0.16, 0.06), (0.12, 0.12), (0.08, 0.22)]:
                glow = plt.Circle(
                    pos[t], radius_scale + 0.02, color=C_PGD, alpha=alpha * s, zorder=1
                )
                ax.add_patch(glow)


def _draw_edges(ax, G, pos):
    for u, v, d in G.edges(data=True):
        w = d.get("w", 1)
        ax.annotate(
            "",
            xy=pos[v],
            xytext=pos[u],
            arrowprops=dict(
                arrowstyle="-|>",
                color=C_BENIGN,
                lw=0.8 + w * 0.6,
                connectionstyle="arc3,rad=0.15",
                mutation_scale=12,
            ),
            alpha=0.5,
            zorder=2,
        )


def _draw_node(ax, t, pos, s, sz, cmap_sens):
    fc = cmap_sens(s)
    circle = plt.Circle(pos[t], sz, color=fc, zorder=3)
    circle.set_edgecolor("white")
    circle.set_linewidth(1.8)
    ax.add_patch(circle)
    ax.text(
        pos[t][0],
        pos[t][1],
        _s(t),
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        fontweight="bold",
        zorder=5,
        path_effects=[pe.withStroke(linewidth=2, foreground=DARK_BG)],
    )
    if s > 0:
        ax.text(
            pos[t][0],
            pos[t][1] - sz - 0.06,
            f"sens={s:.0%}",
            ha="center",
            fontsize=7.5,
            color=cmap_sens(s),
            zorder=6,
        )


def _draw_nodes(ax, all_tools, pos, sensitivity, benign_freq, cmap_sens):
    for t in all_tools:
        s = sensitivity.get(t, 0)
        sz = 0.07 + benign_freq.get(t, 0) * 0.014
        _draw_node(ax, t, pos, s, sz, cmap_sens)


def _draw_colorbar(fig, ax, cmap_sens):
    sm = plt.cm.ScalarMappable(cmap=cmap_sens, norm=mpl.colors.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Adversarial sensitivity (frac. positions changed)", fontsize=9, color=DARK_FG)
    cb.ax.yaxis.set_tick_params(color=DARK_FG)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=DARK_FG)


def graph8_tool_influence():
    pgd_r = load_records("runs/main/pgd/records.jsonl")
    noise_r = load_records("runs/main/noise/records.jsonl")

    all_tools = sorted(
        {
            t
            for r in pgd_r + noise_r
            for key in ["benign", "attacked"]
            for t in r[key]["tool_sequence"]
        }
    )

    sensitivity, benign_freq = _compute_sensitivity(pgd_r, all_tools)
    G = _build_transition_graph(pgd_r, all_tools)
    pos = nx.spring_layout(G, seed=7, k=2.8)

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    _draw_glow_rings(ax, all_tools, pos, sensitivity)
    _draw_edges(ax, G, pos)

    cmap_sens = LinearSegmentedColormap.from_list("sens", [C_BENIGN, "#f0c040", C_PGD], N=256)
    _draw_nodes(ax, all_tools, pos, sensitivity, benign_freq, cmap_sens)
    _draw_colorbar(fig, ax, cmap_sens)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Tool influence graph — glow + colour = adversarial sensitivity under PGD",
        fontsize=13,
        fontweight="bold",
        color=DARK_FG,
        pad=12,
    )
    fig.savefig(GRAPH_OUT / "graph8_tool_influence.png", bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("graph8 ✓")
