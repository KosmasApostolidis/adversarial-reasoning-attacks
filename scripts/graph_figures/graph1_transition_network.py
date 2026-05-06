"""Graph 1: directed tool-transition graph — benign vs PGD overlaid."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from ._common import (
    C_BENIGN,
    C_NODE,
    C_PGD,
    OUT,
    _build_transition_graph,
    _load,
    _short,
)


def graph1_transition_network() -> None:
    pgd_recs = _load("runs/pgd_smoke/records.jsonl")
    Gb = _build_transition_graph(pgd_recs, "benign")
    Gp = _build_transition_graph(pgd_recs, "attacked")

    all_nodes = set(Gb.nodes) | set(Gp.nodes)
    G_layout = nx.DiGraph()
    G_layout.add_nodes_from(all_nodes)
    for u, v, d in Gb.edges(data=True):
        G_layout.add_edge(u, v, weight=d["weight"])
    for u, v, d in Gp.edges(data=True):
        G_layout.add_edge(u, v, weight=d.get("weight", 1))

    pos = nx.spring_layout(G_layout, seed=42, k=2.2, weight="weight")

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_aspect("equal")

    def draw_edges(G, color, alpha, lw_scale=2.5, rad=0.1):
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 1)
            ax.annotate(
                "",
                xy=pos[v],
                xycoords="data",
                xytext=pos[u],
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=w * lw_scale,
                    connectionstyle=f"arc3,rad={rad}",
                    mutation_scale=15,
                ),
                alpha=alpha,
                zorder=2,
            )

    draw_edges(Gb, C_BENIGN, 0.8, lw_scale=2.0, rad=0.12)
    draw_edges(Gp, C_PGD, 0.8, lw_scale=2.0, rad=-0.12)

    # Nodes
    all_counts = {
        n: Gb.nodes.get(n, {}).get("count", 0) + Gp.nodes.get(n, {}).get("count", 0)
        for n in all_nodes
    }
    node_sizes = [200 + all_counts.get(n, 0) * 80 for n in all_nodes]
    nx.draw_networkx_nodes(
        G_layout,
        pos,
        nodelist=list(all_nodes),
        node_size=node_sizes,
        node_color="#21262d",
        edgecolors="#58a6ff",
        linewidths=2.0,
        ax=ax,
    )
    labels = {n: _short(n) for n in all_nodes}
    nx.draw_networkx_labels(
        G_layout,
        pos,
        labels=labels,
        font_size=8,
        font_color=C_NODE,
        font_weight="bold",
        ax=ax,
    )

    legend_elements = [
        mpatches.Patch(color=C_BENIGN, label="Benign trajectory"),
        mpatches.Patch(color=C_PGD, label="PGD-attacked trajectory"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        framealpha=0.2,
        frameon=True,
        facecolor="#161b22",
        edgecolor="#30363d",
        fontsize=10,
    )
    ax.set_title(
        "Tool-call transition graph: benign vs PGD-L∞ attack",
        fontsize=14,
        fontweight="bold",
        pad=15,
        color=C_NODE,
    )
    ax.axis("off")
    fig.savefig(OUT / "graph1_transition_network.png")
    plt.close(fig)
    print("graph1 ✓")
