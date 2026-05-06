"""Graph 2: side-by-side agent decision flow before/after attack."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ._common import C_BENIGN, C_NODE, C_PGD, OUT, _load, _short


def graph2_rewiring() -> None:
    pgd_recs = _load("runs/pgd_smoke/records.jsonl")
    noise_recs = _load("runs/smoke/records.jsonl")

    all_tools: set[str] = set()
    for r in pgd_recs + noise_recs:
        all_tools.update(r["benign"]["tool_sequence"])
        all_tools.update(r["attacked"]["tool_sequence"])
    tools = sorted(all_tools)

    def build_freq(records, key):
        node_freq: Counter = Counter()
        edge_freq: Counter = Counter()
        for r in records:
            seq = r[key]["tool_sequence"]
            node_freq.update(seq)
            edge_freq.update(pairwise(seq))
        return node_freq, edge_freq

    bn_freq, be_freq = build_freq(pgd_recs, "benign")
    pn_freq, pe_freq = build_freq(pgd_recs, "attacked")

    # Fixed circular layout for both panels
    angles = np.linspace(0, 2 * np.pi, len(tools), endpoint=False)
    pos = {t: (np.cos(a) * 1.4, np.sin(a) * 1.4) for t, a in zip(tools, angles, strict=False)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    titles = ["Benign trajectory", "PGD-attacked trajectory"]
    nf_list = [bn_freq, pn_freq]
    ef_list = [be_freq, pe_freq]
    colors = [C_BENIGN, C_PGD]

    for ax, title, nf, ef, ec in zip(axes, titles, nf_list, ef_list, colors, strict=False):
        G = nx.DiGraph()
        G.add_nodes_from(tools)
        for (u, v), w in ef.items():
            G.add_edge(u, v, weight=w)

        # Edges
        for u, v, d in G.edges(data=True):
            w = d["weight"]
            rad = 0.18 if u != v else 0
            ax.annotate(
                "",
                xy=pos[v],
                xycoords="data",
                xytext=pos[u],
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=ec,
                    lw=0.8 + w * 1.2,
                    connectionstyle=f"arc3,rad={rad}",
                    mutation_scale=14,
                ),
                alpha=0.75,
                zorder=2,
            )

        # Nodes
        for t in tools:
            sz = 0.06 + nf.get(t, 0) * 0.018
            circle = plt.Circle(pos[t], sz, color="#21262d", zorder=3, linewidth=2)
            circle.set_edgecolor(ec)
            circle.set_linewidth(2.2)
            ax.add_patch(circle)
            ax.text(
                pos[t][0] * 1.18,
                pos[t][1] * 1.18,
                _short(t),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=C_NODE,
                zorder=5,
                path_effects=[pe.withStroke(linewidth=2, foreground="#0d1117")],
            )

        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", color=ec, pad=10)

    fig.suptitle(
        "Agent decision-flow rewiring under adversarial attack",
        fontsize=15,
        fontweight="bold",
        color=C_NODE,
        y=1.01,
    )
    fig.savefig(OUT / "graph2_rewiring.png")
    plt.close(fig)
    print("graph2 ✓")
