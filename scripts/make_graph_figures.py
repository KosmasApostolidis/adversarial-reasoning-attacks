"""Beautiful graph-based figures for adversarial-reasoning-attacks paper.

Outputs (paper/figures/graphs/):
  graph1_transition_network.png  — directed tool-transition graph: benign vs PGD
  graph2_rewiring.png            — side-by-side agent decision flow before/after attack
  graph3_radial_trajectories.png — polar chart: each patient's tool sequence as radial arcs
  graph4_sankey.png              — Sankey-style flow: what happened to each benign tool
  graph5_similarity_matrix.png   — pairwise trajectory edit-distance heatmap (all conditions)
"""

from __future__ import annotations

from collections import Counter
from itertools import pairwise
from pathlib import Path

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────
mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#0d1117",
        "text.color": "#e6edf3",
        "axes.labelcolor": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "axes.edgecolor": "#30363d",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#0d1117",
        "figure.dpi": 120,
    }
)

C_BENIGN = "#58a6ff"  # blue
C_NOISE = "#3fb950"  # green
C_PGD = "#f85149"  # red
C_NODE = "#e6edf3"

# Tool short names for labels
SHORT = {
    "lookup_pubmed": "PubMed",
    "query_guidelines": "Guidelines",
    "calculate_risk_score": "Risk Score",
    "draft_report": "Draft Report",
    "request_followup": "Followup",
    "escalate_to_specialist": "Escalate",
    "describe_region": "Describe",
}

OUT = Path("paper/figures/graphs")
OUT.mkdir(parents=True, exist_ok=True)


from _plotlib import load_records as _load  # noqa: E402


def _short(t: str) -> str:
    return SHORT.get(t, t.replace("_", "\n"))


def _build_transition_graph(records: list[dict], key: str) -> nx.DiGraph:
    G = nx.DiGraph()
    for r in records:
        seq = r[key]["tool_sequence"]
        for t in seq:
            G.add_node(t)
            G.nodes[t]["count"] = G.nodes[t].get("count", 0) + 1
        for a, b in pairwise(seq):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)
    return G


# ═══════════════════════════════════════════════════════════════════════════
# Graph 1 — Transition network: benign vs PGD overlaid
# ═══════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════
# Graph 2 — Side-by-side rewiring
# ═══════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════
# Graph 3 — Radial trajectory arcs (polar)
# ═══════════════════════════════════════════════════════════════════════════


def graph3_radial_trajectories() -> None:
    pgd_recs = _load("runs/pgd_smoke/records.jsonl")
    noise_recs = {r["sample_id"]: r for r in _load("runs/smoke/records.jsonl")}

    all_tools: list[str] = sorted(
        {
            t
            for r in pgd_recs
            for seq in [r["benign"]["tool_sequence"], r["attacked"]["tool_sequence"]]
            for t in seq
        }
    )
    n_tools = len(all_tools)
    tool_angle = {t: 2 * np.pi * i / n_tools for i, t in enumerate(all_tools)}

    n_patients = len(pgd_recs)
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("#0d1117")

    axes = []
    for i in range(n_patients):
        ax = fig.add_subplot(1, n_patients, i + 1, polar=True)
        ax.set_facecolor("#0d1117")
        axes.append(ax)

    CMAP_B = cm.Blues
    CMAP_N = cm.Greens
    CMAP_P = cm.Reds

    for ax, rec in zip(axes, pgd_recs, strict=False):
        sid = rec["sample_id"]
        nr = noise_recs.get(sid)
        seqs = {
            "benign": rec["benign"]["tool_sequence"],
            "noise": nr["attacked"]["tool_sequence"] if nr else [],
            "PGD": rec["attacked"]["tool_sequence"],
        }
        cmaps = {"benign": CMAP_B, "noise": CMAP_N, "PGD": CMAP_P}
        radii = {"benign": 0.72, "noise": 0.86, "PGD": 1.00}

        for condition, seq in seqs.items():
            if not seq:
                continue
            cmap = cmaps[condition]
            r0 = radii[condition]
            n = len(seq)
            for step_i, tool in enumerate(seq):
                ang = tool_angle[tool]
                frac = step_i / max(n - 1, 1)
                color = cmap(0.4 + 0.55 * frac)
                ax.plot(ang, r0, "o", markersize=9 * r0, color=color, alpha=0.9, zorder=4)
                if step_i > 0:
                    prev_ang = tool_angle[seq[step_i - 1]]
                    ax.annotate(
                        "",
                        xy=(ang, r0),
                        xycoords="data",
                        xytext=(prev_ang, r0),
                        textcoords="data",
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=color,
                            lw=1.4,
                            connectionstyle="arc3,rad=0.3",
                        ),
                        zorder=3,
                    )

        # Angular ticks = tool names
        ax.set_thetagrids(
            [np.degrees(tool_angle[t]) for t in all_tools],
            labels=[_short(t) for t in all_tools],
            fontsize=7,
            color="#8b949e",
        )
        ax.set_ylim(0, 1.15)
        ax.set_yticks([])
        ax.grid(color="#30363d", linewidth=0.5)
        pid = sid.split("_p")[1] if "_p" in sid else sid
        pgd_ed = rec["edit_distance_norm"]
        noise_ed = nr["edit_distance_norm"] if nr else float("nan")
        ax.set_title(
            f"P{pid}\nnoise={noise_ed:.2f}  PGD={pgd_ed:.2f}",
            fontsize=8.5,
            color=C_NODE,
            pad=14,
        )

    # Legend
    handles = [
        mpatches.Patch(color=CMAP_B(0.7), label="Benign"),
        mpatches.Patch(color=CMAP_N(0.7), label="Noise"),
        mpatches.Patch(color=CMAP_P(0.7), label="PGD"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(
        "Radial tool-trajectory chart: tool angles × trajectory rings",
        fontsize=13,
        fontweight="bold",
        color=C_NODE,
        y=1.01,
    )
    fig.savefig(OUT / "graph3_radial_trajectories.png")
    plt.close(fig)
    print("graph3 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# Graph 4 — Sankey-style flow: benign tools → fate under PGD
# ═══════════════════════════════════════════════════════════════════════════


def graph4_sankey() -> None:
    pgd_recs = _load("runs/pgd_smoke/records.jsonl")

    all_tools: list[str] = sorted(
        {
            t
            for r in pgd_recs
            for seq in [r["benign"]["tool_sequence"], r["attacked"]["tool_sequence"]]
            for t in seq
        }
    )
    n = len(all_tools)

    # Count how many times each benign tool appeared, and for each position
    # whether it was kept, substituted, or dropped in the attacked sequence
    kept_ct: Counter = Counter()
    sub_ct: Counter = Counter()
    drop_ct: Counter = Counter()
    ins_ct: Counter = Counter()

    for r in pgd_recs:
        b = r["benign"]["tool_sequence"]
        a = r["attacked"]["tool_sequence"]
        for i, bt in enumerate(b):
            if i < len(a):
                if bt == a[i]:
                    kept_ct[bt] += 1
                else:
                    sub_ct[bt] += 1
            else:
                drop_ct[bt] += 1
        for i in range(len(b), len(a)):
            ins_ct[a[i]] += 1

    fig, ax = plt.subplots(figsize=(13, 7))

    X_LEFT, X_RIGHT = 0.15, 0.85
    ys = np.linspace(0.1, 0.9, n)
    tool_y = {t: ys[i] for i, t in enumerate(all_tools)}
    bar_h = 0.06

    COLORS = {
        "kept": "#3fb950",
        "sub": "#f85149",
        "drop": "#8b949e",
        "ins": "#d2a8ff",
    }

    # Left column — benign tool boxes
    for t in all_tools:
        y = tool_y[t]
        total = kept_ct[t] + sub_ct[t] + drop_ct[t]
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (X_LEFT - 0.09, y - bar_h / 2),
                0.08,
                bar_h,
                boxstyle="round,pad=0.005",
                facecolor="#21262d",
                edgecolor=C_BENIGN,
                linewidth=1.5,
            )
        )
        ax.text(
            X_LEFT - 0.05,
            y,
            _short(t),
            ha="center",
            va="center",
            fontsize=7.5,
            color=C_NODE,
            fontweight="bold",
        )
        if total > 0:
            ax.text(
                X_LEFT - 0.12, y, str(total), ha="right", va="center", fontsize=8, color="#8b949e"
            )

    # Right column — fate boxes stacked
    {t: tool_y[t] for t in all_tools}
    fate_entries = []  # (left_tool, x_right, y_right, color, w)

    for lt in all_tools:
        ly = tool_y[lt]
        # kept
        if kept_ct[lt]:
            ry = ly
            fate_entries.append((lt, lt, kept_ct[lt], "kept"))
        # substituted → target tool
        if sub_ct[lt]:
            # pick first attacked tool at same position
            for r in pgd_recs:
                b = r["benign"]["tool_sequence"]
                a = r["attacked"]["tool_sequence"]
                for i, bt in enumerate(b):
                    if bt == lt and i < len(a) and a[i] != lt:
                        fate_entries.append((lt, a[i], sub_ct[lt], "sub"))
                        break
                else:
                    continue
                break
        if drop_ct[lt]:
            fate_entries.append((lt, None, drop_ct[lt], "drop"))

    for t in all_tools:
        if ins_ct[t]:
            fate_entries.append((None, t, ins_ct[t], "ins"))

    # Draw Sankey flow arcs
    for entry in fate_entries:
        lt, rt, w, fate = entry
        color = COLORS[fate]
        lw = 1.5 + w * 2.5
        ly = tool_y[lt] if lt else -0.05
        ry = tool_y[rt] if rt else 1.05

        ax.annotate(
            "",
            xy=(X_RIGHT, ry),
            xycoords="data",
            xytext=(X_LEFT, ly),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                connectionstyle="arc3,rad=0.0",
                mutation_scale=12,
                alpha=0.65,
            ),
            zorder=2,
        )

    # Right fate labels
    for fate, label in [
        ("kept", "Kept"),
        ("sub", "Substituted"),
        ("drop", "Dropped"),
        ("ins", "Inserted"),
    ]:
        total = sum(w for _, _, w, f in fate_entries if f == fate)
        if total:
            ys_fate = [tool_y[rt] for _, rt, _, f in fate_entries if f == fate and rt]
            yc = np.mean(ys_fate) if ys_fate else 0.5
            ax.text(
                X_RIGHT + 0.02,
                yc,
                f"{label}\n(n={total})",
                fontsize=8.5,
                color=COLORS[fate],
                fontweight="bold",
                va="center",
            )

    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    ax.set_title(
        "Sankey flow: benign tool fate under PGD-L∞ attack",
        fontsize=14,
        fontweight="bold",
        color=C_NODE,
        pad=15,
    )
    ax.text(
        X_LEFT - 0.05,
        0.97,
        "Benign tools",
        ha="center",
        fontsize=10,
        color=C_BENIGN,
        fontweight="bold",
    )
    ax.text(X_RIGHT + 0.02, 0.97, "Fate", ha="left", fontsize=10, color=C_NODE, fontweight="bold")

    fig.savefig(OUT / "graph4_sankey.png")
    plt.close(fig)
    print("graph4 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# Graph 5 — Pairwise trajectory similarity matrix (all conditions)
# ═══════════════════════════════════════════════════════════════════════════


def graph5_similarity_matrix() -> None:
    from adversarial_reasoning.metrics.trajectory import trajectory_edit_distance

    pgd_recs = _load("runs/pgd_smoke/records.jsonl")
    noise_recs = {r["sample_id"]: r for r in _load("runs/smoke/records.jsonl")}

    # Build (label, sequence) list per patient
    [r["sample_id"].split("_p")[1] for r in pgd_recs]
    all_seqs: list[tuple[str, list[str]]] = []
    all_labels: list[str] = []

    for r in pgd_recs:
        pid = r["sample_id"].split("_p")[1]
        nr = noise_recs.get(r["sample_id"])
        all_seqs.append((f"B·{pid}", r["benign"]["tool_sequence"]))
        all_seqs.append((f"N·{pid}", nr["attacked"]["tool_sequence"] if nr else []))
        all_seqs.append((f"P·{pid}", r["attacked"]["tool_sequence"]))
        all_labels += [f"B·{pid}", f"N·{pid}", f"P·{pid}"]

    n = len(all_seqs)
    mat = np.zeros((n, n))
    for i, (_, si) in enumerate(all_seqs):
        for j, (_, sj) in enumerate(all_seqs):
            mat[i, j] = trajectory_edit_distance(si, sj, normalize=True)

    # Cluster by condition groups
    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = cm.get_cmap("magma")
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    labels = [lbl for lbl, _ in all_seqs]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            tcolor = "white" if v > 0.5 else "#cccccc"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=tcolor)

    # Draw group separators every 3 rows/cols
    for k in range(1, len(pgd_recs)):
        ax.axhline(k * 3 - 0.5, color="#30363d", lw=1.5)
        ax.axvline(k * 3 - 0.5, color="#30363d", lw=1.5)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Normalised edit distance", fontsize=10, color=C_NODE)
    cb.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")

    # Legend for condition codes
    legend_txt = "B = benign  ·  N = noise-attacked  ·  P = PGD-attacked"
    ax.set_title(
        f"Pairwise trajectory edit-distance matrix (all conditions, n=5 patients)\n{legend_txt}",
        fontsize=11,
        fontweight="bold",
        color=C_NODE,
        pad=12,
    )
    fig.savefig(OUT / "graph5_similarity_matrix.png")
    plt.close(fig)
    print("graph5 ✓")


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    graph1_transition_network()
    graph2_rewiring()
    graph3_radial_trajectories()
    graph4_sankey()
    graph5_similarity_matrix()
    print(f"\nAll graph figures → {OUT}/")
