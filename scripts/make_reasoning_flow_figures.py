"""Reasoning-flow figures — visualise HOW adversarial attacks change VLM decision chains.

graph11_reasoning_paths.png   — per-patient side-by-side paths through tool space
graph12_alluvial.png          — alluvial stream: tool flow across steps (benign vs PGD)
graph13_transition_delta.png  — delta graph: edges added/removed by PGD
graph14_reasoning_strips.png  — horizontal strip comparison with divergence markers
graph15_multi_condition.png   — all 3 conditions (benign/noise/PGD) in one step×tool grid
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = Path("paper/figures/reasoning_flow")
OUT.mkdir(parents=True, exist_ok=True)

DARK_BG  = "#0d1117"
DARK_AX  = "#161b22"
DARK_FG  = "#e6edf3"
DARK_GRID= "#21262d"
C_BENIGN = "#58a6ff"   # blue
C_NOISE  = "#3fb950"   # green
C_PGD    = "#f85149"   # red
C_BOTH   = "#d2a8ff"   # purple (shared transitions)
C_CHANGE = "#ffa657"   # orange (changed steps)

SHORT = {
    "lookup_pubmed":          "PubMed",
    "query_guidelines":       "Guidelines",
    "calculate_risk_score":   "Risk Score",
    "draft_report":           "Draft Report",
    "request_followup":       "Followup",
    "escalate_to_specialist": "Escalate",
    "describe_region":        "Describe",
}
def _s(t): return SHORT.get(t, t.replace("_", " "))

def _load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]

def _dark_fig(w, h):
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    return fig

def _dark_ax(ax):
    ax.set_facecolor(DARK_AX)
    for s in ax.spines.values():
        s.set_color(DARK_GRID)
    ax.tick_params(colors=DARK_FG, which="both")
    ax.xaxis.label.set_color(DARK_FG)
    ax.yaxis.label.set_color(DARK_FG)
    ax.title.set_color(DARK_FG)


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH 11 — Per-patient reasoning path: benign path vs PGD path in tool space
# ═══════════════════════════════════════════════════════════════════════════

def graph11_reasoning_paths():
    pgd_r = _load("runs/pgd_smoke/records.jsonl")

    all_tools = sorted({
        t for r in pgd_r
        for key in ["benign", "attacked"]
        for t in r[key]["tool_sequence"]
    })
    n_tools = len(all_tools)

    # Fixed 2D positions in a circle
    angles = np.linspace(0, 2 * np.pi, n_tools, endpoint=False)
    tool_pos = {t: (np.cos(a) * 1.0, np.sin(a) * 1.0)
                for t, a in zip(all_tools, angles)}

    n_patients = len(pgd_r)
    fig, axes = plt.subplots(2, n_patients, figsize=(4.2 * n_patients, 8))
    fig.patch.set_facecolor(DARK_BG)
    fig.subplots_adjust(hspace=0.08, wspace=0.06)

    labels_done = set()

    for col, rec in enumerate(pgd_r):
        pid = rec["sample_id"].split("_p")[1] if "_p" in rec["sample_id"] else str(col)
        for row, (key, color, title) in enumerate([
            ("benign",   C_BENIGN, f"P{pid} — Benign"),
            ("attacked", C_PGD,    f"P{pid} — PGD"),
        ]):
            ax = axes[row, col]
            ax.set_facecolor(DARK_BG)

            seq = rec[key]["tool_sequence"]
            b_seq = rec["benign"]["tool_sequence"]

            # Background tool nodes (all tools, dim)
            for t in all_tools:
                x, y = tool_pos[t]
                ax.plot(x, y, "o", ms=18, color="#21262d", markeredgecolor=DARK_GRID,
                        markeredgewidth=1, zorder=2)
                ax.text(x, y, _s(t), ha="center", va="center",
                        fontsize=6, color="#8b949e", zorder=3)

            # Highlight visited tools
            visited = set(seq)
            for t in visited:
                x, y = tool_pos[t]
                in_both = t in set(b_seq)
                fc = color if in_both else C_CHANGE
                ax.plot(x, y, "o", ms=22, color=fc, alpha=0.85,
                        markeredgecolor="white", markeredgewidth=1.5, zorder=4)
                ax.text(x, y, _s(t), ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold", zorder=5,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground=DARK_BG)])

            # Draw path as gradient-colored arrows
            n = len(seq)
            cmap = (mpl.cm.Blues if color == C_BENIGN else mpl.cm.Reds)
            for step_i, (a_tool, b_tool) in enumerate(pairwise(seq)):
                x0, y0 = tool_pos[a_tool]
                x1, y1 = tool_pos[b_tool]
                frac = (step_i + 0.5) / max(n - 1, 1)
                c = cmap(0.4 + 0.55 * frac)
                ax.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=c,
                        lw=1.8,
                        connectionstyle="arc3,rad=0.18",
                        mutation_scale=12,
                    ),
                    alpha=0.85,
                    zorder=6,
                )
                # Step number near midpoint
                mx = (x0 + x1) / 2 + 0.05
                my = (y0 + y1) / 2 + 0.05
                ax.text(mx, my, str(step_i + 1), fontsize=6.5,
                        color=c, fontweight="bold", zorder=7,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground=DARK_BG)])

            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect("equal")
            ax.axis("off")

            # Edit distance annotation (bottom right)
            ed = rec["edit_distance_norm"]
            if row == 1:
                ax.text(1.4, -1.45, f"ed={ed:.2f}", ha="right", va="bottom",
                        fontsize=8.5, color=C_PGD, fontweight="bold")

            if col == 0:
                ax.text(-1.55, 0, title.split(" — ")[1], ha="right", va="center",
                        fontsize=9, color=color, fontweight="bold", rotation=90)

            # Column title above top row only
            if row == 0:
                ax.set_title(f"Patient {pid}", fontsize=10, color=DARK_FG, pad=6)

    fig.suptitle(
        "Reasoning paths through tool space — how PGD redirects the agent's decision flow",
        fontsize=13, fontweight="bold", color=DARK_FG, y=1.01,
    )
    fig.savefig(OUT / "graph11_reasoning_paths.png", bbox_inches="tight",
                facecolor=DARK_BG, dpi=200)
    plt.close(fig)
    print("graph11 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH 12 — Alluvial stream: tool flow at each step (benign vs PGD)
# ═══════════════════════════════════════════════════════════════════════════

def graph12_alluvial():
    pgd_r = _load("runs/pgd_smoke/records.jsonl")

    all_tools = sorted({
        t for r in pgd_r
        for key in ["benign", "attacked"]
        for t in r[key]["tool_sequence"]
    })
    tool_y = {t: i for i, t in enumerate(all_tools)}
    MAX_STEP = 8
    N_TOOLS  = len(all_tools)
    N_PAT    = len(pgd_r)

    # Colour per tool
    cmap20 = plt.get_cmap("tab20")
    tool_color = {t: cmap20(i % 20) for i, t in enumerate(all_tools)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(DARK_BG)
    fig.subplots_adjust(wspace=0.08)

    for ax, (key, cond_label) in zip(axes, [("benign", "Benign"), ("attacked", "PGD-attacked")]):
        ax.set_facecolor(DARK_BG)

        # For each (step, tool) cell: count occupancy
        occ = np.zeros((N_TOOLS, MAX_STEP))
        for r in pgd_r:
            for step, tool in enumerate(r[key]["tool_sequence"][:MAX_STEP]):
                occ[tool_y[tool], step] += 1

        # Draw stream bands
        bar_w = 0.55
        for step in range(MAX_STEP):
            y_cursor = 0.0
            for t in all_tools:
                cnt = occ[tool_y[t], step]
                if cnt > 0:
                    height = cnt / N_PAT  # normalise to [0,1]
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (step - bar_w/2, y_cursor),
                        bar_w, height * (N_TOOLS * 0.7),
                        boxstyle="round,pad=0.02",
                        facecolor=tool_color[t], alpha=0.88,
                        edgecolor="white", linewidth=0.8,
                        zorder=3,
                    ))
                    txt_y = y_cursor + height * N_TOOLS * 0.35
                    if height > 0.12:
                        ax.text(step, txt_y, _s(t), ha="center", va="center",
                                fontsize=7, color="white", fontweight="bold", zorder=4)
                    y_cursor += height * (N_TOOLS * 0.7) + 0.08

        # Flow ribbons between adjacent steps
        for step in range(MAX_STEP - 1):
            # Group by tool, draw connecting bezier-like fill
            for r in pgd_r:
                seq = r[key]["tool_sequence"]
                if step < len(seq) and step + 1 < len(seq):
                    t0, t1 = seq[step], seq[step + 1]
                    # Draw a thin arc
                    y0 = tool_y[t0] * 0.9 + occ[tool_y[t0], step] * 0.15
                    y1 = tool_y[t1] * 0.9 + occ[tool_y[t1], step+1] * 0.15
                    ax.annotate(
                        "", xy=(step + 0.6, y1), xytext=(step + 0.4, y0),
                        arrowprops=dict(
                            arrowstyle="-",
                            color=tool_color[t0],
                            lw=1.2,
                            connectionstyle="arc3,rad=0.0",
                            alpha=0.35,
                        ),
                        zorder=2,
                    )

        ax.set_xticks(range(MAX_STEP))
        ax.set_xticklabels([f"Step {i+1}" for i in range(MAX_STEP)],
                           fontsize=9, color=DARK_FG)
        ax.set_yticks([])
        ax.set_xlim(-0.7, MAX_STEP - 0.3)
        ax.set_title(cond_label, fontsize=13, fontweight="bold",
                     color=C_BENIGN if "Benign" in cond_label else C_PGD, pad=10)
        for sp in ax.spines.values():
            sp.set_color(DARK_GRID)
        ax.tick_params(colors=DARK_FG)
        ax.axhline(0, color=DARK_GRID, lw=0.5)

    # Shared legend
    handles = [mpatches.Patch(color=tool_color[t], label=_s(t)) for t in all_tools]
    fig.legend(handles=handles, loc="lower center", ncol=len(all_tools),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04),
               labelcolor=DARK_FG)
    fig.suptitle(
        "Alluvial stream — tool selection flow at each reasoning step: benign vs PGD",
        fontsize=13, fontweight="bold", color=DARK_FG, y=1.02,
    )
    fig.savefig(OUT / "graph12_alluvial.png", bbox_inches="tight",
                facecolor=DARK_BG, dpi=200)
    plt.close(fig)
    print("graph12 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH 13 — Transition delta graph: what PGD added / removed
# ═══════════════════════════════════════════════════════════════════════════

def graph13_transition_delta():
    pgd_r = _load("runs/pgd_smoke/records.jsonl")

    benign_edges:  Counter = Counter()
    attacked_edges: Counter = Counter()

    for r in pgd_r:
        for a, b in pairwise(r["benign"]["tool_sequence"]):
            benign_edges[(a, b)] += 1
        for a, b in pairwise(r["attacked"]["tool_sequence"]):
            attacked_edges[(a, b)] += 1

    all_tools = sorted({
        t for r in pgd_r
        for key in ["benign", "attacked"]
        for t in r[key]["tool_sequence"]
    })

    # Build graph
    G = nx.DiGraph()
    G.add_nodes_from(all_tools)
    all_edge_keys = set(benign_edges) | set(attacked_edges)
    for e in all_edge_keys:
        G.add_edge(*e)

    pos = nx.spring_layout(G, seed=13, k=2.5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor(DARK_BG)
    fig.subplots_adjust(wspace=0.06)

    panels = [
        ("Benign transitions",    benign_edges,  attacked_edges, C_BENIGN),
        ("PGD transitions",       attacked_edges, benign_edges,  C_PGD),
        ("Δ: PGD − Benign",       None, None, None),
    ]

    def draw_nodes(ax, highlight_tools=None):
        for t in all_tools:
            fc = "#21262d"
            ec = "#58a6ff" if (highlight_tools and t in highlight_tools) else DARK_GRID
            lw = 2 if (highlight_tools and t in highlight_tools) else 1
            circle = plt.Circle(pos[t], 0.08, color=fc, zorder=3)
            circle.set_edgecolor(ec)
            circle.set_linewidth(lw)
            ax.add_patch(circle)
            ax.text(pos[t][0], pos[t][1], _s(t),
                    ha="center", va="center", fontsize=7, color=DARK_FG,
                    fontweight="bold", zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=DARK_BG)])

    for ax_i, (title, primary, other, color) in enumerate(panels):
        ax = axes[ax_i]
        ax.set_facecolor(DARK_BG)

        if ax_i < 2:  # benign or PGD
            for (u, v), w in primary.items():
                ax.annotate(
                    "", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=0.7 + w * 1.2,
                                    connectionstyle="arc3,rad=0.15",
                                    mutation_scale=12),
                    alpha=0.8, zorder=2,
                )
            draw_nodes(ax)

        else:  # Delta panel
            # Added edges (in PGD, not in benign)
            added   = {e: w for e, w in attacked_edges.items() if e not in benign_edges}
            removed = {e: w for e, w in benign_edges.items()   if e not in attacked_edges}
            kept    = {e: w for e, w in benign_edges.items()   if e in attacked_edges}

            for edges_dict, ec, rad, label in [
                (kept,    "#8b949e", 0.0,  "kept"),
                (removed, C_BENIGN, 0.2,  "removed (benign only)"),
                (added,   C_PGD,   -0.2, "added (PGD only)"),
            ]:
                for (u, v), w in edges_dict.items():
                    ax.annotate(
                        "", xy=pos[v], xytext=pos[u],
                        arrowprops=dict(arrowstyle="-|>", color=ec,
                                        lw=0.7 + w * 1.4,
                                        connectionstyle=f"arc3,rad={rad}",
                                        mutation_scale=12),
                        alpha=0.85, zorder=2,
                    )

            draw_nodes(ax, highlight_tools=set(t for e in added for t in e) |
                                           set(t for e in removed for t in e))
            # Delta legend
            delta_handles = [
                mpatches.Patch(color="#8b949e", label=f"Kept ({len(kept)})"),
                mpatches.Patch(color=C_BENIGN,  label=f"Removed by PGD ({len(removed)})"),
                mpatches.Patch(color=C_PGD,     label=f"Added by PGD ({len(added)})"),
            ]
            ax.legend(handles=delta_handles, loc="lower right", fontsize=8.5,
                      framealpha=0.25, facecolor=DARK_AX, edgecolor=DARK_GRID,
                      labelcolor=DARK_FG)

        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal"); ax.axis("off")
        c_title = {0: C_BENIGN, 1: C_PGD, 2: DARK_FG}[ax_i]
        ax.set_title(title, fontsize=12, fontweight="bold", color=c_title, pad=10)

    fig.suptitle(
        "Transition delta graph — how PGD rewires the agent's tool-call graph",
        fontsize=14, fontweight="bold", color=DARK_FG, y=1.01,
    )
    fig.savefig(OUT / "graph13_transition_delta.png", bbox_inches="tight",
                facecolor=DARK_BG, dpi=200)
    plt.close(fig)
    print("graph13 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH 14 — Horizontal reasoning strips with divergence markers
# ═══════════════════════════════════════════════════════════════════════════

def graph14_reasoning_strips():
    pgd_r   = _load("runs/pgd_smoke/records.jsonl")
    noise_r = {r["sample_id"]: r for r in _load("runs/smoke/records.jsonl")}

    all_tools = sorted({
        t for r in pgd_r
        for key in ["benign", "attacked"]
        for t in r[key]["tool_sequence"]
    })
    cmap20 = plt.get_cmap("tab20")
    tool_color = {t: cmap20(i % 20) for i, t in enumerate(all_tools)}

    MAX_STEP = 10
    N_PAT    = len(pgd_r)
    ROW_H    = 2.2   # height per patient block

    fig_h = N_PAT * ROW_H + 1.2
    fig = _dark_fig(16, fig_h)
    ax  = fig.add_axes([0.12, 0.06, 0.85, 0.86])
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    # Step grid lines
    for s in range(MAX_STEP + 1):
        ax.axvline(s, color=DARK_GRID, lw=0.6, alpha=0.6, zorder=1)

    BLOCK_W = 0.88
    BLOCK_H = 0.58

    for pi, rec in enumerate(pgd_r):
        pid    = rec["sample_id"].split("_p")[1] if "_p" in rec["sample_id"] else str(pi)
        nr     = noise_r.get(rec["sample_id"])
        y_base = (N_PAT - 1 - pi) * ROW_H

        seqs = {
            "Benign": (rec["benign"]["tool_sequence"],   C_BENIGN, y_base + 1.35),
            "Noise":  (nr["attacked"]["tool_sequence"] if nr else [], C_NOISE, y_base + 0.68),
            "PGD":    (rec["attacked"]["tool_sequence"], C_PGD,    y_base + 0.02),
        }

        # First divergence step vs benign
        b = rec["benign"]["tool_sequence"]
        a = rec["attacked"]["tool_sequence"]
        div_step = next((i for i in range(min(len(b), len(a))) if b[i] != a[i]),
                        min(len(b), len(a)))

        for cond_label, (seq, cond_c, y_row) in seqs.items():
            # Row label
            ax.text(-0.12, y_row + BLOCK_H / 2, cond_label,
                    ha="right", va="center", fontsize=8.5,
                    color=cond_c, fontweight="bold")

            for step, tool in enumerate(seq[:MAX_STEP]):
                fc = tool_color[tool]
                # Brighter highlight if step is a divergence point
                is_changed = (step >= div_step and cond_label != "Benign"
                              and step < len(b) and step < len(seq)
                              and b[step] != tool)
                edge_c = C_CHANGE if is_changed else "white"
                edge_w = 2.5 if is_changed else 0.8

                ax.add_patch(mpatches.FancyBboxPatch(
                    (step + 0.06, y_row),
                    BLOCK_W, BLOCK_H,
                    boxstyle="round,pad=0.04",
                    facecolor=fc, edgecolor=edge_c,
                    linewidth=edge_w, zorder=3,
                ))
                ax.text(step + 0.06 + BLOCK_W / 2, y_row + BLOCK_H / 2,
                        _s(tool), ha="center", va="center",
                        fontsize=6.8, color="white", fontweight="bold", zorder=4,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground=DARK_BG)])

        # Divergence vertical marker
        if div_step < MAX_STEP:
            ax.axvline(div_step + 0.02, color=C_CHANGE, lw=2, linestyle="--",
                       alpha=0.7, zorder=2,
                       ymin=(y_base) / (N_PAT * ROW_H),
                       ymax=(y_base + ROW_H) / (N_PAT * ROW_H))
            ax.text(div_step + 0.12, y_base + ROW_H - 0.12,
                    f"⚡ div@{div_step+1}",
                    fontsize=8, color=C_CHANGE, fontweight="bold", zorder=5)

        # Patient label on left
        ax.text(-0.12, y_base + ROW_H / 2 + 0.15,
                f"P{pid}",
                ha="right", va="center", fontsize=11, color=DARK_FG,
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground=DARK_BG)])
        # Edit distance
        ax.text(MAX_STEP + 0.08, y_base + 0.69,
                f"ed={rec['edit_distance_norm']:.2f}",
                ha="left", va="center", fontsize=8.5, color=C_PGD, fontweight="bold")

        # Separator
        if pi < N_PAT - 1:
            ax.axhline(y_base + ROW_H, color=DARK_GRID, lw=1, alpha=0.5, zorder=1)

    # Step labels at top
    for s in range(MAX_STEP):
        ax.text(s + 0.5, N_PAT * ROW_H + 0.12, f"step {s+1}",
                ha="center", va="bottom", fontsize=8.5, color="#8b949e")

    ax.set_xlim(-0.25, MAX_STEP + 0.4)
    ax.set_ylim(-0.2, N_PAT * ROW_H + 0.4)

    # Tool colour legend
    handles = [mpatches.Patch(color=tool_color[t], label=_s(t)) for t in all_tools]
    fig.legend(handles=handles, loc="lower center", ncol=len(all_tools),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01),
               labelcolor=DARK_FG)

    fig.suptitle(
        "Reasoning strips — benign / noise / PGD trajectories with divergence markers",
        fontsize=13, fontweight="bold", color=DARK_FG, y=0.99,
    )
    fig.savefig(OUT / "graph14_reasoning_strips.png", bbox_inches="tight",
                facecolor=DARK_BG, dpi=200)
    plt.close(fig)
    print("graph14 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH 15 — All 3 conditions overlaid in step × tool grid (extended graph9)
# ═══════════════════════════════════════════════════════════════════════════

def graph15_multi_condition():
    pgd_r   = _load("runs/pgd_smoke/records.jsonl")
    noise_r = {r["sample_id"]: r for r in _load("runs/smoke/records.jsonl")}

    all_tools = sorted({
        t for r in pgd_r
        for key in ["benign", "attacked"]
        for t in r[key]["tool_sequence"]
    })
    tool_y = {t: i for i, t in enumerate(all_tools)}
    MAX_STEP = 8

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    conditions = [
        ("benign",   pgd_r,                                C_BENIGN, 2.5, "-",  0.80),
        ("noise",    [noise_r[r["sample_id"]] for r in pgd_r if r["sample_id"] in noise_r],
                                                           C_NOISE,  2.0, "--", 0.65),
        ("pgd",      pgd_r,                                C_PGD,    2.5, ":",  0.80),
    ]

    rng = np.random.default_rng(21)

    for cond_key, records, color, lw, ls, alpha in conditions:
        key = "attacked" if cond_key in ("noise", "pgd") else "benign"
        for pi, rec in enumerate(records):
            seq = rec[key]["tool_sequence"]
            xs  = list(range(min(len(seq), MAX_STEP)))
            jit = rng.uniform(-0.08, 0.08, len(xs))
            ys  = [tool_y[seq[s]] + jit[s] + ({"benign": -0.12, "noise": 0.0, "pgd": 0.12}[cond_key]) for s in xs]
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha,
                    linestyle=ls, zorder=3)
            ax.scatter(xs, ys, s=55, color=color, edgecolors="white", lw=0.6,
                       zorder=5, alpha=0.9)

    # Vertical step lines
    for s in range(MAX_STEP):
        ax.axvline(s, color=DARK_GRID, lw=0.7, zorder=1)

    # Horizontal tool band shading (alternating)
    for i, t in enumerate(all_tools):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#161b22", alpha=0.5, zorder=0)

    ax.set_xticks(range(MAX_STEP))
    ax.set_xticklabels([f"Step {i+1}" for i in range(MAX_STEP)],
                       fontsize=11, color=DARK_FG)
    ax.set_yticks(range(len(all_tools)))
    ax.set_yticklabels([_s(t) for t in all_tools], fontsize=10.5, color=DARK_FG)
    ax.set_xlabel("Trajectory step", fontsize=12, color=DARK_FG)
    ax.set_ylabel("Tool", fontsize=12, color=DARK_FG)
    ax.tick_params(colors=DARK_FG)
    for sp in ax.spines.values():
        sp.set_color(DARK_GRID)
    ax.set_xlim(-0.5, MAX_STEP - 0.5)
    ax.set_ylim(-0.6, len(all_tools) - 0.4)

    legend_handles = [
        mpatches.Patch(color=C_BENIGN, label="Benign  (solid)"),
        mpatches.Patch(color=C_NOISE,  label="Noise-attacked  (dashed)"),
        mpatches.Patch(color=C_PGD,    label="PGD-attacked  (dotted)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=11,
              framealpha=0.25, facecolor=DARK_AX, edgecolor=DARK_GRID,
              labelcolor=DARK_FG)
    ax.set_title(
        "All 3 conditions overlaid — benign / noise / PGD reasoning paths (n=5 patients)",
        fontsize=13, fontweight="bold", color=DARK_FG, pad=12,
    )
    fig.savefig(OUT / "graph15_multi_condition.png", bbox_inches="tight",
                facecolor=DARK_BG, dpi=200)
    plt.close(fig)
    print("graph15 ✓")


if __name__ == "__main__":
    graph11_reasoning_paths()
    graph12_alluvial()
    graph13_transition_delta()
    graph14_reasoning_strips()
    graph15_multi_condition()
    print(f"\nAll figures → {OUT}/")
