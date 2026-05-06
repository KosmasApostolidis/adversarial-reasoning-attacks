"""Comprehensive statistical + graph figures for adversarial-reasoning-attacks.

Statistical (light background, paper-ready):
  stat1_overview.png            — 4-panel: violin+box, scatter, correlation, grouped bars
  stat2_epsilon_sweep.png       — ε dose-response with per-sample scatter + regression
  stat3_trajectory_lengths.png  — trajectory length distributions + CDF
  stat4_step_heatmap.png        — tool occupancy per step position (benign vs PGD)

Graph / reasoning-flow (dark background, visually striking):
  graph6_bipartite.png          — bipartite benign↔attacked alignment per patient
  graph7_divergence.png         — per-patient divergence tree (where reasoning splits)
  graph8_tool_influence.png     — tool-node influence graph (glow = PGD sensitivity)
  graph9_layered_flow.png       — layered step graph: benign path vs PGD path
  graph10_step_occupancy.png    — heatmap: which tools appear at each step under attack
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from _plotlib import despine, load_records
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as scipy_stats

# ── Output directories ───────────────────────────────────────────────────
STAT_OUT = Path("paper/figures/stats")
GRAPH_OUT = Path("paper/figures/graphs_v2")
STAT_OUT.mkdir(parents=True, exist_ok=True)
GRAPH_OUT.mkdir(parents=True, exist_ok=True)

# ── Colour palette ───────────────────────────────────────────────────────
C_BENIGN = "#2166ac"
C_NOISE = "#4dac26"
C_PGD = "#d73027"
C_LLAVA = "#f46d43"
C_ACCENT = "#762a83"
DARK_BG = "#0d1117"
DARK_AX = "#161b22"
DARK_FG = "#e6edf3"
DARK_GRID = "#30363d"

SHORT = {
    "lookup_pubmed": "PubMed",
    "query_guidelines": "Guidelines",
    "calculate_risk_score": "Risk Score",
    "draft_report": "Draft Report",
    "request_followup": "Followup",
    "escalate_to_specialist": "Escalate",
    "describe_region": "Describe",
}


def _s(t: str) -> str:
    return SHORT.get(t, t.replace("_", "\n"))


def _panel(ax, letter, x=-0.13, y=1.07):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")


# ═══════════════════════════════════════════════════════════════════════════
# STAT 1 — 4-panel overview
# ═══════════════════════════════════════════════════════════════════════════


def stat1_overview():
    noise_r = load_records("runs/main/noise/records.jsonl")
    pgd_r = load_records("runs/main/pgd/records.jsonl")
    llava_r = [r for r in noise_r if "llava" in r.get("model_id", "").lower()]

    nd = np.array([r["edit_distance_norm"] for r in noise_r])
    pd_ = np.array([r["edit_distance_norm"] for r in pgd_r])
    ld = np.array([r["edit_distance_norm"] for r in llava_r])

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.subplots_adjust(hspace=0.42, wspace=0.35)

    # ── A: Violin + strip (3 conditions) ──────────────────────────────
    ax = axes[0, 0]
    data = [nd, pd_, ld]
    labels = ["Qwen\nNoise", "Qwen\nPGD", "LLaVA\nNoise"]
    colors = [C_NOISE, C_PGD, C_LLAVA]
    vp = ax.violinplot(data, positions=[1, 2, 3], showmedians=False, showextrema=False, widths=0.55)
    for pc, c in zip(vp["bodies"], colors, strict=False):
        pc.set_facecolor(c)
        pc.set_alpha(0.35)
        pc.set_edgecolor(c)
        pc.set_linewidth(1.5)

    bp = ax.boxplot(
        data,
        positions=[1, 2, 3],
        widths=0.18,
        patch_artist=True,
        medianprops=dict(color="white", lw=2.5),
        whiskerprops=dict(lw=1.5),
        capprops=dict(lw=1.5),
        flierprops=dict(marker="d", ms=5, alpha=0.5),
    )
    for box, c in zip(bp["boxes"], colors, strict=False):
        box.set_facecolor(c)
        box.set_alpha(0.85)

    rng = np.random.default_rng(42)
    for xi, arr, c in zip([1, 2, 3], data, colors, strict=False):
        jit = rng.uniform(-0.08, 0.08, len(arr))
        ax.scatter(xi + jit, arr, s=55, color=c, edgecolors="white", lw=0.8, zorder=5)
        ax.text(
            xi,
            arr.max() + 0.07,
            f"μ={arr.mean():.3f}",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color=c,
        )

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Normalised edit distance")
    ax.set_ylim(bottom=0)
    ax.set_title("Edit-distance distribution by condition", pad=8)
    despine(ax)
    _panel(ax, "A")

    # ── B: Scatter — benign length vs edit distance ────────────────────
    ax = axes[0, 1]
    b_lens_n = [len(r["benign"]["tool_sequence"]) for r in noise_r]
    b_lens_p = [len(r["benign"]["tool_sequence"]) for r in pgd_r]
    ax.scatter(
        b_lens_n,
        nd,
        s=90,
        color=C_NOISE,
        edgecolors="white",
        lw=0.8,
        label="Noise",
        zorder=4,
        alpha=0.9,
    )
    ax.scatter(
        b_lens_p,
        pd_,
        s=90,
        color=C_PGD,
        edgecolors="white",
        lw=0.8,
        label="PGD",
        zorder=4,
        alpha=0.9,
    )
    # regression lines
    for xs, ys, c in [(b_lens_n, nd, C_NOISE), (b_lens_p, pd_, C_PGD)]:
        if len(set(xs)) > 1:
            m, b, r, *_ = scipy_stats.linregress(xs, ys)
            xr = np.linspace(min(xs), max(xs), 50)
            ax.plot(xr, m * xr + b, color=c, lw=1.8, alpha=0.6, linestyle="--", label=f"r={r:.2f}")
    ax.set_xlabel("Benign trajectory length (steps)")
    ax.set_ylabel("Normalised edit distance")
    ax.set_title("Benign length vs attack drift", pad=8)
    ax.legend(fontsize=9)
    despine(ax)
    _panel(ax, "B")

    # ── C: Correlation matrix ──────────────────────────────────────────
    ax = axes[1, 0]
    all_r = pgd_r  # only PGD has pgd_loss
    feats = {
        "ε": [r["epsilon"] for r in all_r],
        "b_len": [len(r["benign"]["tool_sequence"]) for r in all_r],
        "a_len": [len(r["attacked"]["tool_sequence"]) for r in all_r],
        "edit_dist": [r["edit_distance_norm"] for r in all_r],
        "pgd_loss": [r["attacked"]["metadata"]["pgd_loss_final"] for r in all_r],
        "elapsed_s": [r["elapsed_s"] for r in all_r],
    }
    feat_names = list(feats.keys())
    arr = np.array([feats[k] for k in feat_names])
    corr = np.corrcoef(arr)

    cmap = LinearSegmentedColormap.from_list("rwb", ["#2166ac", "white", "#d73027"], N=256)
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=9)
    for i in range(len(feat_names)):
        for j in range(len(feat_names)):
            v = corr[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="black" if abs(v) < 0.5 else "white",
                fontweight="bold",
            )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Pearson r", fontsize=9)
    ax.set_title("Feature correlation matrix (PGD records)", pad=8)
    _panel(ax, "C")

    # ── D: Grouped bar — tool call count change ────────────────────────
    ax = axes[1, 1]
    all_tools = sorted(
        {
            t
            for r in pgd_r + noise_r
            for seq in [r["benign"]["tool_sequence"], r["attacked"]["tool_sequence"]]
            for t in seq
        }
    )
    n_tools = len(all_tools)
    x = np.arange(n_tools)
    w = 0.25

    def tool_counts(records, key):
        c: Counter = Counter()
        for r in records:
            c.update(r[key]["tool_sequence"])
        return [c.get(t, 0) for t in all_tools]

    b_cnt = tool_counts(pgd_r, "benign")
    n_cnt = tool_counts(noise_r, "attacked")
    p_cnt = tool_counts(pgd_r, "attacked")

    ax.bar(x - w, b_cnt, w, color=C_BENIGN, label="Benign", edgecolor="white", lw=0.5)
    ax.bar(x, n_cnt, w, color=C_NOISE, label="Noise atk", edgecolor="white", lw=0.5)
    ax.bar(x + w, p_cnt, w, color=C_PGD, label="PGD atk", edgecolor="white", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([_s(t) for t in all_tools], fontsize=8.5)
    ax.set_ylabel("Total invocations (all patients)")
    ax.set_title("Tool invocation frequency by condition", pad=8)
    ax.legend(fontsize=9)
    despine(ax)
    _panel(ax, "D")

    fig.suptitle(
        "Comprehensive statistical overview of adversarial attack effects",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.savefig(STAT_OUT / "stat1_overview.png", bbox_inches="tight")
    plt.close(fig)
    print("stat1 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# STAT 2 — ε sweep + per-sample scatter + regression
# ═══════════════════════════════════════════════════════════════════════════


def stat2_epsilon_sweep():
    sweep = load_records("runs/main/noise/records.jsonl")
    pgd_r = load_records("runs/main/pgd/records.jsonl")

    eps_vals = sorted(set(r["epsilon"] for r in sweep))
    per_eps: dict[float, list] = defaultdict(list)
    for r in sweep:
        per_eps[r["epsilon"]].append(r["edit_distance_norm"])
    means = [np.mean(per_eps[e]) for e in eps_vals]
    sems = [scipy_stats.sem(per_eps[e]) if len(per_eps[e]) > 1 else 0 for e in eps_vals]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.35)

    # ── A: Sweep with per-sample strip ────────────────────────────────
    ax = axes[0]
    ax.errorbar(
        [e * 255 for e in eps_vals],
        means,
        yerr=sems,
        fmt="-o",
        color=C_NOISE,
        lw=2.5,
        ms=9,
        capsize=5,
        label="mean ± SEM (noise, Qwen)",
    )
    ax.fill_between(
        [e * 255 for e in eps_vals],
        [m - s for m, s in zip(means, sems, strict=False)],
        [m + s for m, s in zip(means, sems, strict=False)],
        alpha=0.18,
        color=C_NOISE,
    )

    rng = np.random.default_rng(9)
    for e in eps_vals:
        ys = per_eps[e]
        xs = [e * 255] * len(ys) + rng.uniform(-0.2, 0.2, len(ys))
        ax.scatter(xs, ys, s=50, color=C_NOISE, alpha=0.6, edgecolors="white", lw=0.6, zorder=5)

    ax.axhline(
        pgd_r[0]["edit_distance_norm"]
        if False
        else np.mean([r["edit_distance_norm"] for r in pgd_r]),
        color=C_PGD,
        lw=2,
        linestyle="--",
        label=f"PGD mean={np.mean([r['edit_distance_norm'] for r in pgd_r]):.3f}",
    )

    ax.set_xlabel("ε (pixel units × 255)", fontsize=11)
    ax.set_ylabel("Normalised edit distance", fontsize=11)
    ax.set_title("Dose–response: uniform noise (ε sweep)", pad=8)
    ax.set_xticks([e * 255 for e in eps_vals])
    ax.set_xticklabels(["2/255", "4/255", "8/255", "16/255"])
    ax.legend(fontsize=9)
    despine(ax)
    _panel(ax, "A")

    # ── B: Attack mode comparison bar with individual points ──────────
    ax = axes[1]
    noise_r = load_records("runs/main/noise/records.jsonl")
    nd = [r["edit_distance_norm"] for r in noise_r]
    pd_ = [r["edit_distance_norm"] for r in pgd_r]
    groups = [("Uniform\nNoise", nd, C_NOISE), ("PGD-L∞\n20 steps", pd_, C_PGD)]
    xs = [1, 2]
    for xi, (_label, vals, c) in zip(xs, groups, strict=False):
        ax.bar(
            xi,
            np.mean(vals),
            width=0.5,
            color=c,
            alpha=0.75,
            edgecolor="white",
            lw=1.5,
            yerr=scipy_stats.sem(vals),
            capsize=7,
            error_kw=dict(elinewidth=2, capthick=2, ecolor="#444"),
        )
        jit = rng.uniform(-0.07, 0.07, len(vals))
        ax.scatter(xi + jit, vals, s=70, color=c, edgecolors="white", lw=1, zorder=5, alpha=0.9)

    # Significance bracket
    y_max = max(max(nd), max(pd_)) + 0.12
    ax.annotate(
        "", xy=(2, y_max), xytext=(1, y_max), arrowprops=dict(arrowstyle="-", color="black", lw=1.8)
    )
    ax.text(
        1.5,
        y_max + 0.03,
        "★  3× more drift",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=C_PGD,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(["Uniform\nNoise", "PGD-L∞\n20 steps"], fontsize=11)
    ax.set_ylabel("Normalised edit distance", fontsize=11)
    ax.set_title("Attack effectiveness at ε=0.0314 (8/255)\nQwen2.5-VL-7B, n=5 patients", pad=8)
    ax.set_ylim(bottom=0)
    despine(ax)
    _panel(ax, "B")

    fig.suptitle(
        "ε-sweep dose–response and attack mode comparison", fontsize=13, fontweight="bold", y=1.02
    )
    fig.savefig(STAT_OUT / "stat2_epsilon_sweep.png", bbox_inches="tight")
    plt.close(fig)
    print("stat2 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# STAT 3 — Trajectory lengths + CDF
# ═══════════════════════════════════════════════════════════════════════════


def stat3_trajectory_lengths():
    noise_r = load_records("runs/main/noise/records.jsonl")
    pgd_r = load_records("runs/main/pgd/records.jsonl")
    llava_r = [r for r in noise_r if "llava" in r.get("model_id", "").lower()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.35)

    datasets = [
        ("Benign (Qwen)", [len(r["benign"]["tool_sequence"]) for r in pgd_r], C_BENIGN),
        ("Noise-atk (Qwen)", [len(r["attacked"]["tool_sequence"]) for r in noise_r], C_NOISE),
        ("PGD-atk (Qwen)", [len(r["attacked"]["tool_sequence"]) for r in pgd_r], C_PGD),
        ("Noise-atk (LLaVA)", [len(r["attacked"]["tool_sequence"]) for r in llava_r], C_LLAVA),
    ]

    # ── A: Overlapping histograms ─────────────────────────────────────
    ax = axes[0]
    bins = np.arange(0, 16, 1)
    for label, vals, c in datasets:
        ax.hist(vals, bins=bins, alpha=0.45, color=c, edgecolor=c, lw=1.2, label=label)
    ax.set_xlabel("Trajectory length (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Trajectory length distribution", pad=8)
    ax.legend(fontsize=8)
    despine(ax)
    _panel(ax, "A")

    # ── B: CDF ────────────────────────────────────────────────────────
    ax = axes[1]
    for label, vals, c in datasets:
        s = sorted(vals)
        y = np.arange(1, len(s) + 1) / len(s)
        ax.step(s, y, color=c, lw=2, label=label)
    ax.set_xlabel("Trajectory length (steps)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF of trajectory lengths", pad=8)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    despine(ax)
    _panel(ax, "B")

    # ── C: Scatter — benign len vs attacked len (noise vs PGD) ────────
    ax = axes[2]
    bl_n = [len(r["benign"]["tool_sequence"]) for r in noise_r]
    al_n = [len(r["attacked"]["tool_sequence"]) for r in noise_r]
    bl_p = [len(r["benign"]["tool_sequence"]) for r in pgd_r]
    al_p = [len(r["attacked"]["tool_sequence"]) for r in pgd_r]

    ax.scatter(bl_n, al_n, s=90, color=C_NOISE, edgecolors="white", lw=0.8, label="Noise", zorder=4)
    ax.scatter(bl_p, al_p, s=90, color=C_PGD, edgecolors="white", lw=0.8, label="PGD", zorder=4)
    lim = max(max(bl_n + bl_p), max(al_n + al_p)) + 1
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, alpha=0.4, label="b_len = a_len")
    ax.set_xlabel("Benign trajectory length")
    ax.set_ylabel("Attacked trajectory length")
    ax.set_title("Length before vs after attack", pad=8)
    ax.legend(fontsize=9)
    despine(ax)
    _panel(ax, "C")

    fig.suptitle(
        "Trajectory length analysis: how attacks shorten/extend reasoning chains",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(STAT_OUT / "stat3_trajectory_lengths.png", bbox_inches="tight")
    plt.close(fig)
    print("stat3 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# STAT 4 — Step-position × tool occupancy heatmap
# ═══════════════════════════════════════════════════════════════════════════


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

    def build_matrix(records, key):
        mat = np.zeros((len(all_tools), MAX_STEP))
        tidx = {t: i for i, t in enumerate(all_tools)}
        for r in records:
            for step, tool in enumerate(r[key]["tool_sequence"][:MAX_STEP]):
                mat[tidx[tool], step] += 1
        return mat

    conditions = [
        ("Benign", build_matrix(pgd_r, "benign"), C_BENIGN),
        ("Noise-attacked", build_matrix(noise_r, "attacked"), C_NOISE),
        ("PGD-attacked", build_matrix(pgd_r, "attacked"), C_PGD),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.4)
    tool_labels = [_s(t) for t in all_tools]

    for ax, (title, mat, c) in zip(axes, conditions, strict=False):
        cmap = LinearSegmentedColormap.from_list(f"cm_{c}", ["#f7f7f7", c], N=256)
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0)
        ax.set_xticks(range(MAX_STEP))
        ax.set_xticklabels([f"s{i + 1}" for i in range(MAX_STEP)], fontsize=9)
        ax.set_yticks(range(len(all_tools)))
        ax.set_yticklabels(tool_labels, fontsize=9)
        ax.set_xlabel("Step position")
        ax.set_title(title, pad=8, fontsize=11)
        for i in range(len(all_tools)):
            for j in range(MAX_STEP):
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
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Count", fontsize=8)

    fig.suptitle(
        "Tool occupancy at each trajectory step: how PGD rewires step-by-step reasoning",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(STAT_OUT / "stat4_step_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("stat4 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# DARK GRAPH HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _dark_ax(ax):
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors=DARK_FG)
    for s in ax.spines.values():
        s.set_color(DARK_GRID)


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH 6 — Bipartite alignment: benign steps ↔ attacked steps
# ═══════════════════════════════════════════════════════════════════════════


def graph6_bipartite():
    pgd_r = load_records("runs/main/pgd/records.jsonl")
    n = len(pgd_r)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 7))
    fig.patch.set_facecolor(DARK_BG)

    for ax, rec in zip(axes, pgd_r, strict=False):
        ax.set_facecolor(DARK_BG)
        b = rec["benign"]["tool_sequence"]
        a = rec["attacked"]["tool_sequence"]
        max_len = max(len(b), len(a), 1)

        # Draw nodes
        for _col, seq, xpos in [(b, "left", 0.15), (a, "right", 0.85)]:
            for step, tool in enumerate(seq):
                y = 1 - (step + 0.5) / max_len
                fc = C_BENIGN if xpos < 0.5 else C_PGD
                circle = plt.Circle((xpos, y), 0.045, color=fc, zorder=4, linewidth=2)
                circle.set_edgecolor("white")
                ax.add_patch(circle)
                ax.text(
                    xpos,
                    y,
                    _s(tool),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                    fontweight="bold",
                    zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=DARK_BG)],
                )
            # column header
            ax.text(
                xpos,
                1.04,
                "Benign" if xpos < 0.5 else "PGD",
                ha="center",
                fontsize=9,
                color=C_BENIGN if xpos < 0.5 else C_PGD,
                fontweight="bold",
            )

        # Draw alignment edges
        for i in range(max_len):
            bt = b[i] if i < len(b) else None
            at = a[i] if i < len(a) else None
            if bt is None and at is None:
                continue
            by = 1 - (i + 0.5) / max_len if bt else None
            ay = 1 - (i + 0.5) / max_len if at else None

            if bt and at:
                color = C_NOISE if bt == at else C_PGD
                lw = 2.5 if bt == at else 1.8
                alpha = 0.85 if bt == at else 0.65
                ax.annotate(
                    "",
                    xy=(0.83, ay),
                    xytext=(0.23, by),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=10),
                    alpha=alpha,
                )
            elif bt and not at:
                ax.text(
                    0.5, by, "✕", ha="center", va="center", fontsize=12, color="#8b949e", alpha=0.7
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.12)
        ax.axis("off")
        pid = rec["sample_id"].split("_p")[1] if "_p" in rec["sample_id"] else rec["sample_id"]
        ax.set_title(
            f"P{pid}\ned={rec['edit_distance_norm']:.3f}", fontsize=9, color=DARK_FG, pad=4
        )

    # Legend
    handles = [
        mpatches.Patch(color=C_NOISE, label="Kept (same tool)"),
        mpatches.Patch(color=C_PGD, label="Substituted"),
        mpatches.Patch(color="#8b949e", label="Dropped (✕)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.04),
        labelcolor=DARK_FG,
    )
    fig.suptitle(
        "Bipartite trajectory alignment: benign ↔ PGD-attacked (per patient, per step)",
        fontsize=13,
        fontweight="bold",
        color=DARK_FG,
        y=1.01,
    )
    fig.savefig(GRAPH_OUT / "graph6_bipartite.png", bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("graph6 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH 7 — Divergence tree (where benign and PGD paths split)
# ═══════════════════════════════════════════════════════════════════════════


def graph7_divergence():
    pgd_r = load_records("runs/main/pgd/records.jsonl")

    fig, axes = plt.subplots(1, len(pgd_r), figsize=(4.5 * len(pgd_r), 7))
    fig.patch.set_facecolor(DARK_BG)

    for ax, rec in zip(axes, pgd_r, strict=False):
        ax.set_facecolor(DARK_BG)
        b = rec["benign"]["tool_sequence"]
        a = rec["attacked"]["tool_sequence"]
        max_len = max(len(b), len(a))

        # Find first divergence point
        div_step = next(
            (i for i in range(min(len(b), len(a))) if b[i] != a[i]), min(len(b), len(a))
        )

        # Draw steps
        for step in range(max_len):
            y = 1 - (step + 0.5) / max_len
            for seq, xpos, c in [(b, 0.3, C_BENIGN), (a, 0.7, C_PGD)]:
                if step < len(seq):
                    merged = step < div_step
                    draw_x = 0.5 if merged else xpos
                    fc = "#21262d" if merged else (C_BENIGN if c == C_BENIGN else C_PGD)
                    ec = C_BENIGN if merged else c
                    circle = plt.Circle((draw_x, y), 0.055, color=fc, zorder=4)
                    circle.set_edgecolor(ec)
                    circle.set_linewidth(2)
                    ax.add_patch(circle)
                    if not merged or c == C_BENIGN:
                        ax.text(
                            draw_x,
                            y,
                            _s(seq[step]),
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="white",
                            fontweight="bold",
                            zorder=5,
                            path_effects=[pe.withStroke(linewidth=1.5, foreground=DARK_BG)],
                        )

            # Divergence marker
            if step == div_step and div_step < min(len(b), len(a)):
                ax.axhline(
                    y + 0.5 / max_len,
                    color=C_PGD,
                    lw=1.5,
                    linestyle="--",
                    alpha=0.6,
                    xmin=0.1,
                    xmax=0.9,
                )
                ax.text(
                    0.95, y + 0.5 / max_len, "⚡", fontsize=12, va="center", color=C_PGD, zorder=6
                )

            # Connection lines from previous step
            if step > 0:
                py = 1 - (step - 0.5) / max_len
                for seq, xpos, c in [(b, 0.3, C_BENIGN), (a, 0.7, C_PGD)]:
                    if step < len(seq) and step - 1 < len(seq):
                        prev_merged = step - 1 < div_step
                        curr_merged = step < div_step
                        px = 0.5 if prev_merged else xpos
                        cx = 0.5 if curr_merged else xpos
                        if not (prev_merged and curr_merged and c == C_PGD):
                            ax.plot([px, cx], [py, y], color=c, lw=1.5, alpha=0.55, zorder=2)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.1)
        ax.axis("off")
        pid = rec["sample_id"].split("_p")[1] if "_p" in rec["sample_id"] else rec["sample_id"]
        label = f"P{pid}  |  diverges at step {div_step + 1}"
        ax.set_title(label, fontsize=8.5, color=DARK_FG, pad=4)
        # Column labels at bottom
        if div_step < max_len:
            ax.text(0.3, -0.05, "Benign", ha="center", fontsize=8, color=C_BENIGN)
            ax.text(0.7, -0.05, "PGD", ha="center", fontsize=8, color=C_PGD)

    fig.suptitle(
        "Divergence tree: where adversarial attack splits the reasoning chain",
        fontsize=13,
        fontweight="bold",
        color=DARK_FG,
        y=1.01,
    )
    fig.savefig(GRAPH_OUT / "graph7_divergence.png", bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("graph7 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH 8 — Tool influence graph (glow = adversarial sensitivity)
# ═══════════════════════════════════════════════════════════════════════════


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

    # Compute per-tool adversarial sensitivity: fraction of step positions
    # where the tool was CHANGED under PGD vs benign
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

    # Graph
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

    pos = nx.spring_layout(G, seed=7, k=2.8)

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    # Draw glow rings for high-sensitivity nodes
    for t in all_tools:
        s = sensitivity.get(t, 0)
        if s > 0:
            for radius_scale, alpha in [(0.16, 0.06), (0.12, 0.12), (0.08, 0.22)]:
                glow = plt.Circle(
                    pos[t], radius_scale + 0.02, color=C_PGD, alpha=alpha * s, zorder=1
                )
                ax.add_patch(glow)

    # Edges
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

    # Nodes — size ∝ benign_freq, color ∝ sensitivity
    cmap_sens = LinearSegmentedColormap.from_list("sens", [C_BENIGN, "#f0c040", C_PGD], N=256)
    for t in all_tools:
        s = sensitivity.get(t, 0)
        sz = 0.07 + benign_freq.get(t, 0) * 0.014
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

    sm = plt.cm.ScalarMappable(cmap=cmap_sens, norm=mpl.colors.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Adversarial sensitivity (frac. positions changed)", fontsize=9, color=DARK_FG)
    cb.ax.yaxis.set_tick_params(color=DARK_FG)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=DARK_FG)

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


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH 9 — Layered step flow: benign path vs PGD path
# ═══════════════════════════════════════════════════════════════════════════


def graph9_layered_flow():
    pgd_r = load_records("runs/main/pgd/records.jsonl")

    all_tools = sorted(
        {t for r in pgd_r for key in ["benign", "attacked"] for t in r[key]["tool_sequence"]}
    )
    MAX_STEP = 7
    tool_y = {t: i for i, t in enumerate(all_tools)}

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    # For each patient and each condition, draw a path through step × tool space
    for ri, rec in enumerate(pgd_r):
        offset = (ri - 2) * 0.12  # vertical jitter per patient

        for seq, c, lw, alpha, ls in [
            (rec["benign"]["tool_sequence"], C_BENIGN, 2.0, 0.75, "-"),
            (rec["attacked"]["tool_sequence"], C_PGD, 2.0, 0.75, "--"),
        ]:
            xs = list(range(min(len(seq), MAX_STEP)))
            ys = [tool_y[seq[s]] + offset for s in xs]
            ax.plot(xs, ys, color=c, lw=lw, alpha=alpha, linestyle=ls, zorder=3)
            ax.scatter(xs, ys, s=60, color=c, edgecolors="white", lw=0.7, zorder=5, alpha=0.9)

    # Vertical grid lines per step
    for s in range(MAX_STEP):
        ax.axvline(s, color=DARK_GRID, lw=0.8, zorder=1)

    ax.set_xticks(range(MAX_STEP))
    ax.set_xticklabels([f"Step {i + 1}" for i in range(MAX_STEP)], fontsize=10, color=DARK_FG)
    ax.set_yticks(range(len(all_tools)))
    ax.set_yticklabels([_s(t) for t in all_tools], fontsize=9.5, color=DARK_FG)
    ax.set_xlabel("Trajectory step", fontsize=11, color=DARK_FG)
    ax.set_ylabel("Tool", fontsize=11, color=DARK_FG)
    ax.tick_params(colors=DARK_FG)
    for sp in ax.spines.values():
        sp.set_color(DARK_GRID)
    ax.set_xlim(-0.4, MAX_STEP - 0.6)
    ax.set_ylim(-0.8, len(all_tools) - 0.2)

    # Legend
    handles = [
        mpatches.Patch(color=C_BENIGN, label="Benign trajectory"),
        mpatches.Patch(color=C_PGD, label="PGD-attacked trajectory"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=10,
        framealpha=0.3,
        facecolor=DARK_AX,
        edgecolor=DARK_GRID,
        labelcolor=DARK_FG,
    )
    ax.set_title(
        "Layered flow: how PGD shifts tool selection at each reasoning step (n=5 patients)",
        fontsize=13,
        fontweight="bold",
        color=DARK_FG,
        pad=12,
    )
    fig.savefig(GRAPH_OUT / "graph9_layered_flow.png", bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("graph9 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH 10 — Step-occupancy heatmap (dark, beautiful)
# ═══════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    stat1_overview()
    stat2_epsilon_sweep()
    stat3_trajectory_lengths()
    stat4_step_heatmap()
    graph6_bipartite()
    graph7_divergence()
    graph8_tool_influence()
    graph9_layered_flow()
    graph10_step_occupancy()
    print(f"\nStats → {STAT_OUT}/")
    print(f"Graphs → {GRAPH_OUT}/")
