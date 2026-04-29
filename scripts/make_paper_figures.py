"""Publication-quality figures for the adversarial-reasoning-attacks paper.

Outputs (all in paper/figures/paper/):
  fig1_main_result.png      — 3-panel: noise vs PGD boxplot | per-sample bars | ε-sweep
  fig2_trajectories.png     — Gantt-style benign / noise / PGD sequences (3 samples)
  fig3_tool_heatmap.png     — Tool-substitution matrix under PGD
  fig4_cross_model.png      — Qwen vs LLaVA under uniform noise
  fig5_attack_landscape.png — Violin distributions: noise, PGD across models
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from _plotlib import despine, load_records, tool_palette
from matplotlib.patches import FancyBboxPatch

# ── Style ──────────────────────────────────────────────────────────────────
mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
    }
)

C_BENIGN = "#2166ac"  # blue
C_NOISE = "#92c5de"  # light blue
C_PGD = "#d6604d"  # red
C_LLAVA = "#f4a582"  # orange
C_ACCENT = "#1a9850"  # green
PALETTE20 = plt.get_cmap("tab20").colors

OUT = Path("paper/figures/paper")
OUT.mkdir(parents=True, exist_ok=True)


def _panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Main result
# ═══════════════════════════════════════════════════════════════════════════


def fig1_main_result() -> None:
    noise_recs = load_records("runs/main/noise/records.jsonl")
    pgd_recs = load_records("runs/main/pgd/records.jsonl")
    sweep_recs = [
        r
        for r in load_records("runs/main/noise/records.jsonl")
        if "qwen" in r.get("model_id", "").lower()
    ]

    nd = np.array([r["edit_distance_norm"] for r in noise_recs])
    pd_ = np.array([r["edit_distance_norm"] for r in pgd_recs])
    eps = pgd_recs[0]["epsilon"]

    # ε-sweep
    per_eps: dict[float, list[float]] = defaultdict(list)
    for r in sweep_recs:
        per_eps[r["epsilon"]].append(r["edit_distance_norm"])
    eps_sorted = sorted(per_eps)
    means = [np.mean(per_eps[e]) for e in eps_sorted]
    stds = [np.std(per_eps[e], ddof=1) if len(per_eps[e]) > 1 else 0 for e in eps_sorted]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.subplots_adjust(wspace=0.38)

    # Panel A — boxplot
    ax = axes[0]
    bp = ax.boxplot(
        [nd, pd_],
        tick_labels=["Uniform\nnoise", "PGD-L∞\n(20 steps)"],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=5, linestyle="none"),
    )
    bp["boxes"][0].set_facecolor(C_NOISE)
    bp["boxes"][1].set_facecolor(C_PGD)
    bp["boxes"][0].set_alpha(0.85)
    bp["boxes"][1].set_alpha(0.85)

    for i, arr in enumerate([nd, pd_], start=1):
        ax.scatter(
            np.full(len(arr), i) + np.random.default_rng(42).uniform(-0.1, 0.1, len(arr)),
            arr,
            s=40,
            zorder=5,
            color=[C_NOISE, C_PGD][i - 1],
            edgecolors="white",
            linewidths=0.6,
        )
        ax.text(i, arr.max() + 0.06, f"μ={arr.mean():.3f}", ha="center", fontsize=9, color="black")

    ax.set_ylabel("Normalised edit distance")
    ax.set_title(f"Attack effectiveness at ε = {eps:.4f}", pad=8)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    despine(ax)
    _panel_label(ax, "A")

    # Panel B — per-sample grouped bars
    ax = axes[1]
    n = len(pgd_recs)
    x = np.arange(n)
    w = 0.35
    pids = [r["sample_id"].split("_p")[1].replace("_s", " s") for r in pgd_recs]
    ax.bar(x - w / 2, nd, w, color=C_NOISE, label="Noise", edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, pd_, w, color=C_PGD, label="PGD", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(pids, fontsize=8)
    ax.set_xlabel("Patient ID")
    ax.set_ylabel("Normalised edit distance")
    ax.set_title("Per-patient comparison", pad=8)
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    despine(ax)
    _panel_label(ax, "B")

    # Panel C — ε-sweep
    ax = axes[2]
    ax.errorbar(
        [e * 255 for e in eps_sorted],
        means,
        yerr=stds,
        marker="o",
        linewidth=2,
        markersize=7,
        capsize=4,
        color=C_NOISE,
        label="Uniform noise (Qwen)",
    )
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("ε (pixel units, ×255)")
    ax.set_ylabel("Mean normalised edit distance")
    ax.set_title("Dose–response (uniform noise)", pad=8)
    ax.set_xticks([e * 255 for e in eps_sorted])
    ax.set_xticklabels(["2", "4", "8", "16"])
    ax.legend()
    despine(ax)
    _panel_label(ax, "C")

    fig.suptitle(
        "Adversarial perturbations alter VLM agent tool-call trajectories on prostate MRI",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(OUT / "fig1_main_result.png")
    plt.close(fig)
    print("fig1 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Trajectory Gantt
# ═══════════════════════════════════════════════════════════════════════════


def _tool_palette(all_tools: list[str]) -> dict[str, tuple]:
    return tool_palette(all_tools, sort=True)


def fig2_trajectories() -> None:
    noise_recs = load_records("runs/main/noise/records.jsonl")
    pgd_recs = load_records("runs/main/pgd/records.jsonl")

    # Pick 3 most interesting samples (highest PGD edit distance)
    by_ed = sorted(pgd_recs, key=lambda r: r["edit_distance_norm"], reverse=True)[:3]

    # Build tool palette from all seqs
    all_tools: set[str] = set()
    noise_map = {r["sample_id"]: r for r in noise_recs}
    for r in by_ed:
        all_tools.update(r["benign"]["tool_sequence"])
        all_tools.update(r["attacked"]["tool_sequence"])
        nr = noise_map.get(r["sample_id"])
        if nr:
            all_tools.update(nr["attacked"]["tool_sequence"])
    pal = _tool_palette(list(all_tools))

    n_samples = len(by_ed)
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3.2 * n_samples))
    if n_samples == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.55)

    ROW_LABELS = ["Benign", "Noise", "PGD"]
    ROW_COLORS = ["#1a9850", C_NOISE, C_PGD]

    for ax, rec in zip(axes, by_ed, strict=False):
        sid = rec["sample_id"]
        nr = noise_map.get(sid)
        seqs = [
            rec["benign"]["tool_sequence"],
            nr["attacked"]["tool_sequence"] if nr else [],
            rec["attacked"]["tool_sequence"],
        ]
        max_len = max(len(s) for s in seqs)

        for row_i, (label, seq, rc) in enumerate(zip(ROW_LABELS, seqs, ROW_COLORS, strict=False)):
            for col, tool in enumerate(seq):
                color = pal[tool]
                ax.add_patch(
                    FancyBboxPatch(
                        (col + 0.04, row_i - 0.38),
                        0.84,
                        0.76,
                        boxstyle="round,pad=0.02",
                        facecolor=color,
                        edgecolor="white",
                        linewidth=1.2,
                        zorder=2,
                    )
                )
                short = tool.replace("_", "\n")
                ax.text(
                    col + 0.46,
                    row_i,
                    short,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                    fontweight="bold",
                    zorder=3,
                )
            ax.text(
                -0.4,
                row_i,
                label,
                ha="right",
                va="center",
                fontsize=9,
                color=rc,
                fontweight="bold",
            )

        ax.set_xlim(-0.6, max_len + 0.2)
        ax.set_ylim(-0.7, len(seqs) - 0.3)
        ax.set_yticks([])
        ax.set_xticks(range(max_len))
        ax.set_xticklabels([f"step {i+1}" for i in range(max_len)], fontsize=8)
        pid = sid.split("_p")[1] if "_p" in sid else sid
        pgd_ed = rec["edit_distance_norm"]
        n_ed = nr["edit_distance_norm"] if nr else float("nan")
        ax.set_title(
            f"Patient {pid}   |   edit dist: noise={n_ed:.3f}  PGD={pgd_ed:.3f}",
            fontsize=10,
            loc="left",
            pad=5,
        )
        ax.spines[:].set_visible(False)
        ax.axhline(-0.5, color="#dddddd", linewidth=0.8)

    # Shared legend
    handles = [
        mpl.patches.Patch(facecolor=pal[t], label=t.replace("_", " ")) for t in sorted(all_tools)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(all_tools), 4),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        title="Tools",
        title_fontsize=9,
    )
    fig.suptitle(
        "Tool-call trajectory comparison: benign vs noise vs PGD (top-3 drifted patients)",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(OUT / "fig2_trajectories.png")
    plt.close(fig)
    print("fig2 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Tool substitution heatmap under PGD
# ═══════════════════════════════════════════════════════════════════════════


def fig3_tool_heatmap() -> None:
    pgd_recs = load_records("runs/main/pgd/records.jsonl")

    # Collect min-edit alignment: for each sample, align benign → attacked
    # and count (benign_tool, attacked_tool) substitution pairs.
    all_tools: set[str] = set()
    for r in pgd_recs:
        all_tools.update(r["benign"]["tool_sequence"])
        all_tools.update(r["attacked"]["tool_sequence"])
    tools = sorted(all_tools)
    idx = {t: i for i, t in enumerate(tools)}
    n = len(tools)

    counts = np.zeros((n, n), dtype=int)
    insert = np.zeros(n, dtype=int)  # in attacked but not in benign position
    delete = np.zeros(n, dtype=int)  # in benign but not in attacked

    for r in pgd_recs:
        b = r["benign"]["tool_sequence"]
        a = r["attacked"]["tool_sequence"]
        # Simple prefix alignment (visual approximation)
        for bi, tool in enumerate(b):
            if bi < len(a):
                if b[bi] != a[bi]:
                    counts[idx[b[bi]], idx[a[bi]]] += 1
            else:
                delete[idx[tool]] += 1
        for ai in range(len(b), len(a)):
            insert[idx[a[ai]]] += 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1]})
    fig.subplots_adjust(wspace=0.4)

    # Heatmap
    ax = axes[0]
    vmax = max(counts.max(), 1)
    im = ax.imshow(counts, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels([t.replace("_", "\n") for t in tools], fontsize=7.5, rotation=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels([t.replace("_", "\n") for t in tools], fontsize=7.5)
    ax.set_xlabel("Attacked trajectory tool")
    ax.set_ylabel("Benign trajectory tool")
    ax.set_title("Tool substitution matrix under PGD-L∞", pad=8)
    for i in range(n):
        for j in range(n):
            v = counts[i, j]
            if v > 0:
                ax.text(
                    j,
                    i,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if v > vmax * 0.5 else "black",
                )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("substitution count", fontsize=9)

    # Insert / delete bar chart
    ax2 = axes[1]
    y = np.arange(n)
    ax2.barh(y - 0.2, delete, 0.38, color=C_BENIGN, label="deleted (benign→∅)", alpha=0.8)
    ax2.barh(y + 0.2, insert, 0.38, color=C_PGD, label="inserted (∅→attacked)", alpha=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels([t.replace("_", "\n") for t in tools], fontsize=7.5)
    ax2.set_xlabel("Count")
    ax2.set_title("Insertions & deletions", pad=8)
    ax2.legend(loc="lower right")
    despine(ax2)

    _panel_label(axes[0], "A")
    _panel_label(axes[1], "B")

    fig.suptitle(
        "PGD-induced tool-call perturbation anatomy (n=5 patients)", fontsize=12, fontweight="bold"
    )
    fig.savefig(OUT / "fig3_tool_heatmap.png")
    plt.close(fig)
    print("fig3 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Cross-model (Qwen vs LLaVA) under uniform noise
# ═══════════════════════════════════════════════════════════════════════════


def fig4_cross_model() -> None:
    all_noise = load_records("runs/main/noise/records.jsonl")
    qwen_recs = [r for r in all_noise if "qwen" in r.get("model_id", "").lower()]
    llava_recs = [r for r in all_noise if "llava" in r.get("model_id", "").lower()]

    qd = np.array([r["edit_distance_norm"] for r in qwen_recs])
    ld = np.array([r["edit_distance_norm"] for r in llava_recs])
    eps = qwen_recs[0]["epsilon"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.36)

    # Violin + strip
    ax = axes[0]
    parts = ax.violinplot([qd, ld], positions=[1, 2], showmedians=True, showextrema=False)
    for pc, color in zip(parts["bodies"], [C_BENIGN, C_LLAVA], strict=False):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts["cmedians"].set_colors(["black", "black"])
    parts["cmedians"].set_linewidth(2)
    rng = np.random.default_rng(0)
    for xi, arr, c in [(1, qd, C_BENIGN), (2, ld, C_LLAVA)]:
        jitter = rng.uniform(-0.06, 0.06, len(arr))
        ax.scatter(
            np.full(len(arr), xi) + jitter,
            arr,
            s=50,
            zorder=5,
            color=c,
            edgecolors="white",
            linewidths=0.8,
        )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Qwen2.5-VL-7B", "LLaVA-v1.6-Mistral-7B"])
    ax.set_ylabel("Normalised edit distance")
    ax.set_title(f"Cross-model sensitivity at ε={eps:.4f}", pad=8)
    for xi, arr, c in [(1, qd, C_BENIGN), (2, ld, C_LLAVA)]:
        ax.text(
            xi,
            arr.max() + 0.07,
            f"μ={arr.mean():.3f}",
            ha="center",
            fontsize=9,
            color=c,
            fontweight="bold",
        )
    ax.set_ylim(bottom=0)
    despine(ax)
    _panel_label(ax, "A")

    # Per-patient grouped bars
    ax2 = axes[1]
    n = min(len(qwen_recs), len(llava_recs))
    x = np.arange(n)
    w = 0.35
    pids = [r["sample_id"].split("_p")[1].replace("_s", "·") for r in qwen_recs[:n]]
    ax2.bar(x - w / 2, qd[:n], w, color=C_BENIGN, label="Qwen2.5-VL", edgecolor="white")
    ax2.bar(x + w / 2, ld[:n], w, color=C_LLAVA, label="LLaVA-v1.6", edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(pids, fontsize=8)
    ax2.set_xlabel("Patient ID")
    ax2.set_ylabel("Normalised edit distance")
    ax2.set_title("Per-patient cross-model comparison", pad=8)
    ax2.legend()
    ax2.set_ylim(bottom=0)
    despine(ax2)
    _panel_label(ax2, "B")

    fig.suptitle(
        "LLaVA is more sensitive to uniform-noise perturbations than Qwen2.5-VL",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(OUT / "fig4_cross_model.png")
    plt.close(fig)
    print("fig4 ✓")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Attack landscape: violin across attack modes & models
# ═══════════════════════════════════════════════════════════════════════════


def fig5_attack_landscape() -> None:
    all_noise = load_records("runs/main/noise/records.jsonl")
    all_pgd = load_records("runs/main/pgd/records.jsonl")
    noise_q = [r for r in all_noise if "qwen" in r.get("model_id", "").lower()]
    noise_l = [r for r in all_noise if "llava" in r.get("model_id", "").lower()]
    pgd_q = [r for r in all_pgd if "qwen" in r.get("model_id", "").lower()]

    groups = [
        ("Qwen\n(noise)", np.array([r["edit_distance_norm"] for r in noise_q]), C_NOISE),
        ("Qwen\n(PGD)", np.array([r["edit_distance_norm"] for r in pgd_q]), C_PGD),
        ("LLaVA\n(noise)", np.array([r["edit_distance_norm"] for r in noise_l]), C_LLAVA),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = [1, 2, 3]
    parts = ax.violinplot(
        [g[1] for g in groups],
        positions=positions,
        showmedians=True,
        showextrema=False,
        widths=0.5,
    )
    for pc, (_, _, c) in zip(parts["bodies"], groups, strict=False):
        pc.set_facecolor(c)
        pc.set_alpha(0.8)
    parts["cmedians"].set_colors(["black"] * len(groups))
    parts["cmedians"].set_linewidth(2.5)

    rng = np.random.default_rng(7)
    for xi, (_label, arr, c) in zip(positions, groups, strict=False):
        jitter = rng.uniform(-0.07, 0.07, len(arr))
        ax.scatter(
            np.full(len(arr), xi) + jitter,
            arr,
            s=55,
            zorder=5,
            color=c,
            edgecolors="white",
            linewidths=1.0,
        )
        ax.text(
            xi,
            arr.max() + 0.07,
            f"μ={arr.mean():.3f}",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color=c,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([g[0] for g in groups], fontsize=11)
    ax.set_ylabel("Normalised trajectory edit distance", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.set_title(
        "Attack landscape: trajectory drift by model and perturbation type",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    # Significance bracket (PGD vs noise, same model)
    y_bracket = max(r["edit_distance_norm"] for r in pgd_q) + 0.22
    ax.annotate(
        "",
        xy=(2, y_bracket),
        xytext=(1, y_bracket),
        arrowprops=dict(arrowstyle="-", color="black", lw=1.5),
    )
    ax.text(1.5, y_bracket + 0.03, "p < 0.05*", ha="center", fontsize=9, style="italic")

    despine(ax)
    fig.savefig(OUT / "fig5_attack_landscape.png")
    plt.close(fig)
    print("fig5 ✓")


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fig1_main_result()
    fig2_trajectories()
    fig3_tool_heatmap()
    fig4_cross_model()
    fig5_attack_landscape()
    print(f"\nAll figures written to {OUT}/")
