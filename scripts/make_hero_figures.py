"""Visually-amazing hero figures for the adversarial-reasoning-attacks paper.

Dark editorial theme, neon-coral sunset palette, magazine-style typography.
All figures use only matplotlib + numpy primitives (no external deps).

Outputs (paper/figures/hero/):
  fig1_beeswarm.png         — full-bleed beeswarm with stat-card inset
  fig2_ridgeline.png        — joy-plot of edit-distance distributions per attack
  fig3_heatmap_attack_eps.png — attack × ε heatmap with annotated cells
  fig4_radial_profile.png   — circular radial bars per attack across 5 metrics
  fig5_bento.png            — 6-panel magazine composite (hero stat cards + charts)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from _plotlib import load_records
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import FancyBboxPatch, Rectangle

# ── Theme ──────────────────────────────────────────────────────────────────
BG = "#0B0F1A"
PANEL = "#141A29"
PANEL_LIGHT = "#1B2236"
TEXT = "#ECEFF4"
TEXT_MUTED = "#7A8499"
GRID = "#2A3147"
ACCENT = "#F2C94C"  # warm amber accent

PALETTE = {
    "noise": "#5C6B82",  # slate
    "pgd": "#B794F4",  # lavender
    "apgd": "#FC8181",  # coral
    "targeted_tool": "#4FD1C5",  # teal
    "trajectory_drift": "#F6AD55",  # warm orange
}
LABELS = {
    "noise": "UNIFORM NOISE",
    "pgd": "PGD-L∞",
    "apgd": "APGD-L∞",
    "targeted_tool": "TARGETED-TOOL",
    "trajectory_drift": "TRAJECTORY-DRIFT",
}
ATTACK_ORDER = ["noise", "pgd", "apgd", "targeted_tool", "trajectory_drift"]

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.weight": "400",
        "axes.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.grid": False,
        "axes.titlecolor": TEXT,
        "figure.facecolor": BG,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.facecolor": BG,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.35,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
        "legend.labelcolor": TEXT,
        "text.color": TEXT,
    }
)


def edits(recs):
    return np.array([r["edit_distance_norm"] for r in recs], dtype=np.float64)


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return tuple(np.quantile(boots, [alpha / 2, 1 - alpha / 2]))


def step1_flip_rate(recs):
    flips, total = 0, 0
    for r in recs:
        b = r.get("benign", {}).get("tool_sequence", []) or []
        a = r.get("attacked", {}).get("tool_sequence", []) or []
        if not b:
            continue
        total += 1
        if not a or a[0] != b[0]:
            flips += 1
    return flips / total if total else 0.0


def gather() -> dict[str, list[dict]]:
    root = Path(__file__).resolve().parents[1]
    return {
        "noise": load_records(root / "runs/main/noise/records.jsonl"),
        "pgd": load_records(root / "runs/main/pgd/records.jsonl"),
        "apgd": load_records(root / "runs/main/apgd/records.jsonl"),
        "targeted_tool": load_records(root / "runs/main/targeted_tool/records.jsonl"),
        "trajectory_drift": load_records(root / "runs/main/trajectory_drift/records.jsonl"),
    }


# ── Helpers ───────────────────────────────────────────────────────────────
def add_panel(fig_or_ax, x, y, w, h, *, fc=PANEL, ec=GRID, alpha=1.0, radius=0.012):
    """Draw a rounded panel as a FancyBboxPatch in figure or axes coords."""
    is_fig = hasattr(fig_or_ax, "patches") and not hasattr(fig_or_ax, "transData")
    is_fig = isinstance(fig_or_ax, plt.Figure)
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.001,rounding_size={radius}",
        linewidth=0.8,
        edgecolor=ec,
        facecolor=fc,
        alpha=alpha,
        transform=(fig_or_ax.transFigure if is_fig else fig_or_ax.transAxes),
        zorder=0,
    )
    fig_or_ax.patches.append(box) if is_fig else fig_or_ax.add_patch(box)


def beeswarm_y(values, max_width=0.32, sigma=0.018):
    """Compute y-jitter for a 1-D array, packing points without overlap.

    Simplified force-pack: bin values, jitter within each bin proportional
    to bin density.
    """
    n = len(values)
    if n == 0:
        return np.array([])
    bin_w = sigma
    bins = np.round(values / bin_w).astype(int)
    jitter = np.zeros(n)
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        k = len(idx)
        # Symmetric ladder
        offsets = (np.arange(k) - (k - 1) / 2) * (max_width / max(k, 1)) * 0.85
        rng = np.random.default_rng(int(b) + 13)
        rng.shuffle(offsets)
        jitter[idx] = offsets
    return jitter


def fmt_eps(e: float) -> str:
    return f"{round(e * 255)}/255"


# ───────────────────────────────────────────────────────────────────────
# Figure 1 — HERO beeswarm with stat-card inset
# ───────────────────────────────────────────────────────────────────────
def fig_beeswarm(by_attack, out_path: Path) -> None:
    fig = plt.figure(figsize=(15, 8.5))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.06, 0.17, 0.62, 0.72])
    ax.set_facecolor(BG)

    np.random.seed(0)
    for i, name in enumerate(ATTACK_ORDER):
        vals = edits(by_attack[name])
        if vals.size == 0:
            continue
        jitter = beeswarm_y(vals, max_width=0.34)
        color = PALETTE[name]
        ax.scatter(
            vals,
            np.full_like(vals, i, dtype=float) + jitter,
            s=85,
            color=color,
            alpha=0.78,
            edgecolor=BG,
            linewidth=0.9,
            zorder=3,
        )
        # Median line
        med = float(np.median(vals))
        mn = float(np.mean(vals))
        ax.plot([med, med], [i - 0.42, i + 0.42], color=TEXT, linewidth=1.5, zorder=4, alpha=0.85)
        # Mean glyph
        ax.scatter([mn], [i], marker="D", s=80, color=ACCENT, edgecolor=BG, linewidth=1.2, zorder=5)

    ax.set_yticks(range(len(ATTACK_ORDER)))
    ax.set_yticklabels(
        [LABELS[a] for a in ATTACK_ORDER], color=TEXT, fontsize=12, fontweight="bold"
    )
    ax.invert_yaxis()
    ax.set_xlabel("Normalised trajectory edit distance", color=TEXT_MUTED, fontsize=11, labelpad=10)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(len(ATTACK_ORDER) - 0.5, -0.5)
    for spine in ("bottom",):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(GRID)

    # ε ticks every 0.25
    ax.set_xticks(np.linspace(0, 1, 5))
    for v in np.linspace(0, 1, 5):
        ax.axvline(v, color=GRID, linewidth=0.5, alpha=0.5, zorder=1)

    # Title block
    fig.text(0.06, 0.945, "ATTACK LANDSCAPE", color=TEXT, fontsize=24, fontweight="bold")
    fig.text(
        0.06,
        0.918,
        "Trajectory edit distance per record · Qwen2.5-VL-7B medical agent · ProstateX val=5",
        color=TEXT_MUTED,
        fontsize=11,
    )

    # Stat-card column
    card_x, card_w = 0.71, 0.25
    card_h = 0.13
    card_gap = 0.018
    card_y0 = 0.82
    fig.text(card_x, 0.945, "ATTACK STATISTICS", color=TEXT, fontsize=13, fontweight="bold")
    fig.text(
        card_x,
        0.918,
        "n samples · mean ed [95% CI] · step-1 flip rate",
        color=TEXT_MUTED,
        fontsize=10,
    )

    for i, name in enumerate(ATTACK_ORDER):
        y = card_y0 - i * (card_h + card_gap)
        add_panel(fig, card_x, y - card_h, card_w, card_h, fc=PANEL, ec=GRID, radius=0.012)
        # Color stripe left
        stripe = Rectangle(
            (card_x, y - card_h),
            0.005,
            card_h,
            facecolor=PALETTE[name],
            edgecolor="none",
            transform=fig.transFigure,
        )
        fig.patches.append(stripe)

        recs = by_attack[name]
        eds = edits(recs)
        if eds.size == 0:
            fig.text(card_x + 0.018, y - card_h / 2, "no data", color=TEXT_MUTED, fontsize=10)
            continue
        lo, hi = bootstrap_ci(eds)
        flip = step1_flip_rate(recs)
        # Heading
        fig.text(
            card_x + 0.018,
            y - 0.022,
            LABELS[name],
            color=PALETTE[name],
            fontsize=11,
            fontweight="bold",
        )
        # Big mean number
        fig.text(
            card_x + 0.018,
            y - 0.062,
            f"μ = {eds.mean():.3f}",
            color=TEXT,
            fontsize=18,
            fontweight="bold",
            family="DejaVu Sans Mono",
        )
        # Detail
        fig.text(
            card_x + 0.018,
            y - 0.083,
            f"n={eds.size:>3d}   95% CI [{lo:.2f}, {hi:.2f}]",
            color=TEXT_MUTED,
            fontsize=9,
            family="DejaVu Sans Mono",
        )
        # Flip-rate mini-bar
        bar_x = card_x + 0.018
        bar_w = card_w - 0.036
        bar_y = y - card_h + 0.012
        # Track
        fig.patches.append(
            Rectangle(
                (bar_x, bar_y),
                bar_w,
                0.008,
                facecolor=PANEL_LIGHT,
                edgecolor="none",
                transform=fig.transFigure,
            )
        )
        # Fill
        fig.patches.append(
            Rectangle(
                (bar_x, bar_y),
                bar_w * flip,
                0.008,
                facecolor=PALETTE[name],
                edgecolor="none",
                transform=fig.transFigure,
            )
        )
        fig.text(
            card_x + 0.018, bar_y + 0.012, f"step-1 flip {flip:.0%}", color=TEXT_MUTED, fontsize=8.5
        )

    # Legend strip below main plot — placed below xlabel
    leg_y = 0.065
    fig.text(0.06, leg_y, "MEDIAN", color=TEXT_MUTED, fontsize=8.5)
    fig.add_artist(
        plt.Line2D(
            [0.105, 0.125],
            [leg_y + 0.005, leg_y + 0.005],
            color=TEXT,
            linewidth=1.5,
            transform=fig.transFigure,
        )
    )
    fig.text(0.135, leg_y, "MEAN", color=TEXT_MUTED, fontsize=8.5)
    fig.add_artist(
        plt.Line2D(
            [0.165],
            [leg_y + 0.005],
            marker="D",
            markersize=8,
            color=ACCENT,
            markeredgecolor=BG,
            markeredgewidth=1.0,
            transform=fig.transFigure,
        )
    )
    fig.text(
        0.185, leg_y, "EACH DOT = ONE RECORD (sample × ε × seed)", color=TEXT_MUTED, fontsize=8.5
    )

    # Footer
    fig.text(
        0.06,
        0.018,
        "adversarial-reasoning-attacks · 2026 · github.com/KosmasApostolidis/adversarial-reasoning-attacks",
        color=TEXT_MUTED,
        fontsize=8,
        alpha=0.6,
    )
    fig.text(
        0.96,
        0.018,
        "ε ∈ {2, 4, 8, 16}/255 · seeds {0, 1, 2}",
        color=TEXT_MUTED,
        fontsize=8,
        alpha=0.6,
        ha="right",
    )

    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# Figure 2 — Ridgeline (joy plot)
# ───────────────────────────────────────────────────────────────────────
def fig_ridgeline(by_attack, out_path: Path) -> None:
    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.10, 0.10, 0.85, 0.78])
    ax.set_facecolor(BG)

    xs = np.linspace(-0.05, 1.05, 400)
    spacing = 1.0
    for i, name in enumerate(ATTACK_ORDER):
        vals = edits(by_attack[name])
        if vals.size == 0:
            continue
        # Gaussian KDE manual (no scipy)
        bw = 0.06
        density = np.exp(-0.5 * ((xs[:, None] - vals[None, :]) / bw) ** 2).sum(axis=1)
        if density.max() > 0:
            density = density / density.max()
        y_base = (len(ATTACK_ORDER) - 1 - i) * spacing
        color = PALETTE[name]
        rgba_fill = to_rgba(color, alpha=0.55)
        rgba_edge = to_rgba(color, alpha=1.0)
        ax.fill_between(
            xs,
            y_base,
            y_base + density * 0.85,
            color=rgba_fill,
            edgecolor=rgba_edge,
            linewidth=2.0,
            zorder=3 - i * 0.05,
        )
        # Median + mean tick
        med = float(np.median(vals))
        mn = float(np.mean(vals))
        ax.plot([med, med], [y_base, y_base + 0.22], color=TEXT, linewidth=1.8, zorder=8)
        ax.scatter(
            [mn], [y_base], marker="D", s=70, color=ACCENT, edgecolor=BG, linewidth=1.0, zorder=9
        )
        # Label inside ridge
        ax.text(
            -0.05,
            y_base + 0.06,
            LABELS[name],
            color=color,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax.text(
            1.05,
            y_base + 0.06,
            f"n={vals.size}  μ={mn:.3f}",
            color=TEXT_MUTED,
            fontsize=10,
            ha="right",
            va="bottom",
            family="DejaVu Sans Mono",
        )

    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.4, len(ATTACK_ORDER) * spacing)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Normalised trajectory edit distance", color=TEXT_MUTED, fontsize=11)
    for x in np.linspace(0, 1, 6):
        ax.axvline(x, color=GRID, linewidth=0.4, alpha=0.5, zorder=1)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)

    fig.text(0.10, 0.945, "DISTRIBUTION SHAPES", color=TEXT, fontsize=24, fontweight="bold")
    fig.text(
        0.10,
        0.918,
        "Kernel density estimate of edit-distance per attack · vertical tick = median · diamond = mean",
        color=TEXT_MUTED,
        fontsize=11,
    )
    fig.text(
        0.10,
        0.04,
        "More mass near 1.0 = trajectory rewritten · more mass near 0 = agent unaffected",
        color=TEXT_MUTED,
        fontsize=9.5,
        alpha=0.85,
    )

    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# Figure 3 — Attack × ε heatmap
# ───────────────────────────────────────────────────────────────────────
def fig_heatmap(by_attack, out_path: Path) -> None:
    eps_vals = sorted({float(r["epsilon"]) for recs in by_attack.values() for r in recs})
    attacks_with_data = [a for a in ATTACK_ORDER if by_attack.get(a)]

    cell = np.full((len(attacks_with_data), len(eps_vals)), np.nan)
    counts = np.zeros_like(cell, dtype=int)
    for i, a in enumerate(attacks_with_data):
        groups = defaultdict(list)
        for r in by_attack[a]:
            groups[float(r["epsilon"])].append(r["edit_distance_norm"])
        for j, e in enumerate(eps_vals):
            if e in groups:
                cell[i, j] = float(np.mean(groups[e]))
                counts[i, j] = len(groups[e])

    cmap = LinearSegmentedColormap.from_list(
        "ed_dark", [PANEL, "#3A3F5C", "#7A4F8B", PALETTE["apgd"], "#FFD37A"]
    )
    fig = plt.figure(figsize=(12, 6.8))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.18, 0.18, 0.65, 0.66])
    ax.set_facecolor(BG)
    masked = np.ma.masked_invalid(cell)
    cmap.set_bad(PANEL_LIGHT)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=0.85, aspect="auto", origin="upper")

    for i in range(masked.shape[0]):
        for j in range(masked.shape[1]):
            if np.isnan(cell[i, j]):
                ax.text(
                    j,
                    i,
                    "n/a",
                    ha="center",
                    va="center",
                    color=TEXT_MUTED,
                    fontsize=10,
                    fontstyle="italic",
                )
            else:
                v = cell[i, j]
                color = "black" if v > 0.55 else TEXT
                ax.text(
                    j,
                    i - 0.10,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=14,
                    fontweight="bold",
                    family="DejaVu Sans Mono",
                )
                ax.text(
                    j,
                    i + 0.22,
                    f"n={counts[i, j]}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8.5,
                    alpha=0.85,
                )

    ax.set_xticks(range(len(eps_vals)))
    ax.set_xticklabels([fmt_eps(e) for e in eps_vals], color=TEXT_MUTED, fontsize=11)
    ax.set_yticks(range(len(attacks_with_data)))
    ax.set_yticklabels(
        [LABELS[a] for a in attacks_with_data], color=TEXT, fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Perturbation budget ε", color=TEXT_MUTED, fontsize=12)
    ax.tick_params(length=0)

    cbar_ax = fig.add_axes([0.86, 0.20, 0.018, 0.55])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.outline.set_edgecolor(GRID)
    cbar.outline.set_linewidth(0.7)
    cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    cbar.set_label("Mean edit distance", color=TEXT_MUTED, fontsize=10)

    fig.text(0.06, 0.93, "ATTACK × BUDGET HEATMAP", color=TEXT, fontsize=22, fontweight="bold")
    fig.text(
        0.06,
        0.895,
        "Mean normalised edit distance per (attack, ε) cell · brighter = more disruption",
        color=TEXT_MUTED,
        fontsize=11,
    )
    fig.text(
        0.06,
        0.05,
        "PGD evaluated only at smoke ε=8/255 (n=5); other attacks span full sweep (4 ε × 3 seeds × 5 samples = 60).",
        color=TEXT_MUTED,
        fontsize=9,
        alpha=0.85,
    )

    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# Figure 4 — Radial profile (circular bars)
# ───────────────────────────────────────────────────────────────────────
def fig_radial(by_attack, out_path: Path) -> None:
    metric_names = ["MEAN", "MAX", "FLIP", "P95", "Δ-LEN"]

    def metrics(name):
        recs = by_attack[name]
        eds = edits(recs)
        if eds.size == 0:
            return [0.0] * len(metric_names)
        bens = np.array([len(r.get("benign", {}).get("tool_sequence", []) or []) for r in recs])
        atts = np.array([len(r.get("attacked", {}).get("tool_sequence", []) or []) for r in recs])
        traj_d = np.abs(atts - bens).mean() if bens.size else 0.0
        return [
            float(eds.mean()),
            float(eds.max()),
            step1_flip_rate(recs),
            float(np.quantile(eds, 0.95)),
            float(traj_d),
        ]

    raw = {n: metrics(n) for n in ATTACK_ORDER}
    arr = np.array([raw[n] for n in ATTACK_ORDER])
    col_max = np.where(arr.max(axis=0) == 0, 1.0, arr.max(axis=0))
    norm = arr / col_max

    fig = plt.figure(figsize=(13, 13))
    fig.patch.set_facecolor(BG)
    cols = 3
    rows = 2
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False)

    for idx, name in enumerate(ATTACK_ORDER):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="polar")
        ax.set_facecolor(BG)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.15)

        # Background rings + radial value labels
        for r in [0.25, 0.5, 0.75, 1.0]:
            ax.plot(np.linspace(0, 2 * np.pi, 200), [r] * 200, color=GRID, linewidth=0.6, alpha=0.6)
            ax.text(
                np.pi / 2,
                r,
                f"{r:.2f}",
                color=TEXT_MUTED,
                fontsize=7,
                ha="center",
                va="center",
                alpha=0.55,
                zorder=2,
            )

        # Per-axis bars
        bar_w = 2 * np.pi / len(metric_names) * 0.72
        ax.bar(
            angles,
            norm[idx],
            width=bar_w,
            color=PALETTE[name],
            alpha=0.92,
            edgecolor=BG,
            linewidth=1.6,
            zorder=3,
        )
        # Annotate raw values just outside the bar tip
        for i_m, (ang, val_norm, val_raw) in enumerate(
            zip(angles, norm[idx], raw[name], strict=True)
        ):
            label = f"{val_raw:.0%}" if metric_names[i_m] == "FLIP" else f"{val_raw:.2f}"
            ax.text(
                ang,
                max(val_norm + 0.10, 0.18),
                label,
                ha="center",
                va="center",
                color=TEXT,
                fontsize=10,
                fontweight="bold",
                family="DejaVu Sans Mono",
                zorder=5,
            )

        ax.set_xticks(angles)
        ax.set_xticklabels(metric_names, color=TEXT, fontsize=10.5, fontweight="bold")
        ax.tick_params(axis="x", pad=14)
        ax.set_yticks([])
        ax.spines["polar"].set_color(GRID)
        ax.spines["polar"].set_linewidth(1.0)
        ax.set_title(LABELS[name], color=PALETTE[name], fontsize=14, fontweight="bold", pad=22)
        ax.set_ylim(0, 1.30)

    # Empty 6th panel → big legend explanation
    ax = fig.add_subplot(rows, cols, 6)
    ax.set_facecolor(PANEL)
    ax.axis("off")
    add_panel(ax, 0.02, 0.02, 0.96, 0.96, fc=PANEL, ec=GRID, radius=0.05)
    ax.text(0.06, 0.86, "READING THE RADAR", color=TEXT, fontsize=15, fontweight="bold")
    ax.text(0.06, 0.78, "Each spoke is one metric.", color=TEXT_MUTED, fontsize=10)
    ax.text(
        0.06,
        0.71,
        "Bar length = column-normalised score across\nall attacks (longer = stronger attack).",
        color=TEXT_MUTED,
        fontsize=10,
    )
    ax.text(0.06, 0.58, "METRICS", color=ACCENT, fontsize=11, fontweight="bold")
    metric_help = [
        ("MEAN", "average normalised edit distance"),
        ("MAX", "worst-case sample"),
        ("FLIP", "fraction with first tool flipped"),
        ("P95", "95th-percentile edit distance"),
        ("Δ-LEN", "abs change in trajectory length"),
    ]
    for i, (k, v) in enumerate(metric_help):
        ax.text(0.06, 0.49 - i * 0.075, k, color=TEXT, fontsize=9.5, fontweight="bold")
        ax.text(0.30, 0.49 - i * 0.075, v, color=TEXT_MUTED, fontsize=9.5)
    ax.text(
        0.06,
        0.06,
        "Numbers shown above each bar are raw values\n(not normalised).",
        color=TEXT_MUTED,
        fontsize=9,
        alpha=0.85,
        style="italic",
    )

    fig.text(0.5, 0.965, "ATTACK PROFILES", color=TEXT, fontsize=24, fontweight="bold", ha="center")
    fig.text(
        0.5,
        0.940,
        "Five-axis radar per attack · column-normalised across attacks",
        color=TEXT_MUTED,
        fontsize=11,
        ha="center",
    )

    fig.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.05, hspace=0.55, wspace=0.40)
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# Figure 5 — Bento composite (magazine layout)
# ───────────────────────────────────────────────────────────────────────
def fig_bento(by_attack, out_path: Path) -> None:
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor(BG)

    # Title bar
    fig.text(0.04, 0.955, "ADVERSARIAL ATTACK DOSSIER", color=TEXT, fontsize=26, fontweight="bold")
    fig.text(
        0.04,
        0.928,
        "Qwen2.5-VL-7B · ProstateX val=5 · ε ∈ {2,4,8,16}/255 · seeds {0,1,2}",
        color=TEXT_MUTED,
        fontsize=11,
    )

    # ── Cell A: Hero stat — strongest attack
    add_panel(fig, 0.04, 0.66, 0.24, 0.24, fc=PANEL, ec=GRID, radius=0.018)
    strongest = max(
        ATTACK_ORDER, key=lambda a: edits(by_attack[a]).mean() if edits(by_attack[a]).size else 0
    )
    s_eds = edits(by_attack[strongest])
    s_lo, s_hi = bootstrap_ci(s_eds)
    fig.text(0.06, 0.870, "STRONGEST ATTACK", color=TEXT_MUTED, fontsize=10, fontweight="bold")
    fig.text(
        0.06, 0.835, LABELS[strongest], color=PALETTE[strongest], fontsize=20, fontweight="bold"
    )
    fig.text(
        0.06,
        0.795,
        f"{s_eds.mean():.3f}",
        color=TEXT,
        fontsize=44,
        fontweight="bold",
        family="DejaVu Sans Mono",
        va="top",
    )
    fig.text(0.06, 0.720, "mean normalised edit distance", color=TEXT_MUTED, fontsize=9.5)
    fig.text(
        0.06,
        0.685,
        f"95% CI [{s_lo:.3f}, {s_hi:.3f}]   ·   n = {s_eds.size}",
        color=TEXT_MUTED,
        fontsize=9,
        family="DejaVu Sans Mono",
    )

    # ── Cell B: Hero stat — most-flipping attack
    add_panel(fig, 0.30, 0.66, 0.24, 0.24, fc=PANEL, ec=GRID, radius=0.018)
    flippers = max(ATTACK_ORDER, key=lambda a: step1_flip_rate(by_attack[a]))
    flip_v = step1_flip_rate(by_attack[flippers])
    fig.text(
        0.32, 0.870, "BIGGEST STEP-1 DISRUPTOR", color=TEXT_MUTED, fontsize=10, fontweight="bold"
    )
    fig.text(0.32, 0.835, LABELS[flippers], color=PALETTE[flippers], fontsize=20, fontweight="bold")
    fig.text(
        0.32,
        0.795,
        f"{flip_v:.0%}",
        color=TEXT,
        fontsize=44,
        fontweight="bold",
        family="DejaVu Sans Mono",
        va="top",
    )
    fig.text(
        0.32, 0.720, "of trajectories had the first tool flipped", color=TEXT_MUTED, fontsize=9.5
    )
    fig.text(
        0.32,
        0.685,
        f"n = {len(by_attack[flippers])} records  ·  ε ∈ [2,16]/255",
        color=TEXT_MUTED,
        fontsize=9,
        family="DejaVu Sans Mono",
    )

    # ── Cell C: Hero stat — total records
    add_panel(fig, 0.56, 0.66, 0.20, 0.24, fc=PANEL, ec=GRID, radius=0.018)
    total = sum(len(by_attack[a]) for a in ATTACK_ORDER)
    fig.text(0.58, 0.870, "TOTAL RECORDS", color=TEXT_MUTED, fontsize=10, fontweight="bold")
    fig.text(
        0.58,
        0.815,
        f"{total}",
        color=ACCENT,
        fontsize=54,
        fontweight="bold",
        family="DejaVu Sans Mono",
        va="top",
    )
    fig.text(
        0.58,
        0.720,
        f"across {sum(1 for a in ATTACK_ORDER if by_attack[a])} attack modes",
        color=TEXT_MUTED,
        fontsize=9.5,
    )
    fig.text(
        0.58,
        0.685,
        "5 ProstateX samples × 4 ε × 3 seeds",
        color=TEXT_MUTED,
        fontsize=9,
        style="italic",
    )

    # ── Cell D: Compute time stat
    add_panel(fig, 0.78, 0.66, 0.18, 0.24, fc=PANEL, ec=GRID, radius=0.018)
    fig.text(0.80, 0.870, "TOTAL GPU TIME", color=TEXT_MUTED, fontsize=10, fontweight="bold")
    fig.text(
        0.80,
        0.815,
        "46:00",
        color=PALETTE["targeted_tool"],
        fontsize=54,
        fontweight="bold",
        family="DejaVu Sans Mono",
        va="top",
    )
    fig.text(0.80, 0.720, "minutes : seconds (sequential)", color=TEXT_MUTED, fontsize=9.5)
    fig.text(
        0.80, 0.685, "single H200 NVL · 3 sweeps", color=TEXT_MUTED, fontsize=9, style="italic"
    )

    # ── Cell E: Mean ed bar chart (left big cell)
    ax = fig.add_axes([0.04, 0.08, 0.42, 0.50])
    ax.set_facecolor(BG)
    means = [edits(by_attack[a]).mean() if edits(by_attack[a]).size else 0.0 for a in ATTACK_ORDER]
    cis = [bootstrap_ci(edits(by_attack[a])) for a in ATTACK_ORDER]
    err_lo = np.array([m - lo for m, (lo, _) in zip(means, cis, strict=True)])
    err_hi = np.array([hi - m for m, (_, hi) in zip(means, cis, strict=True)])
    ypos = np.arange(len(ATTACK_ORDER))
    ax.barh(
        ypos,
        means,
        color=[PALETTE[a] for a in ATTACK_ORDER],
        alpha=0.92,
        edgecolor=BG,
        linewidth=1.0,
        height=0.62,
    )
    ax.errorbar(
        means,
        ypos,
        xerr=[err_lo, err_hi],
        fmt="none",
        color=TEXT,
        capsize=5,
        linewidth=1.3,
        alpha=0.85,
    )
    for i, (m, n) in enumerate(
        zip(means, [edits(by_attack[a]).size for a in ATTACK_ORDER], strict=True)
    ):
        ax.text(
            m + 0.02,
            i,
            f"{m:.3f}  ",
            color=TEXT,
            fontsize=11,
            va="center",
            fontweight="bold",
            family="DejaVu Sans Mono",
        )
        ax.text(
            m + 0.02,
            i + 0.35,
            f"n={n}",
            color=TEXT_MUTED,
            fontsize=8.5,
            va="center",
            family="DejaVu Sans Mono",
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [LABELS[a] for a in ATTACK_ORDER], color=TEXT, fontsize=10.5, fontweight="bold"
    )
    ax.set_xticks(np.linspace(0, 0.8, 5))
    for v in np.linspace(0, 0.8, 5):
        ax.axvline(v, color=GRID, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_xlim(0, max(means) * 1.35)
    ax.set_xlabel("Mean ± 95% CI", color=TEXT_MUTED, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(
        "MEAN EDIT DISTANCE PER ATTACK",
        color=TEXT,
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=12,
    )

    # ── Cell F: ε-sweep mini line plot
    ax = fig.add_axes([0.50, 0.30, 0.46, 0.28])
    ax.set_facecolor(BG)
    eps_vals = sorted({float(r["epsilon"]) for recs in by_attack.values() for r in recs})
    for name in ATTACK_ORDER:
        groups = defaultdict(list)
        for r in by_attack[name]:
            groups[float(r["epsilon"])].append(r["edit_distance_norm"])
        xs = sorted(groups)
        if not xs:
            continue
        ys_mean = np.array([np.mean(groups[e]) for e in xs])
        ci = np.array([bootstrap_ci(np.asarray(groups[e])) for e in xs])
        if len(xs) == 1:
            ax.scatter(
                xs,
                ys_mean,
                marker="*",
                s=240,
                color=PALETTE[name],
                edgecolor=BG,
                linewidth=1.2,
                label=f"{LABELS[name]}*",
                zorder=4,
            )
        else:
            ax.plot(
                xs,
                ys_mean,
                marker="o",
                color=PALETTE[name],
                linewidth=2.5,
                markersize=8,
                markeredgecolor=BG,
                markeredgewidth=0.8,
                label=LABELS[name],
                zorder=3,
            )
            ax.fill_between(xs, ci[:, 0], ci[:, 1], color=PALETTE[name], alpha=0.18, zorder=2)
    for e in eps_vals:
        ax.axvline(e, color=GRID, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_xticks(eps_vals)
    ax.set_xticklabels([fmt_eps(e) for e in eps_vals], color=TEXT_MUTED)
    ax.set_xlabel("Perturbation budget ε", color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel("Mean ed", color=TEXT_MUTED, fontsize=10)
    ax.set_title(
        "EFFECTIVENESS VS ε", color=TEXT, fontsize=12, fontweight="bold", loc="left", pad=12
    )
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    # ── Cell G: Tool-flip dial — mini bar chart
    ax = fig.add_axes([0.50, 0.08, 0.46, 0.16])
    ax.set_facecolor(BG)
    flips = [step1_flip_rate(by_attack[a]) for a in ATTACK_ORDER]
    ypos = np.arange(len(ATTACK_ORDER))
    ax.barh(
        ypos,
        flips,
        color=[PALETTE[a] for a in ATTACK_ORDER],
        alpha=0.92,
        edgecolor=BG,
        linewidth=1.0,
        height=0.62,
    )
    for i, f in enumerate(flips):
        ax.text(
            f + 0.012,
            i,
            f"{f:.0%}",
            color=TEXT,
            fontsize=10,
            va="center",
            fontweight="bold",
            family="DejaVu Sans Mono",
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels([LABELS[a] for a in ATTACK_ORDER], color=TEXT, fontsize=9.5)
    ax.set_xlim(0, 1.05)
    ax.set_xticks(np.linspace(0, 1, 6))
    for v in np.linspace(0, 1, 6):
        ax.axvline(v, color=GRID, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_xlabel("Step-1 flip rate", color=TEXT_MUTED, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(
        "STEP-1 TOOL-FLIP RATE", color=TEXT, fontsize=12, fontweight="bold", loc="left", pad=10
    )

    # Footer
    fig.text(
        0.04,
        0.025,
        "github.com/KosmasApostolidis/adversarial-reasoning-attacks   · 2026",
        color=TEXT_MUTED,
        fontsize=8.5,
        alpha=0.7,
    )
    fig.text(
        0.96,
        0.025,
        "* PGD anchor: smoke ε=8/255 only (n=5)",
        color=TEXT_MUTED,
        fontsize=8.5,
        alpha=0.7,
        ha="right",
    )

    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
def main() -> int:
    by_attack = gather()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "paper" / "figures" / "hero"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_beeswarm(by_attack, out_dir / "fig1_beeswarm.png")
    fig_ridgeline(by_attack, out_dir / "fig2_ridgeline.png")
    fig_heatmap(by_attack, out_dir / "fig3_heatmap_attack_eps.png")
    fig_radial(by_attack, out_dir / "fig4_radial_profile.png")
    fig_bento(by_attack, out_dir / "fig5_bento.png")

    print(f"[make_hero_figures] wrote 5 figures → {out_dir}")
    for name in ATTACK_ORDER:
        eds = edits(by_attack[name])
        print(
            f"  {name:18s} n={eds.size:3d}  μ={eds.mean():.3f}  flip={step1_flip_rate(by_attack[name]):.0%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
