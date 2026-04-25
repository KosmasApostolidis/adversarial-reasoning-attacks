"""Publication-quality landscape figures across the full attack matrix.

Outputs (paper/figures/attack_landscape/):
  fig1_landscape_overview.png   — 2×2 composite (box + bar + ε-curve + flip rate)
  fig2_eps_curves_ci.png        — per-attack ε vs edit-distance with 95% bootstrap CI bands
  fig3_attack_radar.png         — radar chart of 4 attacks across 5 metrics
  fig4_tool_substitution.png    — heatmap matrix of benign→attacked tool flips, one panel per attack
  fig5_violin_grid.png          — violin distributions per (attack, ε) cell

Aggregates:
  noise=runs/smoke + runs/smoke_sweep
  pgd=runs/pgd_smoke
  apgd=runs/apgd_sweep + runs/apgd_smoke
  targeted_tool=runs/targeted_tool_sweep + runs/targeted_tool_smoke
  trajectory_drift=runs/trajectory_drift_sweep + runs/trajectory_drift_smoke
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


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

# Curated palette — distinct, perceptually ordered (cool→warm = weak→strong)
PALETTE = {
    "noise":            "#9E9E9E",   # neutral grey
    "pgd":              "#5E35B1",   # deep purple
    "apgd":             "#C62828",   # red
    "targeted_tool":    "#1565C0",   # blue
    "trajectory_drift": "#EF6C00",   # orange
}
LABELS = {
    "noise":            "Uniform noise",
    "pgd":              "PGD-L∞",
    "apgd":             "APGD-L∞",
    "targeted_tool":    "Targeted-Tool",
    "trajectory_drift": "Trajectory-Drift",
}
ATTACK_ORDER = ["noise", "pgd", "apgd", "targeted_tool", "trajectory_drift"]


from _plotlib import load_records  # noqa: E402


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def edits(recs: Iterable[dict]) -> np.ndarray:
    return np.array([r["edit_distance_norm"] for r in recs], dtype=np.float64)


def flip_rate(recs: list[dict]) -> float:
    """Fraction of samples whose first attacked tool differs from first benign tool."""
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


def loss_finals(recs: list[dict], key_suffix: str) -> np.ndarray:
    out: list[float] = []
    for r in recs:
        meta = r.get("attacked", {}).get("metadata", {}) or {}
        v = meta.get(f"{key_suffix}_loss_final")
        if v is not None:
            out.append(float(v))
    return np.array(out, dtype=np.float64)


# ───────────────────────────────────────────────────────────────────────
# Figure 1: 2×2 landscape overview composite
# ───────────────────────────────────────────────────────────────────────
def fig_landscape_overview(by_attack: dict[str, list[dict]], out_path: Path) -> None:
    fig = plt.figure(figsize=(13.5, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.18)
    ax_box = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_eps = fig.add_subplot(gs[1, 0])
    ax_flip = fig.add_subplot(gs[1, 1])

    # ── Top-left: violin+strip overlay
    data = [edits(by_attack[a]) for a in ATTACK_ORDER]
    parts = ax_box.violinplot(data, positions=range(len(ATTACK_ORDER)), widths=0.7, showmeans=False, showmedians=False, showextrema=False)
    for body, name in zip(parts["bodies"], ATTACK_ORDER, strict=True):
        body.set_facecolor(PALETTE[name])
        body.set_edgecolor(PALETTE[name])
        body.set_alpha(0.45)
    for i, (name, arr) in enumerate(zip(ATTACK_ORDER, data, strict=True)):
        if arr.size == 0:
            continue
        jitter = (np.random.default_rng(7).uniform(-0.08, 0.08, arr.size))
        ax_box.scatter(np.full_like(arr, i, dtype=float) + jitter, arr,
                       s=14, color=PALETTE[name], alpha=0.85, edgecolor="white", linewidths=0.4, zorder=3)
        # Median tick
        med = float(np.median(arr))
        ax_box.hlines(med, i - 0.22, i + 0.22, color="black", linewidth=1.6, zorder=4)
    ax_box.set_xticks(range(len(ATTACK_ORDER)))
    ax_box.set_xticklabels([LABELS[a] for a in ATTACK_ORDER], rotation=12, ha="right")
    ax_box.set_ylabel("Normalised trajectory edit distance")
    ax_box.set_title("(a)  Distribution per attack", loc="left", fontweight="bold", fontsize=12)
    ax_box.set_ylim(-0.05, 1.05)
    ax_box.axhline(0, color="#cccccc", linewidth=0.6)

    # ── Top-right: bar with bootstrap CI
    means = [edits(by_attack[a]).mean() if edits(by_attack[a]).size else 0.0 for a in ATTACK_ORDER]
    cis = [bootstrap_ci(edits(by_attack[a])) for a in ATTACK_ORDER]
    err_lo = np.array([m - lo for m, (lo, _) in zip(means, cis)])
    err_hi = np.array([hi - m for m, (_, hi) in zip(means, cis)])
    bars = ax_bar.bar(
        range(len(ATTACK_ORDER)), means,
        color=[PALETTE[a] for a in ATTACK_ORDER], alpha=0.92,
        yerr=[err_lo, err_hi], capsize=6, error_kw={"linewidth": 1.3, "ecolor": "#333333"},
        edgecolor="white", linewidth=1.0,
    )
    ax_bar.set_xticks(range(len(ATTACK_ORDER)))
    ax_bar.set_xticklabels([LABELS[a] for a in ATTACK_ORDER], rotation=12, ha="right")
    ax_bar.set_ylabel("Mean ± 95% bootstrap CI")
    ax_bar.set_title("(b)  Mean attack effectiveness", loc="left", fontweight="bold", fontsize=12)
    for b, m, n in zip(bars, means, [edits(by_attack[a]).size for a in ATTACK_ORDER], strict=True):
        ax_bar.text(b.get_x() + b.get_width() / 2, m + 0.025,
                    f"{m:.3f}\n(n={n})", ha="center", va="bottom", fontsize=8.5, color="#222222")
    ax_bar.set_ylim(0, max(means + [hi for _, hi in cis]) * 1.20 + 0.05)

    # ── Bottom-left: ε vs mean edit-distance with bootstrap CI bands
    eps_vals = sorted({float(r["epsilon"]) for recs in by_attack.values() for r in recs})
    for name in ATTACK_ORDER:
        recs = by_attack[name]
        groups = defaultdict(list)
        for r in recs:
            groups[float(r["epsilon"])].append(r["edit_distance_norm"])
        if len(groups) < 2 and name != "pgd":
            # Skip lines with only 1 ε unless explicitly noting (PGD anchor below)
            pass
        xs = sorted(groups)
        if not xs:
            continue
        ys_mean, ys_lo, ys_hi = [], [], []
        for e in xs:
            arr = np.asarray(groups[e])
            ys_mean.append(arr.mean())
            lo, hi = bootstrap_ci(arr)
            ys_lo.append(lo); ys_hi.append(hi)
        if len(xs) == 1:
            ax_eps.errorbar(xs, ys_mean,
                            yerr=[[ys_mean[0] - ys_lo[0]], [ys_hi[0] - ys_mean[0]]],
                            fmt="*", color=PALETTE[name], markersize=15,
                            capsize=5, label=f"{LABELS[name]} (smoke only)")
        else:
            ax_eps.plot(xs, ys_mean, marker="o", color=PALETTE[name], linewidth=2.2,
                        markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                        label=LABELS[name])
            ax_eps.fill_between(xs, ys_lo, ys_hi, color=PALETTE[name], alpha=0.18)
    ax_eps.set_xlabel("ε (normalised pixel domain)")
    ax_eps.set_ylabel("Mean edit distance ± 95% CI")
    ax_eps.set_title("(c)  Effectiveness vs perturbation budget", loc="left", fontweight="bold", fontsize=12)
    ax_eps.set_xticks(eps_vals)
    ax_eps.set_xticklabels([f"{e:.4g}\n({int(round(e * 255))}/255)" for e in eps_vals], fontsize=8.5)
    ax_eps.grid(linestyle=":", alpha=0.35)
    ax_eps.legend(loc="upper left", framealpha=0.9, frameon=True, edgecolor="#dddddd")

    # ── Bottom-right: first-step flip rate per attack
    flips = [flip_rate(by_attack[a]) for a in ATTACK_ORDER]
    bars = ax_flip.barh(
        range(len(ATTACK_ORDER)), flips,
        color=[PALETTE[a] for a in ATTACK_ORDER], alpha=0.92,
        edgecolor="white", linewidth=1.0,
    )
    ax_flip.set_yticks(range(len(ATTACK_ORDER)))
    ax_flip.set_yticklabels([LABELS[a] for a in ATTACK_ORDER])
    ax_flip.set_xlabel("Fraction of trajectories whose first tool flipped")
    ax_flip.set_title("(d)  Step-1 tool-flip rate", loc="left", fontweight="bold", fontsize=12)
    ax_flip.set_xlim(0, 1.05)
    for b, f in zip(bars, flips, strict=True):
        ax_flip.text(f + 0.015, b.get_y() + b.get_height() / 2, f"{f:.0%}",
                     va="center", fontsize=10, color="#222222")
    ax_flip.invert_yaxis()
    ax_flip.grid(axis="x", linestyle=":", alpha=0.35)

    fig.suptitle(
        "Adversarial Attack Landscape on Qwen2.5-VL-7B Medical Agent (ProstateX val=5)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.savefig(out_path)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# Figure 2: ε curves with CI bands (zoomed, dedicated)
# ───────────────────────────────────────────────────────────────────────
def fig_eps_curves(by_attack: dict[str, list[dict]], out_path: Path) -> None:
    eps_vals = sorted({float(r["epsilon"]) for recs in by_attack.values() for r in recs})
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name in ATTACK_ORDER:
        recs = by_attack[name]
        groups: dict[float, list[float]] = defaultdict(list)
        for r in recs:
            groups[float(r["epsilon"])].append(r["edit_distance_norm"])
        xs = sorted(groups)
        if not xs:
            continue
        ys_mean = np.array([np.mean(groups[e]) for e in xs])
        ci = np.array([bootstrap_ci(np.asarray(groups[e])) for e in xs])
        ys_lo, ys_hi = ci[:, 0], ci[:, 1]
        if len(xs) == 1:
            ax.errorbar(xs, ys_mean, yerr=[ys_mean - ys_lo, ys_hi - ys_mean],
                        fmt="*", color=PALETTE[name], markersize=18,
                        markeredgecolor="white", markeredgewidth=1.0,
                        capsize=5, label=f"{LABELS[name]} (smoke only)")
        else:
            ax.plot(xs, ys_mean, marker="o", color=PALETTE[name], linewidth=2.5,
                    markersize=8, markeredgecolor="white", markeredgewidth=0.8,
                    label=LABELS[name], zorder=3)
            ax.fill_between(xs, ys_lo, ys_hi, color=PALETTE[name], alpha=0.20, zorder=2)
    ax.set_xlabel("Perturbation budget ε (normalised pixel domain)", fontsize=12)
    ax.set_ylabel("Mean normalised trajectory edit distance", fontsize=12)
    ax.set_title("Attack effectiveness vs ε (95% bootstrap CI bands)", fontsize=13, pad=12)
    ax.set_xticks(eps_vals)
    ax.set_xticklabels([f"{e:.4g}\n({int(round(e * 255))}/255)" for e in eps_vals], fontsize=10)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.95, frameon=True, edgecolor="#dddddd", fontsize=10)
    ax.set_ylim(0, max(0.85, ax.get_ylim()[1]))
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# Figure 3: Radar chart across multiple metrics
# ───────────────────────────────────────────────────────────────────────
def fig_radar(by_attack: dict[str, list[dict]], out_path: Path) -> None:
    # Metrics, each normalised to [0, 1] via global max so larger = stronger attack.
    metric_names = ["Mean edit\ndistance", "Max edit\ndistance", "Step-1\nflip rate", "P95 edit\ndistance", "Trajectory\nlength Δ"]

    def compute(name: str) -> list[float]:
        recs = by_attack[name]
        eds = edits(recs)
        if eds.size == 0:
            return [0.0] * len(metric_names)
        bens_len = np.array([len(r.get("benign", {}).get("tool_sequence", []) or []) for r in recs])
        atts_len = np.array([len(r.get("attacked", {}).get("tool_sequence", []) or []) for r in recs])
        traj_delta = np.abs(atts_len - bens_len).mean() if bens_len.size else 0.0
        return [
            float(eds.mean()),
            float(eds.max()),
            flip_rate(recs),
            float(np.quantile(eds, 0.95)),
            float(traj_delta),
        ]

    raw = {n: compute(n) for n in ATTACK_ORDER if n != "pgd" or len(by_attack[n]) > 0}
    # Normalise each metric column to [0, 1] using attack max
    arr = np.array([raw[n] for n in raw])
    col_max = np.where(arr.max(axis=0) == 0, 1.0, arr.max(axis=0))
    norm = arr / col_max

    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for i, name in enumerate(raw):
        values = norm[i].tolist() + [norm[i][0]]
        ax.plot(angles, values, color=PALETTE[name], linewidth=2.4, label=LABELS[name], zorder=3)
        ax.fill(angles, values, color=PALETTE[name], alpha=0.16, zorder=2)
        # Annotate raw values
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8.5, color="#666666")
    ax.set_ylim(0, 1.12)
    ax.spines["polar"].set_color("#dddddd")
    ax.grid(color="#dddddd", linewidth=0.7)
    ax.set_title("Attack profile across 5 metrics (column-normalised, larger = stronger attack)",
                 fontsize=12, pad=22, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), frameon=False, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# Figure 4: Tool-substitution heatmap matrix per attack
# ───────────────────────────────────────────────────────────────────────
def fig_tool_substitution(by_attack: dict[str, list[dict]], out_path: Path) -> None:
    attacks_with_data = [a for a in ATTACK_ORDER if by_attack.get(a)]
    n_panels = len(attacks_with_data)
    cols = min(3, n_panels)
    rows = int(np.ceil(n_panels / cols))

    # Collect global tool universe
    universe: set[str] = set()
    for recs in by_attack.values():
        for r in recs:
            universe.update(r.get("benign", {}).get("tool_sequence", []) or [])
            universe.update(r.get("attacked", {}).get("tool_sequence", []) or [])
    tools = sorted(universe)
    idx = {t: i for i, t in enumerate(tools)}

    cmap = LinearSegmentedColormap.from_list("flip", ["#f5f5f5", "#1565C0", "#5E35B1", "#C62828"])
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows), squeeze=False)

    for ax_idx, name in enumerate(attacks_with_data):
        ax = axes[ax_idx // cols][ax_idx % cols]
        mat = np.zeros((len(tools), len(tools)), dtype=np.float64)
        recs = by_attack[name]
        for r in recs:
            b = r.get("benign", {}).get("tool_sequence", []) or []
            a = r.get("attacked", {}).get("tool_sequence", []) or []
            for bi, ai in zip(b, a):
                if bi in idx and ai in idx:
                    mat[idx[bi], idx[ai]] += 1
        # Normalise rows (P(attacked | benign))
        row_sums = mat.sum(axis=1, keepdims=True)
        norm = np.where(row_sums == 0, 0, mat / np.where(row_sums == 0, 1, row_sums))
        im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        # Diagonal = identity (no flip); off-diagonal = substitutions
        for i in range(len(tools)):
            for j in range(len(tools)):
                if mat[i, j] > 0:
                    color = "white" if norm[i, j] > 0.55 else "#333333"
                    ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center",
                            fontsize=7.5, color=color)
        ax.set_xticks(range(len(tools)))
        ax.set_yticks(range(len(tools)))
        ax.set_xticklabels(tools, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tools, fontsize=8)
        ax.set_xlabel("Attacked tool", fontsize=10)
        ax.set_ylabel("Benign tool", fontsize=10)
        ax.set_title(LABELS[name], fontsize=11, color=PALETTE[name], fontweight="bold")
        # Soft diagonal indicator
        ax.plot([-0.5, len(tools) - 0.5], [-0.5, len(tools) - 0.5],
                color="#888888", linewidth=0.6, linestyle=":", alpha=0.7)

    # Hide unused subplots
    for k in range(n_panels, rows * cols):
        axes[k // cols][k % cols].axis("off")

    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label("P(attacked tool | benign tool)", fontsize=10)
    fig.suptitle("Tool-substitution matrices per attack (counts shown; diagonal = no flip)",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.savefig(out_path)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
# Figure 5: Violin grid: per (attack, ε)
# ───────────────────────────────────────────────────────────────────────
def fig_violin_grid(by_attack: dict[str, list[dict]], out_path: Path) -> None:
    attacks_eps = [a for a in ATTACK_ORDER if any(by_attack.get(a, []))]
    eps_vals = sorted({float(r["epsilon"]) for a in attacks_eps for r in by_attack[a]})

    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_attacks = len(attacks_eps)
    width = 0.78 / n_attacks
    positions_list = []
    for ai, name in enumerate(attacks_eps):
        groups = defaultdict(list)
        for r in by_attack[name]:
            groups[float(r["epsilon"])].append(r["edit_distance_norm"])
        xs = [eps_vals.index(e) for e in eps_vals if e in groups]
        data = [groups[eps_vals[x]] for x in xs]
        if not data:
            continue
        offset = (ai - (n_attacks - 1) / 2) * width
        positions = [x + offset for x in xs]
        positions_list.append((name, positions))
        parts = ax.violinplot(data, positions=positions, widths=width * 0.9,
                              showmeans=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(PALETTE[name]); body.set_edgecolor(PALETTE[name])
            body.set_alpha(0.65)
        if "cmeans" in parts:
            parts["cmeans"].set_color("black"); parts["cmeans"].set_linewidth(1.2)
    ax.set_xticks(range(len(eps_vals)))
    ax.set_xticklabels([f"{e:.4g}\n({int(round(e * 255))}/255)" for e in eps_vals], fontsize=10)
    ax.set_xlabel("Perturbation budget ε", fontsize=12)
    ax.set_ylabel("Normalised trajectory edit distance", fontsize=12)
    ax.set_title("Edit-distance distribution per (attack, ε) cell", fontsize=13, pad=12)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[a], alpha=0.65)
               for a, _ in positions_list]
    labels = [LABELS[a] for a, _ in positions_list]
    ax.legend(handles, labels, loc="upper left", frameon=True, edgecolor="#dddddd", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
def main() -> int:
    root = Path(__file__).resolve().parents[1]
    by_attack = {
        "noise":            load_records(root / "runs/smoke/records.jsonl",
                                          root / "runs/smoke_sweep/records.jsonl"),
        "pgd":              load_records(root / "runs/pgd_smoke/records.jsonl"),
        "apgd":             load_records(root / "runs/apgd_sweep/records.jsonl",
                                          root / "runs/apgd_smoke/records.jsonl"),
        "targeted_tool":    load_records(root / "runs/targeted_tool_sweep/records.jsonl",
                                          root / "runs/targeted_tool_smoke/records.jsonl"),
        "trajectory_drift": load_records(root / "runs/trajectory_drift_sweep/records.jsonl",
                                          root / "runs/trajectory_drift_smoke/records.jsonl"),
    }

    out_dir = root / "paper" / "figures" / "attack_landscape"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_landscape_overview(by_attack, out_dir / "fig1_landscape_overview.png")
    fig_eps_curves(by_attack, out_dir / "fig2_eps_curves_ci.png")
    fig_radar(by_attack, out_dir / "fig3_attack_radar.png")
    fig_tool_substitution(by_attack, out_dir / "fig4_tool_substitution.png")
    fig_violin_grid(by_attack, out_dir / "fig5_violin_grid.png")

    print(f"[make_attack_landscape] wrote 5 figures → {out_dir}")
    for name, recs in by_attack.items():
        eds = edits(recs)
        print(f"  {name:18s} n={eds.size:3d}  μ={eds.mean():.3f}  flip={flip_rate(recs):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
