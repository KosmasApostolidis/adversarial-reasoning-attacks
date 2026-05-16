"""Cross-model attack landscape: Qwen2.5-VL-7B vs LLaVA-v1.6-Mistral-7B.

Reads all 6 smoke run dirs, produces:
  - paper/figures/cross_model/edit_dist_grouped.png  (grouped bar, model × attack)
  - paper/figures/cross_model/per_sample_dot.png     (dot plot, sample-level pairing)
  - paper/figures/cross_model/summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ATTACKS = ["apgd", "targeted", "drift"]
MODELS = [("qwen", "Qwen2.5-VL-7B"), ("llava", "LLaVA-v1.6-Mistral-7B")]
ATTACK_LABEL = {
    "apgd": "APGD-L∞",
    "targeted": "Targeted-Tool",
    "drift": "Trajectory-Drift",
}
ATTACK_COLOR = {"apgd": "#ef6c00", "targeted": "#1976d2", "drift": "#6a1b9a"}
MODEL_HATCH = {"qwen": "", "llava": "///"}


def _load(run_dir: Path) -> list[dict]:
    p = run_dir / "records.jsonl"
    if not p.exists() or p.stat().st_size == 0:
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _collect(runs_root: Path) -> dict[tuple[str, str], list[dict]]:
    map_ = {}
    for atk in ATTACKS:
        for mdl, _ in MODELS:
            tag = "trajectory_drift_smoke" if atk == "drift" else f"{atk}_smoke"
            tag = "targeted_tool_smoke" if atk == "targeted" else tag
            tag = "apgd_smoke" if atk == "apgd" else tag
            d = runs_root / (tag if mdl == "qwen" else f"{tag}_llava")
            map_[(atk, mdl)] = _load(d)
    return map_


def _compute_model_stats(data: dict, mdl: str) -> tuple[list[float], list[float]]:
    means, errs = [], []
    for atk in ATTACKS:
        recs = data[(atk, mdl)]
        eds = [r["edit_distance_norm"] for r in recs]
        means.append(statistics.mean(eds) if eds else 0.0)
        errs.append(statistics.stdev(eds) / np.sqrt(len(eds)) if len(eds) > 1 else 0.0)
    return means, errs


def _draw_model_bars(
    ax,
    mdl: str,
    mdl_label: str,
    x: np.ndarray,
    w: float,
    i: int,
    means: list[float],
    errs: list[float],
) -> None:
    offset = (i - 0.5) * w
    bars = ax.bar(
        x + offset,
        means,
        w,
        yerr=errs,
        label=mdl_label,
        color=[ATTACK_COLOR[a] for a in ATTACKS],
        edgecolor="black",
        hatch=MODEL_HATCH[mdl],
        linewidth=0.8,
        capsize=3,
        alpha=0.95 if mdl == "qwen" else 0.7,
    )
    for b, v in zip(bars, means, strict=False):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.012,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _build_model_legend_handles() -> list:
    return [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            fc="#888",
            ec="black",
            hatch=MODEL_HATCH[m],
            alpha=0.95 if m == "qwen" else 0.7,
        )
        for m, _ in MODELS
    ]


def _decorate_grouped_axes(ax, x: np.ndarray) -> None:
    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_LABEL[a] for a in ATTACKS])
    ax.set_ylabel("Tool-sequence edit distance (normalized)")
    ax.set_ylim(0, max(0.8, ax.get_ylim()[1]))
    ax.set_title("Cross-model attack landscape (smoke, n=5, ε=8/255)", fontsize=11, weight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    handles = _build_model_legend_handles()
    ax.legend(handles, [lbl for _, lbl in MODELS], loc="upper left", frameon=False)


def _grouped_bar(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=200)
    x = np.arange(len(ATTACKS))
    w = 0.36
    for i, (mdl, mdl_label) in enumerate(MODELS):
        means, errs = _compute_model_stats(data, mdl)
        _draw_model_bars(ax, mdl, mdl_label, x, w, i, means, errs)
    _decorate_grouped_axes(ax, x)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _dot_plot(data: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(ATTACKS), figsize=(11, 4.2), dpi=200, sharey=True)
    for ax, atk in zip(axes, ATTACKS, strict=False):
        for j, (mdl, _mdl_label) in enumerate(MODELS):
            eds = [r["edit_distance_norm"] for r in data[(atk, mdl)]]
            jitter = np.random.normal(0, 0.04, size=len(eds))
            ax.scatter(
                np.full(len(eds), j) + jitter,
                eds,
                s=60,
                color=ATTACK_COLOR[atk],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.85,
                zorder=3,
            )
            if eds:
                ax.hlines(
                    statistics.mean(eds), j - 0.18, j + 0.18, color="black", linewidth=2, zorder=4
                )
        ax.set_xticks([0, 1])
        ax.set_xticklabels([m_lbl.split("-")[0] for _, m_lbl in MODELS])
        ax.set_title(ATTACK_LABEL[atk], fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Edit distance (normalized)")
    fig.suptitle("Per-sample edit distance, Qwen vs LLaVA", fontsize=12, weight="bold", y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _summary(data: dict) -> dict:
    out = {}
    for (atk, mdl), recs in data.items():
        eds = [r["edit_distance_norm"] for r in recs]
        out[f"{mdl}_{atk}"] = {
            "n": len(eds),
            "mean": statistics.mean(eds) if eds else None,
            "median": statistics.median(eds) if eds else None,
            "min": min(eds) if eds else None,
            "max": max(eds) if eds else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures/cross_model"))
    args = ap.parse_args()

    data = _collect(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    _grouped_bar(data, args.out / "edit_dist_grouped.png")
    _dot_plot(data, args.out / "per_sample_dot.png")
    (args.out / "summary.json").write_text(json.dumps(_summary(data), indent=2))
    print(f"Wrote {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
