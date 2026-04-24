"""Render result figures from runner JSONL + noise-floor gate reports.

Outputs to ``paper/figures/`` (created if absent). One-shot CLI.

Figures
-------
1. ``tool_sequence_comparison_<model>_<sample>.png`` — benign vs attacked
   tool-call sequence per sample, Gantt-style.
2. ``edit_distance_distribution_<model>.png`` — histogram + boxplot of
   normalized edit distances, with noise-floor threshold overlay.
3. ``tool_frequency_<model>.png`` — stacked/side-by-side bars of tool
   invocation counts, benign vs attacked.
4. ``eps_sweep_<model>.png`` — mean edit distance vs ε (only when multi-ε
   data present).
5. ``example_attack_panel_<model>_<sample>.png`` — clean | δ | adversarial
   triptych for the first sample (uses saved perturbed PIL if present; else
   regenerates deterministic noise perturbation from record).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


def _load_records(jsonl_path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]


def _load_gate(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _unique_tools(records: list[dict]) -> list[str]:
    names: set[str] = set()
    for r in records:
        names.update(r["benign"]["tool_sequence"])
        names.update(r["attacked"]["tool_sequence"])
    return sorted(names)


def fig_sequence_comparison(
    record: dict, palette: dict[str, Any], out_path: Path
) -> None:
    benign = record["benign"]["tool_sequence"]
    attacked = record["attacked"]["tool_sequence"]
    max_len = max(len(benign), len(attacked), 1)

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * max_len), 3.2))
    for row, (label, seq) in enumerate([("attacked", attacked), ("benign", benign)]):
        for step, name in enumerate(seq):
            ax.add_patch(
                Rectangle(
                    (step, row - 0.35),
                    0.9,
                    0.7,
                    facecolor=palette[name],
                    edgecolor="black",
                    linewidth=0.6,
                )
            )
            ax.text(
                step + 0.45,
                row,
                name.replace("_", "\n"),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                weight="bold",
            )
    ax.set_xlim(-0.2, max_len + 0.2)
    ax.set_ylim(-0.8, 1.8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["attacked", "benign"], fontsize=11)
    ax.set_xlabel("trajectory step", fontsize=11)
    ed = record["edit_distance_norm"]
    ax.set_title(
        f"{record['model_key']} · {record['sample_id']} · ε={record['epsilon']:.4f} · "
        f"normalized edit distance={ed:.3f}",
        fontsize=11,
    )
    ax.set_xticks(range(max_len))
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_edit_distance_distribution(
    records: list[dict], noise_threshold: float, out_path: Path
) -> None:
    dists = np.array([r["edit_distance_norm"] for r in records])
    fig, (ax_h, ax_b) = plt.subplots(
        1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 1]}
    )
    bins = np.linspace(0, max(0.5, dists.max() * 1.1), 16)
    ax_h.hist(dists, bins=bins, color="#4a7ab8", edgecolor="black", linewidth=0.6)
    ax_h.axvline(noise_threshold, color="red", linestyle="--", linewidth=1.2,
                 label=f"noise floor threshold={noise_threshold:.3f}")
    ax_h.axvline(dists.mean(), color="black", linestyle="-", linewidth=1.0,
                 label=f"mean={dists.mean():.3f}")
    ax_h.set_xlabel("normalized trajectory edit distance", fontsize=11)
    ax_h.set_ylabel("sample count", fontsize=11)
    ax_h.set_title(f"{records[0]['model_key']} · attack effect distribution (n={len(dists)})",
                   fontsize=11)
    ax_h.legend(fontsize=9)
    ax_h.grid(axis="y", linestyle=":", alpha=0.4)

    bp = ax_b.boxplot(dists, vert=True, patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#4a7ab8")
    ax_b.axhline(noise_threshold, color="red", linestyle="--", linewidth=1.2)
    ax_b.set_xticks([])
    ax_b.set_ylabel("edit distance", fontsize=10)
    ax_b.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_tool_frequency(records: list[dict], palette: dict[str, Any], out_path: Path) -> None:
    benign_counts: Counter[str] = Counter()
    attacked_counts: Counter[str] = Counter()
    for r in records:
        benign_counts.update(r["benign"]["tool_sequence"])
        attacked_counts.update(r["attacked"]["tool_sequence"])
    tools = sorted(set(benign_counts) | set(attacked_counts))
    if not tools:
        return
    x = np.arange(len(tools))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(tools)), 4.5))
    ax.bar(x - w / 2, [benign_counts[t] for t in tools], w, label="benign",
           color="#2e7d32", edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, [attacked_counts[t] for t in tools], w, label="attacked",
           color="#c62828", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tools], fontsize=9)
    ax.set_ylabel("tool-call count across samples", fontsize=11)
    ax.set_title(f"{records[0]['model_key']} · tool invocation frequency (n={len(records)})",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_eps_sweep(records: list[dict], out_path: Path) -> None:
    if len({r["epsilon"] for r in records}) < 2:
        return
    per_eps: dict[float, list[float]] = {}
    for r in records:
        per_eps.setdefault(r["epsilon"], []).append(r["edit_distance_norm"])
    eps_sorted = sorted(per_eps)
    means = [np.mean(per_eps[e]) for e in eps_sorted]
    stds = [np.std(per_eps[e], ddof=1) if len(per_eps[e]) > 1 else 0 for e in eps_sorted]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(eps_sorted, means, yerr=stds, marker="o", linewidth=1.5,
                capsize=4, color="#4a7ab8", label="mean ± 1σ")
    ax.set_xlabel("ε (L∞ radius)", fontsize=11)
    ax.set_ylabel("mean normalized edit distance", fontsize=11)
    ax.set_title(f"{records[0]['model_key']} · dose–response", fontsize=11)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_attack_panel(
    records: list[dict], task_id: str, out_path: Path, split: str = "val"
) -> None:
    """Render clean | δ | adversarial triptych for first sample."""
    from adversarial_reasoning.runner import perturb_noise
    from adversarial_reasoning.tasks.loader import load_task

    rec = records[0]
    samples = list(load_task(task_id, split=split, n=1, synthetic=False))
    if not samples:
        return
    clean = samples[0].image
    adv = perturb_noise(clean, rec["epsilon"], rec["seed"])
    clean_arr = np.asarray(clean, dtype=np.float32)
    adv_arr = np.asarray(adv, dtype=np.float32)
    delta = (adv_arr - clean_arr) / 255.0  # normalized
    delta_vis = ((delta - delta.min()) / (delta.max() - delta.min() + 1e-9) * 255.0).astype(
        np.uint8
    )

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(clean_arr.astype(np.uint8))
    axes[0].set_title("clean", fontsize=11)
    axes[1].imshow(delta_vis)
    axes[1].set_title(
        f"δ (normalized, ε={rec['epsilon']:.4f}, seed={rec['seed']})", fontsize=11
    )
    axes[2].imshow(adv_arr.astype(np.uint8))
    axes[2].set_title("adversarial", fontsize=11)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{rec['model_key']} · {rec['sample_id']}", fontsize=12, y=0.98
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_palette(tools: list[str]) -> dict[str, Any]:
    cmap = plt.get_cmap("tab20")
    return {t: cmap(i % 20) for i, t in enumerate(tools)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True, help="Path to records.jsonl")
    p.add_argument("--noise-floor", default=None, help="Optional gate JSON path")
    p.add_argument("--out", default="paper/figures", help="Output dir")
    p.add_argument("--task", default="prostate_mri_workup")
    args = p.parse_args()

    records = _load_records(Path(args.records))
    if not records:
        print("[make_figures] no records")
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nf = _load_gate(Path(args.noise_floor)) if args.noise_floor else None
    noise_thr = float(nf["threshold_for_signal"]) if nf else 0.0

    palette = make_palette(_unique_tools(records))
    model_key = records[0]["model_key"]

    fig_edit_distance_distribution(records, noise_thr, out_dir / f"edit_distance_distribution_{model_key}.png")
    fig_tool_frequency(records, palette, out_dir / f"tool_frequency_{model_key}.png")
    fig_eps_sweep(records, out_dir / f"eps_sweep_{model_key}.png")
    fig_attack_panel(records, args.task, out_dir / f"example_attack_panel_{model_key}.png")

    for r in records:
        fig_sequence_comparison(
            r, palette,
            out_dir / f"tool_sequence_{model_key}_{r['sample_id']}.png",
        )
    print(f"[make_figures] wrote figures → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
