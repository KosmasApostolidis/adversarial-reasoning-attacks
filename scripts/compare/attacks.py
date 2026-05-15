"""Compare N attack modes on shared evaluation samples.

Aggregates one or more ``runs/<mode>/records.jsonl`` files and emits:
  - <out>/edit_distance_box.png   — boxplot of trajectory edit distance
  - <out>/edit_distance_bar.png   — mean ± 95% bootstrap CI bar chart
  - <out>/edit_distance_vs_eps.png — line plot per attack across ε (sweep runs)
  - <out>/targeted_hit_rate.png   — bar chart restricted to targeted_tool runs
  - <out>/summary.json            — per-attack means, medians, CIs

Usage
-----
    python scripts/compare_attacks.py \
        --runs noise=runs/smoke pgd=runs/pgd_smoke apgd=runs/apgd_smoke \
               targeted_tool=runs/targeted_tool_smoke \
               trajectory_drift=runs/trajectory_drift_smoke \
        --out paper/figures/attack_comparison
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ATTACK_COLORS = {
    "noise": "#888888",
    "pgd": "#c62828",
    "apgd": "#ef6c00",
    "targeted_tool": "#1976d2",
    "trajectory_drift": "#6a1b9a",
}
ATTACK_LABELS = {
    "noise": "Uniform noise",
    "pgd": "PGD-L∞",
    "apgd": "APGD-L∞",
    "targeted_tool": "Targeted-Tool",
    "trajectory_drift": "Trajectory-Drift",
}


def _load_records(records_path: Path) -> list[dict]:
    return [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]


def _bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05
) -> tuple[float, float]:
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(0)
    boots = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _parse_runs(specs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--runs entry must be 'name=path', got {spec!r}")
        name, path = spec.split("=", 1)
        records = Path(path) / "records.jsonl" if Path(path).is_dir() else Path(path)
        if not records.exists():
            raise FileNotFoundError(records)
        out[name] = records
    return out


def boxplot(by_attack: dict[str, np.ndarray], out_path: Path) -> None:
    names = list(by_attack)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(names)), 4.5))
    bp = ax.boxplot(
        [by_attack[n] for n in names],
        labels=[ATTACK_LABELS.get(n, n) for n in names],
        patch_artist=True,
        widths=0.55,
    )
    for patch, name in zip(bp["boxes"], names, strict=False):
        patch.set_facecolor(ATTACK_COLORS.get(name, "#444444"))
        patch.set_alpha(0.85)
    ax.set_ylabel("Normalized trajectory edit distance", fontsize=11)
    ax.set_title("Edit-distance distribution per attack", fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for i, n in enumerate(names, start=1):
        arr = by_attack[n]
        if arr.size:
            ax.text(i, arr.max() + 0.04, f"μ={arr.mean():.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def bar_with_ci(by_attack: dict[str, np.ndarray], out_path: Path) -> None:
    names = list(by_attack)
    means = np.array([by_attack[n].mean() if by_attack[n].size else 0.0 for n in names])
    cis = [_bootstrap_ci(by_attack[n]) for n in names]
    err_lo = np.array([m - lo for m, (lo, _) in zip(means, cis, strict=False)])
    err_hi = np.array([hi - m for m, (_, hi) in zip(means, cis, strict=False)])
    colors = [ATTACK_COLORS.get(n, "#444444") for n in names]

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(names)), 4.5))
    ax.bar(
        names,
        means,
        color=colors,
        alpha=0.9,
        yerr=[err_lo, err_hi],
        capsize=6,
        tick_label=[ATTACK_LABELS.get(n, n) for n in names],
    )
    ax.set_ylabel("Mean edit distance (95% bootstrap CI)", fontsize=11)
    ax.set_title("Attack effectiveness on Qwen2.5-VL-7B (ProstateX, val-5)", fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def line_vs_eps(records_by_attack: dict[str, list[dict]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, records in records_by_attack.items():
        groups: dict[float, list[float]] = defaultdict(list)
        for r in records:
            groups[float(r["epsilon"])].append(r["edit_distance_norm"])
        if not groups:
            continue
        eps_vals = sorted(groups)
        means = [float(np.mean(groups[e])) for e in eps_vals]
        sems = [float(np.std(groups[e]) / max(1, len(groups[e]) ** 0.5)) for e in eps_vals]
        ax.errorbar(
            eps_vals,
            means,
            yerr=sems,
            marker="o",
            capsize=4,
            color=ATTACK_COLORS.get(name, "#444444"),
            label=ATTACK_LABELS.get(name, name),
        )
    ax.set_xlabel("ε (normalized pixel domain)", fontsize=11)
    ax.set_ylabel("Mean edit distance ± SEM", fontsize=11)
    ax.set_title("Attack effectiveness vs perturbation budget", fontsize=12)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def targeted_hit_rate(records_by_attack: dict[str, list[dict]], out_path: Path) -> float | None:
    if "targeted_tool" not in records_by_attack:
        return None
    recs = records_by_attack["targeted_tool"]
    hits, total = 0, 0
    targets: list[str] = []
    for r in recs:
        meta = r.get("attacked", {}).get("metadata", {}) or {}
        target = meta.get("target_tool")
        if target is None:
            continue
        targets.append(target)
        seq = r.get("attacked", {}).get("tool_sequence", []) or []
        hits += int(target in seq)
        total += 1
    if total == 0:
        return None
    rate = hits / total
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Targeted-Tool"], [rate], color=ATTACK_COLORS["targeted_tool"], alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Hit rate (target tool appears in trajectory)", fontsize=11)
    target_label = targets[0] if len(set(targets)) == 1 else "various"
    ax.set_title(f"Targeted-Tool hit rate (n={total}, target={target_label})", fontsize=11)
    ax.text(0, rate + 0.03, f"{rate:.2%}", ha="center", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return rate


def _pgd_noise_compare(noise_path: Path, pgd_path: Path, out_dir: Path) -> int:
    """Boxplot of PGD vs uniform-noise edit-distance at matched ε."""
    out_dir.mkdir(parents=True, exist_ok=True)
    noise = _load_records(noise_path)
    pgd = _load_records(pgd_path)
    nd = np.array([r["edit_distance_norm"] for r in noise])
    pd_ = np.array([r["edit_distance_norm"] for r in pgd])
    eps = pgd[0]["epsilon"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(
        [nd, pd_],
        labels=["uniform noise", "PGD-L∞ (20 steps)"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#888888")
    bp["boxes"][1].set_facecolor("#c62828")
    ax.set_ylabel("normalized trajectory edit distance", fontsize=11)
    ax.set_title(
        f"PGD vs noise on Qwen2.5-VL-7B at ε={eps:.4f} (n={len(pd_)})",
        fontsize=11,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for i, (_label, arr) in enumerate([("noise", nd), ("PGD", pd_)], start=1):
        ax.text(i, max(arr.max(), 0.05) + 0.04, f"mean={arr.mean():.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "pgd_vs_noise_box.png", dpi=140)
    plt.close(fig)

    print(
        f"[compare_attacks pgd_noise] noise mean={nd.mean():.3f}  "
        f"PGD mean={pd_.mean():.3f}  Δ={pd_.mean() - nd.mean():+.3f}"
    )
    print(f"[compare_attacks pgd_noise] wrote → {out_dir}")
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["default", "pgd_noise"],
        default="default",
        help="default: N-attack aggregate. pgd_noise: PGD-vs-noise boxplot.",
    )
    ap.add_argument(
        "--runs",
        nargs="+",
        help="[default mode] One or more 'name=path' entries (dir or records.jsonl).",
    )
    ap.add_argument("--noise", help="[pgd_noise mode] records.jsonl from noise run.")
    ap.add_argument("--pgd", help="[pgd_noise mode] records.jsonl from PGD run.")
    ap.add_argument("--out", default=None)
    return ap


def _compute_summary(
    edit_by_attack: dict[str, np.ndarray], hit_rate: float | None
) -> dict[str, dict]:
    summary = {
        name: {
            "n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else 0.0,
            "median": float(np.median(arr)) if arr.size else 0.0,
            "ci95": list(_bootstrap_ci(arr)),
        }
        for name, arr in edit_by_attack.items()
    }
    if hit_rate is not None:
        summary["targeted_tool"]["hit_rate"] = hit_rate
    return summary


def _print_summary(
    paths: dict[str, Path],
    edit_by_attack: dict[str, np.ndarray],
    hit_rate: float | None,
    out_dir: Path,
) -> None:
    print(f"[compare_attacks] wrote {len(paths)} attacks → {out_dir}")
    for name, arr in edit_by_attack.items():
        ci = _bootstrap_ci(arr)
        print(f"  {name:18s} n={arr.size:3d}  μ={arr.mean():.3f}  CI95=[{ci[0]:.3f}, {ci[1]:.3f}]")
    if hit_rate is not None:
        print(f"  targeted_tool hit rate: {hit_rate:.2%}")


def _run_default_mode(args: argparse.Namespace) -> int:
    paths = _parse_runs(args.runs)
    records_by_attack = {name: _load_records(p) for name, p in paths.items()}
    edit_by_attack = {
        name: np.array([r["edit_distance_norm"] for r in recs], dtype=np.float64)
        for name, recs in records_by_attack.items()
    }
    out_dir = Path(args.out or "paper/figures/attack_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    boxplot(edit_by_attack, out_dir / "edit_distance_box.png")
    bar_with_ci(edit_by_attack, out_dir / "edit_distance_bar.png")
    line_vs_eps(records_by_attack, out_dir / "edit_distance_vs_eps.png")
    hit_rate = targeted_hit_rate(records_by_attack, out_dir / "targeted_hit_rate.png")
    summary = _compute_summary(edit_by_attack, hit_rate)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _print_summary(paths, edit_by_attack, hit_rate, out_dir)
    return 0


def main() -> int:
    ap = _build_argparser()
    args = ap.parse_args()

    if args.mode == "pgd_noise":
        if not (args.noise and args.pgd):
            ap.error("--noise and --pgd are required for --mode pgd_noise")
        out_dir = Path(args.out or "paper/figures/pgd_vs_noise")
        return _pgd_noise_compare(Path(args.noise), Path(args.pgd), out_dir)

    if not args.runs:
        ap.error("--runs is required for --mode default")

    return _run_default_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
