"""Trajectory length before vs after adversarial attacks, averaged across CV folds.

For each (model, attack_mode), aggregates the mean number of tool calls in the
benign vs attacked agent trajectory across the 3 ProstateX CV folds. Folds are
parsed from the runner-emitted sample_id (e.g. ``bhi_f1_val_p000_s06``).

The cross-fold mean drives the bar height; the cross-fold std drives the error
bar. Per-(model, attack, fold) means are averaged across all samples, seeds,
and epsilons in that bucket -- this is the global before/after view; an
epsilon-resolved breakdown would be a separate panel.

Inputs   runs/main/<mode>/<fold_tag>/records.jsonl  (5 modes x 3 folds = 15 files)
Outputs  paper/figures/paper/fig_trajectory_length_before_after.{png,csv}
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib.pyplot as plt

# Allow ``from _plotlib import ...`` when invoked as a module-by-path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _plotlib import despine, load_records, panel_label

_FOLD_RE = re.compile(r"_f(\d)_")
_ATTACK_ORDER = ("noise", "pgd", "apgd", "targeted_tool", "trajectory_drift")


def _parse_fold(sample_id: str) -> str | None:
    m = _FOLD_RE.search(sample_id)
    return f"fold_{m.group(1)}" if m else None


def _collect_paths(runs_root: Path) -> list[Path]:
    return sorted(runs_root.glob("*/fold_*/records.jsonl"))


def _aggregate(
    records: list[dict[str, Any]],
    *,
    task_id: str,
) -> tuple[dict[tuple[str, str, str], tuple[float, float, int]], list[str], list[str]]:
    """Return ``per_fold[(model, attack, fold)] = (mean_benign, mean_attacked, n)``.

    Also return ordered lists of (models, attacks) that actually appear.
    """
    bucket: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    seen_models: set[str] = set()
    seen_attacks: set[str] = set()

    for r in records:
        if r.get("task_id") != task_id:
            continue
        sample_id = r.get("sample_id", "")
        fold = _parse_fold(sample_id)
        if fold is None:
            continue
        model = r.get("model_key")
        attack = r.get("attack_mode")
        benign_seq = r.get("benign", {}).get("tool_sequence") or []
        attacked_seq = r.get("attacked", {}).get("tool_sequence") or []
        if not isinstance(model, str) or not isinstance(attack, str):
            continue
        bucket[(model, attack, fold)].append((len(benign_seq), len(attacked_seq)))
        seen_models.add(model)
        seen_attacks.add(attack)

    per_fold: dict[tuple[str, str, str], tuple[float, float, int]] = {}
    for key, pairs in bucket.items():
        b_lens = [b for b, _ in pairs]
        a_lens = [a for _, a in pairs]
        per_fold[key] = (mean(b_lens), mean(a_lens), len(pairs))

    models = sorted(seen_models)
    attacks = [a for a in _ATTACK_ORDER if a in seen_attacks] + sorted(
        seen_attacks - set(_ATTACK_ORDER)
    )
    return per_fold, models, attacks


def _avg_across_folds(
    per_fold: dict[tuple[str, str, str], tuple[float, float, int]],
    *,
    model: str,
    attack: str,
) -> tuple[float, float, float, float, int] | None:
    """Average per-fold means across folds; return (b_mean, a_mean, b_std, a_std, n_folds)."""
    b_means = [v[0] for k, v in per_fold.items() if k[0] == model and k[1] == attack]
    a_means = [v[1] for k, v in per_fold.items() if k[0] == model and k[1] == attack]
    if not b_means:
        return None
    b_std = pstdev(b_means) if len(b_means) > 1 else 0.0
    a_std = pstdev(a_means) if len(a_means) > 1 else 0.0
    return mean(b_means), mean(a_means), b_std, a_std, len(b_means)


def _write_csv(
    out_csv: Path,
    per_fold: dict[tuple[str, str, str], tuple[float, float, int]],
) -> None:
    rows = []
    for (model, attack, fold), (b_mean, a_mean, n) in sorted(per_fold.items()):
        rows.append((model, attack, fold, f"{b_mean:.4f}", f"{a_mean:.4f}", n))
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model_key",
                "attack_mode",
                "fold",
                "mean_len_benign",
                "mean_len_attacked",
                "n_samples",
            ]
        )
        w.writerows(rows)


def _compute_panel_data(
    per_fold: dict[tuple[str, str, str], tuple[float, float, int]],
    model: str,
    attacks: list[str],
) -> tuple[list[float], list[float], list[float], list[float]]:
    b_heights, a_heights, b_errs, a_errs = [], [], [], []
    for attack in attacks:
        agg = _avg_across_folds(per_fold, model=model, attack=attack)
        if agg is None:
            b_heights.append(0.0)
            a_heights.append(0.0)
            b_errs.append(0.0)
            a_errs.append(0.0)
            continue
        b_mean, a_mean, b_std, a_std, _ = agg
        b_heights.append(b_mean)
        a_heights.append(a_mean)
        b_errs.append(b_std)
        a_errs.append(a_std)
    return b_heights, a_heights, b_errs, a_errs


def _draw_one_panel(
    ax,
    model: str,
    x: list[int],
    bar_w: float,
    attacks: list[str],
    col: int,
    heights_and_errs: tuple[list[float], list[float], list[float], list[float]],
) -> None:
    b_heights, a_heights, b_errs, a_errs = heights_and_errs
    ax.bar(
        [xi - bar_w / 2 for xi in x], b_heights, width=bar_w, yerr=b_errs,
        capsize=3, color="#4c78a8", label="benign",
    )
    ax.bar(
        [xi + bar_w / 2 for xi in x], a_heights, width=bar_w, yerr=a_errs,
        capsize=3, color="#e45756", label="attacked",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=20, ha="right")
    ax.set_title(model, fontsize=11)
    if col == 0:
        ax.set_ylabel("mean tool calls per trajectory")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    despine(ax)
    panel_label(ax, "abcdefg"[col])


def _plot(
    per_fold: dict[tuple[str, str, str], tuple[float, float, int]],
    *,
    models: list[str],
    attacks: list[str],
    out_png: Path,
    task_id: str,
) -> None:
    n_panels = max(1, len(models))
    fig, axes = plt.subplots(1, n_panels, figsize=(5.6 * n_panels, 4.4), sharey=True, squeeze=False)
    axes = axes[0]

    bar_w = 0.38
    x = list(range(len(attacks)))

    for col, model in enumerate(models):
        heights_and_errs = _compute_panel_data(per_fold, model, attacks)
        _draw_one_panel(axes[col], model, x, bar_w, attacks, col, heights_and_errs)

    if axes.size > 0:
        axes[-1].legend(loc="upper right", frameon=False)

    fig.suptitle(
        f"Trajectory length before vs after attack, averaged across CV folds  ({task_id})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-root", default="runs/main", help="Root of <mode>/<fold>/records.jsonl tree"
    )
    p.add_argument("--task", default="prostate_mri_workup")
    p.add_argument(
        "--out",
        default="paper/figures/paper/fig_trajectory_length_before_after.png",
        help="Output PNG path; sibling .csv is written alongside",
    )
    args = p.parse_args(argv)

    runs_root = Path(args.runs_root)
    out_png = Path(args.out)
    out_csv = out_png.with_suffix(".csv")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    paths = _collect_paths(runs_root)
    if not paths:
        print(f"[plot] no records.jsonl under {runs_root}/*/fold_*/", file=sys.stderr)
        return 2

    print(f"[plot] loading {len(paths)} records.jsonl files from {runs_root}")
    records = load_records(*paths)
    print(f"[plot] loaded {len(records)} rows; filtering task={args.task}")

    per_fold, models, attacks = _aggregate(records, task_id=args.task)
    if not per_fold:
        print(f"[plot] no rows match task_id={args.task}", file=sys.stderr)
        return 2

    print(f"[plot] models={models}  attacks={attacks}  per_fold cells={len(per_fold)}")
    _write_csv(out_csv, per_fold)
    _plot(per_fold, models=models, attacks=attacks, out_png=out_png, task_id=args.task)

    print(f"[plot] wrote {out_png}")
    print(f"[plot] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
