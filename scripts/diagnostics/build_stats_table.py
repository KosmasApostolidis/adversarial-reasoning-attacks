"""Build the main paper benchmark stats table from ``runs/main/``.

Per cell ``(model_key, task_id, attack_mode, epsilon)``:
  - median of ``edit_distance_norm``,
  - 95% bootstrap CI on the *paired* delta (attacked minus noise baseline),
  - paired Wilcoxon signed-rank vs the matched ``noise`` baseline.

P-values are pooled per ``(model_key, task_id)`` family and corrected with
Benjamini-Hochberg at ``q=0.05``. Rows surviving correction are tagged ``*``.

Output: a booktabs LaTeX table ready to ``\\input{}`` from the manuscript.

Usage
-----
    python scripts/build_stats_table.py \\
        --runs-dir runs/main \\
        --out paper/tables/main_benchmark.tex
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from adversarial_reasoning.metrics.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    wilcoxon_signed_rank,
)

# (benign_value, attacked_value) per record, or None if metric absent.
MetricExtractor = Callable[[dict], tuple[float, float] | None]

logger = logging.getLogger(__name__)

ATTACK_MODES: tuple[str, ...] = (
    "noise",
    "pgd",
    "apgd",
    "trajectory_drift",
    "targeted_tool",
)
NON_NOISE_MODES: tuple[str, ...] = tuple(m for m in ATTACK_MODES if m != "noise")
ATTACK_LABELS: dict[str, str] = {
    "pgd": "PGD-L$_\\infty$",
    "apgd": "APGD-L$_\\infty$",
    "trajectory_drift": "Trajectory-Drift",
    "targeted_tool": "Targeted-Tool",
}

PairKey = tuple[str, str, int, str]
CellKey = tuple[str, str, str, float]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_runs_dir(runs_dir: Path) -> dict[str, list[dict]]:
    return {mode: _load_jsonl(runs_dir / mode / "records.jsonl") for mode in ATTACK_MODES}


def _record_attack_mode(rec: dict) -> str:
    return rec.get("attack_mode") or rec.get("attack_name") or "unknown"


def _pair_key(rec: dict) -> PairKey:
    return (rec["model_key"], rec["task_id"], int(rec["seed"]), str(rec["sample_id"]))


def _noise_baseline(noise_records: Iterable[dict]) -> dict[PairKey, float]:
    """Mean ``edit_distance_norm`` per pair-key across all noise epsilons."""
    by_key: dict[PairKey, list[float]] = defaultdict(list)
    for rec in noise_records:
        by_key[_pair_key(rec)].append(float(rec["edit_distance_norm"]))
    return {k: float(np.mean(v)) for k, v in by_key.items()}


def _build_cells(
    attacked_records: Iterable[dict],
    noise_baseline: dict[PairKey, float],
) -> dict[CellKey, dict[str, list[float]]]:
    cells: dict[CellKey, dict[str, list[float]]] = defaultdict(
        lambda: {"benign": [], "attacked": []}
    )
    for rec in attacked_records:
        key = _pair_key(rec)
        if key not in noise_baseline:
            continue
        cell_key: CellKey = (
            rec["model_key"],
            rec["task_id"],
            _record_attack_mode(rec),
            float(rec["epsilon"]),
        )
        cells[cell_key]["benign"].append(noise_baseline[key])
        cells[cell_key]["attacked"].append(float(rec["edit_distance_norm"]))
    return cells


def _build_cells_for_metric(
    attacked_records: Iterable[dict],
    extractor: MetricExtractor,
) -> dict[CellKey, dict[str, list[float]]]:
    """Generic per-cell collector: extractor yields (benign, attacked) per record.

    Skips records where the extractor returns ``None`` (missing metric fields),
    so legacy records without CoT scores never enter the CoT table.
    """
    cells: dict[CellKey, dict[str, list[float]]] = defaultdict(
        lambda: {"benign": [], "attacked": []}
    )
    for rec in attacked_records:
        pair = extractor(rec)
        if pair is None:
            continue
        b, a = pair
        cell_key: CellKey = (
            rec["model_key"],
            rec["task_id"],
            _record_attack_mode(rec),
            float(rec["epsilon"]),
        )
        cells[cell_key]["benign"].append(float(b))
        cells[cell_key]["attacked"].append(float(a))
    return cells


# --- CoT metric extractors ---------------------------------------------------
def _extract_drift(rec: dict) -> tuple[float, float] | None:
    """Drift is already a benign-vs-attacked distance — pair with constant 0."""
    v = rec.get("cot_drift_score")
    if v is None:
        return None
    return (0.0, float(v))


def _extract_faith(rec: dict) -> tuple[float, float] | None:
    b = rec.get("cot_faithfulness_benign")
    a = rec.get("cot_faithfulness_attacked")
    if b is None or a is None:
        return None
    return (float(b), float(a))


def _extract_halluc(rec: dict) -> tuple[float, float] | None:
    b = rec.get("cot_hallucination_benign")
    a = rec.get("cot_hallucination_attacked")
    if b is None or a is None:
        return None
    return (float(b), float(a))


def _extract_refusal(rec: dict) -> tuple[float, float] | None:
    b = rec.get("cot_refusal_benign")
    a = rec.get("cot_refusal_attacked")
    if b is None or a is None:
        return None
    return (1.0 if b else 0.0, 1.0 if a else 0.0)


COT_METRICS: tuple[tuple[str, MetricExtractor], ...] = (
    ("drift", _extract_drift),
    ("faith", _extract_faith),
    ("halluc", _extract_halluc),
    ("refusal", _extract_refusal),
)


def _safe_pvalue(benign: np.ndarray, attacked: np.ndarray, cell_key: CellKey) -> tuple[float, str]:
    # Distinguish three p-value outcomes: clean compute (ok), scipy
    # raised ValueError (e.g. all-zero diffs), or scipy returned NaN.
    # All three previously collapsed to pvalue=1.0 silently.
    pvalue_status = "ok"
    try:
        wlx = wilcoxon_signed_rank(benign, attacked)
        pvalue = wlx.pvalue
    except ValueError as exc:
        logger.warning(
            "wilcoxon_signed_rank ValueError for cell=%s: %s — pvalue→1.0",
            cell_key,
            exc,
        )
        pvalue = 1.0
        pvalue_status = "valuerror"
    if not np.isfinite(pvalue):
        logger.warning(
            "wilcoxon_signed_rank returned non-finite pvalue for cell=%s — pvalue→1.0",
            cell_key,
        )
        pvalue = 1.0
        pvalue_status = "nan"
    return float(pvalue), pvalue_status


def _stats_per_cell(
    cells: dict[CellKey, dict[str, list[float]]],
    *,
    n_resamples: int,
    ci_level: float,
    bootstrap_seed: int | None = 0,
) -> list[dict]:
    rows: list[dict] = []
    for cell_key, arrs in cells.items():
        benign = np.asarray(arrs["benign"], dtype=float)
        attacked = np.asarray(arrs["attacked"], dtype=float)
        if benign.size < 2:
            continue
        delta = attacked - benign
        ci = bootstrap_ci(
            delta,
            statistic="median",
            n_resamples=n_resamples,
            ci_level=ci_level,
            rng_seed=bootstrap_seed,
        )
        pvalue, pvalue_status = _safe_pvalue(benign, attacked, cell_key)
        model, task, attack, eps = cell_key
        rows.append(
            {
                "model_key": model,
                "task_id": task,
                "attack_mode": attack,
                "epsilon": eps,
                "n": int(benign.size),
                "median_delta": float(np.median(delta)),
                "ci_lower": float(ci.lower),
                "ci_upper": float(ci.upper),
                "pvalue": pvalue,
                "pvalue_status": pvalue_status,
            }
        )
    return rows


def _apply_bh(rows: list[dict], q: float) -> None:
    """Tag each row with ``significant`` after BH correction per (model, task)."""
    by_strat: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_strat[(row["model_key"], row["task_id"])].append(i)
    for indices in by_strat.values():
        pvals = np.array([rows[i]["pvalue"] for i in indices], dtype=float)
        rejected = benjamini_hochberg(pvals, q=q)
        for ix, rej in zip(indices, rejected, strict=True):
            rows[ix]["significant"] = bool(rej)


def _format_cell(row: dict) -> str:
    star = "$^{*}$" if row.get("significant") else ""
    return f"{row['median_delta']:+.3f} [{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}]{star}"


def _emit_latex(rows: list[dict], out_path: Path) -> None:
    rows_sorted = sorted(
        rows,
        key=lambda r: (r["model_key"], r["task_id"], r["attack_mode"], r["epsilon"]),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "% Auto-generated by scripts/build_stats_table.py — DO NOT EDIT.",
        "\\begin{tabular}{llllr}",
        "\\toprule",
        "Model & Task & Attack & $\\varepsilon$ & median $\\Delta$edit-dist [95\\% CI] \\\\",
        "\\midrule",
    ]
    for row in rows_sorted:
        attack_label = ATTACK_LABELS.get(row["attack_mode"], row["attack_mode"])
        lines.append(
            f"{row['model_key']} & {row['task_id']} & {attack_label} & "
            f"{row['epsilon']:.4f} & {_format_cell(row)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _emit_cot_latex(rows_by_metric: dict[str, list[dict]], out_path: Path) -> None:
    """Emit a wider CoT table: 1 row per cell × 4 metric columns.

    rows_by_metric maps metric_name → list of stat rows (already BH-tagged).
    Cell join key: (model, task, attack, epsilon).
    """
    cell_rows: dict[CellKey, dict[str, dict]] = defaultdict(dict)
    for metric, rows in rows_by_metric.items():
        for r in rows:
            ck: CellKey = (r["model_key"], r["task_id"], r["attack_mode"], r["epsilon"])
            cell_rows[ck][metric] = r

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "% Auto-generated by scripts/build_stats_table.py — DO NOT EDIT.",
        "\\begin{tabular}{llllllll}",
        "\\toprule",
        (
            "Model & Task & Attack & $\\varepsilon$ & "
            "median CoT-drift [95\\% CI] & "
            "median $\\Delta$faith [95\\% CI] & "
            "median $\\Delta$hall [95\\% CI] & "
            "refusal rate (atk) \\\\"
        ),
        "\\midrule",
    ]
    for ck in sorted(cell_rows.keys()):
        model, task, attack, eps = ck
        attack_label = ATTACK_LABELS.get(attack, attack)
        cells = cell_rows[ck]
        drift_cell = _format_cell(cells["drift"]) if "drift" in cells else "--"
        faith_cell = _format_cell(cells["faith"]) if "faith" in cells else "--"
        hall_cell = _format_cell(cells["halluc"]) if "halluc" in cells else "--"
        # Refusal column: attacked rate as percent (mean over attacked column),
        # not the paired Δ — Δ on {0,1} pairs is too coarse for a headline number.
        if "refusal" in cells and cells["refusal"].get("attacked_rate") is not None:
            atk_rate_str = f"{100.0 * cells['refusal']['attacked_rate']:.1f}\\%"
        else:
            atk_rate_str = "--"
        lines.append(
            f"{model} & {task} & {attack_label} & {eps:.4f} & "
            f"{drift_cell} & {faith_cell} & {hall_cell} & {atk_rate_str} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _annotate_attacked_rate(rows: list[dict], cells: dict[CellKey, dict[str, list[float]]]) -> None:
    """Attach 'attacked_rate' = mean(attacked) per row's cell — used for refusal column."""
    for r in rows:
        ck: CellKey = (r["model_key"], r["task_id"], r["attack_mode"], r["epsilon"])
        atk = cells[ck]["attacked"]
        r["attacked_rate"] = float(np.mean(atk)) if atk else 0.0


def _build_cot_metric_rows(
    metric_name: str,
    extractor: MetricExtractor,
    attacked_records: list[dict],
    *,
    n_resamples: int,
    ci_level: float,
    q: float,
    bootstrap_seed: int | None,
) -> list[dict]:
    cells = _build_cells_for_metric(attacked_records, extractor)
    if not cells:
        return []
    rows = _stats_per_cell(
        cells,
        n_resamples=n_resamples,
        ci_level=ci_level,
        bootstrap_seed=bootstrap_seed,
    )
    _apply_bh(rows, q=q)
    if metric_name == "refusal":
        _annotate_attacked_rate(rows, cells)
    return rows


def build_cot_table(
    runs_dir: Path,
    out_path: Path,
    *,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    q: float = 0.05,
    bootstrap_seed: int | None = 0,
) -> int:
    """Build the CoT-axis stats table from CoT-enriched records.

    Reads ``runs/main/<mode>/records.jsonl`` (records must already have
    cot_drift_score + faith/halluc/refusal pairs, e.g. via
    scripts/backfill_cot_metrics.py). Emits a sibling LaTeX table.
    """
    runs = _load_runs_dir(runs_dir)
    attacked_records: list[dict] = []
    for mode in NON_NOISE_MODES:
        attacked_records.extend(runs[mode])
    if not attacked_records:
        sys.stderr.write("[build_cot_table] no non-noise records — abort\n")
        return 1

    rows_by_metric: dict[str, list[dict]] = {
        m: _build_cot_metric_rows(
            m,
            ex,
            attacked_records,
            n_resamples=n_resamples,
            ci_level=ci_level,
            q=q,
            bootstrap_seed=bootstrap_seed,
        )
        for m, ex in COT_METRICS
    }

    if not any(rows_by_metric.values()):
        sys.stderr.write("[build_cot_table] no records contained CoT metrics — skip\n")
        return 1

    _emit_cot_latex(rows_by_metric, out_path)
    print(
        f"[build_cot_table] wrote CoT table → {out_path}"
        f" (drift n={len(rows_by_metric['drift'])},"
        f" faith n={len(rows_by_metric['faith'])},"
        f" halluc n={len(rows_by_metric['halluc'])},"
        f" refusal n={len(rows_by_metric['refusal'])})"
    )
    return 0


def build_stats_table(
    runs_dir: Path,
    out_path: Path,
    *,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    q: float = 0.05,
    bootstrap_seed: int | None = 0,
) -> int:
    runs = _load_runs_dir(runs_dir)
    if not runs["noise"]:
        sys.stderr.write(
            f"[build_stats_table] no noise records under {runs_dir / 'noise'} — abort\n"
        )
        return 1
    noise_baseline = _noise_baseline(runs["noise"])
    attacked_records: list[dict] = []
    for mode in NON_NOISE_MODES:
        attacked_records.extend(runs[mode])
    if not attacked_records:
        sys.stderr.write("[build_stats_table] no non-noise records — abort\n")
        return 1
    cells = _build_cells(attacked_records, noise_baseline)
    rows = _stats_per_cell(
        cells,
        n_resamples=n_resamples,
        ci_level=ci_level,
        bootstrap_seed=bootstrap_seed,
    )
    _apply_bh(rows, q=q)
    _emit_latex(rows, out_path)
    print(
        f"[build_stats_table] wrote {len(rows)} rows → {out_path}"
        f" (n_resamples={n_resamples}, ci={ci_level}, q={q},"
        f" bootstrap_seed={bootstrap_seed})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path, default=Path("runs/main"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("paper/tables/main_benchmark.tex"),
    )
    p.add_argument("--n-resamples", type=int, default=10_000)
    p.add_argument("--ci-level", type=float, default=0.95)
    p.add_argument("--fdr-q", type=float, default=0.05)
    p.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="RNG seed for bootstrap CI; pin for deterministic table output",
    )
    p.add_argument(
        "--cot-out",
        type=Path,
        default=None,
        help=(
            "Optional sibling table for CoT metrics (drift, faith Δ, halluc Δ, "
            "refusal rate). Skipped if records contain no CoT fields."
        ),
    )
    args = p.parse_args(argv)
    rc = build_stats_table(
        args.runs_dir,
        args.out,
        n_resamples=args.n_resamples,
        ci_level=args.ci_level,
        q=args.fdr_q,
        bootstrap_seed=args.bootstrap_seed,
    )
    if rc == 0 and args.cot_out is not None:
        # Soft-fail: missing CoT fields produce a skip note, not a hard error.
        cot_rc = build_cot_table(
            args.runs_dir,
            args.cot_out,
            n_resamples=args.n_resamples,
            ci_level=args.ci_level,
            q=args.fdr_q,
            bootstrap_seed=args.bootstrap_seed,
        )
        if cot_rc != 0:
            sys.stderr.write("[build_stats_table] CoT table skipped (no CoT-enriched records)\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
