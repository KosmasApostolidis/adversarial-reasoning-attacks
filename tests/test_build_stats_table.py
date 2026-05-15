"""End-to-end tests for ``scripts/diagnostics/build_stats_table.py`` on synthetic JSONL."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "diagnostics" / "build_stats_table.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_stats_table", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bst = _load_module()


def _record(
    *,
    model_key: str,
    task_id: str,
    attack_mode: str,
    attack_name: str,
    epsilon: float,
    seed: int,
    sample_id: str,
    edit_distance: float,
) -> dict:
    return {
        "model_key": model_key,
        "task_id": task_id,
        "sample_id": sample_id,
        "attack_name": attack_name,
        "attack_mode": attack_mode,
        "epsilon": epsilon,
        "seed": seed,
        "edit_distance_norm": edit_distance,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _populate_runs_dir(root: Path, *, attacked_offset: float = 0.4) -> None:
    """Synthetic noise + pgd legs with a clear attacked > noise gap."""
    rng = np.random.default_rng(42)
    noise_rows: list[dict] = []
    pgd_rows: list[dict] = []
    for seed in range(5):
        for sample_idx in range(8):
            sample_id = f"s{sample_idx:02d}"
            noise_dist = float(rng.uniform(0.0, 0.2))
            noise_rows.append(
                _record(
                    model_key="qwen2_5_vl_7b",
                    task_id="prostate_mri_workup",
                    attack_mode="noise",
                    attack_name="pgd_linf",
                    epsilon=0.0157,
                    seed=seed,
                    sample_id=sample_id,
                    edit_distance=noise_dist,
                )
            )
            pgd_rows.append(
                _record(
                    model_key="qwen2_5_vl_7b",
                    task_id="prostate_mri_workup",
                    attack_mode="pgd",
                    attack_name="pgd_linf",
                    epsilon=0.0157,
                    seed=seed,
                    sample_id=sample_id,
                    edit_distance=noise_dist + attacked_offset,
                )
            )
    _write_jsonl(root / "noise" / "records.jsonl", noise_rows)
    _write_jsonl(root / "pgd" / "records.jsonl", pgd_rows)


def test_pair_key_is_stable_across_models_and_tasks() -> None:
    rec = {"model_key": "m", "task_id": "t", "seed": 3, "sample_id": "s7"}
    assert bst._pair_key(rec) == ("m", "t", 3, "s7")


def test_record_attack_mode_prefers_attack_mode_field() -> None:
    rec = {"attack_mode": "pgd", "attack_name": "pgd_linf"}
    assert bst._record_attack_mode(rec) == "pgd"
    assert bst._record_attack_mode({"attack_name": "pgd_linf"}) == "pgd_linf"
    assert bst._record_attack_mode({}) == "unknown"


def test_noise_baseline_averages_across_eps() -> None:
    records = [
        _record(
            model_key="m",
            task_id="t",
            attack_mode="noise",
            attack_name="pgd_linf",
            epsilon=0.0078,
            seed=0,
            sample_id="s0",
            edit_distance=0.10,
        ),
        _record(
            model_key="m",
            task_id="t",
            attack_mode="noise",
            attack_name="pgd_linf",
            epsilon=0.0157,
            seed=0,
            sample_id="s0",
            edit_distance=0.30,
        ),
    ]
    baseline = bst._noise_baseline(records)
    assert baseline == {("m", "t", 0, "s0"): pytest.approx(0.20)}


def test_build_stats_table_writes_booktabs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _populate_runs_dir(runs_dir)
    out_path = tmp_path / "paper" / "tables" / "main_benchmark.tex"
    rc = bst.build_stats_table(runs_dir, out_path, n_resamples=200, ci_level=0.95, q=0.05)
    assert rc == 0
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in content
    assert "\\toprule" in content
    assert "\\bottomrule" in content
    assert "PGD-L$_\\infty$" in content


def test_build_stats_table_marks_significant_cells(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _populate_runs_dir(runs_dir, attacked_offset=0.5)
    out_path = tmp_path / "out.tex"
    rc = bst.build_stats_table(runs_dir, out_path, n_resamples=200, ci_level=0.95, q=0.05)
    assert rc == 0
    content = out_path.read_text(encoding="utf-8")
    assert "$^{*}$" in content, "Large attacked-vs-noise gap must survive BH at q=0.05"


def test_build_stats_table_is_deterministic_with_pinned_seed(tmp_path: Path) -> None:
    """Two runs with the same ``bootstrap_seed`` must emit byte-identical
    output — guards against silent CI drift across paper recompiles."""
    runs_dir = tmp_path / "runs"
    _populate_runs_dir(runs_dir)
    out_a = tmp_path / "a.tex"
    out_b = tmp_path / "b.tex"
    bst.build_stats_table(runs_dir, out_a, n_resamples=200, bootstrap_seed=0)
    bst.build_stats_table(runs_dir, out_b, n_resamples=200, bootstrap_seed=0)
    assert out_a.read_bytes() == out_b.read_bytes(), (
        "bootstrap_seed=0 must produce identical output across runs"
    )


def _build_zero_delta_rows() -> list[dict]:
    """Build noise+pgd rows that force scipy.wilcoxon onto a degenerate path."""
    rows: list[dict] = []
    for seed in range(5):
        for sample_idx in range(4):
            for mode, name in [("noise", "noise"), ("pgd", "pgd_linf")]:
                rows.append(
                    _record(
                        model_key="m", task_id="t",
                        attack_mode=mode, attack_name=name,
                        epsilon=0.0157, seed=seed,
                        sample_id=f"s{sample_idx}", edit_distance=0.1,
                    )
                )
    return rows


def test_stats_rows_carry_pvalue_status(tmp_path: Path) -> None:
    """Each row must report whether the Wilcoxon p-value was computed
    cleanly or fell back. Distinguishes 'truly non-significant' from
    'computation failed' — both previously collapsed to pvalue=1.0."""
    # Force all-zero deltas → Wilcoxon raises ValueError ("zero-method='wilcox' ... all zero")
    rows = _build_zero_delta_rows()
    runs_dir = tmp_path / "runs"
    _write_jsonl(
        runs_dir / "noise" / "records.jsonl", [r for r in rows if r["attack_mode"] == "noise"]
    )
    _write_jsonl(runs_dir / "pgd" / "records.jsonl", [r for r in rows if r["attack_mode"] == "pgd"])
    # Build cells + stats directly to inspect row dicts (LaTeX hides the column).
    runs = bst._load_runs_dir(runs_dir)
    baseline = bst._noise_baseline(runs["noise"])
    cells = bst._build_cells(runs["pgd"], baseline)
    stat_rows = bst._stats_per_cell(cells, n_resamples=100, ci_level=0.95, bootstrap_seed=0)
    assert stat_rows, "should have at least one stats row"
    for row in stat_rows:
        assert "pvalue_status" in row, "missing pvalue_status sentinel"
        assert row["pvalue_status"] in {
            "ok",
            "valuerror",
            "nan",
        }, f"unexpected status {row['pvalue_status']!r}"
    # All-zero deltas force scipy.wilcoxon into a degenerate path
    # (NaN p-value, or ValueError depending on scipy version). Either
    # is fine — the contract is that it MUST NOT silently report ok.
    assert any(row["pvalue_status"] in {"valuerror", "nan"} for row in stat_rows), (
        "all-zero deltas must surface as a non-ok status, not silently ok"
    )


def test_build_stats_table_aborts_when_no_noise_records(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_jsonl(runs_dir / "pgd" / "records.jsonl", [])
    rc = bst.build_stats_table(runs_dir, tmp_path / "x.tex", n_resamples=50)
    assert rc == 1


# -------- CoT extension --------


def _cot_record(
    *,
    sample_id: str,
    seed: int,
    drift: float | None = 0.0,
    faith_b: float | None = 0.9,
    faith_a: float | None = 0.5,
    halluc_b: float | None = 0.0,
    halluc_a: float | None = 0.4,
    refusal_b: bool | None = False,
    refusal_a: bool | None = True,
) -> dict:
    rec = _record(
        model_key="qwen2_5_vl_7b",
        task_id="prostate_mri_workup",
        attack_mode="pgd",
        attack_name="pgd_linf",
        epsilon=0.0157,
        seed=seed,
        sample_id=sample_id,
        edit_distance=0.4,
    )
    if drift is not None:
        rec["cot_drift_score"] = drift
    if faith_b is not None:
        rec["cot_faithfulness_benign"] = faith_b
    if faith_a is not None:
        rec["cot_faithfulness_attacked"] = faith_a
    if halluc_b is not None:
        rec["cot_hallucination_benign"] = halluc_b
    if halluc_a is not None:
        rec["cot_hallucination_attacked"] = halluc_a
    if refusal_b is not None:
        rec["cot_refusal_benign"] = refusal_b
    if refusal_a is not None:
        rec["cot_refusal_attacked"] = refusal_a
    return rec


def test_extract_drift_pairs_zero_with_score() -> None:
    assert bst._extract_drift({"cot_drift_score": 0.42}) == (0.0, 0.42)
    assert bst._extract_drift({"cot_drift_score": None}) is None
    assert bst._extract_drift({}) is None


def test_extract_faith_requires_both_sides() -> None:
    assert bst._extract_faith(
        {"cot_faithfulness_benign": 0.8, "cot_faithfulness_attacked": 0.3}
    ) == (0.8, 0.3)
    assert bst._extract_faith({"cot_faithfulness_benign": 0.8}) is None


def test_extract_refusal_coerces_bool_to_float() -> None:
    pair = bst._extract_refusal({"cot_refusal_benign": False, "cot_refusal_attacked": True})
    assert pair == (0.0, 1.0)


def test_build_cells_for_metric_skips_records_missing_metric() -> None:
    rows = [
        _cot_record(sample_id="s00", seed=0, drift=0.5),
        _cot_record(sample_id="s01", seed=0, drift=None),  # skipped
        _cot_record(sample_id="s02", seed=1, drift=0.7),
    ]
    cells = bst._build_cells_for_metric(rows, bst._extract_drift)
    # All three records share the same cell; only the two with drift land.
    only_cell = next(iter(cells.values()))
    assert only_cell["benign"] == [0.0, 0.0]
    assert only_cell["attacked"] == [0.5, 0.7]


def test_build_cot_table_emits_eight_column_table(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    rows: list[dict] = []
    for seed in range(5):
        for i in range(6):
            rows.append(_cot_record(sample_id=f"s{i:02d}", seed=seed))
    _write_jsonl(runs_dir / "pgd" / "records.jsonl", rows)
    out_path = tmp_path / "paper" / "tables" / "cot_benchmark.tex"
    rc = bst.build_cot_table(runs_dir, out_path, n_resamples=100, ci_level=0.95, q=0.05)
    assert rc == 0
    content = out_path.read_text(encoding="utf-8")
    assert "\\begin{tabular}{llllllll}" in content
    assert "median CoT-drift" in content
    assert "$\\Delta$faith" in content
    assert "$\\Delta$hall" in content
    assert "refusal rate" in content
    # Refusal rate column rendered as "100.0\%" since attacked_a=True every row.
    assert "100.0\\%" in content


def test_build_cot_table_skips_when_no_cot_fields(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    plain_rows: list[dict] = []
    for seed in range(3):
        for i in range(4):
            plain_rows.append(
                _record(
                    model_key="m",
                    task_id="t",
                    attack_mode="pgd",
                    attack_name="pgd_linf",
                    epsilon=0.0157,
                    seed=seed,
                    sample_id=f"s{i}",
                    edit_distance=0.3,
                )
            )
    _write_jsonl(runs_dir / "pgd" / "records.jsonl", plain_rows)
    rc = bst.build_cot_table(runs_dir, tmp_path / "x.tex", n_resamples=50)
    assert rc == 1, "must signal skip when records lack CoT metrics"


def test_main_with_cot_out_writes_both_tables(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    # Need both noise + pgd legs for the main table; pgd records carry CoT.
    noise_rows: list[dict] = []
    pgd_rows: list[dict] = []
    for seed in range(5):
        for i in range(6):
            sid = f"s{i:02d}"
            noise_rows.append(
                _record(
                    model_key="qwen2_5_vl_7b",
                    task_id="prostate_mri_workup",
                    attack_mode="noise",
                    attack_name="pgd_linf",
                    epsilon=0.0157,
                    seed=seed,
                    sample_id=sid,
                    edit_distance=0.05,
                )
            )
            pgd_rows.append(_cot_record(sample_id=sid, seed=seed))
    _write_jsonl(runs_dir / "noise" / "records.jsonl", noise_rows)
    _write_jsonl(runs_dir / "pgd" / "records.jsonl", pgd_rows)
    main_out = tmp_path / "main.tex"
    cot_out = tmp_path / "cot.tex"
    rc = bst.main(
        [
            "--runs-dir", str(runs_dir),
            "--out", str(main_out),
            "--cot-out", str(cot_out),
            "--n-resamples", "100",
        ]
    )
    assert rc == 0
    assert main_out.exists()
    assert cot_out.exists()
    assert "$\\Delta$faith" in cot_out.read_text(encoding="utf-8")
