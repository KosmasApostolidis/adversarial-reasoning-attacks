"""End-to-end tests for ``scripts/build_stats_table.py`` on synthetic JSONL."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_stats_table.py"


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
    rc = bst.build_stats_table(
        runs_dir, out_path, n_resamples=200, ci_level=0.95, q=0.05
    )
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
    rc = bst.build_stats_table(
        runs_dir, out_path, n_resamples=200, ci_level=0.95, q=0.05
    )
    assert rc == 0
    content = out_path.read_text(encoding="utf-8")
    assert "$^{*}$" in content, "Large attacked-vs-noise gap must survive BH at q=0.05"


def test_build_stats_table_aborts_when_no_noise_records(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_jsonl(runs_dir / "pgd" / "records.jsonl", [])
    rc = bst.build_stats_table(runs_dir, tmp_path / "x.tex", n_resamples=50)
    assert rc == 1
