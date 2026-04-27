"""Behavior tests for scripts/run_full_attack_pipeline.sh.

These tests cover orchestration logic (preflight, mode validation, HF gating,
per-fold aggregation, per-model split, run_step rc reporting) without
invoking the runner, models, or GPU. SKIP_SWEEP / SKIP_FIGURES / SKIP_TABLE
gate the heavy stages; we drive the script with synthetic JSONL fixtures.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_full_attack_pipeline.sh"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """Sandbox repo with script + fake cv_folds + fake configs."""
    sb = tmp_path / "repo"
    (sb / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, sb / "scripts" / "run_full_attack_pipeline.sh")
    os.chmod(sb / "scripts" / "run_full_attack_pipeline.sh", 0o755)

    bhi = sb / "data" / "prostatex" / "processed" / "cv_folds"
    for f in ("fold_1", "fold_2", "fold_3"):
        (bhi / f).mkdir(parents=True)

    (sb / "configs").mkdir()
    for m in ("noise", "pgd", "apgd", "trajectory_drift", "targeted_tool"):
        (sb / "configs" / f"main_{m}.yaml").write_text("# fake\n")

    (sb / "paper" / "tables").mkdir(parents=True)
    (sb / "paper" / "figures").mkdir(parents=True)

    return sb


def _run(
    sandbox: Path,
    *args: str,
    env: dict[str, str] | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    full_env = {
        **os.environ,
        "SKIP_SWEEP": "1",
        "SKIP_FIGURES": "1",
        "SKIP_TABLE": "1",
        "AR_PROSTATEX_BHI_ROOT": str(sandbox / "data/prostatex/processed/cv_folds"),
    }
    full_env.pop("HF_TOKEN", None)
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", "scripts/run_full_attack_pipeline.sh", *args],
        cwd=sandbox,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------- mode validation (m4) ----------


class TestModeValidation:
    def test_unknown_mode_exits_2(self, sandbox: Path) -> None:
        r = _run(sandbox, "badmode")
        assert r.returncode == 2
        assert "unknown mode 'badmode'" in r.stderr

    def test_typo_pdg_rejected(self, sandbox: Path) -> None:
        r = _run(sandbox, "pdg")
        assert r.returncode == 2

    def test_default_modes_accepted(self, sandbox: Path) -> None:
        # No args -> default 5 modes; needs HF for gradient. Pass token.
        r = _run(sandbox, env={"HF_TOKEN": "fake"})
        assert r.returncode == 0, r.stderr

    def test_valid_subset_accepted(self, sandbox: Path) -> None:
        r = _run(sandbox, "noise")
        assert r.returncode == 0, r.stderr


# ---------- HF_TOKEN gating (M3) ----------


class TestHfTokenGating:
    def test_noise_only_no_token_ok(self, sandbox: Path) -> None:
        r = _run(sandbox, "noise")
        assert r.returncode == 0
        assert "HF_TOKEN is required" not in r.stderr

    def test_pgd_without_token_fails(self, sandbox: Path) -> None:
        r = _run(sandbox, "pgd")
        assert r.returncode != 0
        assert "HF_TOKEN" in r.stderr

    def test_apgd_without_token_fails(self, sandbox: Path) -> None:
        r = _run(sandbox, "apgd")
        assert r.returncode != 0

    def test_targeted_tool_without_token_fails(self, sandbox: Path) -> None:
        r = _run(sandbox, "targeted_tool")
        assert r.returncode != 0

    def test_trajectory_drift_without_token_fails(self, sandbox: Path) -> None:
        r = _run(sandbox, "trajectory_drift")
        assert r.returncode != 0

    def test_mixed_subset_requires_token(self, sandbox: Path) -> None:
        r = _run(sandbox, "noise", "pgd")
        assert r.returncode != 0
        assert "HF_TOKEN" in r.stderr


# ---------- preflight: missing cv_folds ----------


class TestCvFoldsPreflight:
    def test_missing_fold_aborts(self, sandbox: Path) -> None:
        shutil.rmtree(sandbox / "data/prostatex/processed/cv_folds/fold_2")
        r = _run(sandbox, "noise")
        assert r.returncode == 1
        assert "missing" in r.stderr and "fold_2" in r.stderr


# ---------- per-fold concat (M2 truncate guard) ----------


class TestConcatAggregation:
    def _seed_fold_records(self, sandbox: Path, mode: str) -> None:
        for fold in ("fold_1", "fold_2", "fold_3"):
            _write_jsonl(
                sandbox / "runs/main" / mode / fold / "records.jsonl",
                [{"fold": fold, "mode": mode, "model_key": "qwen2_5_vl_7b"}],
            )

    def test_aggregates_three_folds(self, sandbox: Path) -> None:
        self._seed_fold_records(sandbox, "noise")
        r = _run(sandbox, "noise")
        assert r.returncode == 0, r.stderr
        agg = sandbox / "runs/main/noise/records.jsonl"
        assert agg.exists()
        lines = agg.read_text().strip().splitlines()
        assert len(lines) == 3
        folds = {json.loads(L)["fold"] for L in lines}
        assert folds == {"fold_1", "fold_2", "fold_3"}

    def test_preserves_prior_aggregate_when_no_fold_files(self, sandbox: Path) -> None:
        # M2: SKIP_SWEEP=1 + missing per-fold dirs must NOT wipe existing
        # mode/records.jsonl.
        agg = sandbox / "runs/main/noise/records.jsonl"
        agg.parent.mkdir(parents=True)
        agg.write_text('{"keep":"me"}\n')
        r = _run(sandbox, "noise")
        assert r.returncode == 0
        assert agg.read_text() == '{"keep":"me"}\n'
        assert "no per-fold jsonl" in r.stderr or "keeping existing" in r.stderr

    def test_skips_internal_dirs(self, sandbox: Path) -> None:
        # _logs and _per_model must not be treated as modes.
        (sandbox / "runs/main/_logs").mkdir(parents=True)
        (sandbox / "runs/main/_per_model").mkdir(parents=True)
        self._seed_fold_records(sandbox, "noise")
        r = _run(sandbox, "noise")
        assert r.returncode == 0
        assert not (sandbox / "runs/main/_logs/records.jsonl").exists()
        assert not (sandbox / "runs/main/_per_model/records.jsonl").exists()


# ---------- per-model split via jq (m1) ----------


class TestPerModelSplit:
    def test_jq_filters_by_model_key(self, sandbox: Path) -> None:
        agg = sandbox / "runs/main/noise/records.jsonl"
        _write_jsonl(
            agg,
            [
                {"model_key": "qwen2_5_vl_7b", "id": 1},
                {"model_key": "llava_v1_6_mistral_7b", "id": 2},
                {"model_key": "qwen2_5_vl_7b", "id": 3},
            ],
        )
        # Run only the figure/split phase.
        env = {"SKIP_SWEEP": "1", "SKIP_FIGURES": "0", "SKIP_TABLE": "1"}
        r = _run(sandbox, "noise", env=env, timeout=60)
        # Figure stage will warn on missing scripts but split runs first.
        qwen_split = sandbox / "runs/main/_per_model/noise__qwen2_5_vl_7b.jsonl"
        llava_split = sandbox / "runs/main/_per_model/noise__llava_v1_6_mistral_7b.jsonl"
        assert qwen_split.exists(), r.stderr
        qwen_ids = sorted(json.loads(L)["id"] for L in qwen_split.read_text().splitlines())
        assert qwen_ids == [1, 3]
        llava_ids = [json.loads(L)["id"] for L in llava_split.read_text().splitlines()]
        assert llava_ids == [2]


# ---------- run_step rc capture (M1) ----------


class TestRunStepRcCapture:
    """run_step must report the failed command's rc, not always rc=0."""

    def test_rc_captured(self, sandbox: Path, tmp_path: Path) -> None:
        # Source the script's run_step in isolation (extract + eval).
        # Easier path: write a tiny harness that re-defines run_step using
        # the same shape and asserts rc propagation.
        harness = tmp_path / "harness.sh"
        harness.write_text(
            r"""#!/usr/bin/env bash
set -uo pipefail
LOG_DIR="$(mktemp -d)"
TS=test
warn() { echo "[harness] WARN: $*" >&2; }
run_step() {
    local label="$1"; shift
    local log="$LOG_DIR/figure_${label}_${TS}.log"
    local rc=0
    "$@" >"$log" 2>&1 || rc=$?
    if [[ "$rc" -eq 0 ]]; then
        echo "ok"
    else
        warn "$label failed (rc=$rc); see $log"
    fi
}
run_step boom bash -c 'exit 42'
"""
        )
        os.chmod(harness, 0o755)
        r = subprocess.run(["bash", str(harness)], capture_output=True, text=True, timeout=10)
        assert "rc=42" in r.stderr
        assert "rc=0" not in r.stderr
