"""Tests for tasks.loader — task config loading + dataset resolution.

Covers the synthetic image fallback, the file-based PNG/JPG split, and
the BHI ``.npy`` cv-fold path (using a tmp fold directory of synthetic
arrays so we don't depend on the real ProstateX cohort).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from adversarial_reasoning.tasks.loader import (
    TaskSample,
    _best_slice_index,
    _bhi_root,
    _normalize_slice_to_uint8,
    _stable_seed,
    load_task,
    load_task_config,
    load_task_sample,
)


def _write_tasks_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "tasks.yaml"
    p.write_text(
        textwrap.dedent(
            """
            tasks:
              synth_task:
                dataset: synthetic
                dataset_split:
                  dev: 3
                  test: 0
                prompt_template: |
                  Synthetic task prompt.

              file_task:
                dataset: file_dataset
                dataset_split:
                  dev: 2
                prompt_template: file prompt

              bhi_task:
                dataset: prostatex_bhi
                dataset_split:
                  dev: 2
                bhi_split_to_fold:
                  dev: 1
                prompt_template: bhi prompt
            """
        ).strip()
    )
    return p


# ---------- load_task_config ----------


def test_load_task_config_returns_dict(tmp_path: Path) -> None:
    cfg = _write_tasks_yaml(tmp_path)
    out = load_task_config("synth_task", config_path=cfg)
    assert out["dataset"] == "synthetic"
    assert "prompt_template" in out


def test_load_task_config_missing_task_raises(tmp_path: Path) -> None:
    cfg = _write_tasks_yaml(tmp_path)
    with pytest.raises(KeyError, match="not in"):
        load_task_config("nope", config_path=cfg)


# ---------- load_task synthetic ----------


def test_load_task_synthetic_yields_n_samples(tmp_path: Path) -> None:
    cfg = _write_tasks_yaml(tmp_path)
    samples = list(load_task("synth_task", split="dev", synthetic=True, config_path=cfg))
    assert len(samples) == 3
    assert all(isinstance(s, TaskSample) for s in samples)
    assert samples[0].image.size == (512, 512)
    assert samples[0].prompt == "Synthetic task prompt."


def test_load_task_synthetic_split_with_zero_count(tmp_path: Path) -> None:
    cfg = _write_tasks_yaml(tmp_path)
    samples = list(load_task("synth_task", split="test", synthetic=True, config_path=cfg))
    assert samples == []


def test_load_task_n_override(tmp_path: Path) -> None:
    cfg = _write_tasks_yaml(tmp_path)
    samples = list(load_task("synth_task", split="dev", n=1, synthetic=True, config_path=cfg))
    assert len(samples) == 1


def test_load_task_n_zero_returns_empty(tmp_path: Path) -> None:
    cfg = _write_tasks_yaml(tmp_path)
    samples = list(load_task("synth_task", split="dev", n=0, synthetic=True, config_path=cfg))
    assert samples == []


# ---------- load_task file-based dataset ----------


def test_load_task_file_dataset_with_pngs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_tasks_yaml(tmp_path)
    # Create a fake PNG split under data/file_dataset/dev relative to cwd.
    monkeypatch.chdir(tmp_path)
    split_dir = tmp_path / "data" / "file_dataset" / "dev"
    split_dir.mkdir(parents=True)
    from PIL import Image as _Image

    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    _Image.fromarray(arr).save(split_dir / "img1.png")
    _Image.fromarray(arr).save(split_dir / "img2.png")

    samples = list(load_task("file_task", split="dev", config_path=cfg))
    assert len(samples) == 2
    assert samples[0].sample_id == "img1"


def test_load_task_file_dataset_missing_dir_falls_back_to_synthetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _write_tasks_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)  # no data/file_dataset dir
    samples = list(load_task("file_task", split="dev", config_path=cfg))
    assert len(samples) == 2
    # All synthetic since nothing on disk.
    assert samples[0].sample_id.startswith("synthetic_")


# ---------- BHI .npy loader ----------


def _write_bhi_fold(root: Path, fold: int, n: int = 2) -> None:
    fold_dir = root / f"fold_{fold}"
    fold_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    # X shape: (n, depth, H, W, channels=1) — 5D path
    x = rng.uniform(-1, 1, size=(n, 4, 16, 16, 1)).astype(np.float32)
    y = (rng.uniform(0, 1, size=(n, 4, 16, 16)) > 0.7).astype(np.uint8)
    np.save(fold_dir / f"fold_{fold}_X_val_3D.npy", x)
    np.save(fold_dir / f"fold_{fold}_y_val_3D.npy", y)


def test_load_task_bhi_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_tasks_yaml(tmp_path)
    bhi_root = tmp_path / "cv_folds"
    _write_bhi_fold(bhi_root, fold=1, n=2)
    monkeypatch.setenv("AR_PROSTATEX_BHI_ROOT", str(bhi_root))

    samples = list(load_task("bhi_task", split="dev", config_path=cfg))
    assert len(samples) == 2
    assert samples[0].sample_id.startswith("bhi_f1_val_p")


def test_load_task_bhi_missing_x_falls_back_to_synthetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _write_tasks_yaml(tmp_path)
    bhi_root = tmp_path / "cv_folds_empty"
    bhi_root.mkdir()
    monkeypatch.setenv("AR_PROSTATEX_BHI_ROOT", str(bhi_root))
    samples = list(load_task("bhi_task", split="dev", config_path=cfg))
    # No on-disk files — should fully synthetic.
    assert len(samples) == 2
    assert all(s.sample_id.startswith("synthetic_") for s in samples)


# ---------- helpers ----------


def test_bhi_root_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AR_PROSTATEX_BHI_ROOT", raising=False)
    assert _bhi_root() == Path("data/prostatex/processed/cv_folds")


def test_bhi_root_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AR_PROSTATEX_BHI_ROOT", "/tmp/somewhere")
    assert _bhi_root() == Path("/tmp/somewhere")


def test_normalize_constant_slice_returns_zeros() -> None:
    s = np.full((4, 4), 7.0, dtype=np.float32)
    out = _normalize_slice_to_uint8(s)
    assert out.dtype == np.uint8
    assert (out == 0).all()


def test_normalize_varying_slice_stretches_to_full_range() -> None:
    s = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    out = _normalize_slice_to_uint8(s)
    assert out.min() == 0
    assert out.max() == 255


def test_best_slice_index_returns_max_lesion_slice() -> None:
    mask = np.zeros((5, 4, 4), dtype=np.uint8)
    mask[2] = 1
    assert _best_slice_index(mask) == 2


def test_best_slice_index_no_lesions_returns_middle() -> None:
    mask = np.zeros((5, 4, 4), dtype=np.uint8)
    assert _best_slice_index(mask) == 2


def test_best_slice_index_4d_squeezes_last_axis() -> None:
    mask = np.zeros((3, 4, 4, 1), dtype=np.uint8)
    mask[1, ..., 0] = 1
    assert _best_slice_index(mask) == 1


# ---------- load_task_sample ----------


def test_load_task_sample_returns_index(tmp_path: Path) -> None:
    cfg = _write_tasks_yaml(tmp_path)
    s = load_task_sample("synth_task", index=0, split="dev", synthetic=True, config_path=cfg)
    assert isinstance(s, TaskSample)


def test_load_task_sample_high_index_returns_synthetic(tmp_path: Path) -> None:
    """load_task_sample passes n=index+1 which always yields enough synthetic
    samples — IndexError is unreachable through this entrypoint, but the
    function still returns a TaskSample at the requested index."""
    cfg = _write_tasks_yaml(tmp_path)
    s = load_task_sample("synth_task", index=5, split="dev", synthetic=True, config_path=cfg)
    assert isinstance(s, TaskSample)


# ---------- _stable_seed ----------


def test_stable_seed_is_deterministic_within_process() -> None:
    """Identical inputs must yield identical seeds within one Python run."""
    a = _stable_seed("synth_task", "dev", 0)
    b = _stable_seed("synth_task", "dev", 0)
    assert a == b


def test_stable_seed_distinct_inputs_produce_distinct_seeds() -> None:
    """Trivial collision check — different keys should rarely collide."""
    seeds = {
        _stable_seed("a", "dev", 0),
        _stable_seed("a", "dev", 1),
        _stable_seed("a", "test", 0),
        _stable_seed("b", "dev", 0),
    }
    assert len(seeds) == 4


def test_stable_seed_does_not_depend_on_pythonhashseed(tmp_path: Path) -> None:
    """Synthetic-image seed must be byte-stable across interpreter runs.

    Regression for ``hash((task_id, split, i)) & 0xFFFFFFFF`` which depended
    on the per-process ``PYTHONHASHSEED`` randomisation.
    """
    import subprocess
    import sys

    code = (
        "from adversarial_reasoning.tasks.loader import _stable_seed; "
        "print(_stable_seed('t', 'dev', 0))"
    )
    out_a = subprocess.check_output(
        [sys.executable, "-c", code],
        env={"PYTHONHASHSEED": "1", "PATH": ""},
        text=True,
    ).strip()
    out_b = subprocess.check_output(
        [sys.executable, "-c", code],
        env={"PYTHONHASHSEED": "random", "PATH": ""},
        text=True,
    ).strip()
    assert out_a == out_b
