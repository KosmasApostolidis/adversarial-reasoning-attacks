"""Smoke tests for the two sanity scripts (null distribution + confusion matrix).

These tests do not exercise plotting; they unit-test the data extraction
helpers that drive the figures.
"""

from __future__ import annotations

from scripts.diagnostics.cot_confusion_matrix import _drift, _flip, confusion
from scripts.diagnostics.cot_null_distribution import (
    _attack_drift_distribution,
    _drift_floor,
    _is_null_row,
)


def _row(*, attack="pgd", epsilon=0.0314, drift=None, ed=0.0, bts=None, ats=None,
         benign_trace="", attacked_trace=""):
    return {
        "model_key": "qwen2_5_vl_7b",
        "task_id": "t",
        "sample_id": "s001",
        "attack_name": attack,
        "epsilon": epsilon,
        "edit_distance_norm": ed,
        "cot_drift_score": drift,
        "benign": {
            "tool_sequence": bts or ["a", "b"],
            "reasoning_trace": benign_trace,
        },
        "attacked": {
            "tool_sequence": ats or ["a", "b"],
            "reasoning_trace": attacked_trace,
        },
    }


# -------- cot_confusion_matrix --------


def test_flip_detected_via_edit_distance() -> None:
    assert _flip(_row(ed=0.5))


def test_flip_detected_via_first_step_change() -> None:
    assert _flip(_row(ed=0.0, bts=["a", "b"], ats=["x", "b"]))


def test_no_flip_when_first_step_same() -> None:
    assert not _flip(_row(ed=0.0, bts=["a", "b"], ats=["a", "b"]))


def test_drift_threshold() -> None:
    assert _drift({"cot_drift_score": 0.4}, 0.3)
    assert not _drift({"cot_drift_score": 0.2}, 0.3)
    assert not _drift({}, 0.3)


def test_confusion_matrix_counts_each_quadrant() -> None:
    rows = [
        _row(drift=0.0, ed=0.0, bts=["a"], ats=["a"]),       # robust
        _row(drift=0.0, ed=0.5, bts=["a"], ats=["x"]),       # flip only
        _row(drift=0.5, ed=0.0, bts=["a"], ats=["a"]),       # silent drift
        _row(drift=0.5, ed=0.5, bts=["a"], ats=["x"]),       # both
        _row(drift=0.5, ed=0.5, bts=["a"], ats=["x"]),       # both (dup)
    ]
    cm = confusion(rows, threshold=0.3)
    assert cm[0, 0] == 1  # robust
    assert cm[0, 1] == 1  # flip only
    assert cm[1, 0] == 1  # silent drift
    assert cm[1, 1] == 2  # both


def test_confusion_skips_rows_without_drift_field() -> None:
    rows = [_row(drift=None)]
    rows[0].pop("cot_drift_score")  # truly absent
    cm = confusion(rows, threshold=0.3)
    assert cm.sum() == 0


def test_confusion_skips_rows_with_null_drift_value() -> None:
    """JSON null -> Python None -> field present but unscored."""
    rows = [_row(drift=None)]  # cot_drift_score key present, value None
    assert "cot_drift_score" in rows[0]
    assert rows[0]["cot_drift_score"] is None
    cm = confusion(rows, threshold=0.3)
    assert cm.sum() == 0


# -------- cot_null_distribution --------


def test_is_null_row_detects_attack_name_null() -> None:
    assert _is_null_row({"attack_name": "null", "epsilon": 0.0314})


def test_is_null_row_detects_zero_epsilon() -> None:
    assert _is_null_row({"attack_name": "pgd", "epsilon": 0.0})


def test_is_null_row_rejects_real_attack() -> None:
    assert not _is_null_row({"attack_name": "pgd", "epsilon": 0.0314})


def test_attack_drift_distribution_excludes_null_rows() -> None:
    records = [
        _row(attack="null", epsilon=0.0, drift=0.05),
        _row(attack="pgd", epsilon=0.0314, drift=0.6),
        _row(attack="pgd", epsilon=0.0314, drift=0.7),
    ]
    out = _attack_drift_distribution(records)
    assert out.tolist() == [0.6, 0.7]


def test_drift_floor_pairs_within_cell() -> None:
    # Two null rows for same (model, task, sample) -> one pair scored.
    rows = [
        _row(attack="null", epsilon=0.0, benign_trace="trace one"),
        _row(attack="null", epsilon=0.0, benign_trace="trace two"),
    ]
    out = _drift_floor(rows, nli=lambda p, h: 0.9)
    # 1 - 0.5*(0.9 + 0.9) = 0.1
    assert len(out) == 1
    assert out[0] == 0.10000000000000009 or abs(out[0] - 0.1) < 1e-6


def test_drift_floor_skips_singleton_cells() -> None:
    rows = [_row(attack="null", epsilon=0.0, benign_trace="only one")]
    out = _drift_floor(rows, nli=lambda p, h: 0.0)
    assert out.size == 0
