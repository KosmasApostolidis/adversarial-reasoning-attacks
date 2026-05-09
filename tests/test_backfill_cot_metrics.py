"""Tests for the CoT-metric backfill script.

Uses a constant-stub NLI so DeBERTa never loads. Verifies idempotency,
rows-without-reasoning pass-through, and counts accounting.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.dataprep.backfill_cot_metrics import CoT_FIELDS, backfill


def _nli_stub(p: str, h: str) -> float:
    return 0.5


def _row(*, with_trace: bool, with_cot: bool = False) -> dict:
    base = {
        "schema_version": "0.4.0",
        "model_key": "qwen2_5_vl_7b",
        "task_id": "t",
        "sample_id": "s001",
        "attack_name": "pgd",
        "attack_mode": "linf",
        "epsilon": 0.0314,
        "seed": 0,
        "benign": {
            "task_id": "t",
            "model_id": "qwen2_5_vl_7b",
            "seed": 0,
            "tool_sequence": ["query_guidelines"],
            "tool_calls": [{"name": "query_guidelines", "args": {}, "result": {"x": 1}}],
            "final_answer": "ok",
            "reasoning_trace": "step 0 reasoning" if with_trace else "",
            "metadata": {},
        },
        "attacked": {
            "task_id": "t",
            "model_id": "qwen2_5_vl_7b",
            "seed": 0,
            "tool_sequence": ["calculate_risk_score"],
            "tool_calls": [{"name": "calculate_risk_score", "args": {}, "result": {"y": 2}}],
            "final_answer": "different",
            "reasoning_trace": "step 0 attacked reasoning" if with_trace else "",
            "metadata": {},
        },
        "edit_distance_norm": 1.0,
        "elapsed_s": 1.0,
    }
    if with_cot:
        for k in CoT_FIELDS:
            base[k] = 0.5 if "score" in k or "faithfulness" in k or "hallucination" in k else False
    return base


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_backfill_scores_rows_with_reasoning(tmp_path: Path) -> None:
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, [_row(with_trace=True)])

    counts = backfill(in_path, out_path, nli=_nli_stub)

    assert counts == {"total": 1, "scored": 1, "skipped_existing": 0, "skipped_no_trace": 0}
    out = _read_jsonl(out_path)
    assert len(out) == 1
    for k in CoT_FIELDS:
        assert k in out[0]


def test_backfill_skips_rows_without_reasoning(tmp_path: Path) -> None:
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, [_row(with_trace=False)])

    counts = backfill(in_path, out_path, nli=_nli_stub)

    assert counts["scored"] == 0
    assert counts["skipped_no_trace"] == 1
    out = _read_jsonl(out_path)
    assert all(k not in out[0] for k in CoT_FIELDS)


def test_backfill_skips_already_scored(tmp_path: Path) -> None:
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, [_row(with_trace=True, with_cot=True)])

    counts = backfill(in_path, out_path, nli=_nli_stub)

    assert counts["scored"] == 0
    assert counts["skipped_existing"] == 1


def test_backfill_idempotent(tmp_path: Path) -> None:
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, [_row(with_trace=True)])

    backfill(in_path, out_path, nli=_nli_stub)
    out_first = _read_jsonl(out_path)

    # Second pass: feed the output back in.
    out_path2 = tmp_path / "out2.jsonl"
    counts = backfill(out_path, out_path2, nli=lambda p, h: 0.99)

    # No re-scoring -- existing fields preserved verbatim.
    assert counts["skipped_existing"] == 1
    out_second = _read_jsonl(out_path2)
    for k in CoT_FIELDS:
        assert out_second[0][k] == out_first[0][k]


def test_backfill_handles_blank_lines(tmp_path: Path) -> None:
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    in_path.write_text(
        "\n"
        + json.dumps(_row(with_trace=True))
        + "\n\n"
        + json.dumps(_row(with_trace=False))
        + "\n"
    )

    counts = backfill(in_path, out_path, nli=_nli_stub)

    assert counts["total"] == 2
    assert counts["scored"] == 1
    assert counts["skipped_no_trace"] == 1


def test_backfill_creates_output_directory(tmp_path: Path) -> None:
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "deeper" / "out.jsonl"
    _write_jsonl(in_path, [_row(with_trace=True)])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = backfill(in_path, out_path, nli=_nli_stub)
    assert counts["scored"] == 1
    assert out_path.exists()
