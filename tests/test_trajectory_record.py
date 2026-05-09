"""Tests for the records.jsonl serialiser."""

from __future__ import annotations

from adversarial_reasoning.agents.base import ToolCall, Trajectory
from adversarial_reasoning.runner import pair_record, trajectory_record
from adversarial_reasoning.runner.records import RECORDS_SCHEMA_VERSION


def _trajectory_with_calls() -> Trajectory:
    return Trajectory(
        task_id="prostate_mri_workup",
        model_id="qwen2_5_vl_7b",
        seed=0,
        tool_calls=[
            ToolCall(
                step=1,
                name="query_guidelines",
                args={"guideline": "NCCN", "topic": "pi_rads_3"},
                result={"recommendation": "biopsy"},
            ),
            ToolCall(
                step=2,
                name="calculate_risk_score",
                args={"psa": 6.0, "volume_cc": 40.0},
                result={"score": 0.72},
            ),
        ],
        final_answer="Recommend MRI-targeted biopsy.",
        reasoning_trace="...",
        metadata={"sample_id": "s001"},
    )


def test_trajectory_record_includes_tool_calls() -> None:
    record = trajectory_record(_trajectory_with_calls())

    assert record["tool_sequence"] == ["query_guidelines", "calculate_risk_score"]
    assert "tool_calls" in record
    assert len(record["tool_calls"]) == 2
    assert record["tool_calls"][0]["name"] == "query_guidelines"
    assert record["tool_calls"][0]["args"] == {
        "guideline": "NCCN",
        "topic": "pi_rads_3",
    }
    assert record["tool_calls"][1]["args"] == {"psa": 6.0, "volume_cc": 40.0}


def test_trajectory_record_tool_calls_serialise_to_dicts() -> None:
    record = trajectory_record(_trajectory_with_calls())

    for call in record["tool_calls"]:
        assert isinstance(call, dict)
        assert set(call.keys()) >= {"step", "name", "args", "result", "error"}


def test_trajectory_record_handles_empty_tool_calls() -> None:
    trajectory = Trajectory(
        task_id="t",
        model_id="m",
        seed=0,
        tool_calls=[],
        final_answer="",
    )
    record = trajectory_record(trajectory)

    assert record["tool_calls"] == []
    assert record["tool_sequence"] == []


def test_trajectory_record_includes_reasoning_trace() -> None:
    trajectory = _trajectory_with_calls()
    record = trajectory_record(trajectory)

    assert "reasoning_trace" in record
    assert record["reasoning_trace"] == trajectory.reasoning_trace


def test_trajectory_record_reasoning_trace_default_empty() -> None:
    trajectory = Trajectory(task_id="t", model_id="m", seed=0)
    record = trajectory_record(trajectory)

    assert record["reasoning_trace"] == ""


def _bare_pair_kwargs() -> dict:
    benign = Trajectory(task_id="t", model_id="m", seed=0)
    attacked = Trajectory(task_id="t", model_id="m", seed=0)
    return {
        "model_key": "qwen2_5_vl_7b",
        "task_id": "t",
        "sample_id": "s001",
        "attack_name": "pgd",
        "attack_mode": "linf",
        "epsilon": 8 / 255,
        "seed": 0,
        "benign": benign,
        "attacked": attacked,
        "edit_distance": 0.0,
        "elapsed_s": 1.0,
    }


def test_pair_record_emits_schema_version() -> None:
    record = pair_record(**_bare_pair_kwargs())
    assert record["schema_version"] == RECORDS_SCHEMA_VERSION


def test_pair_record_without_cot_metrics_omits_cot_fields() -> None:
    record = pair_record(**_bare_pair_kwargs())
    cot_keys = {
        "cot_drift_score",
        "cot_faithfulness_benign",
        "cot_faithfulness_attacked",
        "cot_hallucination_benign",
        "cot_hallucination_attacked",
        "cot_refusal_benign",
        "cot_refusal_attacked",
    }
    assert cot_keys.isdisjoint(record.keys())


def test_pair_record_with_cot_metrics_surfaces_fields() -> None:
    cot = {
        "cot_drift_score": 0.42,
        "cot_faithfulness_benign": 0.91,
        "cot_faithfulness_attacked": 0.55,
        "cot_hallucination_benign": 0.0,
        "cot_hallucination_attacked": 0.33,
        "cot_refusal_benign": False,
        "cot_refusal_attacked": True,
    }
    record = pair_record(cot_metrics=cot, **_bare_pair_kwargs())
    for k, v in cot.items():
        assert record[k] == v
