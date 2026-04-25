"""Tests for the records.jsonl serialiser."""

from __future__ import annotations

from adversarial_reasoning.agents.base import ToolCall, Trajectory
from adversarial_reasoning.runner import trajectory_record


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
