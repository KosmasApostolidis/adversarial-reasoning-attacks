"""Output record schema for benign/attacked trajectory pairs."""

from __future__ import annotations

from typing import Any

from ..agents.base import Trajectory

RECORDS_SCHEMA_VERSION = "0.4.0"


def trajectory_record(t: Trajectory) -> dict[str, Any]:
    return {
        "task_id": t.task_id,
        "model_id": t.model_id,
        "seed": t.seed,
        "tool_sequence": t.tool_sequence(),
        "tool_calls": [c.to_dict() for c in t.tool_calls],
        "final_answer": t.final_answer,
        "reasoning_trace": t.reasoning_trace,
        "metadata": t.metadata,
    }


def pair_record(
    *,
    model_key: str,
    task_id: str,
    sample_id: str,
    attack_name: str,
    attack_mode: str,
    epsilon: float,
    seed: int,
    benign: Trajectory,
    attacked: Trajectory,
    edit_distance: float,
    elapsed_s: float,
    cot_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": RECORDS_SCHEMA_VERSION,
        "model_key": model_key,
        "task_id": task_id,
        "sample_id": sample_id,
        "attack_name": attack_name,
        "attack_mode": attack_mode,
        "epsilon": epsilon,
        "seed": seed,
        "benign": trajectory_record(benign),
        "attacked": trajectory_record(attacked),
        "edit_distance_norm": edit_distance,
        "elapsed_s": elapsed_s,
    }
    if cot_metrics is not None:
        record.update(cot_metrics)
    return record
