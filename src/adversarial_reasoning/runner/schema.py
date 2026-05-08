"""Pydantic v2 schema for experiment YAML configs.

Wire-compatible with :class:`RunnerConfig` — validation runs first, then the
dataclass is constructed from validated fields. The schema layer adds
``extra="forbid"`` so typos in YAML keys fail fast instead of silently
disappearing into ``raw.get("attacks")``-style lookups.

Design notes
------------
- Every field that exists in ``RunnerConfig`` is required-or-default-mirrored
  here.
- ``statistics``, ``transfer_evaluation``, ``loss_ablations`` are
  experiment-level keys read by downstream stats/figure scripts (not by the
  attack runner). They are accepted-and-ignored by ``to_runner_config``.
- ``phase`` is coerced to ``str`` to match the runtime contract — historical
  YAMLs spell it ``phase: 0`` or ``phase: "0"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import RunnerConfig


class StatisticsBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test: str = "wilcoxon_signed_rank"
    bootstrap_resamples: int = 1000
    ci_level: float = 0.95
    multiple_comparison_correction: str = "none"
    bh_q: float | None = None  # Benjamini-Hochberg FDR q value


class TransferEvaluationBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    same_images: bool = True


class ExperimentConfig(BaseModel):
    """Validated experiment block (the value under the YAML's ``experiment:`` key)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    phase: str = "0"
    output_dir: str = "runs"
    seeds: list[int] = Field(default_factory=lambda: [0])
    models: list[str]
    tasks: list[str]
    attacks: list[str]
    task_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    attack_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    epsilons_linf: list[float] = Field(default_factory=list)
    statistics: StatisticsBlock | None = None
    transfer_evaluation: TransferEvaluationBlock | None = None
    loss_ablations: dict[str, Any] | None = None

    @field_validator("phase", mode="before")
    @classmethod
    def _coerce_phase(cls, v: Any) -> str:
        return str(v)

    def to_runner_config(self, *, split: str = "dev") -> RunnerConfig:
        """Project the validated config onto the runtime dataclass."""
        return RunnerConfig(
            name=self.name,
            phase=self.phase,
            output_dir=Path(self.output_dir),
            seeds=list(self.seeds),
            models=list(self.models),
            tasks=list(self.tasks),
            attacks=list(self.attacks),
            task_overrides=dict(self.task_overrides),
            attack_overrides=dict(self.attack_overrides),
            epsilons_linf=list(self.epsilons_linf),
            split=split,
        )
