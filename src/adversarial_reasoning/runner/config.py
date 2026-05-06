"""RunnerConfig dataclass + YAML loading + epsilon resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

GRADIENT_MODES = {"pgd", "apgd", "targeted_tool", "trajectory_drift"}


@dataclass
class RunnerConfig:
    name: str
    phase: str
    output_dir: Path
    seeds: list[int]
    models: list[str]
    tasks: list[str]
    attacks: list[str]
    task_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    attack_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    epsilons_linf: list[float] = field(default_factory=list)
    split: str = "dev"


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})


def load_runner_config(exp_path: str | Path) -> RunnerConfig:
    raw = _load_yaml(exp_path)["experiment"]
    return RunnerConfig(
        name=raw["name"],
        phase=str(raw.get("phase", "0")),
        output_dir=Path(raw.get("output_dir", "runs") + ""),
        seeds=list(raw.get("seeds", [0])),
        models=list(raw["models"]),
        tasks=list(raw["tasks"]),
        attacks=list(raw["attacks"]),
        task_overrides=dict(raw.get("task_overrides", {})),
        attack_overrides=dict(raw.get("attack_overrides", {})),
        epsilons_linf=list(raw.get("epsilons_linf", [])),
    )


def resolve_epsilons(cfg: RunnerConfig, attack_name: str, attacks_yaml: dict) -> list[float]:
    overrides = cfg.attack_overrides.get(attack_name, {})
    if "epsilons" in overrides:
        return list(overrides["epsilons"])
    attack_cfg = attacks_yaml["attacks"].get(attack_name, {})
    eps = attack_cfg.get("epsilons")
    resolved = list(cfg.epsilons_linf) if eps in (None, []) else list(eps)
    if not resolved:
        raise ValueError(
            f"No epsilons configured for attack {attack_name!r}. "
            f"Set epsilons_linf in the experiment YAML or epsilons in "
            f"attacks config under attacks.{attack_name}."
        )
    return resolved
