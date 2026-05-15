"""RunnerConfig dataclass + YAML loading + epsilon resolution.

YAML configs may declare a top-level ``_extends: <relative-path>`` key. The
referenced file is loaded first and its ``experiment:`` block is deep-merged
under the child's ``experiment:`` block (child wins on conflicts; nested
dicts merge recursively; lists are replaced wholesale).

Validation runs through :class:`ExperimentConfig` (pydantic v2,
``extra="forbid"``) before the dataclass is built. Set
``_LEGACY_CONFIG_LOADER=1`` in the environment to bypass schema validation
during the v0.3.x → v0.4.x migration window.
"""

from __future__ import annotations

import os
from copy import deepcopy
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


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two dicts. ``over`` wins; nested dicts merge; lists replace."""
    out: dict[str, Any] = deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _resolve_extends(
    raw: dict[str, Any],
    *,
    here: Path,
    visited: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Recursively resolve ``_extends:`` paths relative to *here*.

    Cycle-safe via a visited tuple. Strips ``_extends`` from the returned
    dict so downstream validation does not see it.
    """
    extends = raw.pop("_extends", None)
    if extends is None:
        return raw
    base_path = (here.parent / extends).resolve()
    if base_path in visited:
        cycle = " -> ".join(str(p) for p in (*visited, base_path))
        raise ValueError(f"Circular _extends in YAML configs: {cycle}")
    base_raw = _resolve_extends(
        _load_yaml(base_path), here=base_path, visited=(*visited, base_path)
    )
    return _deep_merge(base_raw, raw)


def load_runner_config(exp_path: str | Path) -> RunnerConfig:
    here = Path(exp_path).resolve()
    raw_full = _resolve_extends(_load_yaml(here), here=here)
    raw = raw_full.get("experiment", {})
    # Resolve relative output_dir against the config file's directory so
    # Path("..") in a YAML always resolves to the expected absolute path
    # regardless of the CWD the runner is launched from.
    out_dir_raw = raw.get("output_dir", "runs")
    raw["output_dir"] = str((here.parent / Path(out_dir_raw)).resolve())
    if os.environ.get("_LEGACY_CONFIG_LOADER") == "1":
        return RunnerConfig(
            name=raw["name"],
            phase=str(raw.get("phase", "0")),
            output_dir=Path(raw["output_dir"]),
            seeds=list(raw.get("seeds", [0])),
            models=list(raw["models"]),
            tasks=list(raw["tasks"]),
            attacks=list(raw["attacks"]),
            task_overrides=dict(raw.get("task_overrides", {})),
            attack_overrides=dict(raw.get("attack_overrides", {})),
            epsilons_linf=list(raw.get("epsilons_linf", [])),
        )
    # Local import avoids a runtime circular: schema imports RunnerConfig.
    from .schema import ExperimentConfig

    return ExperimentConfig.model_validate(raw).to_runner_config()


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
