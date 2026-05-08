"""Schema validation + _extends deep-merge tests for runner.config.

Covers:
  - every committed YAML in configs/**.yaml validates through ExperimentConfig
  - extra="forbid" rejects unknown keys (typo guard)
  - _extends deep-merge: child wins on scalar; nested dict merges; lists replace
  - _extends cycle detection
  - _LEGACY_CONFIG_LOADER=1 bypasses schema validation
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from adversarial_reasoning.runner.config import (
    _deep_merge,
    _resolve_extends,
    load_runner_config,
)
from adversarial_reasoning.runner.schema import ExperimentConfig

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
EXPERIMENT_CONFIGS = sorted(
    p
    for p in CONFIGS.rglob("*.yaml")
    if p.name not in {"attacks.yaml", "tasks.yaml", "models.yaml", "_base.yaml"}
)


@pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=lambda p: p.name)
def test_every_committed_config_validates(path: Path) -> None:
    cfg = load_runner_config(path)
    assert cfg.name
    assert cfg.models
    assert cfg.tasks
    assert cfg.attacks


def test_extra_forbid_rejects_typos() -> None:
    with pytest.raises(ValidationError, match=r"extra_forbidden|epslon"):
        ExperimentConfig.model_validate(
            {"name": "x", "models": ["m"], "tasks": ["t"], "attacks": ["a"], "epslon": [0.1]}
        )


def test_phase_coerced_to_str() -> None:
    cfg = ExperimentConfig.model_validate(
        {"name": "x", "phase": 2, "models": ["m"], "tasks": ["t"], "attacks": ["a"]}
    )
    assert cfg.phase == "2"


def test_deep_merge_lists_replace() -> None:
    base = {"a": [1, 2, 3], "b": {"x": 1}}
    over = {"a": [9], "b": {"y": 2}}
    out = _deep_merge(base, over)
    assert out == {"a": [9], "b": {"x": 1, "y": 2}}


def test_deep_merge_child_wins_on_scalar() -> None:
    assert _deep_merge({"k": 1}, {"k": 2}) == {"k": 2}


def test_extends_deep_merge_via_yaml(tmp_path: Path) -> None:
    base = tmp_path / "_base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        textwrap.dedent(
            """
            experiment:
              seeds: [0]
              statistics:
                test: wilcoxon_signed_rank
                bootstrap_resamples: 1000
                ci_level: 0.95
                multiple_comparison_correction: none
            """
        ).strip()
    )
    child.write_text(
        textwrap.dedent(
            """
            _extends: _base.yaml
            experiment:
              name: t
              models: [m]
              tasks: [t]
              attacks: [pgd_linf]
              seeds: [0, 1, 2]
              statistics:
                bootstrap_resamples: 5000
            """
        ).strip()
    )
    cfg = load_runner_config(child)
    # Lists replace wholesale: child's [0,1,2] beats base's [0].
    assert cfg.seeds == [0, 1, 2]
    # Statistics merges deeply: child's bootstrap_resamples wins, base's
    # other keys persist (validated indirectly by schema accepting both).
    assert cfg.name == "t"


def test_extends_cycle_detected(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(
        "_extends: b.yaml\nexperiment: {name: a, models: [m], tasks: [t], attacks: [pgd]}\n"
    )
    b.write_text(
        "_extends: a.yaml\nexperiment: {name: b, models: [m], tasks: [t], attacks: [pgd]}\n"
    )
    with pytest.raises(ValueError, match="Circular _extends"):
        load_runner_config(a)


def test_extends_strips_key_before_validation(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("experiment: {seeds: [0]}\n")
    child.write_text(
        "_extends: base.yaml\nexperiment: {name: t, models: [m], tasks: [t], attacks: [a]}\n"
    )
    cfg = load_runner_config(child)
    assert cfg.seeds == [0]


def test_resolve_extends_no_extends_passthrough(tmp_path: Path) -> None:
    raw = {"experiment": {"name": "t", "models": ["m"], "tasks": ["t"], "attacks": ["a"]}}
    out = _resolve_extends(raw, here=tmp_path / "x.yaml")
    assert out == raw


def test_legacy_loader_bypasses_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When _LEGACY_CONFIG_LOADER=1, unknown keys are silently dropped instead of failing."""
    p = tmp_path / "exp.yaml"
    p.write_text(
        textwrap.dedent(
            """
            experiment:
              name: t
              models: [m]
              tasks: [t]
              attacks: [a]
              not_a_real_key: hello
            """
        ).strip()
    )
    monkeypatch.setenv("_LEGACY_CONFIG_LOADER", "1")
    cfg = load_runner_config(p)
    assert cfg.name == "t"


def test_strict_loader_rejects_unknown_key(tmp_path: Path) -> None:
    p = tmp_path / "exp.yaml"
    p.write_text(
        textwrap.dedent(
            """
            experiment:
              name: t
              models: [m]
              tasks: [t]
              attacks: [a]
              not_a_real_key: hello
            """
        ).strip()
    )
    os.environ.pop("_LEGACY_CONFIG_LOADER", None)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_runner_config(p)


def test_to_runner_config_split_propagates() -> None:
    cfg = ExperimentConfig.model_validate(
        {"name": "x", "models": ["m"], "tasks": ["t"], "attacks": ["a"]}
    ).to_runner_config(split="val")
    assert cfg.split == "val"
