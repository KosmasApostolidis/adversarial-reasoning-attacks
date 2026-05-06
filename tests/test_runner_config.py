"""Tests for runner.py pure-config / pure-Python paths.

Avoids GPU/HF model loading — we don't construct VLMs here. Covers:
  - load_runner_config YAML parsing
  - resolve_epsilons fallback chain
  - perturb_noise determinism + ε bound
  - perturb dispatcher (noise / gradient / unknown)
  - build_attack factory for all 4 gradient modes
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from adversarial_reasoning.runner import (
    GRADIENT_MODES,
    RunnerConfig,
    build_attack,
    load_runner_config,
    perturb,
    perturb_noise,
    resolve_epsilons,
)


def _exp_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "exp.yaml"
    p.write_text(
        textwrap.dedent(
            """
            experiment:
              name: t
              phase: "1"
              output_dir: out
              seeds: [0, 1]
              models: [m1]
              tasks: [t1]
              attacks: [pgd_linf, apgd_linf]
              epsilons_linf: [0.01, 0.02]
              attack_overrides:
                pgd_linf:
                  epsilons: [0.05]
            """
        ).strip()
    )
    return p


def test_load_runner_config(tmp_path: Path) -> None:
    cfg = load_runner_config(_exp_yaml(tmp_path))
    assert cfg.name == "t"
    assert cfg.phase == "1"
    assert cfg.seeds == [0, 1]
    assert cfg.models == ["m1"]
    assert cfg.attacks == ["pgd_linf", "apgd_linf"]
    assert cfg.epsilons_linf == [0.01, 0.02]


def test_resolve_epsilons_uses_attack_override(tmp_path: Path) -> None:
    cfg = load_runner_config(_exp_yaml(tmp_path))
    eps = resolve_epsilons(cfg, "pgd_linf", attacks_yaml={"attacks": {}})
    assert eps == [0.05]


def test_resolve_epsilons_falls_back_to_attacks_yaml(tmp_path: Path) -> None:
    cfg = load_runner_config(_exp_yaml(tmp_path))
    attacks_yaml = {"attacks": {"apgd_linf": {"epsilons": [0.1, 0.2]}}}
    eps = resolve_epsilons(cfg, "apgd_linf", attacks_yaml=attacks_yaml)
    assert eps == [0.1, 0.2]


def test_resolve_epsilons_falls_back_to_runner_default(tmp_path: Path) -> None:
    cfg = load_runner_config(_exp_yaml(tmp_path))
    attacks_yaml = {"attacks": {}}
    eps = resolve_epsilons(cfg, "apgd_linf", attacks_yaml=attacks_yaml)
    assert eps == cfg.epsilons_linf


def test_resolve_epsilons_attacks_yaml_empty_list_falls_back(tmp_path: Path) -> None:
    cfg = load_runner_config(_exp_yaml(tmp_path))
    attacks_yaml = {"attacks": {"apgd_linf": {"epsilons": []}}}
    eps = resolve_epsilons(cfg, "apgd_linf", attacks_yaml=attacks_yaml)
    assert eps == cfg.epsilons_linf


# --------- perturb_noise / perturb dispatcher ---------


def _img() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))


def test_perturb_noise_deterministic_for_same_seed() -> None:
    img = _img()
    a = perturb_noise(img, epsilon=0.05, seed=7)
    b = perturb_noise(img, epsilon=0.05, seed=7)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_perturb_noise_respects_epsilon_in_pixel_domain() -> None:
    img = _img()
    out = perturb_noise(img, epsilon=0.05, seed=1)
    delta = np.asarray(out, dtype=np.float32) / 255.0 - np.asarray(img, dtype=np.float32) / 255.0
    # uint8 round-trip introduces at most ~1/255 quantization slack.
    assert float(np.abs(delta).max()) <= 0.05 + 1.5 / 255.0


def test_perturb_dispatcher_noise() -> None:
    img = _img()
    out = perturb("noise", img, epsilon=0.05, seed=0)
    assert isinstance(out, Image.Image)


def test_perturb_dispatcher_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown attack mode"):
        perturb("not_a_mode", _img(), epsilon=0.05, seed=0)


@pytest.mark.parametrize("mode", sorted(GRADIENT_MODES))
def test_perturb_dispatcher_gradient_modes_raise_not_implemented(mode: str) -> None:
    with pytest.raises(NotImplementedError, match="pre-normalised"):
        perturb(mode, _img(), epsilon=0.05, seed=0)


# --------- build_attack factory ---------


@pytest.mark.parametrize(
    "mode,name",
    [
        ("pgd", "pgd_linf"),
        ("apgd", "apgd_linf"),
        ("targeted_tool", "targeted_tool_pgd"),
        ("trajectory_drift", "trajectory_drift_pgd"),
    ],
)
def test_build_attack_returns_correct_attack(mode: str, name: str) -> None:
    a = build_attack(mode, epsilon=0.05, steps=8)
    assert a.epsilon == 0.05
    assert a.steps == 8
    assert a.name == name


def test_build_attack_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown gradient attack mode"):
        build_attack("nope", epsilon=0.05, steps=8)


def test_runner_config_default_split_is_dev() -> None:
    """Memory note: runner --split default is 'dev', not 'val'."""
    cfg = RunnerConfig(
        name="x",
        phase="0",
        output_dir=Path("/tmp/x"),
        seeds=[0],
        models=["m"],
        tasks=["t"],
        attacks=["pgd_linf"],
    )
    assert cfg.split == "dev"
