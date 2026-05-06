"""Experiment runner package — public API re-exports.

Loads ``configs/<exp>.yaml``, iterates over
``models x tasks x attacks x epsilons x seeds x samples``, and records one
JSONL row per (benign, attacked) pair.
"""

from __future__ import annotations

from .attacks import (
    build_attack,
    perturb,
    perturb_noise,
    run_gradient_attack,
)
from .cli import main
from .config import (
    GRADIENT_MODES,
    RunnerConfig,
    load_runner_config,
    resolve_epsilons,
)
from .records import pair_record, trajectory_record

__all__ = [
    "GRADIENT_MODES",
    "RunnerConfig",
    "build_attack",
    "load_runner_config",
    "main",
    "pair_record",
    "perturb",
    "perturb_noise",
    "resolve_epsilons",
    "run_gradient_attack",
    "trajectory_record",
]
