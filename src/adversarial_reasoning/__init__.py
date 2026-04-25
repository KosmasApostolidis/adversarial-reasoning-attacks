"""Adversarial reasoning attacks on medical imaging VLM agents.

Public API. Heavy submodules (torch / transformers) are deferred via
lazy attribute access so ``import adversarial_reasoning`` is cheap.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

__all__ = [
    "APGDAttack",
    "AttackBase",
    "AttackResult",
    "LlavaNext",
    "MedicalAgent",
    "PGDAttack",
    "QwenVL",
    "RunnerConfig",
    "TargetedToolPGD",
    "TrajectoryDriftPGD",
    "VLMBase",
    "__version__",
    "load_runner_config",
    "main",
]

# Static type-checker-visible imports (no runtime cost).
if TYPE_CHECKING:
    from .agents.medical_agent import MedicalAgent
    from .attacks import (
        APGDAttack,
        AttackBase,
        AttackResult,
        PGDAttack,
        TargetedToolPGD,
        TrajectoryDriftPGD,
    )
    from .models.base import VLMBase
    from .models.llava import LlavaNext
    from .models.qwen_vl import QwenVL
    from .runner import RunnerConfig, load_runner_config, main


_LAZY_IMPORTS = {
    "MedicalAgent": ("adversarial_reasoning.agents.medical_agent", "MedicalAgent"),
    "APGDAttack": ("adversarial_reasoning.attacks", "APGDAttack"),
    "AttackBase": ("adversarial_reasoning.attacks", "AttackBase"),
    "AttackResult": ("adversarial_reasoning.attacks", "AttackResult"),
    "PGDAttack": ("adversarial_reasoning.attacks", "PGDAttack"),
    "TargetedToolPGD": ("adversarial_reasoning.attacks", "TargetedToolPGD"),
    "TrajectoryDriftPGD": ("adversarial_reasoning.attacks", "TrajectoryDriftPGD"),
    "VLMBase": ("adversarial_reasoning.models.base", "VLMBase"),
    "LlavaNext": ("adversarial_reasoning.models.llava", "LlavaNext"),
    "QwenVL": ("adversarial_reasoning.models.qwen_vl", "QwenVL"),
    "RunnerConfig": ("adversarial_reasoning.runner", "RunnerConfig"),
    "load_runner_config": ("adversarial_reasoning.runner", "load_runner_config"),
    "main": ("adversarial_reasoning.runner", "main"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        import importlib
        module_name, attr = _LAZY_IMPORTS[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module 'adversarial_reasoning' has no attribute {name!r}")
