"""APGD L∞ — Auto-PGD with adaptive step size (Croce & Hein 2020).

Backend: IBM ART's ``AutoProjectedGradientDescent`` or torchattacks. Loss is
CE on tool-call tokens (identical target to :class:`PGDAttack`); the
adaptive scheduler replaces the fixed α and restart logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .base import AttackBase, AttackResult


@dataclass
class APGDAttack(AttackBase):
    name: str = "apgd_linf"
    epsilon: float = 8.0 / 255.0
    steps: int = 100
    random_restarts: int = 1
    targeted: bool = False
    backend: str = "art"
    clip_min: float = 0.0
    clip_max: float = 1.0

    def run(
        self,
        vlm: Any,
        image: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target: Any,
        **_: Any,
    ) -> AttackResult:
        raise NotImplementedError(
            "APGDAttack is a planned implementation. Backend hook (ART / "
            "torchattacks) still to be wired; see configs/attacks.yaml."
        )
