"""Targeted-Tool PGD — force a specific tool choice at step k.

Loss: CE forcing the target tool-name token sequence at ``target_step_k``.
Shares the PGD-L∞ optimiser with :class:`PGDAttack` but uses
``targeted=True`` and a positive-sign CE loss against the attacker-chosen
tool-call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .base import AttackBase, AttackResult


@dataclass
class TargetedToolPGD(AttackBase):
    name: str = "targeted_tool_pgd"
    epsilon: float = 8.0 / 255.0
    steps: int = 40
    random_restarts: int = 1
    targeted: bool = True
    target_tool: str = "escalate_to_specialist"
    target_step_k: int = 0
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
            "TargetedToolPGD is a planned implementation. Needs prefix-"
            "forcing target-token construction at step k; see "
            "configs/attacks.yaml."
        )
