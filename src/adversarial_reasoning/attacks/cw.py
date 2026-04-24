"""Carlini & Wagner L2 — margin-based targeted attack.

Unlike PGD, C&W uses a continuous c-sweep over the trade-off between
``|δ|_2`` and the margin term; it minimises ``|δ|_2 + c * margin(δ)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .base import AttackBase, AttackResult


@dataclass
class CWAttack(AttackBase):
    name: str = "cw_l2"
    c_search_steps: int = 9
    steps: int = 1000
    confidence: float = 0.0
    targeted: bool = True

    def run(
        self,
        vlm: Any,
        image: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target: Any,
        **_: Any,
    ) -> AttackResult:
        raise NotImplementedError(
            "CWAttack is a planned implementation. Margin loss + c-sweep "
            "binary search still to be wired; see configs/attacks.yaml."
        )
