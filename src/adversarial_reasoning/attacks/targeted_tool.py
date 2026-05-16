"""Targeted-Tool PGD — force a specific tool as the next tool call.

Loss: CE forcing the target tool-name token sequence into the position
immediately following the prompt. Reuses :class:`PGDAttack` with
``targeted=True``; this module only handles tagging the result with the
target-tool metadata so downstream metrics can group by it.

Refactor (2026-04-25): the ``build_target_tokens`` helper moved to
:mod:`adversarial_reasoning.attacks.targets`; we re-export it here for
back-compat with ``runner.py`` and any external scripts that imported
``from .attacks.targeted_tool import build_target_tokens``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ._epsilon import _LINF_EPSILON_8
from .base import AttackBase, AttackResult
from .pgd import PGDAttack
from .targets import build_target_tokens

__all__ = ["TargetedToolPGD", "build_target_tokens"]


@dataclass
class TargetedToolPGD(AttackBase):
    name: str = "targeted_tool_pgd"
    epsilon: float = _LINF_EPSILON_8
    alpha: float | None = None
    steps: int = 40
    random_restarts: int = 1
    targeted: bool = True
    target_tool: str = "escalate_to_specialist"
    clip_min: float = 0.0
    clip_max: float = 1.0
    seed: int | None = None

    def run(
        self,
        vlm: Any,
        image: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target: torch.Tensor,
        *,
        forward_kwargs: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> AttackResult:
        proxy = PGDAttack(
            epsilon=self.epsilon,
            alpha=self.alpha,
            steps=self.steps,
            random_restarts=self.random_restarts,
            targeted=True,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            seed=self.seed,
        )
        result = proxy.run(
            vlm=vlm,
            image=image,
            prompt_tokens=prompt_tokens,
            target=target,
            forward_kwargs=forward_kwargs,
        )
        result.metadata.setdefault("attack", self.name)
        result.metadata["target_tool"] = self.target_tool
        return result
