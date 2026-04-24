"""Base attack abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class AttackResult:
    """Result of running an attack on a single image.

    Fields:
        perturbed_image: adversarial image tensor, same shape as input
        delta: perturbation (perturbed - original), useful for ε verification
        loss_final: loss value at the last iteration
        loss_trajectory: per-iteration loss values (for diagnostics)
        iterations: number of steps actually run (may be <= requested)
        success: implementation-defined success flag
        metadata: extra info (target tool name, step budget, etc.)
    """

    perturbed_image: torch.Tensor
    delta: torch.Tensor
    loss_final: float
    loss_trajectory: list[float] = field(default_factory=list)
    iterations: int = 0
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class AttackBase(ABC):
    """Each attack operates on pixel_values in [0, 1] — normalization is the
    model-family's job and happens inside the differentiable forward pass.

    Inputs:
        vlm: VLMBase with supports_gradients=True
        image: tensor in [0, 1], shape (C, H, W) or (B, C, H, W)
        prompt_tokens: pre-tokenized prompt, shape (B, T)
        target: attack-specific target (token ids, tool-name, trajectory, etc.)
    """

    name: str

    @abstractmethod
    def run(
        self,
        vlm: Any,
        image: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target: Any,
        **kwargs: Any,
    ) -> AttackResult: ...
