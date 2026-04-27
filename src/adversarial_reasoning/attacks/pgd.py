"""L∞ PGD on VLM image inputs, targeting tool-call token logits.

Design notes
------------
- We operate in the clean pixel-value domain ([0, 1]), not the normalized
  post-processing domain. The VLM's forward pass composes its own
  normalization on top of `image_tensor`, so gradients propagate correctly
  back to the pixel domain.
- The loss target is a sequence of token ids corresponding to the expected
  tool-call in the benign trajectory. We maximise cross-entropy against
  those targets for untargeted attacks and (separately in `targeted_tool.py`)
  minimise CE towards an attacker-chosen target tool for targeted attacks.
- Random restart picks the perturbation with the lowest ``loss_final``
  (smaller-is-better convention from the attacker's perspective).

Refactor (2026-04-25): the loss body and step+restart loop were extracted
into :mod:`adversarial_reasoning.attacks.loss` and
:mod:`adversarial_reasoning.attacks._loop` so PGD, APGD, and
TrajectoryDriftPGD share one loss-computation path. Numeric outputs are
byte-identical to the prior monolithic implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ._loop import linf_pgd_loop
from .base import AttackBase, AttackResult
from .loss import TokenTargetLoss


@dataclass
class PGDAttack(AttackBase):
    name: str = "pgd_linf"
    epsilon: float = 8.0 / 255.0
    alpha: float | None = None  # step size; default = epsilon / 4
    steps: int = 40
    random_restarts: int = 1
    targeted: bool = False
    clip_min: float = 0.0
    clip_max: float = 1.0
    seed: int | None = None  # pin RNG for reproducible restarts

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
        """Run PGD-L∞.

        target: token-id LongTensor (B, T_target) representing the token
            sequence whose log-prob we push (up for targeted, down otherwise).
        forward_kwargs: model-specific extras (e.g. Qwen `image_grid_thw`,
            LLaVA-Next `image_sizes`, plus the `attention_mask` covering
            `[prompt ‖ target]`). The attack harness constructs these via
            `vlm.prepare_attack_inputs(...)` before calling `run`.
        """
        if not getattr(vlm, "supports_gradients", False):
            raise ValueError(f"VLM backend {vlm.__class__.__name__} does not support gradients.")

        x0 = image.detach().clone()
        if x0.ndim == 3:
            x0 = x0.unsqueeze(0)

        alpha = self.alpha if self.alpha is not None else self.epsilon / 4.0
        # TokenTargetLoss already encodes targeted/untargeted semantics via
        # ±CE. The loop always descends on the returned scalar (step_sign=-1)
        # so that:
        #   untargeted (loss=-CE):  -sign(grad(-CE)) = +sign(grad(CE)) → CE↑
        #   targeted   (loss=+CE):  -sign(grad(+CE)) = -sign(grad(CE)) → CE↓
        # APGD uses the same convention (``- sign * eta * grad.sign()``).
        step_sign = -1.0

        return linf_pgd_loop(
            loss_fn=TokenTargetLoss(targeted=self.targeted),
            vlm=vlm,
            x0=x0,
            prompt_tokens=prompt_tokens,
            target=target,
            gen_kwargs=forward_kwargs or {},
            epsilon=self.epsilon,
            alpha=alpha,
            n_iter=self.steps,
            n_restarts=self.random_restarts,
            step_sign=step_sign,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            static_metadata={"targeted": self.targeted},
            seed=self.seed,
        )
