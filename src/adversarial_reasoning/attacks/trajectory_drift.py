"""Trajectory-Drift PGD — untargeted KL ascent on full benign trajectory.

Loss
----
``loss = -KL(softmax(logits_attacked) ‖ softmax(logits_benign).detach())``
evaluated at the token positions of the benign tool-call trajectory.
The attacker ascends KL (so we minimise its negation), pushing the
attacked next-token distribution away from the benign reference. Unlike
plain PGD's CE-against-targets, this captures *trajectory-wide* drift
because ``target`` here is the entire concatenated benign tool-call
sequence, not a single tool block.

Caller contract
---------------
- ``target`` is the concatenated token-id tensor of the full benign
  trajectory (shape ``(1, T_target)``).
- The attack computes benign reference logits **once** with the original
  image (no_grad, via :meth:`TrajectoryDriftLoss.from_benign`), caches
  them, then runs PGD-L∞ ascent on KL via :func:`linf_pgd_loop`.

Refactor (2026-04-25): the inline KL block + per-restart loop body now
delegate to :mod:`adversarial_reasoning.attacks.loss` and
:mod:`adversarial_reasoning.attacks._loop`. Numeric outputs are
byte-identical to the prior monolithic implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from ._loop import linf_pgd_loop
from .base import AttackBase, AttackResult
from .loss import TrajectoryDriftLoss


@dataclass
class TrajectoryDriftPGD(AttackBase):
    name: str = "trajectory_drift_pgd"
    epsilon: float = 8.0 / 255.0
    alpha: float | None = None
    steps: int = 40
    random_restarts: int = 1
    targeted: bool = False
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
        if not getattr(vlm, "supports_gradients", False):
            raise ValueError(f"VLM backend {vlm.__class__.__name__} does not support gradients.")

        x0 = image.detach().clone()
        if x0.ndim == 3:
            x0 = x0.unsqueeze(0)

        gen_kwargs = forward_kwargs or {}
        alpha = self.alpha if self.alpha is not None else self.epsilon / 4.0

        # Compute & cache the benign reference distribution once (no_grad).
        loss_fn = TrajectoryDriftLoss.from_benign(
            vlm=vlm,
            x0=x0,
            prompt_tokens=prompt_tokens,
            target=target,
            gen_kwargs=gen_kwargs,
        )

        result = linf_pgd_loop(
            loss_fn=loss_fn,
            vlm=vlm,
            x0=x0,
            prompt_tokens=prompt_tokens,
            target=target,
            gen_kwargs=gen_kwargs,
            epsilon=self.epsilon,
            alpha=alpha,
            n_iter=self.steps,
            n_restarts=self.random_restarts,
            step_sign=1.0,  # always ascend on -KL
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            seed=self.seed,
        )
        # Preserve the legacy metadata key. ``loss_final = -KL`` for the
        # winning restart, so ``kl_final = -loss_final``.
        return replace(
            result,
            metadata={**result.metadata, "kl_final": -result.loss_final},
        )
