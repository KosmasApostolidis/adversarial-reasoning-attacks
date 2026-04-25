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
  image (no_grad), caches them, then runs PGD-L∞ ascent on KL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .base import AttackBase, AttackResult


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
            raise ValueError(
                f"VLM backend {vlm.__class__.__name__} does not support gradients."
            )

        x0 = image.detach().clone()
        if x0.ndim == 3:
            x0 = x0.unsqueeze(0)

        fwd_kwargs = forward_kwargs or {}
        alpha = self.alpha if self.alpha is not None else self.epsilon / 4.0
        t_prompt = prompt_tokens.shape[-1]
        t_target = target.shape[-1]

        with torch.no_grad():
            input_ids = torch.cat([prompt_tokens, target], dim=-1)
            benign_logits = vlm.forward_with_logits(x0, input_ids, **fwd_kwargs)
            benign_slice = benign_logits[:, t_prompt - 1 : t_prompt - 1 + t_target, :].detach()
            log_benign = F.log_softmax(benign_slice, dim=-1)
            p_benign = log_benign.exp()

        best: AttackResult | None = None
        for restart in range(self.random_restarts):
            delta = torch.empty_like(x0).uniform_(-self.epsilon, self.epsilon)
            delta = torch.clamp(x0 + delta, self.clip_min, self.clip_max) - x0
            delta.requires_grad_(True)

            loss_traj: list[float] = []
            for _step in range(self.steps):
                input_ids = torch.cat([prompt_tokens, target], dim=-1)
                logits = vlm.forward_with_logits(x0 + delta, input_ids, **fwd_kwargs)
                attacked_slice = logits[:, t_prompt - 1 : t_prompt - 1 + t_target, :]
                log_attacked = F.log_softmax(attacked_slice, dim=-1)
                kl = F.kl_div(log_attacked, p_benign, reduction="batchmean")
                loss = -kl  # ascend KL → minimise -KL
                loss_traj.append(float(loss.detach().cpu()))

                grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
                with torch.no_grad():
                    delta.add_(alpha * grad.sign())
                    delta.clamp_(-self.epsilon, self.epsilon)
                    delta_data = torch.clamp(x0 + delta, self.clip_min, self.clip_max) - x0
                    delta.copy_(delta_data)
                    delta.grad = None

            perturbed = torch.clamp(x0 + delta.detach(), self.clip_min, self.clip_max)
            candidate = AttackResult(
                perturbed_image=perturbed.squeeze(0) if perturbed.shape[0] == 1 else perturbed,
                delta=delta.detach().squeeze(0) if delta.shape[0] == 1 else delta.detach(),
                loss_final=loss_traj[-1],
                loss_trajectory=loss_traj,
                iterations=self.steps,
                success=math.isfinite(loss_traj[-1]),
                metadata={
                    "restart": restart,
                    "epsilon": self.epsilon,
                    "alpha": alpha,
                    "kl_final": -loss_traj[-1],
                },
            )
            if best is None or candidate.loss_final < best.loss_final:
                best = candidate

        assert best is not None
        return best
