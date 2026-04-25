"""L∞ sign-SGD gradient loop with random restarts.

Shared by :class:`PGDAttack` and :class:`TrajectoryDriftPGD`. APGD has its
own adaptive-step loop (heavy-ball momentum + Croce-Hein checkpoint
schedule with warm restart from best iterate) and does NOT route through
this helper — preserving APGD's byte-identical output across the
refactor was simpler than abstracting both update rules.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from .base import AttackResult
from .loss import LossFn


def linf_pgd_loop(
    *,
    loss_fn: LossFn,
    vlm: Any,
    x0: torch.Tensor,
    prompt_tokens: torch.Tensor,
    target: torch.Tensor,
    gen_kwargs: Mapping[str, Any],
    epsilon: float,
    alpha: float,
    n_iter: int,
    n_restarts: int,
    step_sign: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    static_metadata: dict[str, Any] | None = None,
) -> AttackResult:
    """L-inf sign-SGD: ``delta := clip[ delta + step_sign * alpha * sign(grad) ]``.

    The caller owns the sign convention. ``step_sign`` and the sign of
    the value returned by ``loss_fn`` together determine the attacker's
    direction. This function does not inspect targeted/untargeted
    semantics — it walks the loop and returns the best-loss restart.

    Random restart selection: smallest ``loss_final`` wins (matching the
    legacy ``_select_better`` convention; smaller-is-better from the
    attacker's perspective for both ascent-on-loss and descent-on-loss
    parameterisations).
    """
    best: AttackResult | None = None
    extra = static_metadata or {}

    for restart in range(n_restarts):
        delta = torch.empty_like(x0).uniform_(-epsilon, epsilon)
        delta = torch.clamp(x0 + delta, clip_min, clip_max) - x0
        delta.requires_grad_(True)

        loss_traj: list[float] = []
        for _step in range(n_iter):
            loss = loss_fn(vlm, x0 + delta, prompt_tokens, target, gen_kwargs)
            loss_traj.append(float(loss.detach().cpu()))
            grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
            with torch.no_grad():
                delta.add_(step_sign * alpha * grad.sign())
                delta.clamp_(-epsilon, epsilon)
                delta_data = torch.clamp(x0 + delta, clip_min, clip_max) - x0
                delta.copy_(delta_data)
                delta.grad = None

        perturbed = torch.clamp(x0 + delta.detach(), clip_min, clip_max)
        candidate = AttackResult(
            perturbed_image=perturbed.squeeze(0) if perturbed.shape[0] == 1 else perturbed,
            delta=delta.detach().squeeze(0) if delta.shape[0] == 1 else delta.detach(),
            loss_final=loss_traj[-1],
            loss_trajectory=loss_traj,
            iterations=n_iter,
            success=math.isfinite(loss_traj[-1]),
            metadata={
                "restart": restart,
                "epsilon": epsilon,
                "alpha": alpha,
                **extra,
            },
        )
        if best is None or candidate.loss_final < best.loss_final:
            best = candidate

    assert best is not None
    return best


__all__ = ["linf_pgd_loop"]
