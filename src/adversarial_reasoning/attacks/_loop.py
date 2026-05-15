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


def _pgd_step(
    *,
    loss_fn: LossFn,
    vlm: Any,
    x0: torch.Tensor,
    delta: torch.Tensor,
    prompt_tokens: torch.Tensor,
    target: torch.Tensor,
    gen_kwargs: Mapping[str, Any],
    epsilon: float,
    alpha: float,
    step_sign: float,
    clip_min: float,
    clip_max: float,
    losses: list[float],
) -> None:
    """Single sign-SGD step. Mutates ``delta`` in place; appends loss to ``losses``.

    CQS note: both loss and gradient come from the same forward pass, so the
    loss is recorded as a side channel rather than returned — avoiding the
    command/query collision without a second forward pass.
    """
    loss = loss_fn(vlm, x0 + delta, prompt_tokens, target, gen_kwargs)
    losses.append(float(loss.detach().cpu()))
    grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
    with torch.no_grad():
        delta.add_(step_sign * alpha * grad.sign())
        delta.clamp_(-epsilon, epsilon)
        projected = torch.clamp(x0 + delta, clip_min, clip_max) - x0
        delta.copy_(projected)
        delta.grad = None


def _zero_epsilon_result(
    *,
    x0: torch.Tensor,
    epsilon: float,
    alpha: float,
    clip_min: float,
    clip_max: float,
    extra: Mapping[str, Any],
) -> AttackResult:
    """ε=0 short-circuit. Running the loop would burn ``n_iter * n_restarts``
    forward passes producing exactly zero δ.
    """
    zero_delta = torch.zeros_like(x0)
    perturbed = torch.clamp(x0, clip_min, clip_max)
    return AttackResult(
        perturbed_image=perturbed.squeeze(0) if perturbed.shape[0] == 1 else perturbed,
        delta=zero_delta.squeeze(0) if zero_delta.shape[0] == 1 else zero_delta,
        loss_final=float("nan"),
        loss_trajectory=[],
        iterations=0,
        success=False,
        metadata={"epsilon": epsilon, "alpha": alpha, "short_circuit": "epsilon_zero", **extra},
    )


def _run_one_restart(
    *,
    restart: int,
    loss_fn: LossFn,
    vlm: Any,
    x0: torch.Tensor,
    prompt_tokens: torch.Tensor,
    target: torch.Tensor,
    gen_kwargs: Mapping[str, Any],
    epsilon: float,
    alpha: float,
    n_iter: int,
    step_sign: float,
    clip_min: float,
    clip_max: float,
    extra: Mapping[str, Any],
) -> AttackResult:
    """One PGD random restart: init δ → run n_iter sign-SGD steps → AttackResult."""
    delta = torch.empty_like(x0).uniform_(-epsilon, epsilon)
    delta = torch.clamp(x0 + delta, clip_min, clip_max) - x0
    delta.requires_grad_(True)

    loss_traj: list[float] = []
    for _step in range(n_iter):
        _pgd_step(
            loss_fn=loss_fn,
            vlm=vlm,
            x0=x0,
            delta=delta,
            prompt_tokens=prompt_tokens,
            target=target,
            gen_kwargs=gen_kwargs,
            epsilon=epsilon,
            alpha=alpha,
            step_sign=step_sign,
            clip_min=clip_min,
            clip_max=clip_max,
            losses=loss_traj,
        )

    perturbed = torch.clamp(x0 + delta.detach(), clip_min, clip_max)
    return AttackResult(
        perturbed_image=perturbed.squeeze(0) if perturbed.shape[0] == 1 else perturbed,
        delta=delta.detach().squeeze(0) if delta.shape[0] == 1 else delta.detach(),
        loss_final=loss_traj[-1],
        loss_trajectory=loss_traj,
        iterations=n_iter,
        success=math.isfinite(loss_traj[-1]),
        metadata={"restart": restart, "epsilon": epsilon, "alpha": alpha, **extra},
    )


def _seed_torch_all(seed: int | None) -> None:
    """Pin torch CPU and CUDA RNGs for reproducible restarts."""
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _better_restart(best: AttackResult | None, candidate: AttackResult) -> AttackResult:
    """Pick smaller ``loss_final``; prefer any finite restart over NaN."""
    # NaN guard: ``finite < NaN`` is False, so a NaN-loss first restart
    # would otherwise lock ``best`` and discard every legitimate later
    # restart.
    if best is None:
        return candidate
    if math.isfinite(candidate.loss_final) and (
        not math.isfinite(best.loss_final) or candidate.loss_final < best.loss_final
    ):
        return candidate
    return best


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
    seed: int | None = None,
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
    extra = static_metadata or {}
    if epsilon == 0.0:
        return _zero_epsilon_result(
            x0=x0,
            epsilon=epsilon,
            alpha=alpha,
            clip_min=clip_min,
            clip_max=clip_max,
            extra=extra,
        )

    _seed_torch_all(seed)

    best: AttackResult | None = None
    for restart in range(n_restarts):
        candidate = _run_one_restart(
            restart=restart,
            loss_fn=loss_fn,
            vlm=vlm,
            x0=x0,
            prompt_tokens=prompt_tokens,
            target=target,
            gen_kwargs=gen_kwargs,
            epsilon=epsilon,
            alpha=alpha,
            n_iter=n_iter,
            step_sign=step_sign,
            clip_min=clip_min,
            clip_max=clip_max,
            extra=extra,
        )
        best = _better_restart(best, candidate)

    assert best is not None
    return best


__all__ = ["linf_pgd_loop"]
