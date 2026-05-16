"""L∞ sign-SGD gradient loop with random restarts.

Shared by :class:`PGDAttack` and :class:`TrajectoryDriftPGD`. APGD has its
own adaptive-step loop (heavy-ball momentum + Croce-Hein checkpoint
schedule with warm restart from best iterate) and does NOT route through
this helper — preserving APGD's byte-identical output across the
refactor was simpler than abstracting both update rules.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .base import AttackResult
from .loss import LossFn


@dataclass(frozen=True)
class LinfPGDConfig:
    """Bundled L∞ PGD hyperparameters shared by loop / restart / step."""

    epsilon: float
    alpha: float
    n_iter: int
    n_restarts: int
    step_sign: float = 1.0
    clip_min: float = 0.0
    clip_max: float = 1.0
    seed: int | None = None
    static_metadata: Mapping[str, Any] = field(default_factory=dict)


def _pgd_step(
    *,
    loss_fn: LossFn,
    vlm: Any,
    x0: torch.Tensor,
    delta: torch.Tensor,
    prompt_tokens: torch.Tensor,
    target: torch.Tensor,
    gen_kwargs: Mapping[str, Any],
    cfg: LinfPGDConfig,
    losses: list[float],
) -> None:
    """Single sign-SGD step. Mutates ``delta`` in place; appends loss to ``losses``.

    CQS note: loss + grad come from one forward pass; loss is recorded as a
    side channel rather than returned, avoiding a second forward pass.
    NaN/Inf grads (typical in fp16 overflow) skip the step rather than
    propagating NaN into δ and poisoning every subsequent iterate.
    Loss `.item()` is taken *after* `autograd.grad` releases the graph.
    """
    loss = loss_fn(vlm, x0 + delta, prompt_tokens, target, gen_kwargs)
    grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
    losses.append(loss.item())
    with torch.no_grad():
        if not torch.isfinite(grad).all():
            # Skip the step; δ unchanged. Loss recorded so caller can see
            # the non-finite step in the trajectory.
            delta.grad = None
            return
        delta.add_(cfg.step_sign * cfg.alpha * grad.sign())
        delta.clamp_(-cfg.epsilon, cfg.epsilon)
        projected = torch.clamp(x0 + delta, cfg.clip_min, cfg.clip_max) - x0
        delta.copy_(projected)
        delta.grad = None


def _zero_epsilon_result(x0: torch.Tensor, cfg: LinfPGDConfig) -> AttackResult:
    """ε=0 short-circuit. Loop would burn n_iter * n_restarts forwards for zero δ.

    ``loss_final`` is 0.0 (not NaN) so the value does not poison aggregate
    statistics when ε=0 rows are mixed into a sweep. ``metadata.short_circuit``
    flags the row for figure-scripts that want to exclude it.
    """
    zero_delta = torch.zeros_like(x0)
    perturbed = torch.clamp(x0, cfg.clip_min, cfg.clip_max)
    return AttackResult(
        perturbed_image=perturbed.squeeze(0) if perturbed.shape[0] == 1 else perturbed,
        delta=zero_delta.squeeze(0) if zero_delta.shape[0] == 1 else zero_delta,
        loss_final=0.0,
        loss_trajectory=[],
        iterations=0,
        success=False,
        metadata={
            "epsilon": cfg.epsilon,
            "alpha": cfg.alpha,
            "short_circuit": "epsilon_zero",
            **cfg.static_metadata,
        },
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
    cfg: LinfPGDConfig,
) -> AttackResult:
    """One PGD random restart: init δ → run cfg.n_iter sign-SGD steps → AttackResult.

    Returns the *best-loss* iterate, not the last. Sign-SGD is non-monotonic
    so the final iterate can be arbitrarily worse than a mid-run one;
    APGD already tracks ``x_best`` and PGD now matches.
    """
    delta = torch.empty_like(x0).uniform_(-cfg.epsilon, cfg.epsilon)
    delta = torch.clamp(x0 + delta, cfg.clip_min, cfg.clip_max) - x0
    delta.requires_grad_(True)

    # Pre-allocate the best-iterate buffer once; copy_ in-place when we
    # see an improvement (avoids per-step allocation).
    best_delta = delta.detach().clone()
    best_loss: float = float("inf")
    loss_traj: list[float] = []
    for _step in range(cfg.n_iter):
        _pgd_step(
            loss_fn=loss_fn,
            vlm=vlm,
            x0=x0,
            delta=delta,
            prompt_tokens=prompt_tokens,
            target=target,
            gen_kwargs=gen_kwargs,
            cfg=cfg,
            losses=loss_traj,
        )
        last_loss = loss_traj[-1]
        if math.isfinite(last_loss) and last_loss < best_loss:
            best_loss = last_loss
            best_delta.copy_(delta.detach())

    # If every step produced NaN/Inf, fall back to the random-initialised δ;
    # caller's `_better_restart` will drop the candidate against any finite
    # restart, but the attack still returns a well-formed AttackResult.
    if not math.isfinite(best_loss):
        best_loss = loss_traj[-1] if loss_traj else float("nan")

    # Loud failure: ε>0 must produce a non-zero perturbation. Zero δ at
    # ε>0 means every step was skipped (all-NaN grads) or the gradient
    # was identically zero — both bugs the caller needs to see.
    if cfg.epsilon > 0.0 and float(best_delta.abs().max()) == 0.0:
        raise RuntimeError(
            f"PGD produced zero perturbation at ε={cfg.epsilon} after "
            f"{cfg.n_iter} steps (restart {restart}); every grad step was "
            "skipped (likely all NaN/Inf gradients) or the loss surface "
            "is degenerate. Inspect the loss/forward path."
        )

    perturbed = torch.clamp(x0 + best_delta, cfg.clip_min, cfg.clip_max)
    return AttackResult(
        perturbed_image=perturbed.squeeze(0) if perturbed.shape[0] == 1 else perturbed,
        delta=best_delta.squeeze(0) if best_delta.shape[0] == 1 else best_delta,
        loss_final=best_loss,
        loss_trajectory=loss_traj,
        iterations=cfg.n_iter,
        success=math.isfinite(best_loss),
        metadata={
            "restart": restart,
            "epsilon": cfg.epsilon,
            "alpha": cfg.alpha,
            **cfg.static_metadata,
        },
    )


def _seed_torch_all(seed: int | None) -> None:
    """Pin every RNG that affects an attack restart for full reproducibility.

    cuDNN convolution algorithm selection is non-deterministic by default;
    `deterministic=True, benchmark=False` is the documented recipe.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _better_restart(best: AttackResult | None, candidate: AttackResult) -> AttackResult:
    """Pick smaller ``loss_final``; prefer any finite restart over NaN."""
    # NaN guard: ``finite < NaN`` is False, so a NaN-loss first restart
    # would otherwise lock ``best`` and discard every legitimate later restart.
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
    cfg: LinfPGDConfig,
) -> AttackResult:
    """L-inf sign-SGD with random restarts; returns best-loss restart.

    Caller owns the sign convention via ``cfg.step_sign`` + the sign of
    ``loss_fn``'s return value. Random restart selection: smallest
    ``loss_final`` wins (smaller-is-better for the attacker, matching the
    legacy ``_select_better`` convention).
    """
    if cfg.epsilon == 0.0:
        return _zero_epsilon_result(x0, cfg)

    if cfg.n_restarts < 1:
        raise ValueError(
            f"n_restarts must be >= 1, got {cfg.n_restarts}. "
            "Use the ε=0 short-circuit if you want a no-op attack."
        )

    _seed_torch_all(cfg.seed)

    best: AttackResult | None = None
    for restart in range(cfg.n_restarts):
        candidate = _run_one_restart(
            restart=restart,
            loss_fn=loss_fn,
            vlm=vlm,
            x0=x0,
            prompt_tokens=prompt_tokens,
            target=target,
            gen_kwargs=gen_kwargs,
            cfg=cfg,
        )
        best = _better_restart(best, candidate)

    if best is None:
        # Unreachable while n_restarts >= 1, but `assert` is stripped under
        # `python -O`; an explicit raise survives optimisation.
        raise RuntimeError("linf_pgd_loop produced no restart candidates")
    return best


__all__ = ["LinfPGDConfig", "linf_pgd_loop"]
