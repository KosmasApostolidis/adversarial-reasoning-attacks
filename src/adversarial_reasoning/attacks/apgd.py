"""APGD L∞ — Auto-PGD with adaptive step size (Croce & Hein, 2020).

Hand-rolled implementation (no IBM ART dependency) so it composes with
the same VLM ``forward_with_logits`` path that :class:`PGDAttack` uses.

Differences vs PGD:
  - Step size η starts at ``2 * ε`` and halves at "checkpoints" when the
    optimizer stagnates (loss plateau or low success-rate of steps).
  - Heavy-ball momentum: each iterate combines the projected gradient
    step with a momentum term over the previous iterate.
  - On halving, the search restarts from the best-loss iterate seen so
    far (warm restart), not from random.

Checkpoint schedule (paper §3.2): ``p_0 = 0``, ``p_1 = 0.22``,
``p_{j+1} = p_j + max(p_j - p_{j-1} - 0.03, 0.06)``. The j-th
checkpoint is at iteration ``ceil(p_j * N)``.

Refactor note (2026-04-25): the cross-entropy loss body that APGD
shared with PGD via ``proxy._loss`` is now provided by
:class:`adversarial_reasoning.attacks.loss.TokenTargetLoss`. The
adaptive-step + warm-restart logic stays APGD-private — APGD's update
rule is too distinct to share with PGD's plain sign-SGD loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .base import AttackBase, AttackResult
from .loss import TokenTargetLoss


def _checkpoints(n_iter: int) -> list[int]:
    """Return APGD checkpoint iteration indices for ``n_iter`` total steps."""
    p = [0.0, 0.22]
    while p[-1] < 1.0:
        nxt = p[-1] + max(p[-1] - p[-2] - 0.03, 0.06)
        p.append(min(nxt, 1.0))
    pts = sorted({max(1, int(math.ceil(pj * n_iter))) for pj in p[1:]})
    return [c for c in pts if c <= n_iter]


def _step_is_improvement(loss_val: float, loss_prev: float, tol: float = 1e-12) -> bool:
    """Croce-Hein 2020 ρ_w semantics: strict step-over-step decrease.

    Returns True iff ``loss_val`` is finite and strictly less than the
    previous step's loss by more than ``tol``. Replaces an earlier
    improvement-over-running-best heuristic that under-counted halvings
    once ``loss_best`` saturated.
    """
    return math.isfinite(loss_val) and loss_val < loss_prev - tol


@dataclass
class APGDAttack(AttackBase):
    name: str = "apgd_linf"
    epsilon: float = 8.0 / 255.0
    steps: int = 100
    random_restarts: int = 1
    targeted: bool = False
    momentum: float = 0.75
    rho: float = 0.75
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
        if not getattr(vlm, "supports_gradients", False):
            raise ValueError(f"VLM backend {vlm.__class__.__name__} does not support gradients.")

        x0 = image.detach().clone()
        if x0.ndim == 3:
            x0 = x0.unsqueeze(0)

        # ε=0 ⇒ no admissible perturbation; skip the inner loop. Same
        # rationale as ``linf_pgd_loop`` short-circuit. Note ``grad.sign()``
        # returns 0 in saturated regions — sign-SGD limitation, not fixed
        # here.
        if self.epsilon == 0.0:
            zero_delta = torch.zeros_like(x0)
            perturbed = torch.clamp(x0, self.clip_min, self.clip_max)
            return AttackResult(
                perturbed_image=perturbed.squeeze(0) if perturbed.shape[0] == 1 else perturbed,
                delta=zero_delta.squeeze(0) if zero_delta.shape[0] == 1 else zero_delta,
                loss_final=float("nan"),
                loss_trajectory=[],
                iterations=0,
                success=False,
                metadata={
                    "epsilon": self.epsilon,
                    "targeted": self.targeted,
                    "short_circuit": "epsilon_zero",
                },
            )

        loss_fn = TokenTargetLoss(targeted=self.targeted)
        gen_kwargs = forward_kwargs or {}
        checkpoints = _checkpoints(self.steps)
        sign = -1.0 if self.targeted else 1.0  # smaller-loss-is-better convention

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        best: AttackResult | None = None
        for restart in range(self.random_restarts):
            delta = torch.empty_like(x0).uniform_(-self.epsilon, self.epsilon)
            delta = torch.clamp(x0 + delta, self.clip_min, self.clip_max) - x0
            delta.requires_grad_(True)

            eta = 2.0 * self.epsilon
            x_prev = (x0 + delta).detach().clone()
            x_best = x_prev.clone()
            loss_best = float("inf")
            loss_prev = float("inf")
            loss_traj: list[float] = []
            success_count = 0
            loss_at_last_ckpt = float("inf")
            eta_at_last_ckpt = eta
            ckpt_idx = 0

            for step in range(self.steps):
                loss = loss_fn(vlm, x0 + delta, prompt_tokens, target, gen_kwargs)
                loss_val = float(loss.detach().cpu())
                loss_traj.append(loss_val)

                if loss_val < loss_best:
                    loss_best = loss_val
                    x_best = (x0 + delta).detach().clone()
                # ρ_w (Croce-Hein 2020): success counts strict step-over-step
                # improvement within the current checkpoint window, not
                # improvement over the running best.
                if _step_is_improvement(loss_val, loss_prev):
                    success_count += 1
                loss_prev = loss_val

                grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]

                with torch.no_grad():
                    z = x0 + delta - sign * eta * grad.sign()
                    z = torch.clamp(z, x0 - self.epsilon, x0 + self.epsilon)
                    z = torch.clamp(z, self.clip_min, self.clip_max)
                    x_new = (
                        (x0 + delta)
                        + self.momentum * (z - (x0 + delta))
                        + (1.0 - self.momentum) * ((x0 + delta) - x_prev)
                    )
                    x_new = torch.clamp(x_new, x0 - self.epsilon, x0 + self.epsilon)
                    x_new = torch.clamp(x_new, self.clip_min, self.clip_max)
                    x_prev = (x0 + delta).detach().clone()
                    delta_data = (x_new - x0).detach()
                    delta.copy_(delta_data)
                    delta.grad = None

                if ckpt_idx < len(checkpoints) and (step + 1) == checkpoints[ckpt_idx]:
                    window = checkpoints[ckpt_idx] - (
                        checkpoints[ckpt_idx - 1] if ckpt_idx > 0 else 0
                    )
                    cond1 = success_count < self.rho * window
                    cond2 = math.isclose(eta, eta_at_last_ckpt, rel_tol=1e-12) and not (
                        loss_best < loss_at_last_ckpt - 1e-9
                    )
                    if cond1 or cond2:
                        with torch.no_grad():
                            eta = max(eta / 2.0, 1e-8)
                            delta_data = (x_best - x0).detach()
                            delta.copy_(delta_data)
                            delta.grad = None
                            x_prev = x_best.clone()
                    eta_at_last_ckpt = eta
                    loss_at_last_ckpt = loss_best
                    success_count = 0
                    ckpt_idx += 1

            perturbed = torch.clamp(x_best, self.clip_min, self.clip_max)
            candidate = AttackResult(
                perturbed_image=perturbed.squeeze(0) if perturbed.shape[0] == 1 else perturbed,
                delta=(perturbed - x0).squeeze(0) if perturbed.shape[0] == 1 else (perturbed - x0),
                loss_final=loss_best,
                loss_trajectory=loss_traj,
                iterations=self.steps,
                success=math.isfinite(loss_best),
                metadata={
                    "restart": restart,
                    "epsilon": self.epsilon,
                    "eta_final": eta,
                    "targeted": self.targeted,
                    "checkpoints": checkpoints,
                },
            )
            if best is None or candidate.loss_final < best.loss_final:
                best = candidate

        assert best is not None
        return best
