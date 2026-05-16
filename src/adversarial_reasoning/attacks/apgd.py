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
from dataclasses import dataclass, field
from typing import Any

import torch

from ._epsilon import _LINF_EPSILON_8
from ._loop import _better_restart
from .base import AttackBase, AttackResult
from .loss import TokenTargetLoss

# Croce & Hein 2020, §3.2 — checkpoint schedule:
#   p_0 = 0, p_1 = _APGD_P1_INIT,
#   p_{j+1} = p_j + max(p_j - p_{j-1} - _APGD_P_DECAY, _APGD_P_FLOOR)
_APGD_P1_INIT: float = 0.22
_APGD_P_DECAY: float = 0.03
_APGD_P_FLOOR: float = 0.06

# Strict step-over-step improvement tolerance for ρ_w success counting.
_APGD_IMPROVEMENT_TOL: float = 1e-12

# Checkpoint-condition tolerances (Croce & Hein 2020):
#   _APGD_ETA_CONVERGED_RTOL — η has not been halved since last checkpoint.
#   _APGD_LOSS_PROGRESS_TOL  — loss_best has not improved by this margin.
#   _APGD_ETA_FLOOR          — minimum η; halving never goes below this.
_APGD_ETA_CONVERGED_RTOL: float = 1e-12
_APGD_LOSS_PROGRESS_TOL: float = 1e-9
_APGD_ETA_FLOOR: float = 1e-8

_APGD_ETA_INIT_MULTIPLIER: float = 2.0  # η starts at multiplier × ε


def _checkpoints(n_iter: int) -> list[int]:
    """Return APGD checkpoint iteration indices for ``n_iter`` total steps."""
    p = [0.0, _APGD_P1_INIT]
    while p[-1] < 1.0:
        nxt = p[-1] + max(p[-1] - p[-2] - _APGD_P_DECAY, _APGD_P_FLOOR)
        p.append(min(nxt, 1.0))
    pts = sorted({max(1, math.ceil(pj * n_iter)) for pj in p[1:]})
    return [c for c in pts if c <= n_iter]


def _checkpoint_triggered(
    success_count: int,
    window: int,
    rho: float,
    eta: float,
    eta_at_last_ckpt: float,
    loss_best: float,
    loss_at_last_ckpt: float,
) -> bool:
    """True when APGD step-size should halve (Croce & Hein 2020 §3.2)."""
    low_success = success_count < rho * window
    eta_stuck = math.isclose(eta, eta_at_last_ckpt, rel_tol=_APGD_ETA_CONVERGED_RTOL)
    loss_stuck = not (loss_best < loss_at_last_ckpt - _APGD_LOSS_PROGRESS_TOL)
    return low_success or (eta_stuck and loss_stuck)


def _step_is_improvement(
    loss_val: float, loss_prev: float, tol: float = _APGD_IMPROVEMENT_TOL
) -> bool:
    """Croce-Hein 2020 ρ_w semantics: strict step-over-step decrease.

    Returns True iff ``loss_val`` is finite and strictly less than the
    previous step's loss by more than ``tol``. Replaces an earlier
    improvement-over-running-best heuristic that under-counted halvings
    once ``loss_best`` saturated.
    """
    return math.isfinite(loss_val) and loss_val < loss_prev - tol


@dataclass
class _ApgdLoopState:
    """Mutable per-restart state for the APGD inner loop."""

    delta: torch.Tensor
    eta: float
    x_prev: torch.Tensor
    x_best: torch.Tensor
    loss_best: float
    loss_prev: float
    success_count: int
    loss_at_last_ckpt: float
    eta_at_last_ckpt: float
    ckpt_idx: int
    loss_traj: list[float] = field(default_factory=list)


@dataclass
class APGDAttack(AttackBase):
    name: str = "apgd_linf"
    epsilon: float = _LINF_EPSILON_8
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
        """Run APGD-Linf and return the best-loss restart."""
        if not getattr(vlm, "supports_gradients", False):
            raise ValueError(f"VLM backend {vlm.__class__.__name__} does not support gradients.")

        x0 = image.detach().clone()
        if x0.ndim == 3:
            x0 = x0.unsqueeze(0)

        if self.epsilon == 0.0:
            return self._zero_epsilon_result(x0)

        loss_fn = TokenTargetLoss(targeted=self.targeted)
        gen_kwargs = forward_kwargs or {}
        checkpoints = _checkpoints(self.steps)
        # Always descend on the loss returned by ``loss_fn``. ``TokenTargetLoss``
        # already encodes the attacker's intent via its ``targeted`` flag
        # (returns ``-CE`` untargeted, ``+CE`` targeted).
        step_sign = -1.0

        self._seed_rngs()

        best: AttackResult | None = None
        for restart in range(self.random_restarts):
            candidate = self._one_restart(
                vlm,
                x0,
                prompt_tokens,
                target,
                loss_fn,
                gen_kwargs,
                checkpoints,
                step_sign,
                restart,
            )
            # ``_better_restart`` handles the NaN-locks-out-finite case that
            # ``a < b`` returns False for: a finite candidate strictly beats a
            # NaN-best, so we never discard a real restart in favour of a
            # poisoned one.
            best = _better_restart(best, candidate)

        if best is None:
            # Unreachable under ``random_restarts >= 1`` but survives ``python -O``.
            raise RuntimeError("APGDAttack produced no restart candidates")
        return best

    def _seed_rngs(self) -> None:
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

    def _zero_epsilon_result(self, x0: torch.Tensor) -> AttackResult:
        """ε=0 ⇒ no admissible perturbation; skip the inner loop. Same
        rationale as ``linf_pgd_loop`` short-circuit."""
        zero_delta = torch.zeros_like(x0)
        perturbed = torch.clamp(x0, self.clip_min, self.clip_max)
        return AttackResult(
            perturbed_image=perturbed.squeeze(0) if perturbed.shape[0] == 1 else perturbed,
            delta=zero_delta.squeeze(0) if zero_delta.shape[0] == 1 else zero_delta,
            # 0.0 (not NaN) so the row does not poison sweep aggregates;
            # ``short_circuit`` lets analysis scripts filter explicitly.
            loss_final=0.0,
            loss_trajectory=[],
            iterations=0,
            success=False,
            metadata={
                "epsilon": self.epsilon,
                "targeted": self.targeted,
                "short_circuit": "epsilon_zero",
            },
        )

    def _init_restart_state(self, x0: torch.Tensor) -> _ApgdLoopState:
        delta = torch.empty_like(x0).uniform_(-self.epsilon, self.epsilon)
        delta = torch.clamp(x0 + delta, self.clip_min, self.clip_max) - x0
        delta.requires_grad_(True)
        eta = _APGD_ETA_INIT_MULTIPLIER * self.epsilon
        x_prev = (x0 + delta).detach().clone()
        return _ApgdLoopState(
            delta=delta,
            eta=eta,
            x_prev=x_prev,
            x_best=x_prev.clone(),
            loss_best=float("inf"),
            loss_prev=float("inf"),
            success_count=0,
            loss_at_last_ckpt=float("inf"),
            eta_at_last_ckpt=eta,
            ckpt_idx=0,
        )

    def _one_restart(
        self,
        vlm: Any,
        x0: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target: torch.Tensor,
        loss_fn: TokenTargetLoss,
        gen_kwargs: dict[str, Any],
        checkpoints: list[int],
        step_sign: float,
        restart: int,
    ) -> AttackResult:
        state = self._init_restart_state(x0)
        for step in range(self.steps):
            self._apgd_step(
                vlm,
                x0,
                prompt_tokens,
                target,
                loss_fn,
                gen_kwargs,
                state,
                step_sign,
                step,
                checkpoints,
            )

        perturbed = torch.clamp(state.x_best, self.clip_min, self.clip_max)
        delta_final = perturbed - x0
        return AttackResult(
            perturbed_image=perturbed.squeeze(0) if perturbed.shape[0] == 1 else perturbed,
            delta=delta_final.squeeze(0) if perturbed.shape[0] == 1 else delta_final,
            loss_final=state.loss_best,
            loss_trajectory=state.loss_traj,
            iterations=self.steps,
            success=math.isfinite(state.loss_best),
            metadata={
                "restart": restart,
                "epsilon": self.epsilon,
                "eta_final": state.eta,
                "targeted": self.targeted,
                "checkpoints": checkpoints,
            },
        )

    def _apgd_step(
        self,
        vlm: Any,
        x0: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target: torch.Tensor,
        loss_fn: TokenTargetLoss,
        gen_kwargs: dict[str, Any],
        state: _ApgdLoopState,
        step_sign: float,
        step: int,
        checkpoints: list[int],
    ) -> None:
        loss = loss_fn(vlm, x0 + state.delta, prompt_tokens, target, gen_kwargs)
        # ``.item()`` syncs once and produces a Python float without an extra
        # device-to-host tensor allocation; the prior ``float(.detach().cpu())``
        # was a ~25 µs/step sync that compounded into seconds across a sweep.
        loss_val = loss.item()
        state.loss_traj.append(loss_val)

        if loss_val < state.loss_best:
            state.loss_best = loss_val
            # In-place copy into pre-allocated x_best (allocated in
            # _init_restart_state) — no per-step tensor allocation.
            state.x_best.copy_((x0 + state.delta).detach())
        # ρ_w (Croce-Hein 2020): success counts strict step-over-step
        # improvement within the current checkpoint window.
        if _step_is_improvement(loss_val, state.loss_prev):
            state.success_count += 1
        state.loss_prev = loss_val

        grad = torch.autograd.grad(loss, state.delta, retain_graph=False)[0]
        state.x_prev = self._apply_momentum_update(
            x0,
            state.delta,
            grad,
            state.eta,
            state.x_prev,
            step_sign,
        )
        self._maybe_halve_step_size(state, x0, step, checkpoints)

    def _apply_momentum_update(
        self,
        x0: torch.Tensor,
        delta: torch.Tensor,
        grad: torch.Tensor,
        eta: float,
        x_prev: torch.Tensor,
        step_sign: float,
    ) -> torch.Tensor:
        with torch.no_grad():
            z = x0 + delta + step_sign * eta * grad.sign()
            z = torch.clamp(z, x0 - self.epsilon, x0 + self.epsilon)
            z = torch.clamp(z, self.clip_min, self.clip_max)
            x_new = (
                (x0 + delta)
                + self.momentum * (z - (x0 + delta))
                + (1.0 - self.momentum) * ((x0 + delta) - x_prev)
            )
            x_new = torch.clamp(x_new, x0 - self.epsilon, x0 + self.epsilon)
            x_new = torch.clamp(x_new, self.clip_min, self.clip_max)
            new_x_prev = (x0 + delta).detach().clone()
            delta_data = (x_new - x0).detach()
            delta.copy_(delta_data)
            delta.grad = None
        return new_x_prev

    def _maybe_halve_step_size(
        self,
        state: _ApgdLoopState,
        x0: torch.Tensor,
        step: int,
        checkpoints: list[int],
    ) -> None:
        if state.ckpt_idx >= len(checkpoints) or (step + 1) != checkpoints[state.ckpt_idx]:
            return
        window = checkpoints[state.ckpt_idx] - (
            checkpoints[state.ckpt_idx - 1] if state.ckpt_idx > 0 else 0
        )
        if _checkpoint_triggered(
            state.success_count,
            window,
            self.rho,
            state.eta,
            state.eta_at_last_ckpt,
            state.loss_best,
            state.loss_at_last_ckpt,
        ):
            with torch.no_grad():
                state.eta = max(state.eta / _APGD_ETA_INIT_MULTIPLIER, _APGD_ETA_FLOOR)
                delta_data = (state.x_best - x0).detach()
                state.delta.copy_(delta_data)
                state.delta.grad = None
                # In-place copy into pre-allocated x_prev avoids the
                # per-checkpoint tensor allocation that the prior .clone()
                # introduced inside the hot loop.
                state.x_prev.copy_(state.x_best)
        state.eta_at_last_ckpt = state.eta
        state.loss_at_last_ckpt = state.loss_best
        state.success_count = 0
        state.ckpt_idx += 1
