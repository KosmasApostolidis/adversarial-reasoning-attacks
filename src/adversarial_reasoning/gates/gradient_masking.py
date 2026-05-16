"""Phase-0 gate: Athalye-Carlini-Wagner 2018 gradient-masking checklist.

A gradient-based adversarial attack reports honest robustness numbers
only if the loss surface actually exposes useful gradients. Athalye,
Carlini & Wagner 2018 (`Obfuscated gradients give a false sense of
security`) list four canonical sanity checks; this gate runs all four
and gates further attack work behind their joint pass.

Checks
------
(a) huge-ε loss drop — at ε → ∞ (here ``huge_epsilon``) the attack must
    drive the loss far below the benign baseline. Failure indicates the
    optimiser is stuck (gradient masking, vanishing gradients, or a
    degenerate loss surface).
(b) PGD beats uniform noise — at the same ε budget, the attack's
    final loss must be ≤ the loss reached by uniform L∞-noise of the
    same magnitude. If random noise matches or beats PGD, the gradients
    are not informative.
(c) Loss monotonicity — across PGD iterates the attacker loss should
    decrease monotonically (modulo small step-size noise). A loss
    trajectory that wanders or rises after early iterates is the
    classic obfuscated-gradients fingerprint.
(d) Gradient-norm non-collapse — ``||∇_x L||`` at the attacked iterate
    must not collapse relative to the benign baseline. A ratio near
    zero means ``sign(grad)`` is amplifying numerical noise rather
    than following signal.

Each check produces a boolean; the gate passes iff all four pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


_MONOTONICITY_THRESHOLD = 0.8
"""Fraction of consecutive iterates with non-increasing loss required."""

_HUGE_EPS_LOSS_DROP_FRAC = 0.5
"""Huge-ε attack must reduce loss by at least this fraction of benign."""

_GRAD_NORM_FLOOR_RATIO = 0.1
"""Attacked-iterate ``||∇L||`` must be ≥ this · benign ``||∇L||``."""


@dataclass
class GradientMaskingResult:
    model_name: str
    epsilon: float
    huge_epsilon: float
    benign_loss: float
    pgd_loss: float
    noise_loss: float
    huge_eps_loss: float
    benign_grad_norm: float
    attacked_grad_norm: float
    loss_trajectory: list[float] = field(default_factory=list)

    # Per-check verdicts (computed in __post_init__).
    huge_eps_passes: bool = False
    pgd_beats_noise: bool = False
    loss_monotonic: bool = False
    grad_norm_alive: bool = False

    def __post_init__(self) -> None:
        self.huge_eps_passes = self._check_huge_eps_passes()
        self.pgd_beats_noise = self.pgd_loss <= self.noise_loss
        self.loss_monotonic = _is_monotonic(self.loss_trajectory)
        self.grad_norm_alive = self._check_grad_norm_alive()

    def _check_huge_eps_passes(self) -> bool:
        """Huge-ε must drop loss by at least ``_HUGE_EPS_LOSS_DROP_FRAC``
        of the benign-baseline magnitude. We compute the drop on
        ``|benign_loss|`` so the threshold is well-defined regardless
        of the (signed) loss convention used by the attack."""
        if self.benign_loss == 0.0:
            return self.huge_eps_loss < 0.0
        drop = self.benign_loss - self.huge_eps_loss
        return drop >= _HUGE_EPS_LOSS_DROP_FRAC * abs(self.benign_loss)

    def _check_grad_norm_alive(self) -> bool:
        if self.benign_grad_norm == 0.0:
            return self.attacked_grad_norm > 0.0
        ratio = self.attacked_grad_norm / self.benign_grad_norm
        return ratio >= _GRAD_NORM_FLOOR_RATIO

    @property
    def passes(self) -> bool:
        return (
            self.huge_eps_passes
            and self.pgd_beats_noise
            and self.loss_monotonic
            and self.grad_norm_alive
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "epsilon": self.epsilon,
            "huge_epsilon": self.huge_epsilon,
            "benign_loss": self.benign_loss,
            "pgd_loss": self.pgd_loss,
            "noise_loss": self.noise_loss,
            "huge_eps_loss": self.huge_eps_loss,
            "benign_grad_norm": self.benign_grad_norm,
            "attacked_grad_norm": self.attacked_grad_norm,
            "loss_trajectory": [float(x) for x in self.loss_trajectory],
            "huge_eps_passes": self.huge_eps_passes,
            "pgd_beats_noise": self.pgd_beats_noise,
            "loss_monotonic": self.loss_monotonic,
            "grad_norm_alive": self.grad_norm_alive,
            "passes": self.passes,
        }


def _is_monotonic(trajectory: Sequence[float]) -> bool:
    """Return True if ≥ ``_MONOTONICITY_THRESHOLD`` of consecutive
    iterates are non-increasing. We allow small slips because PGD's
    sign-step + ε-projection is not strictly monotone on every step."""
    if len(trajectory) < 2:
        return False
    pairs = list(zip(trajectory[:-1], trajectory[1:]))
    non_increasing = sum(1 for prev, curr in pairs if curr <= prev)
    return (non_increasing / len(pairs)) >= _MONOTONICITY_THRESHOLD


def run_gradient_masking(
    *,
    model_name: str,
    epsilon: float,
    huge_epsilon: float,
    benign_loss: float,
    pgd_loss: float,
    noise_loss: float,
    huge_eps_loss: float,
    benign_grad_norm: float,
    attacked_grad_norm: float,
    loss_trajectory: Sequence[float],
) -> GradientMaskingResult:
    """Run the four-check gate from already-computed attack telemetry.

    The caller is responsible for actually invoking PGD at ``epsilon``
    and ``huge_epsilon``, sampling uniform noise at ``epsilon``, and
    extracting the gradient norms; this function only checks the
    invariants. That keeps the gate testable without requiring a real
    VLM and matches how :mod:`noise_floor` decouples agent execution
    from the gate verdict.
    """
    return GradientMaskingResult(
        model_name=model_name,
        epsilon=epsilon,
        huge_epsilon=huge_epsilon,
        benign_loss=benign_loss,
        pgd_loss=pgd_loss,
        noise_loss=noise_loss,
        huge_eps_loss=huge_eps_loss,
        benign_grad_norm=benign_grad_norm,
        attacked_grad_norm=attacked_grad_norm,
        loss_trajectory=list(loss_trajectory),
    )


def write_gate_report(result: GradientMaskingResult, out_path: str | Path) -> None:
    import json

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
