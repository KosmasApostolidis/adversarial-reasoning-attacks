"""Differentiable losses for gradient-based attacks on VLM agents.

Each loss returns a scalar tensor consumed by an attack's gradient loop.
The sign convention matches the existing ascent-on-loss formulation in
:class:`PGDAttack` / :class:`APGDAttack` / :class:`TrajectoryDriftPGD`:

- :class:`TokenTargetLoss` (untargeted): returns ``-CE(target)`` so an
  ascent step pushes the attacker AWAY from ``target`` token likelihood.
- :class:`TokenTargetLoss` (targeted):   returns ``+CE(target)`` so the
  step direction (typically inverted by the loop's per-attack sign flag)
  pulls the attacker TOWARD ``target``.
- :class:`TrajectoryDriftLoss`: returns ``-KL(attacked || benign)``.

Refactor note (2026-04-25): these classes consolidate three duplicated
loss bodies — ``PGDAttack._loss`` (used by APGD via proxy), the inline
KL block inside ``TrajectoryDriftPGD.run``, and the cross-entropy logic
that was previously coupled to the runner's target-construction helpers.
The numeric outputs are byte-identical to the prior implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from torch.nn import functional as F

# ``GenKwargs`` is the *documented* shape (TypedDict in
# :mod:`adversarial_reasoning.types`); we accept ``Mapping[str, Any]``
# at the parameter to keep callers free to pass plain ``dict``s (which
# is what the runner builds) without tripping TypedDict invariance.


@runtime_checkable
class LossFn(Protocol):
    """Stateless (or pre-cached) differentiable loss for one attack step.

    Implementations are responsible for the sign convention. The attack
    loop applies a fixed sign-step to the gradient of the value returned
    here; flipping the meaning of "attacker-favorable direction" lives
    inside :class:`LossFn`.
    """

    def __call__(
        self,
        vlm: Any,
        perturbed_pixels: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target: torch.Tensor,
        gen_kwargs: Mapping[str, Any],
    ) -> torch.Tensor: ...


def _logits_for_target(logits: torch.Tensor, t_prompt: int, t_target: int) -> torch.Tensor:
    """Slice causal-LM logits scoring the teacher-forced ``target`` tokens.

    Causal LM invariant: logits at position ``i`` predict the token at
    position ``i + 1``. Feeding ``[prompt ‖ target]`` puts target-scoring
    logits at ``[t_prompt - 1 : t_prompt - 1 + t_target]`` — NOT at the
    tail of the sequence.
    """
    # t_prompt==0 produces an off-by-one slice: logits[:, -1:t_target-1] is
    # an empty or wrong slice; the length check below would still pass.
    if t_prompt < 1:
        raise ValueError(
            f"t_prompt must be >= 1 (got {t_prompt}); causal-LM slicing "
            "requires at least one prompt token to score the first target."
        )
    if logits.shape[1] < t_prompt + t_target:
        raise ValueError(
            f"Logits length {logits.shape[1]} < prompt+target length "
            f"{t_prompt + t_target}. Ensure forward_with_logits returns "
            "per-token logits for the full teacher-forced sequence."
        )
    return logits[:, t_prompt - 1 : t_prompt - 1 + t_target, :]


@dataclass
class TokenTargetLoss:
    """Teacher-forced cross-entropy against a fixed target token sequence.

    Args:
        targeted: when ``True``, return ``+CE`` (caller intends to pull the
            attacker TOWARD the target); when ``False``, return ``-CE``
            (caller intends to push AWAY from the target).
    """

    targeted: bool = False

    def __call__(
        self,
        vlm: Any,
        perturbed_pixels: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target: torch.Tensor,
        gen_kwargs: Mapping[str, Any],
    ) -> torch.Tensor:
        input_ids = torch.cat([prompt_tokens, target], dim=-1)
        logits = vlm.forward_with_logits(perturbed_pixels, input_ids, **gen_kwargs)
        sliced = _logits_for_target(logits, prompt_tokens.shape[-1], target.shape[-1])
        ce = F.cross_entropy(
            sliced.reshape(-1, sliced.shape[-1]),
            target.reshape(-1),
        )
        return ce if self.targeted else -ce


# Floor used when taking log of cached benign probabilities. Avoids -inf
# when a vocab entry has p≈0; small enough not to perturb finite values.
_LOG_PROB_FLOOR: float = 1e-12


@dataclass
class TrajectoryDriftLoss:
    """Negative KL divergence from cached benign next-token distribution.

    Construction is eager: build via :meth:`from_benign` once with the
    *unperturbed* image; subsequent calls reuse the cached benign
    reference (no_grad, detached). Returns ``-KL(attacked || benign)`` so
    an ascent step pushes the attacked distribution AWAY from benign.

    The KL is computed manually as ``Σ p_attacked · (log p_attacked −
    log p_benign)`` averaged over (B × T). PyTorch's ``F.kl_div(input,
    target)`` computes ``KL(target || input)`` — i.e. the reverse
    direction — and ``reduction="batchmean"`` divides by B only (not
    B·T), so loss magnitude scales with target length and ε-sweeps
    across tasks of different trajectory length are not comparable.
    Both bugs (#review 2026-05-16) are fixed here.

    .. note::
        ``p_benign`` already lives on the same device as the model; the
        loop must not move it across devices between calls.
    """

    p_benign: torch.Tensor
    log_p_benign: torch.Tensor
    t_prompt: int
    t_target: int

    @classmethod
    def from_benign(
        cls,
        vlm: Any,
        x0: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target: torch.Tensor,
        gen_kwargs: Mapping[str, Any],
    ) -> TrajectoryDriftLoss:
        with torch.no_grad():
            input_ids = torch.cat([prompt_tokens, target], dim=-1)
            logits = vlm.forward_with_logits(x0, input_ids, **gen_kwargs)
            sliced = _logits_for_target(logits, prompt_tokens.shape[-1], target.shape[-1]).detach()
            log_benign = F.log_softmax(sliced, dim=-1)
            p_benign = log_benign.exp()
        return cls(
            p_benign=p_benign,
            log_p_benign=log_benign,
            t_prompt=prompt_tokens.shape[-1],
            t_target=target.shape[-1],
        )

    def __call__(
        self,
        vlm: Any,
        perturbed_pixels: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target: torch.Tensor,
        gen_kwargs: Mapping[str, Any],
    ) -> torch.Tensor:
        input_ids = torch.cat([prompt_tokens, target], dim=-1)
        logits = vlm.forward_with_logits(perturbed_pixels, input_ids, **gen_kwargs)
        sliced = _logits_for_target(logits, self.t_prompt, self.t_target)
        log_attacked = F.log_softmax(sliced, dim=-1)
        p_attacked = log_attacked.exp()
        # Manual KL(p_attacked || p_benign), token-averaged so loss is
        # comparable across tasks with different target lengths.
        kl_per_token = (p_attacked * (log_attacked - self.log_p_benign)).sum(dim=-1)
        kl = kl_per_token.mean()  # mean over (B, T)
        return -kl  # ascent on -KL ⇒ KL diverges


__all__ = [
    "LossFn",
    "TokenTargetLoss",
    "TrajectoryDriftLoss",
]
