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
- Random restart picks the perturbation with the worst (highest) loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .base import AttackBase, AttackResult


@dataclass
class PGDAttack(AttackBase):
    name: str = "pgd_linf"
    epsilon: float = 8.0 / 255.0
    alpha: float | None = None           # step size; default = epsilon / 4
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
        """Run PGD-L∞.

        target: token-id LongTensor (B, T_target) representing the token
            sequence whose log-prob we push (up for targeted, down otherwise).
        forward_kwargs: model-specific extras (e.g. Qwen `image_grid_thw`,
            MLlama `aspect_ratio_ids` / `cross_attention_mask`, and the
            `attention_mask` that covers the concatenated [prompt ‖ target]
            sequence). The attack harness constructs these via
            `vlm.prepare_attack_inputs(...)` before calling `run`.
        """
        if not getattr(vlm, "supports_gradients", False):
            raise ValueError(
                f"VLM backend {vlm.__class__.__name__} does not support gradients."
            )

        x0 = image.detach().clone()
        if x0.ndim == 3:
            x0 = x0.unsqueeze(0)

        alpha = self.alpha if self.alpha is not None else self.epsilon / 4.0
        best: AttackResult | None = None
        fwd_kwargs = forward_kwargs or {}

        for restart in range(self.random_restarts):
            delta = torch.empty_like(x0).uniform_(-self.epsilon, self.epsilon)
            delta = torch.clamp(x0 + delta, self.clip_min, self.clip_max) - x0
            delta.requires_grad_(True)

            loss_traj: list[float] = []
            for _step in range(self.steps):
                loss = self._loss(vlm, x0 + delta, prompt_tokens, target, fwd_kwargs)
                loss_traj.append(float(loss.detach().cpu()))
                grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]

                # Untargeted: ascend (maximize CE) → +alpha * sign.
                # Targeted:   descend (minimize CE towards target) → -alpha * sign.
                direction = -1.0 if self.targeted else 1.0
                with torch.no_grad():
                    delta.add_(direction * alpha * grad.sign())
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
                    "targeted": self.targeted,
                },
            )
            best = self._select_better(best, candidate)

        assert best is not None
        return best

    def _loss(
        self,
        vlm: Any,
        x: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target: torch.Tensor,
        forward_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Teacher-forced cross-entropy against the target token sequence.

        Causal LM invariant: logits at position *i* predict the token at
        position *i+1*. So if we feed [prompt ‖ target] into the VLM, the
        logits that score the target tokens live at positions
        [len(prompt) - 1 : len(prompt) - 1 + len(target)], NOT at the tail.

        For targeted attack: return +CE (descend ⇒ match target).
        For untargeted: return −CE (ascend ⇒ diverge from benign target).
        """
        input_ids = torch.cat([prompt_tokens, target], dim=-1)
        logits = vlm.forward_with_logits(x, input_ids, **(forward_kwargs or {}))

        t_prompt = prompt_tokens.shape[-1]
        t_target = target.shape[-1]
        if logits.shape[1] < t_prompt + t_target:
            raise ValueError(
                f"Logits length {logits.shape[1]} < prompt+target length "
                f"{t_prompt + t_target}. Ensure forward_with_logits returns "
                "per-token logits for the full teacher-forced sequence."
            )

        logits_for_target = logits[:, t_prompt - 1 : t_prompt - 1 + t_target, :]
        ce = F.cross_entropy(
            logits_for_target.reshape(-1, logits_for_target.shape[-1]),
            target.reshape(-1),
        )
        return ce if self.targeted else -ce

    @staticmethod
    def _select_better(
        current: AttackResult | None,
        candidate: AttackResult,
    ) -> AttackResult:
        """Keep the worst-loss restart for untargeted, best-loss for targeted."""
        if current is None:
            return candidate
        # For untargeted (loss = -CE), smaller loss means larger CE → worse for model.
        # For targeted (loss = +CE), smaller loss means better for attacker.
        # In both cases, smaller is better from the attacker's perspective.
        return candidate if candidate.loss_final < current.loss_final else current
