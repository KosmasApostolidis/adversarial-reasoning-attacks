"""Smoke test: PGD converges on a synthetic linear-classifier toy model.

Rationale: we don't load a 7B VLM in CI. Instead we verify PGD's invariants —
ε-bounded δ, monotone-ish loss decrease (targeted) — against a differentiable
stub that exposes the same `forward_with_logits` + `supports_gradients`
contract the real VLM implements.
"""

from __future__ import annotations

import pytest
import torch

from adversarial_reasoning.attacks.pgd import PGDAttack


class _LinearStubVLM:
    """Differentiable toy VLM. Maps flattened pixels → logit over a vocab."""

    supports_gradients = True
    model_id = "toy/linear-stub"

    def __init__(self, in_dim: int = 3 * 16 * 16, vocab: int = 8) -> None:
        torch.manual_seed(0)
        self.weight = torch.randn(vocab, in_dim, requires_grad=False)

    def forward_with_logits(
        self,
        image_tensor: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        flat = image_tensor.reshape(image_tensor.shape[0], -1)
        logits_per_batch = torch.matmul(flat, self.weight.t())  # (B, vocab)
        seq_len = int(input_ids.shape[-1])
        # Broadcast image-derived logits across all seq positions (stub only).
        # PGD's causal alignment selects logits[:, t_prompt-1 : t_prompt-1+t_target, :];
        # returning full (B, seq_len, vocab) satisfies the length check and keeps
        # gradient flow from image pixels to the target position.
        return logits_per_batch.unsqueeze(1).expand(-1, seq_len, -1).contiguous()


@pytest.mark.smoke
def test_pgd_respects_epsilon_budget():
    vlm = _LinearStubVLM()
    image = torch.rand(3, 16, 16)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.zeros(1, 1, dtype=torch.long)

    attack = PGDAttack(
        epsilon=8.0 / 255.0,
        steps=20,
        random_restarts=1,
        targeted=False,
    )
    result = attack.run(vlm, image, prompt, target)

    linf = result.delta.abs().max().item()
    assert linf <= 8.0 / 255.0 + 1e-6


@pytest.mark.smoke
def test_pgd_targeted_reduces_ce_to_target():
    vlm = _LinearStubVLM()
    image = torch.rand(3, 16, 16)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.tensor([[3]], dtype=torch.long)  # push the logits towards token 3

    attack = PGDAttack(
        epsilon=16.0 / 255.0,
        steps=40,
        random_restarts=1,
        targeted=True,
    )
    result = attack.run(vlm, image, prompt, target)

    # Loss trajectory should end lower than it started under a targeted attack.
    assert result.loss_trajectory[-1] <= result.loss_trajectory[0]


@pytest.mark.smoke
def test_pgd_rejects_gradient_free_backend():
    class _NoGradVLM:
        supports_gradients = False
        model_id = "toy/nograd"

    attack = PGDAttack(epsilon=4.0 / 255.0, steps=5)
    with pytest.raises(ValueError):
        attack.run(
            _NoGradVLM(),
            torch.rand(3, 8, 8),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
        )
