"""Smoke test for TrajectoryDriftPGD on a synthetic linear stub.

Mirrors test_attacks.py — verifies the KL-ascent attack respects ε,
exposes ``kl_final`` in metadata, and rejects gradient-free backends.
"""

from __future__ import annotations

import pytest
import torch

from adversarial_reasoning.attacks.trajectory_drift import TrajectoryDriftPGD


class _LinearStubVLM:
    supports_gradients = True
    model_id = "toy/linear-stub"

    def __init__(self, in_dim: int = 3 * 8 * 8, vocab: int = 8) -> None:
        torch.manual_seed(0)
        self.weight = torch.randn(vocab, in_dim, requires_grad=False)

    def forward_with_logits(
        self,
        image_tensor: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        flat = image_tensor.reshape(image_tensor.shape[0], -1)
        logits = torch.matmul(flat, self.weight.t())
        seq_len = int(input_ids.shape[-1])
        return logits.unsqueeze(1).expand(-1, seq_len, -1).contiguous()


@pytest.mark.smoke
def test_drift_respects_epsilon_budget() -> None:
    vlm = _LinearStubVLM()
    image = torch.rand(3, 8, 8)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    # Trajectory target: a small concatenated tool sequence.
    target = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    attack = TrajectoryDriftPGD(
        epsilon=8.0 / 255.0,
        steps=10,
        random_restarts=1,
    )
    result = attack.run(vlm, image, prompt, target)
    assert result.delta.abs().max().item() <= 8.0 / 255.0 + 1e-6


@pytest.mark.smoke
def test_drift_metadata_includes_kl_final() -> None:
    vlm = _LinearStubVLM()
    image = torch.rand(3, 8, 8)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.tensor([[1, 2, 3]], dtype=torch.long)

    result = TrajectoryDriftPGD(epsilon=4.0 / 255.0, steps=8).run(vlm, image, prompt, target)
    assert "kl_final" in result.metadata
    # KL ≥ 0 by definition.
    assert result.metadata["kl_final"] >= 0.0


@pytest.mark.smoke
def test_drift_uses_default_alpha_when_unspecified() -> None:
    vlm = _LinearStubVLM()
    image = torch.rand(3, 8, 8)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.tensor([[2, 3]], dtype=torch.long)

    a = TrajectoryDriftPGD(epsilon=8.0 / 255.0, alpha=None, steps=5)
    result = a.run(vlm, image, prompt, target)
    assert result.iterations == 5


@pytest.mark.smoke
def test_drift_rejects_gradient_free_backend() -> None:
    class _NoGradVLM:
        supports_gradients = False
        model_id = "toy/nograd"

    with pytest.raises(ValueError, match="does not support gradients"):
        TrajectoryDriftPGD(epsilon=4.0 / 255.0, steps=4).run(
            _NoGradVLM(),
            torch.rand(3, 8, 8),
            torch.zeros(1, 1, dtype=torch.long),
            torch.tensor([[1, 2]], dtype=torch.long),
        )


@pytest.mark.smoke
def test_drift_accepts_4d_input_image() -> None:
    vlm = _LinearStubVLM()
    image = torch.rand(1, 3, 8, 8)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.tensor([[1, 2]], dtype=torch.long)
    result = TrajectoryDriftPGD(epsilon=4.0 / 255.0, steps=4).run(vlm, image, prompt, target)
    assert result.delta.abs().max().item() <= 4.0 / 255.0 + 1e-6
