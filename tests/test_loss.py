"""Unit tests for the loss classes in ``adversarial_reasoning.attacks.loss``.

These tests pin the sign convention and the target-position slicing
described in the module docstring. They use a tiny linear stub VLM so
gradients flow back to the input image without loading a real backbone.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from adversarial_reasoning.attacks.loss import (
    TokenTargetLoss,
    TrajectoryDriftLoss,
    _logits_for_target,
)


class _LinearStubVLM:
    supports_gradients = True
    model_id = "toy/loss-stub"

    def __init__(self, in_dim: int = 3 * 8 * 8, vocab: int = 6) -> None:
        torch.manual_seed(0)
        self.weight = torch.randn(vocab, in_dim, requires_grad=False)

    def forward_with_logits(
        self, image_tensor: torch.Tensor, input_ids: torch.Tensor, **_: object
    ) -> torch.Tensor:
        flat = image_tensor.reshape(image_tensor.shape[0], -1)
        per_batch = flat @ self.weight.t()
        seq_len = int(input_ids.shape[-1])
        return per_batch.unsqueeze(1).expand(-1, seq_len, -1).contiguous()


@pytest.mark.smoke
def test_token_target_loss_sign_convention_untargeted():
    """Untargeted: returns ``-CE``. Sign should match ``-CE`` exactly."""
    vlm = _LinearStubVLM()
    image = torch.rand(1, 3, 8, 8)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.tensor([[2]], dtype=torch.long)

    loss = TokenTargetLoss(targeted=False)
    val = loss(vlm, image, prompt, target, {})

    # Reference: compute CE manually and confirm the sign flip.
    input_ids = torch.cat([prompt, target], dim=-1)
    logits = vlm.forward_with_logits(image, input_ids)
    ref_ce = F.cross_entropy(
        logits[:, 0:1, :].reshape(-1, logits.shape[-1]), target.reshape(-1)
    )
    assert torch.allclose(val, -ref_ce, atol=1e-6)


@pytest.mark.smoke
def test_token_target_loss_sign_convention_targeted():
    """Targeted: returns ``+CE``."""
    vlm = _LinearStubVLM()
    image = torch.rand(1, 3, 8, 8)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.tensor([[2]], dtype=torch.long)

    val_t = TokenTargetLoss(targeted=True)(vlm, image, prompt, target, {})
    val_u = TokenTargetLoss(targeted=False)(vlm, image, prompt, target, {})
    # Targeted loss = -1 * untargeted loss.
    assert torch.allclose(val_t, -val_u, atol=1e-6)


@pytest.mark.smoke
def test_logits_for_target_slices_correct_positions():
    """``_logits_for_target`` must hit the causal-LM teacher-forced slice."""
    logits = torch.randn(1, 10, 5)
    sliced = _logits_for_target(logits, t_prompt=4, t_target=3)
    assert sliced.shape == (1, 3, 5)
    # The slice should be at positions [3, 4, 5] (i.e. t_prompt-1 ... +t_target).
    assert torch.equal(sliced, logits[:, 3:6, :])


@pytest.mark.smoke
def test_logits_for_target_rejects_short_sequence():
    logits = torch.randn(1, 4, 5)
    with pytest.raises(ValueError, match="Logits length"):
        _logits_for_target(logits, t_prompt=3, t_target=3)


@pytest.mark.smoke
def test_trajectory_drift_loss_returns_negative_kl():
    """``TrajectoryDriftLoss`` evaluated at the benign image returns ≈ 0."""
    vlm = _LinearStubVLM()
    image = torch.rand(1, 3, 8, 8)
    prompt = torch.zeros(1, 2, dtype=torch.long)
    target = torch.tensor([[1, 2]], dtype=torch.long)

    loss_fn = TrajectoryDriftLoss.from_benign(
        vlm=vlm, x0=image, prompt_tokens=prompt, target=target, gen_kwargs={}
    )
    # KL(p_benign || p_benign) ≈ 0 ⇒ -KL ≈ 0.
    val = loss_fn(vlm, image, prompt, target, {})
    assert val.abs().item() < 1e-5


@pytest.mark.smoke
def test_trajectory_drift_loss_caches_benign_no_grad():
    """The cached ``p_benign`` tensor must not carry gradients."""
    vlm = _LinearStubVLM()
    image = torch.rand(1, 3, 8, 8)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.tensor([[3]], dtype=torch.long)

    loss_fn = TrajectoryDriftLoss.from_benign(
        vlm=vlm, x0=image, prompt_tokens=prompt, target=target, gen_kwargs={}
    )
    assert loss_fn.p_benign.requires_grad is False
    assert loss_fn.t_prompt == 1
    assert loss_fn.t_target == 1


@pytest.mark.smoke
def test_trajectory_drift_loss_grows_under_perturbation():
    """Perturbing the image should make ``-KL`` more negative (KL grows)."""
    vlm = _LinearStubVLM()
    image = torch.rand(1, 3, 8, 8)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.tensor([[2]], dtype=torch.long)

    loss_fn = TrajectoryDriftLoss.from_benign(
        vlm=vlm, x0=image, prompt_tokens=prompt, target=target, gen_kwargs={}
    )
    val_clean = loss_fn(vlm, image, prompt, target, {}).item()
    perturbed = torch.clamp(image + 0.3, 0.0, 1.0)
    val_pert = loss_fn(vlm, perturbed, prompt, target, {}).item()
    # Perturbed input drifts ⇒ KL > 0 ⇒ -KL < 0 < val_clean.
    assert val_pert < val_clean
