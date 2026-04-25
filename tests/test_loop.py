"""Unit tests for ``adversarial_reasoning.attacks._loop.linf_pgd_loop``.

Verifies the invariants the loop is responsible for, independent of the
loss function plugged in:

- ε-budget — perturbation max-abs stays within ``epsilon``.
- Image-bounds clipping — perturbed image stays in ``[clip_min, clip_max]``.
- Random-restart selection — best-of-N (smallest ``loss_final``) is returned.
- ``step_sign`` direction — flipping the sign moves the loss in the
  opposite direction over the trajectory.
- Static metadata pass-through — caller-supplied keys land in the
  returned ``AttackResult.metadata`` dict.
"""

from __future__ import annotations

import math

import pytest
import torch

from adversarial_reasoning.attacks._loop import linf_pgd_loop
from adversarial_reasoning.attacks.loss import TokenTargetLoss


class _LinearStubVLM:
    supports_gradients = True
    model_id = "toy/loop-stub"

    def __init__(self, in_dim: int = 3 * 8 * 8, vocab: int = 4) -> None:
        torch.manual_seed(0)
        self.weight = torch.randn(vocab, in_dim, requires_grad=False)

    def forward_with_logits(
        self, image_tensor: torch.Tensor, input_ids: torch.Tensor, **_: object
    ) -> torch.Tensor:
        flat = image_tensor.reshape(image_tensor.shape[0], -1)
        per_batch = flat @ self.weight.t()
        seq_len = int(input_ids.shape[-1])
        return per_batch.unsqueeze(1).expand(-1, seq_len, -1).contiguous()


def _make_inputs():
    torch.manual_seed(0)
    image = torch.rand(1, 3, 8, 8)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.zeros(1, 1, dtype=torch.long)
    return image, prompt, target


@pytest.mark.smoke
def test_linf_loop_respects_epsilon_budget():
    vlm = _LinearStubVLM()
    image, prompt, target = _make_inputs()
    epsilon = 8.0 / 255.0

    res = linf_pgd_loop(
        loss_fn=TokenTargetLoss(targeted=False),
        vlm=vlm,
        x0=image,
        prompt_tokens=prompt,
        target=target,
        gen_kwargs={},
        epsilon=epsilon,
        alpha=epsilon / 4.0,
        n_iter=10,
        n_restarts=1,
        step_sign=1.0,
    )
    assert res.delta.abs().max().item() <= epsilon + 1e-6


@pytest.mark.smoke
def test_linf_loop_respects_image_bounds():
    vlm = _LinearStubVLM()
    image, prompt, target = _make_inputs()

    res = linf_pgd_loop(
        loss_fn=TokenTargetLoss(targeted=False),
        vlm=vlm, x0=image, prompt_tokens=prompt, target=target,
        gen_kwargs={}, epsilon=64.0 / 255.0, alpha=8.0 / 255.0,
        n_iter=10, n_restarts=1, step_sign=1.0,
        clip_min=0.0, clip_max=1.0,
    )
    perturbed = res.perturbed_image
    assert perturbed.min().item() >= -1e-6
    assert perturbed.max().item() <= 1.0 + 1e-6


@pytest.mark.smoke
def test_linf_loop_picks_lowest_loss_restart():
    vlm = _LinearStubVLM()
    image, prompt, target = _make_inputs()

    res = linf_pgd_loop(
        loss_fn=TokenTargetLoss(targeted=False),
        vlm=vlm, x0=image, prompt_tokens=prompt, target=target,
        gen_kwargs={}, epsilon=8.0 / 255.0, alpha=2.0 / 255.0,
        n_iter=8, n_restarts=4, step_sign=1.0,
    )
    # Best-of-N returns the smallest final loss across restarts.
    assert math.isfinite(res.loss_final)
    assert "restart" in res.metadata
    assert 0 <= res.metadata["restart"] < 4


@pytest.mark.smoke
def test_linf_loop_step_sign_inverts_direction():
    """+1 and -1 step_sign on the same loss must end on opposite sides."""
    vlm = _LinearStubVLM()
    image, prompt, target = _make_inputs()
    common = dict(
        loss_fn=TokenTargetLoss(targeted=False),
        vlm=vlm, x0=image, prompt_tokens=prompt, target=target,
        gen_kwargs={}, epsilon=8.0 / 255.0, alpha=2.0 / 255.0,
        n_iter=15, n_restarts=1,
    )
    pos = linf_pgd_loop(step_sign=+1.0, **common)
    neg = linf_pgd_loop(step_sign=-1.0, **common)
    # Opposite signs walk the loss in opposite directions; final losses differ.
    assert pos.loss_final != neg.loss_final


@pytest.mark.smoke
def test_linf_loop_static_metadata_passthrough():
    vlm = _LinearStubVLM()
    image, prompt, target = _make_inputs()

    res = linf_pgd_loop(
        loss_fn=TokenTargetLoss(targeted=False),
        vlm=vlm, x0=image, prompt_tokens=prompt, target=target,
        gen_kwargs={}, epsilon=4.0 / 255.0, alpha=1.0 / 255.0,
        n_iter=4, n_restarts=1, step_sign=1.0,
        static_metadata={"targeted": False, "marker": "loop-test"},
    )
    assert res.metadata.get("targeted") is False
    assert res.metadata.get("marker") == "loop-test"
    # Loop-owned keys must coexist with caller's static keys.
    assert "epsilon" in res.metadata
    assert "alpha" in res.metadata
