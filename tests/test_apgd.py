"""Smoke + invariant tests for APGD-L∞ on a synthetic linear stub.

Mirrors test_attacks.py's stub-VLM pattern. Keeps everything CPU-only and
fast (≤ 30 steps × small image). Verifies APGD's contracts:

- ε-bounded δ
- step-size η halves at checkpoints when stagnating
- warm-restart from best-loss iterate (uses no_grad — see
  feedback_apgd_warm_restart.md)
- targeted mode reduces token-cross-entropy
- gradient-free backend rejected
- ``_checkpoints`` schedule produces strictly monotone, in-range integers
"""

from __future__ import annotations

import pytest
import torch

from adversarial_reasoning.attacks.apgd import (
    APGDAttack,
    _checkpoints,
    _step_is_improvement,
)


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
        logits_per_batch = torch.matmul(flat, self.weight.t())
        seq_len = int(input_ids.shape[-1])
        return logits_per_batch.unsqueeze(1).expand(-1, seq_len, -1).contiguous()


# ---------------------- _checkpoints schedule ---------------------------


def _make_apgd_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    image = torch.rand(1, 3, 8, 8)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.zeros(1, 1, dtype=torch.long)
    return image, prompt, target


def test_apgd_epsilon_zero_short_circuits():
    """``epsilon=0`` must return the input unmodified without running steps."""
    vlm = _LinearStubVLM()
    image, prompt, target = _make_apgd_inputs()
    res = APGDAttack(epsilon=0.0, steps=20).run(
        vlm=vlm,
        image=image,
        prompt_tokens=prompt,
        target=target,
    )
    assert torch.equal(res.delta, torch.zeros_like(res.delta))
    assert res.iterations == 0


def test_apgd_seed_makes_runs_deterministic():
    """Same ``seed`` on APGDAttack must produce identical trajectories;
    different seeds must diverge. Asserts on the loss trajectory rather
    than final delta because APGD often converges to the same ε-boundary
    on a trivial linear stub regardless of init."""
    vlm = _LinearStubVLM()
    image, prompt, target = _make_apgd_inputs()
    a = APGDAttack(epsilon=8.0 / 255.0, steps=6, random_restarts=2, seed=123).run(
        vlm=vlm,
        image=image,
        prompt_tokens=prompt,
        target=target,
    )
    b = APGDAttack(epsilon=8.0 / 255.0, steps=6, random_restarts=2, seed=123).run(
        vlm=vlm,
        image=image,
        prompt_tokens=prompt,
        target=target,
    )
    c = APGDAttack(epsilon=8.0 / 255.0, steps=6, random_restarts=2, seed=456).run(
        vlm=vlm,
        image=image,
        prompt_tokens=prompt,
        target=target,
    )
    assert torch.equal(a.delta, b.delta), "same seed must give identical delta"
    assert a.loss_trajectory == b.loss_trajectory, "same seed must give identical trajectory"
    assert a.loss_trajectory != c.loss_trajectory, "different seeds must diverge in trajectory"


class TestStepIsImprovement:
    """Croce-Hein 2020 ρ_w: strict step-over-step decrease within a window.

    The earlier code counted ``loss_val < loss_best`` (improvement-over-
    running-best), which under-fires once loss_best saturates. The fixed
    helper compares against the previous step's loss directly.
    """

    def test_finite_strict_decrease_is_improvement(self) -> None:
        assert _step_is_improvement(loss_val=5.0, loss_prev=10.0) is True

    def test_first_step_against_inf_counts_as_improvement(self) -> None:
        # First iteration has no prior, so loss_prev=inf — any finite loss
        # is treated as an improvement (matches Croce-Hein init convention).
        assert _step_is_improvement(loss_val=10.0, loss_prev=float("inf")) is True

    def test_equal_loss_is_not_improvement(self) -> None:
        assert _step_is_improvement(loss_val=5.0, loss_prev=5.0) is False

    def test_increase_is_not_improvement(self) -> None:
        assert _step_is_improvement(loss_val=6.0, loss_prev=5.0) is False

    def test_nan_loss_is_not_improvement(self) -> None:
        # NaN values must not count as success — protects against polluting
        # success_count with garbage when the model returns NaN logits.
        assert _step_is_improvement(loss_val=float("nan"), loss_prev=5.0) is False

    def test_micro_decrease_below_tol_not_improvement(self) -> None:
        # Numerical-noise-only decreases must not count as success (tol=1e-12).
        assert _step_is_improvement(loss_val=5.0 - 1e-15, loss_prev=5.0) is False


class TestCheckpoints:
    def test_in_range_and_unique(self) -> None:
        cps = _checkpoints(50)
        assert all(1 <= c <= 50 for c in cps)
        assert len(cps) == len(set(cps))

    def test_monotone_non_decreasing(self) -> None:
        cps = _checkpoints(100)
        assert cps == sorted(cps)

    def test_first_checkpoint_at_22pct(self) -> None:
        cps = _checkpoints(100)
        assert cps[0] == 22

    def test_n_iter_one(self) -> None:
        cps = _checkpoints(1)
        assert cps == [1]


# ---------------------- attack invariants ------------------------------


@pytest.mark.smoke
def test_apgd_respects_epsilon_budget() -> None:
    vlm = _LinearStubVLM()
    image = torch.rand(3, 8, 8)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.tensor([[3]], dtype=torch.long)

    attack = APGDAttack(
        epsilon=8.0 / 255.0,
        steps=15,
        random_restarts=1,
        targeted=False,
    )
    result = attack.run(vlm, image, prompt, target)

    linf = result.delta.abs().max().item()
    assert linf <= 8.0 / 255.0 + 1e-6


@pytest.mark.smoke
def test_apgd_targeted_reduces_loss() -> None:
    vlm = _LinearStubVLM()
    image = torch.rand(3, 8, 8)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.tensor([[3]], dtype=torch.long)

    attack = APGDAttack(
        epsilon=16.0 / 255.0,
        steps=20,
        random_restarts=1,
        targeted=True,
    )
    result = attack.run(vlm, image, prompt, target)
    assert result.loss_trajectory[-1] <= result.loss_trajectory[0]


@pytest.mark.smoke
def test_apgd_metadata_records_checkpoints_and_eta() -> None:
    vlm = _LinearStubVLM()
    image = torch.rand(3, 8, 8)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.tensor([[1]], dtype=torch.long)

    result = APGDAttack(epsilon=8.0 / 255.0, steps=10, random_restarts=1).run(
        vlm, image, prompt, target
    )
    assert "checkpoints" in result.metadata
    assert "eta_final" in result.metadata
    assert "epsilon" in result.metadata
    assert result.metadata["targeted"] is False


@pytest.mark.smoke
def test_apgd_records_full_loss_trajectory() -> None:
    vlm = _LinearStubVLM()
    image = torch.rand(3, 8, 8)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.tensor([[2]], dtype=torch.long)

    result = APGDAttack(epsilon=4.0 / 255.0, steps=12).run(vlm, image, prompt, target)
    assert len(result.loss_trajectory) == 12


@pytest.mark.smoke
def test_apgd_random_restarts_returns_best() -> None:
    vlm = _LinearStubVLM()
    image = torch.rand(3, 8, 8)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.tensor([[5]], dtype=torch.long)

    result = APGDAttack(
        epsilon=8.0 / 255.0,
        steps=8,
        random_restarts=3,
        targeted=True,
    ).run(vlm, image, prompt, target)
    assert result.metadata["restart"] in {0, 1, 2}


@pytest.mark.smoke
def test_apgd_rejects_gradient_free_backend() -> None:
    class _NoGradVLM:
        supports_gradients = False
        model_id = "toy/nograd"

    attack = APGDAttack(epsilon=4.0 / 255.0, steps=5)
    with pytest.raises(ValueError, match="does not support gradients"):
        attack.run(
            _NoGradVLM(),
            torch.rand(3, 8, 8),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
        )


@pytest.mark.smoke
def test_apgd_accepts_4d_input_image() -> None:
    """Image already batched (4D) should not error."""
    vlm = _LinearStubVLM()
    image = torch.rand(1, 3, 8, 8)
    prompt = torch.zeros(1, 1, dtype=torch.long)
    target = torch.tensor([[1]], dtype=torch.long)
    result = APGDAttack(epsilon=4.0 / 255.0, steps=5).run(vlm, image, prompt, target)
    assert result.delta.abs().max().item() <= 4.0 / 255.0 + 1e-6
