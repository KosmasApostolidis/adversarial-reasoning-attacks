"""Smoke test for Phase 0 gate logic (no real model load)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from adversarial_reasoning.attacks._epsilon import (
    _LINF_EPSILON_2,
    _LINF_EPSILON_8,
    _LINF_EPSILON_16,
)
from adversarial_reasoning.gates.preprocessing_transfer import run_preprocessing_transfer


def test_preprocessing_transfer_png_roundtrip_preserves_signal():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    # Use a minimal stub with just a model_id attribute.
    class _Stub:
        model_id = "stub/vlm"

    result = run_preprocessing_transfer(_Stub(), sample_image=img, epsilon=_LINF_EPSILON_16)
    # PNG is lossless, so at ε=16/255 we expect ~full signal preserved.
    assert result.effective_linf_post_roundtrip > 0.0
    assert result.passed is True


def test_preprocessing_transfer_passes_typical_epsilon():
    img = Image.new("RGB", (64, 64), color=(123, 45, 200))

    class _Stub:
        model_id = "stub/vlm"

    r = run_preprocessing_transfer(_Stub(), sample_image=img, epsilon=_LINF_EPSILON_8)
    assert r.gate_threshold == _LINF_EPSILON_2
    # 8/255 > 2/255 threshold → gate should pass for a PNG round-trip.
    assert r.passed is True


# --- gradient_masking gate ------------------------------------------------


from adversarial_reasoning.gates.gradient_masking import (  # noqa: E402
    _is_monotonic,
    run_gradient_masking,
)


def _good_run() -> dict:
    """Healthy telemetry: all four Athalye checks should pass."""
    return dict(
        model_name="stub/vlm",
        epsilon=8 / 255,
        huge_epsilon=64 / 255,
        benign_loss=1.0,
        pgd_loss=-2.0,           # PGD pushes loss far down
        noise_loss=0.5,          # noise barely moves it
        huge_eps_loss=-3.0,      # huge ε pushes further
        benign_grad_norm=1.0,
        attacked_grad_norm=0.8,  # gradients alive
        loss_trajectory=[1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0],
    )


def test_gradient_masking_healthy_run_passes_all_checks():
    res = run_gradient_masking(**_good_run())
    assert res.huge_eps_passes is True
    assert res.pgd_beats_noise is True
    assert res.loss_monotonic is True
    assert res.grad_norm_alive is True
    assert res.passes is True


def test_gradient_masking_pgd_worse_than_noise_fails():
    kw = _good_run()
    kw["pgd_loss"] = 0.9  # worse than noise (0.5)
    res = run_gradient_masking(**kw)
    assert res.pgd_beats_noise is False
    assert res.passes is False


def test_gradient_masking_huge_eps_no_progress_fails():
    kw = _good_run()
    kw["huge_eps_loss"] = 0.9  # barely below benign
    res = run_gradient_masking(**kw)
    assert res.huge_eps_passes is False
    assert res.passes is False


def test_gradient_masking_collapsed_grad_norm_fails():
    kw = _good_run()
    kw["attacked_grad_norm"] = 1e-6  # gradients vanished
    res = run_gradient_masking(**kw)
    assert res.grad_norm_alive is False
    assert res.passes is False


def test_gradient_masking_non_monotone_loss_fails():
    kw = _good_run()
    # Loss rises after each step (gradient-masking fingerprint).
    kw["loss_trajectory"] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    res = run_gradient_masking(**kw)
    assert res.loss_monotonic is False
    assert res.passes is False


def test_gradient_masking_serialisation_round_trips():
    res = run_gradient_masking(**_good_run())
    d = res.to_dict()
    assert d["passes"] is True
    assert d["loss_trajectory"] == [1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0]
    assert "huge_eps_passes" in d
    assert d["model_name"] == "stub/vlm"


def test_is_monotonic_threshold_boundary():
    """80% of pairs must be non-increasing. Construct exactly 80% pass."""
    # 5 pairs: 4 non-increasing, 1 increasing → 4/5 = 0.80 — at threshold.
    assert _is_monotonic([5.0, 4.0, 3.0, 2.0, 3.0, 1.0]) is True
    # 4/6 ≈ 0.67 — below threshold.
    assert _is_monotonic([5.0, 4.0, 3.0, 4.0, 5.0, 2.0, 1.0]) is False
    # Empty / single-element trajectories cannot be monotonic.
    assert _is_monotonic([]) is False
    assert _is_monotonic([1.0]) is False
