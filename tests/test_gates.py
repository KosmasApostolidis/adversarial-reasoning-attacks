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
