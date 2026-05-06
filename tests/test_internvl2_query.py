"""Cheap unit test for InternVL2._build_query: IMG_CONTEXT_TOKEN bookkeeping.

No HF weights pulled. Uses stub model + bound-method dispatch to verify the
single load-bearing invariant: the number of <IMG_CONTEXT> tokens emitted
equals num_image_token * num_patches. Off-by-one here corrupts the visual
embedding alignment in InternVLChatModel.forward.
"""

from __future__ import annotations

import pytest

from adversarial_reasoning.models.internvl2 import (
    _IMG_CONTEXT_TOKEN,
    _IMG_END_TOKEN,
    _IMG_START_TOKEN,
    InternVL2,
)


class _StubModel:
    system_message = "You are a helpful assistant."


class _StubInternVL2:
    """Minimal attribute surface so the unbound _build_query works."""

    def __init__(self, num_image_token: int) -> None:
        self._num_image_token = num_image_token
        self.model = _StubModel()


def _call(stub: _StubInternVL2, prompt: str, num_patches: int) -> str:
    # Real call sites pass the output of _format_prompt, which always prefixes
    # the prompt with `<image>\n` — mirror that contract here.
    formatted = f"<image>\n{prompt}"
    return InternVL2._build_query(stub, formatted, num_patches)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "num_image_token,num_patches",
    [(256, 1), (256, 5), (256, 13), (64, 12), (1, 1)],
)
def test_img_context_token_count(num_image_token: int, num_patches: int) -> None:
    stub = _StubInternVL2(num_image_token=num_image_token)
    out = _call(stub, "describe", num_patches=num_patches)
    expected = num_image_token * num_patches
    assert out.count(_IMG_CONTEXT_TOKEN) == expected
    assert out.count(_IMG_START_TOKEN) == 1
    assert out.count(_IMG_END_TOKEN) == 1


def test_image_placeholder_consumed() -> None:
    stub = _StubInternVL2(num_image_token=256)
    out = _call(stub, "find lesion", num_patches=2)
    assert "<image>" not in out
    assert _IMG_START_TOKEN in out and _IMG_END_TOKEN in out


def test_chatml_envelope_present() -> None:
    stub = _StubInternVL2(num_image_token=256)
    out = _call(stub, "find lesion", num_patches=1)
    assert "<|im_start|>system" in out
    assert "<|im_start|>user" in out
    assert "<|im_start|>assistant" in out
    assert "find lesion" in out
    assert _StubModel.system_message in out


def test_only_first_image_placeholder_replaced() -> None:
    """If a stray `<image>` ever survives in the user prompt body, only the
    first occurrence is consumed by token expansion — second remains literal.
    """
    stub = _StubInternVL2(num_image_token=4)
    formatted = "<image>\nrefer to <image> below"
    out = InternVL2._build_query(stub, formatted, num_patches=1)  # type: ignore[arg-type]
    assert out.count(_IMG_CONTEXT_TOKEN) == 4
    assert out.count("<image>") == 1
