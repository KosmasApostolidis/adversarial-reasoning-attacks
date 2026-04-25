"""Typed dictionaries for VLM attack pipelines.

These are :class:`typing.TypedDict` definitions (not dataclasses) so existing
code that builds and indexes plain dicts continues to work without any
runtime change.

The shapes documented here are the *contract* between
:meth:`VLMBase.prepare_attack_inputs` and the runner / attack loop / agent
pipeline. Adding a new key to a model wrapper only requires extending the
relevant TypedDict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import torch


class AttackInputs(TypedDict, total=False):
    """Output of :meth:`VLMBase.prepare_attack_inputs`.

    Required keys (every HF-backed model that supports gradients):
        pixel_values: tensor in the model's pre-normalized image domain.
        input_ids:    tokenized prompt, shape ``(B, T_prompt)``.

    Optional keys (model-family-specific; threaded into forward/generate
    via ``**`` unpacking by the runner):
        attention_mask:  prompt attention mask, shape ``(B, T_prompt)``.
        image_grid_thw:  Qwen2.5-VL anyres grid ``(T, H, W)``.
        image_sizes:     LLaVA-Next per-image ``(H, W)`` shape list.
        image_token_id:  scalar id of the image placeholder token.
    """

    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    image_grid_thw: torch.Tensor
    image_sizes: torch.Tensor
    image_token_id: int


class GenKwargs(TypedDict, total=False):
    """Subset of :class:`AttackInputs` re-exposed to ``generate`` /
    ``forward_with_logits``.

    Holds only the model-family-specific extras (``pixel_values`` and
    ``input_ids`` are passed positionally). The runner expands these via
    ``**gen_kwargs`` so optional keys are omitted (not ``None``) when the
    model does not need them.

    Note that ``attention_mask`` here, when present, covers the full
    teacher-forced ``[prompt ‖ target]`` sequence used by attack forward
    passes — not the prompt-only mask carried in :class:`AttackInputs`.
    """

    attention_mask: torch.Tensor
    image_grid_thw: torch.Tensor
    image_sizes: torch.Tensor


__all__ = ["AttackInputs", "GenKwargs"]
