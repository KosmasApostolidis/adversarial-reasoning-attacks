"""Base abstraction for VLM backends (HF fp16 surrogate or Ollama Q4 server)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from ..types import AttackInputs


@dataclass(frozen=True)
class VLMGenerateResult:
    """Output of a single VLM generation call.

    Both HF and Ollama backends return this shape so the agent loop and
    attack pipeline are backend-agnostic.
    """

    text: str
    tokens: list[int] = field(default_factory=list)
    logits: np.ndarray | None = None
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict)


class VLMBase(ABC):
    """Uniform interface across HF `transformers` and Ollama backends.

    The HF backend exposes :meth:`forward_with_logits`,
    :meth:`prepare_attack_inputs`, and :meth:`generate_from_pixel_values`
    for gradient-based attacks. The Ollama backend raises
    :class:`NotImplementedError` for these; it is only used for transfer
    evaluation.
    """

    family: str
    model_id: str
    supports_gradients: bool

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        seed: int | None = None,
        tools_schema: list[dict] | None = None,
    ) -> VLMGenerateResult:
        """Run generation on a single (image, prompt) pair."""

    def forward_with_logits(
        self, image_tensor: Any, prompt_tokens: Any, **forward_kwargs: Any
    ) -> Any:
        """Gradient-enabled forward pass. HF-only.

        ``forward_kwargs`` carries the model-specific tensors documented in
        :class:`adversarial_reasoning.types.GenKwargs` (e.g. Qwen
        ``image_grid_thw``, LLaVA-Next ``image_sizes``, the full
        ``[prompt ‖ target]`` ``attention_mask``).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support gradient-enabled forward pass."
        )

    def preprocess_image(self, image: Image.Image) -> Any:
        """Model-specific image preprocessing (resize / normalize / patchify)."""
        raise NotImplementedError

    @property
    def pixel_std(self) -> float:
        """L∞ scaling factor for converting pixel-domain ε to model-input ε.

        Pixel-domain ε (e.g. ``8/255``) refers to perturbation magnitude on the
        raw image in ``[0, 1]``. After the preprocessor normalises with
        ``y = (x − μ) / σ``, the equivalent normalised-space L∞ bound is
        ``ε / σ`` per channel. We return ``max(σ)`` so a single scalar ε
        uniformly applied in the normalised domain stays *within* pixel
        budget on every channel (conservative bound).

        Backends that operate directly on pixel domain (e.g. Ollama with raw
        bytes) leave the default ``1.0``. HF backends MUST override.
        """
        return 1.0

    def prepare_attack_inputs(
        self,
        image: Image.Image,
        prompt: str,
        tools_schema: list[dict] | None = None,
    ) -> AttackInputs:
        """Tokenize prompt + preprocess image into tensors ready for an attack.

        Returns a dict matching :class:`adversarial_reasoning.types.AttackInputs`:
        ``pixel_values``, ``input_ids``, optional ``attention_mask`` plus any
        model-family-specific keys (Qwen ``image_grid_thw``, LLaVA-Next
        ``image_sizes``, MLlama ``aspect_ratio_*``).

        Backends without gradient support (Ollama) should leave the default
        :class:`NotImplementedError` in place.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement prepare_attack_inputs; "
            "needed for gradient-based attacks."
        )

    def generate_from_pixel_values(
        self,
        pixel_values: Any,
        prompt: str,
        *,
        template_image: Image.Image,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        seed: int | None = None,
        tools_schema: list[dict] | None = None,
        **model_kwargs: Any,
    ) -> VLMGenerateResult:
        """Generate from a pre-prepared (perturbed) ``pixel_values`` tensor.

        Sidesteps the image processor (which would re-quantise an
        adversarial perturbation) by handing the already-normalised tensor
        straight to the underlying model. ``template_image`` provides the
        reference H/W so the chat template emits the right number of
        ``<image>`` placeholders.

        ``model_kwargs`` carries the family-specific extras
        (``image_grid_thw`` for Qwen, ``image_sizes`` for LLaVA-Next).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement generate_from_pixel_values; "
            "needed for adversarial inference."
        )
