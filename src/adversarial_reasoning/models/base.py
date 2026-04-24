"""Base abstraction for VLM backends (HF fp16 surrogate or Ollama Q4 server)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


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

    The HF backend exposes `forward_with_logits` for gradient-based attacks.
    The Ollama backend raises `NotImplementedError` for gradient ops; it is
    only used for transfer evaluation.
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

    def forward_with_logits(self, image_tensor: Any, prompt_tokens: Any) -> Any:
        """Gradient-enabled forward pass. HF-only."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support gradient-enabled forward pass."
        )

    def preprocess_image(self, image: Image.Image) -> Any:
        """Model-specific image preprocessing (resize / normalize / patchify)."""
        raise NotImplementedError
