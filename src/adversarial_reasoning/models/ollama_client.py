"""Ollama HTTP client for VLMs. Transfer-evaluation backend only — no gradients."""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import VLMBase, VLMGenerateResult

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None  # type: ignore[assignment]


@dataclass
class OllamaSettings:
    host: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    request_timeout_s: float = 120.0
    max_retries: int = 3


class OllamaVLMClient(VLMBase):
    """Call a VLM via Ollama's `/api/chat` endpoint with an image attached.

    Used for transfer evaluation only: adversarial images generated on the
    HF fp16 surrogate are sent here as base64 PNGs to see whether the attack
    survives quantization + preprocessing differences.
    """

    supports_gradients = False

    def __init__(
        self,
        ollama_tag: str,
        family: str,
        settings: OllamaSettings | None = None,
    ) -> None:
        if ollama is None:
            raise RuntimeError(
                "`ollama` package not installed. Install with `pip install ollama>=0.4`."
            )
        self.model_id = ollama_tag
        self.family = family
        self.settings = settings or OllamaSettings()
        self._client = ollama.Client(host=self.settings.host)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _chat(self, messages: list[dict[str, Any]], temperature: float) -> dict[str, Any]:
        return self._client.chat(
            model=self.model_id,
            messages=messages,
            options={"temperature": temperature},
        )

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
        # Encode image as base64 PNG. This is the exact payload path an adversary
        # would use when delivering perturbed images to an Ollama server, so we
        # test attack transfer along this same route.
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        messages = [{"role": "user", "content": prompt, "images": [b64]}]
        response = self._chat(messages, temperature)
        return VLMGenerateResult(
            text=response.get("message", {}).get("content", ""),
            finish_reason=response.get("done_reason", "stop"),
            raw=response,
        )
