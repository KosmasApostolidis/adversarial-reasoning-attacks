"""Ollama HTTP client for VLMs. Transfer-evaluation backend only — no gradients."""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from typing import Any, cast

from PIL import Image
from tenacity import Retrying, stop_after_attempt, wait_exponential

from .base import VLMBase, VLMGenerateResult

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None  # type: ignore[assignment]


@dataclass
class OllamaSettings:
    host: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    )
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
        self._client = ollama.Client(
            host=self.settings.host, timeout=self.settings.request_timeout_s
        )

    def _chat_once(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        tools: list[dict] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "options": options,
        }
        if tools is not None:
            kwargs["tools"] = tools
        return dict(self._client.chat(**kwargs))

    def _chat(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        tools: list[dict] | None,
    ) -> dict[str, Any]:
        retryer = Retrying(
            stop=stop_after_attempt(self.settings.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        return cast(dict[str, Any], retryer(self._chat_once, messages, options, tools))

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
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_new_tokens,
        }
        if seed is not None:
            options["seed"] = seed
        response = self._chat(messages, options, tools_schema)
        return VLMGenerateResult(
            text=response.get("message", {}).get("content", ""),
            finish_reason=response.get("done_reason", "stop"),
            raw=response,
        )
