"""VLM loaders and inference glue (HF `transformers` surrogate + Ollama deployment)."""

from .base import VLMBase, VLMGenerateResult
from .loader import load_hf_vlm, load_ollama_vlm
from .ollama_client import OllamaSettings, OllamaVLMClient

__all__ = [
    "VLMBase",
    "VLMGenerateResult",
    "OllamaSettings",
    "OllamaVLMClient",
    "load_hf_vlm",
    "load_ollama_vlm",
]
