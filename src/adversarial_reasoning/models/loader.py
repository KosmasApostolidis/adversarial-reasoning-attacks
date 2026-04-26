"""Config-driven VLM loader. Reads configs/models.yaml and instantiates backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import VLMBase
from .ollama_client import OllamaSettings, OllamaVLMClient


def load_hf_vlm(model_name: str, config_path: str | Path = "configs/models.yaml") -> VLMBase:
    """Load an HF surrogate VLM by config key."""
    cfg = _read_config(config_path)["models"][model_name]
    family = cfg["family"]

    if family == "qwen_vl":
        from .qwen_vl import QwenVL

        return QwenVL(
            hf_id=cfg["hf_id"],
            device_map=cfg.get("device_map", "auto"),
            revision=cfg.get("hf_revision", "main"),
        )
    if family == "llava_next":
        from .llava import LlavaNext

        return LlavaNext(
            hf_id=cfg["hf_id"],
            device_map=cfg.get("device_map", "auto"),
            revision=cfg.get("hf_revision", "main"),
        )
    if family == "llama_vision":
        from .llama_vision import LlamaVision

        return LlamaVision(
            hf_id=cfg["hf_id"],
            device_map=cfg.get("device_map", "auto"),
            revision=cfg.get("hf_revision", "main"),
        )
    raise ValueError(f"Unknown VLM family: {family}")


def load_ollama_vlm(
    model_name: str,
    config_path: str | Path = "configs/models.yaml",
    settings: OllamaSettings | None = None,
) -> OllamaVLMClient:
    """Load the Ollama-served twin of a VLM by config key."""
    cfg = _read_config(config_path)["models"][model_name]
    return OllamaVLMClient(
        ollama_tag=cfg["ollama_tag"],
        family=cfg["family"],
        settings=settings,
    )


def _read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})
