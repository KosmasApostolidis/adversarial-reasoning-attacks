"""Config-driven VLM loader. Reads configs/models.yaml and instantiates backends.

Supply-chain safety: remote HF ids MUST pin ``hf_revision`` to an immutable
git SHA (40-hex). Movable refs (``main``, branch names) are rejected at load
time — they let an upstream repo silently swap weights/code between runs and
defeat reproducibility. Local checkpoint paths are exempt. Set
``ADREASON_ALLOW_MUTABLE_HF_REVISION=1`` in env to bypass for dev only; the
loader prints a loud WARN. Production runs MUST NOT set this.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .base import VLMBase
from .ollama_client import OllamaSettings, OllamaVLMClient

_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_ALLOW_MUTABLE_ENV = "ADREASON_ALLOW_MUTABLE_HF_REVISION"


def _resolve_revision(hf_id: str, revision: str) -> str:
    """Enforce immutable revision pinning for *remote* HF ids.

    Local paths are passed through unchanged (the local filesystem is the
    trust boundary). Remote ids that point at a mutable ref raise unless the
    dev-only bypass env var is set.
    """
    if Path(hf_id).is_dir():
        return revision
    if _SHA_PATTERN.match(revision):
        return revision
    if os.environ.get(_ALLOW_MUTABLE_ENV) == "1":
        print(
            f"[loader] WARN: hf_revision={revision!r} for {hf_id!r} is mutable; "
            f"bypassing pinning via {_ALLOW_MUTABLE_ENV}=1. NOT FOR PRODUCTION.",
            flush=True,
        )
        return revision
    raise ValueError(
        f"hf_revision for remote model {hf_id!r} must be an immutable 40-char "
        f"git SHA (got {revision!r}). Pin via huggingface_hub:\n"
        f"  >>> from huggingface_hub import HfApi\n"
        f"  >>> HfApi().model_info({hf_id!r}).sha\n"
        f"Set {_ALLOW_MUTABLE_ENV}=1 to bypass for local dev only."
    )


def load_hf_vlm(model_name: str, config_path: str | Path = "configs/models.yaml") -> VLMBase:
    """Load an HF surrogate VLM by config key."""
    cfg = _read_config(config_path)["models"][model_name]
    family = cfg["family"]
    hf_id = cfg["hf_id"]
    revision = _resolve_revision(hf_id, cfg.get("hf_revision", "main"))

    if family == "qwen_vl":
        from .qwen_vl import QwenVL

        return QwenVL(
            hf_id=hf_id,
            device_map=cfg.get("device_map", "auto"),
            revision=revision,
        )
    if family == "llava_next":
        from .llava import LlavaNext

        return LlavaNext(
            hf_id=hf_id,
            device_map=cfg.get("device_map", "auto"),
            revision=revision,
        )
    if family == "llava_onevision":
        from .llava import LlavaNext

        model = LlavaNext(
            hf_id=hf_id,
            device_map=cfg.get("device_map", "auto"),
            revision=revision,
        )
        model.family = "llava_onevision"
        return model
    if family == "internvl2":
        from .internvl2 import InternVL2

        return InternVL2(
            hf_id=hf_id,
            device_map=cfg.get("device_map", "auto"),
            revision=revision,
        )
    raise ValueError(f"Unknown VLM family: {family}")


def load_ollama_vlm(
    model_name: str,
    config_path: str | Path = "configs/models.yaml",
    settings: OllamaSettings | None = None,
) -> OllamaVLMClient:
    """Load the Ollama-served twin of a VLM by config key."""
    cfg = _read_config(config_path)["models"][model_name]
    if "ollama_tag" not in cfg:
        raise NotImplementedError(
            f"No Ollama image registered for model {model_name!r} "
            f"(family={cfg.get('family')!r}). Transfer evaluation via Ollama is "
            "not supported for this model; use the HF surrogate (load_hf_vlm)."
        )
    return OllamaVLMClient(
        ollama_tag=cfg["ollama_tag"],
        family=cfg["family"],
        settings=settings,
    )


def _read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})
