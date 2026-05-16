"""Best-available ``attn_implementation`` selector for HF model loads.

Tries ``flash_attention_2`` first (Ampere+ GPUs with ``flash-attn>=2``
installed), falls back to ``sdpa`` (PyTorch ≥ 2.0 scaled-dot-product
attention — still kernel-fused, no Python loop), finally the model's
own default. ``flash_attention_2`` gives ~1.5–2× forward speedup on
the VLMs we benchmark; without it the surrogate's fp16 forward becomes
the dominant cost of every PGD step.

Selection is logged once per process via a one-shot WARN so reviewers can
verify which kernel served their run.
"""

from __future__ import annotations

import sys
from typing import Any

_ATTN_FALLBACK_LOGGED: set[str] = set()


def _log_once(impl: str, hf_id: str) -> None:
    key = f"{impl}:{hf_id}"
    if key in _ATTN_FALLBACK_LOGGED:
        return
    _ATTN_FALLBACK_LOGGED.add(key)
    print(
        f"[attention] using attn_implementation={impl!r} for {hf_id!r}",
        file=sys.stderr,
        flush=True,
    )


def _load_with_best_attention(
    model_cls: Any,
    hf_id: str,
    **from_pretrained_kwargs: Any,
) -> Any:
    """Call ``model_cls.from_pretrained`` with the best available attention kernel.

    Try order: ``flash_attention_2`` → ``sdpa`` → model default. Each
    fallback catches ``ImportError`` (flash-attn not installed) and
    ``ValueError`` (kernel unsupported for this architecture / dtype).
    """
    for impl in ("flash_attention_2", "sdpa"):
        try:
            model = model_cls.from_pretrained(
                hf_id,
                attn_implementation=impl,
                **from_pretrained_kwargs,
            )
            _log_once(impl, hf_id)
            return model
        except (ImportError, ValueError, TypeError):
            continue
    model = model_cls.from_pretrained(hf_id, **from_pretrained_kwargs)
    _log_once("<model default>", hf_id)
    return model
