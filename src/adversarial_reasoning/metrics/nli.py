"""Lazy NLI judge for CoT scoring.

The NLI model is a 440M-parameter DeBERTa-v3-large fine-tuned on MNLI.
We do not import torch or transformers at module load time — that would
balloon CI cold-start cost on machines that never touch CoT metrics.

Public API
----------
get_nli() -> Callable[[str, str], float]
    Returns a function ``entailment_prob(premise, hypothesis)`` that
    yields a probability in [0, 1]. Loads the model on first call,
    caches the resulting closure on subsequent calls.

set_nli(fn) -> None
    Override the cached judge. Used in tests with a deterministic stub.

reset_nli() -> None
    Drop the cache. Mostly for tests.

NLI_MODEL_ID
    Pinned model identifier. Surface-level constant so the runbook can
    cross-reference. Revision pin lives in ``requirements.lock``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-large"

EntailmentFn = Callable[[str, str], float]

_OVERRIDE: EntailmentFn | None = None


def set_nli(fn: EntailmentFn | None) -> None:
    """Replace (or clear) the cached NLI callable. Test hook."""
    global _OVERRIDE
    _OVERRIDE = fn
    clear = getattr(_build_real_nli, "cache_clear", None)
    if clear is not None:
        clear()


def reset_nli() -> None:
    """Drop the cached real-model loader and any test override."""
    global _OVERRIDE
    _OVERRIDE = None
    clear = getattr(_build_real_nli, "cache_clear", None)
    if clear is not None:
        clear()


def get_nli() -> EntailmentFn:
    if _OVERRIDE is not None:
        return _OVERRIDE
    return _build_real_nli()


@lru_cache(maxsize=1)
def _build_real_nli() -> EntailmentFn:
    """Construct the real DeBERTa NLI callable. Lazy: imports only here."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID)
    model.train(False)  # inference mode (equivalent to .eval())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    label2id = {label.lower(): idx for label, idx in model.config.label2id.items()}
    if "entailment" not in label2id:
        raise RuntimeError(
            f"NLI model {NLI_MODEL_ID} has no 'entailment' label "
            f"(found: {list(label2id)})"
        )
    entail_idx = label2id["entailment"]

    @torch.no_grad()
    def entailment_prob(premise: str, hypothesis: str) -> float:
        if not premise.strip() or not hypothesis.strip():
            return 0.0
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        return float(probs[entail_idx].item())

    return entailment_prob
