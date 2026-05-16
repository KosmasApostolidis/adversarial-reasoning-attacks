"""Lazy DeBERTa-v3-large-MNLI judge for CoT scoring.

The 440M-parameter cross-encoder is loaded on **first call** so importing
:mod:`adversarial_reasoning.metrics` no longer pulls torch/transformers
weights into CI runs that only need the deterministic stub (this was a
~10s import-time penalty + weight download per CI invocation).

Public API
----------
NLI_MODEL_ID : str
    HF model identifier.
NLI_MODEL_REVISION : str
    HF revision. Defaults to ``main`` with a runtime WARN. Set
    ``NLI_MODEL_REVISION`` env var to an immutable 40-char SHA for paper
    reproducibility.
entailment_prob(premise, hypothesis) -> float
    Single-pair probability in [0, 1].
entailment_probs(pairs) -> list[float]
    Batched probability for ``[(premise, hypothesis), ...]``. ~10× faster
    than a Python loop over ``entailment_prob`` once you exceed ~4 pairs.

The unit tests for :mod:`metrics.cot` pass their own deterministic stub
into the metric functions, so they never need to import this module.
"""

from __future__ import annotations

import os
import sys
from typing import Any

NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-large"
NLI_MODEL_REVISION = os.environ.get("NLI_MODEL_REVISION", "main")
_NLI_BATCH_SIZE = int(os.environ.get("NLI_BATCH_SIZE", "16"))

# Populated on first call. Tuple so a partial init is impossible.
_NLI_CACHE: dict[str, Any] = {}


def _warned_about_mutable_revision() -> bool:
    if NLI_MODEL_REVISION == "main" and "_warned" not in _NLI_CACHE:
        print(
            f"[nli] WARN: NLI_MODEL_REVISION='main' — set to an immutable "
            f"40-char SHA for paper reproducibility (current: {NLI_MODEL_ID}).",
            file=sys.stderr,
            flush=True,
        )
        _NLI_CACHE["_warned"] = True
    return True


def _get_nli() -> tuple[Any, Any, str, int]:
    """Lazy-load tokenizer + model on first call; cached thereafter."""
    if "model" in _NLI_CACHE:
        return (
            _NLI_CACHE["tokenizer"],
            _NLI_CACHE["model"],
            _NLI_CACHE["device"],
            _NLI_CACHE["entail_idx"],
        )
    _warned_about_mutable_revision()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        NLI_MODEL_ID, revision=NLI_MODEL_REVISION
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL_ID, revision=NLI_MODEL_REVISION
    )
    model.train(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    label2id = {label.lower(): idx for label, idx in model.config.label2id.items()}
    if "entailment" not in label2id:
        raise RuntimeError(
            f"NLI model {NLI_MODEL_ID} has no 'entailment' label "
            f"(found: {list(label2id)})"
        )
    _NLI_CACHE.update(
        tokenizer=tokenizer,
        model=model,
        device=device,
        entail_idx=label2id["entailment"],
    )
    return tokenizer, model, device, label2id["entailment"]


def entailment_prob(premise: str, hypothesis: str) -> float:
    """Return P(premise entails hypothesis) under DeBERTa-v3-large-MNLI."""
    if not premise.strip() or not hypothesis.strip():
        return 0.0
    return entailment_probs([(premise, hypothesis)])[0]


def entailment_probs(pairs: list[tuple[str, str]]) -> list[float]:
    """Batched entailment probabilities. ~10× faster than a python loop.

    Empty / whitespace-only pairs short-circuit to 0.0 without invoking the
    model (the original loop allocated a forward pass for them).
    """
    if not pairs:
        return []
    import torch

    tokenizer, model, device, entail_idx = _get_nli()
    out: list[float] = [0.0] * len(pairs)
    todo: list[tuple[int, str, str]] = []
    for i, (p, h) in enumerate(pairs):
        if p.strip() and h.strip():
            todo.append((i, p, h))
    with torch.no_grad():
        for batch_start in range(0, len(todo), _NLI_BATCH_SIZE):
            batch = todo[batch_start : batch_start + _NLI_BATCH_SIZE]
            premises = [p for _, p, _ in batch]
            hypotheses = [h for _, _, h in batch]
            inputs = tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(device)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, entail_idx]
            for (orig_i, _, _), p_val in zip(batch, probs.tolist()):
                out[orig_i] = float(p_val)
    return out
