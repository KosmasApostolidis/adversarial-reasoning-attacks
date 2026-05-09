"""Eager DeBERTa-v3-large-MNLI judge for CoT scoring.

Loads the 440M-parameter cross-encoder NLI model at module import time.
Importing this module pulls in torch and transformers, downloads the
weights on first run, and pins the model to ``model.train(False)`` /
``no_grad`` for inference.

Public API
----------
NLI_MODEL_ID : str
    Pinned HF model identifier. Revision pin lives in ``requirements.lock``.

entailment_prob(premise: str, hypothesis: str) -> float
    Probability in [0, 1] that ``premise`` entails ``hypothesis``.

The unit tests for ``metrics.cot`` pass their own deterministic stub
into the metric functions, so they never need to import this module.
A real-model smoke test lives behind ``pytest -m slow``.
"""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-large"

_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID)
_model.train(False)  # inference mode (equivalent to .eval())
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = _model.to(_device)

_label2id = {label.lower(): idx for label, idx in _model.config.label2id.items()}
if "entailment" not in _label2id:
    raise RuntimeError(
        f"NLI model {NLI_MODEL_ID} has no 'entailment' label "
        f"(found: {list(_label2id)})"
    )
_ENTAIL_IDX = _label2id["entailment"]


@torch.no_grad()
def entailment_prob(premise: str, hypothesis: str) -> float:
    """Return P(premise entails hypothesis) under DeBERTa-v3-large-MNLI."""
    if not premise.strip() or not hypothesis.strip():
        return 0.0
    inputs = _tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(_device)
    logits = _model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    return float(probs[_ENTAIL_IDX].item())
