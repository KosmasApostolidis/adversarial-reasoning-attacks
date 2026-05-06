"""Target-token builders for gradient-based attacks.

Each builder returns a ``LongTensor`` of shape ``(1, T_target)`` containing
the teacher-forced target sequence that the attack's loss is computed
against. Lives next to :mod:`adversarial_reasoning.attacks.loss` so the
two halves of an attack's specification — *what to push toward/away
from* (here) and *how to score it* (loss.py) — sit in one package.

These helpers were extracted from the runner's private ``_build_*``
helpers and from :mod:`adversarial_reasoning.attacks.targeted_tool` in
the 2026-04-25 refactor; numeric outputs (token sequences) are
byte-identical to the prior implementations.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..agents.base import Trajectory


def _tokenize(vlm: Any, text: str, device: Any | None = None) -> torch.Tensor:
    """Run the VLM's own tokenizer over ``text`` (no special tokens)."""
    enc = vlm.processor.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids: torch.Tensor = enc["input_ids"]
    return ids.to(device) if device is not None else ids


def _tool_call_block(name: str, args: dict | None) -> str:
    """Qwen-style tool-call wrapper around a single ``{"name", "arguments"}`` JSON."""
    return f'<tool_call>\n{{"name": "{name}", "arguments": {json.dumps(args or {})}}}\n</tool_call>'


def target_from_benign(
    vlm: Any, benign: Trajectory, prompt_input_ids: torch.Tensor
) -> torch.Tensor:
    """Untargeted PGD: target = the first benign tool call (or a
    placeholder if the benign trajectory had no tool calls).

    For untargeted PGD we want the model's likelihood of *the benign tool
    call* to drop. Build a Qwen-style ``<tool_call>...</tool_call>`` block
    around the first benign tool call and tokenize via the model's own
    tokenizer.
    """
    if benign.tool_calls:
        tc = benign.tool_calls[0]
        target_text = _tool_call_block(tc.name, tc.args)
    else:
        target_text = _tool_call_block("describe_region", {})
    return _tokenize(vlm, target_text, device=prompt_input_ids.device)


def target_from_trajectory(
    vlm: Any, benign: Trajectory, prompt_input_ids: torch.Tensor
) -> torch.Tensor:
    """Trajectory-Drift PGD: target = concat of every benign tool-call block.

    Falls back to :func:`target_from_benign` when the benign trajectory
    is empty (preserves original behaviour).
    """
    if not benign.tool_calls:
        return target_from_benign(vlm, benign, prompt_input_ids)
    blocks = [_tool_call_block(tc.name, tc.args) for tc in benign.tool_calls]
    target_text = "\n".join(blocks)
    return _tokenize(vlm, target_text, device=prompt_input_ids.device)


def target_from_tool(
    vlm: Any,
    target_tool: str,
    target_args: dict | None = None,
    *,
    device: Any | None = None,
) -> torch.Tensor:
    """Targeted-Tool PGD: target = a tool-call block forcing ``target_tool``."""
    return _tokenize(vlm, _tool_call_block(target_tool, target_args), device=device)


# Back-compat alias. ``targeted_tool.build_target_tokens`` was the public
# helper before the refactor; some external scripts/tests may import it.
build_target_tokens = target_from_tool


__all__ = [
    "build_target_tokens",
    "target_from_benign",
    "target_from_tool",
    "target_from_trajectory",
]
