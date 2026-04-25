"""Targeted-Tool PGD — force a specific tool choice at step k.

Loss: CE forcing the target tool-name token sequence into the position
``target_step_k`` of the trajectory. Reuses :class:`PGDAttack` with
``targeted=True``; this module only handles target-token construction.

The caller is expected to pass ``target`` already tokenized. To build it
from a tool-name string + step index, use :func:`build_target_tokens`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch

from .base import AttackBase, AttackResult
from .pgd import PGDAttack


def build_target_tokens(
    vlm: Any,
    target_tool: str,
    target_args: dict | None = None,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Tokenize a Qwen-style tool-call block forcing ``target_tool``."""
    args_json = json.dumps(target_args or {})
    target_text = (
        "<tool_call>\n"
        f'{{"name": "{target_tool}", "arguments": {args_json}}}'
        "\n</tool_call>"
    )
    enc = vlm.processor.tokenizer(
        target_text, return_tensors="pt", add_special_tokens=False
    )
    ids = enc["input_ids"]
    return ids.to(device) if device is not None else ids


@dataclass
class TargetedToolPGD(AttackBase):
    name: str = "targeted_tool_pgd"
    epsilon: float = 8.0 / 255.0
    alpha: float | None = None
    steps: int = 40
    random_restarts: int = 1
    targeted: bool = True
    target_tool: str = "escalate_to_specialist"
    target_step_k: int = 0
    clip_min: float = 0.0
    clip_max: float = 1.0

    def run(
        self,
        vlm: Any,
        image: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target: torch.Tensor,
        *,
        forward_kwargs: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> AttackResult:
        proxy = PGDAttack(
            epsilon=self.epsilon,
            alpha=self.alpha,
            steps=self.steps,
            random_restarts=self.random_restarts,
            targeted=True,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
        )
        result = proxy.run(
            vlm=vlm,
            image=image,
            prompt_tokens=prompt_tokens,
            target=target,
            forward_kwargs=forward_kwargs,
        )
        result.metadata.setdefault("attack", self.name)
        result.metadata["target_tool"] = self.target_tool
        result.metadata["target_step_k"] = self.target_step_k
        return result
