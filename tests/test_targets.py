"""Unit tests for ``adversarial_reasoning.attacks.targets``.

The target-token builders are pure tokenizer wrappers — exercising them
without loading a real VLM only requires a stub ``vlm.processor.tokenizer``
that returns deterministic ``input_ids``. We assert structure, not byte
equivalence to a specific HF tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from adversarial_reasoning.attacks.targets import (
    build_target_tokens,
    target_from_benign,
    target_from_tool,
    target_from_trajectory,
)


@dataclass
class _ToolCall:
    name: str
    args: dict


@dataclass
class _Trajectory:
    """Minimal stand-in for ``adversarial_reasoning.agents.base.Trajectory``."""
    tool_calls: list[_ToolCall]


class _StubTokenizer:
    """Maps each character in the input text to its ord(), padded to a tensor.

    Deterministic + fast; lets tests assert that:
      - shape is (1, T) where T == len(text)
      - longer text ⇒ longer token tensor
      - device routing works
    """

    def __call__(self, text: str, *, return_tensors: str, add_special_tokens: bool):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        ids = torch.tensor([[ord(c) for c in text]], dtype=torch.long)
        return {"input_ids": ids}


class _StubProcessor:
    def __init__(self) -> None:
        self.tokenizer = _StubTokenizer()


class _StubVLM:
    def __init__(self) -> None:
        self.processor = _StubProcessor()


@pytest.mark.smoke
def test_target_from_benign_uses_first_tool_call():
    vlm = _StubVLM()
    benign = _Trajectory(tool_calls=[
        _ToolCall("query_guidelines", {"topic": "pca"}),
        _ToolCall("calculate_risk_score", {"psa": 6.5}),
    ])
    prompt_ids = torch.zeros(1, 1, dtype=torch.long)
    out = target_from_benign(vlm, benign, prompt_ids)
    assert out.dim() == 2
    assert out.shape[0] == 1
    # First tool call wraps in <tool_call>...</tool_call>; output should grow
    # roughly with the json arg payload.
    assert out.shape[-1] > len("<tool_call>")


@pytest.mark.smoke
def test_target_from_benign_falls_back_when_empty():
    vlm = _StubVLM()
    benign = _Trajectory(tool_calls=[])
    prompt_ids = torch.zeros(1, 1, dtype=torch.long)
    out = target_from_benign(vlm, benign, prompt_ids)
    # Placeholder is the describe_region block; non-empty tensor.
    assert out.numel() > 0


@pytest.mark.smoke
def test_target_from_trajectory_concatenates_all_tool_calls():
    vlm = _StubVLM()
    single = _Trajectory(tool_calls=[_ToolCall("a", {})])
    full = _Trajectory(tool_calls=[
        _ToolCall("a", {}),
        _ToolCall("b", {}),
        _ToolCall("c", {}),
    ])
    prompt_ids = torch.zeros(1, 1, dtype=torch.long)
    short = target_from_trajectory(vlm, single, prompt_ids).shape[-1]
    long_ = target_from_trajectory(vlm, full, prompt_ids).shape[-1]
    assert long_ > short


@pytest.mark.smoke
def test_target_from_trajectory_falls_back_to_benign_when_empty():
    vlm = _StubVLM()
    benign = _Trajectory(tool_calls=[])
    prompt_ids = torch.zeros(1, 1, dtype=torch.long)
    a = target_from_trajectory(vlm, benign, prompt_ids)
    b = target_from_benign(vlm, benign, prompt_ids)
    assert torch.equal(a, b)


@pytest.mark.smoke
def test_target_from_tool_includes_target_name_in_output():
    vlm = _StubVLM()
    out = target_from_tool(
        vlm, target_tool="escalate_to_specialist", target_args={"urgency": "high"},
    )
    # Stub tokenizer maps each char to its ord(); reconstruct text.
    text = "".join(chr(int(c)) for c in out[0].tolist())
    assert "escalate_to_specialist" in text
    assert "urgency" in text
    assert text.startswith("<tool_call>")
    assert text.endswith("</tool_call>")


@pytest.mark.smoke
def test_target_from_tool_routes_to_device():
    vlm = _StubVLM()
    cpu = torch.device("cpu")
    out = target_from_tool(vlm, target_tool="x", device=cpu)
    assert out.device.type == "cpu"


@pytest.mark.smoke
def test_build_target_tokens_is_back_compat_alias_for_target_from_tool():
    assert build_target_tokens is target_from_tool
