"""Cover the small lazy-import / stub modules and the targeted_tool wrapper.

These modules have low statement counts but were uncovered by prior tests
because their bodies only execute on import-time attribute access or via
specific dispatch paths. Each test here exercises one such path.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest
import torch

# --- adversarial_reasoning/__init__.py — lazy __getattr__ -------------------


def test_pkg_getattr_resolves_known_lazy_imports() -> None:
    import adversarial_reasoning as ar

    # Each lazy name maps to (module, attr) — touching it must import + return.
    assert ar.RunnerConfig.__name__ == "RunnerConfig"
    assert ar.AttackBase.__name__ == "AttackBase"
    assert callable(ar.load_runner_config)
    assert callable(ar.main)


def test_pkg_getattr_unknown_raises_attribute_error() -> None:
    import adversarial_reasoning as ar

    with pytest.raises(AttributeError, match="no attribute 'definitely_not_real'"):
        ar.definitely_not_real  # noqa: B018


# --- gates/__init__.py — PEP 562 lazy import branch -------------------------


def test_gates_lazy_imports_resolve() -> None:
    from adversarial_reasoning import gates

    assert gates.PreprocessingTransferResult.__name__ == "PreprocessingTransferResult"
    assert callable(gates.run_preprocessing_transfer)
    assert gates.NoiseFloorResult.__name__ == "NoiseFloorResult"
    assert callable(gates.run_noise_floor)


def test_gates_unknown_attribute_raises() -> None:
    from adversarial_reasoning import gates

    with pytest.raises(AttributeError, match="bogus"):
        gates.bogus  # noqa: B018


# --- runner/__main__.py — module entrypoint ---------------------------------


def test_runner_main_module_importable() -> None:
    """Importing the module covers its top-level statements without invoking
    the ``if __name__ == '__main__'`` block."""
    import importlib

    mod = importlib.import_module("adversarial_reasoning.runner.__main__")
    assert callable(mod.main)


def test_runner_main_module_help_exits_zero() -> None:
    """``python -m adversarial_reasoning.runner --help`` must exit 0.

    Subprocess covers the ``if __name__ == '__main__'`` branch end-to-end
    (coverage data isn't shared back to the parent process — this test
    asserts behavior, the import-time test above asserts coverage)."""
    proc = subprocess.run(
        [sys.executable, "-m", "adversarial_reasoning.runner", "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 0
    assert "Adversarial reasoning runner" in proc.stdout


# --- tools/pubmed_stub.py — _lookup + tool() factory ------------------------


def test_pubmed_stub_lookup_hit() -> None:
    from adversarial_reasoning.tools.pubmed_stub import _lookup

    out = _lookup(["pi_rads", "biopsy"])
    assert len(out) == 1
    assert out[0]["pmid"] == "33197547"


def test_pubmed_stub_lookup_case_insensitive_and_extra_terms() -> None:
    from adversarial_reasoning.tools.pubmed_stub import _lookup

    # Fixture uses lowercase + a 2-element subset; uppercase + extra terms hit.
    out = _lookup(["PI_RADS", "BIOPSY", "extra"])
    assert len(out) == 1


def test_pubmed_stub_lookup_miss_returns_empty() -> None:
    from adversarial_reasoning.tools.pubmed_stub import _lookup

    assert _lookup(["nothing", "matches"]) == []


def test_pubmed_stub_tool_factory_returns_registered_tool() -> None:
    from adversarial_reasoning.tools.pubmed_stub import tool

    t = tool()
    assert t.name == "lookup_pubmed"
    assert t.parameters_schema["required"] == ["terms"]
    # Handler is _lookup — call through to confirm wiring.
    assert t.handler(["prostate", "active_surveillance"])[0]["pmid"] == "29910363"


# --- attacks/targeted_tool.py — TargetedToolPGD.run wrapper -----------------


def test_targeted_tool_pgd_stamps_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """The run() wrapper delegates to PGDAttack and stamps target metadata."""
    from adversarial_reasoning.attacks import targeted_tool
    from adversarial_reasoning.attacks.base import AttackResult

    captured = {}

    def fake_run(self, **kw):
        captured.update(kw)
        return AttackResult(
            perturbed_image=torch.zeros(1, 3, 4, 4),
            delta=torch.zeros(1, 3, 4, 4),
            loss_final=0.0,
            iterations=1,
        )

    monkeypatch.setattr(targeted_tool.PGDAttack, "run", fake_run)

    attack = targeted_tool.TargetedToolPGD(
        epsilon=0.02,
        steps=2,
        target_tool="custom_tool",
    )
    out = attack.run(
        vlm=object(),
        image=torch.zeros(1, 3, 4, 4),
        prompt_tokens=torch.zeros(1, 1, dtype=torch.long),
        target=torch.zeros(1, 1, dtype=torch.long),
        forward_kwargs={},
    )
    # PGDAttack.run got the expected kwargs from the wrapper.
    assert "image" in captured
    assert "prompt_tokens" in captured
    assert "target" in captured
    # Metadata was stamped on the result.
    assert out.metadata["attack"] == "targeted_tool_pgd"
    assert out.metadata["target_tool"] == "custom_tool"


# --- agents/medical_agent.py — pure parsers + branch coverage --------------


def _make_agent(vlm: Any):
    from adversarial_reasoning.agents.medical_agent import MedicalAgent
    from adversarial_reasoning.tools.registry import default_registry

    return MedicalAgent(vlm=vlm, tools=default_registry())


def _scripted_vlm(script: list[str]):
    """Build a stub VLM with both ``generate`` and ``generate_from_pixel_values``."""
    from dataclasses import dataclass

    from adversarial_reasoning.models.base import VLMBase, VLMGenerateResult

    @dataclass
    class _Vlm(VLMBase):
        family: str = "scripted"
        model_id: str = "stub/scripted"
        supports_gradients: bool = False

        def __post_init__(self) -> None:
            self._script = list(script)
            self._step = 0

        def _next(self) -> VLMGenerateResult:
            text = self._script[self._step % len(self._script)]
            self._step += 1
            return VLMGenerateResult(text=text)

        def generate(self, image, prompt, **kw):
            return self._next()

        def generate_from_pixel_values(self, *, pixel_values, prompt, template_image, **kw):
            return self._next()

    return _Vlm()


def test_find_balanced_close_handles_strings_and_escapes() -> None:
    """``_find_balanced_close`` must ignore braces inside JSON strings,
    including escaped quotes."""
    from adversarial_reasoning.agents.medical_agent import MedicalAgent

    # Brace inside string — must NOT close the outer object early.
    text = '{"k": "}{not real{"}'
    end = MedicalAgent._find_balanced_close(text, 0)
    assert text[end] == "}"
    # Escaped quote inside string ("\"") must not flip out of string mode.
    text2 = '{"k": "with \\"quote\\" inside"}'
    end2 = MedicalAgent._find_balanced_close(text2, 0)
    assert text2[end2] == "}"


def test_find_balanced_close_returns_minus_one_when_unbalanced() -> None:
    from adversarial_reasoning.agents.medical_agent import MedicalAgent

    assert MedicalAgent._find_balanced_close('{"k": "v"', 0) == -1


def test_extract_tool_calls_skips_invalid_json_keeps_scanning() -> None:
    """Invalid JSON before a valid tool-call object must not abort the scan."""
    agent = _make_agent(_scripted_vlm(["unused"]))
    text = (
        "noise {bad json no quotes} more noise "
        '{"name": "query_guidelines", "arguments": {"condition": "x", "query": "y"}}'
    )
    calls = agent._extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "query_guidelines"


def test_extract_tool_calls_handles_unclosed_brace() -> None:
    """An unclosed brace at end of text must terminate the scan cleanly."""
    agent = _make_agent(_scripted_vlm(["unused"]))
    assert agent._extract_tool_calls('{"name": "x", "arguments": ') == []


def test_extract_tool_calls_returns_empty_for_no_braces() -> None:
    agent = _make_agent(_scripted_vlm(["unused"]))
    assert agent._extract_tool_calls("plain text no JSON") == []


def test_dispatch_unknown_tool_records_error() -> None:
    """Calling _dispatch with a name not in the registry returns an error
    ToolCall instead of raising."""
    agent = _make_agent(_scripted_vlm(["unused"]))
    tc = agent._dispatch(step=0, call_spec={"name": "not_a_real_tool", "arguments": {}})
    assert tc.error == "unknown_tool: not_a_real_tool"
    assert tc.result is None


def test_run_with_pixel_values_requires_method_on_vlm(dummy_image) -> None:
    """If the VLM lacks ``generate_from_pixel_values``, an attempt to use it
    for adversarial inference must raise NotImplementedError."""
    from dataclasses import dataclass

    import torch

    from adversarial_reasoning.models.base import VLMBase, VLMGenerateResult

    @dataclass
    class NoPVL(VLMBase):
        family: str = "noPV"
        model_id: str = "stub/noPV"
        supports_gradients: bool = False

        def generate(self, image, prompt, **kw):
            return VLMGenerateResult(text="")

    agent = _make_agent(NoPVL())
    with pytest.raises(NotImplementedError, match="generate_from_pixel_values"):
        agent.run_with_pixel_values(
            task_id="t",
            pixel_values=torch.zeros(1, 3, 4, 4),
            prompt="?",
            template_image=dummy_image,
        )


def test_run_with_pixel_values_happy_path(dummy_image) -> None:
    """run_with_pixel_values dispatches one tool call then exits when the next
    response carries no tool call (final answer branch)."""
    import torch

    vlm = _scripted_vlm(
        [
            '{"name": "query_guidelines", "arguments": {"condition": "x", "query": "y"}}',
            "Final answer: ok.",
        ]
    )
    agent = _make_agent(vlm)
    traj = agent.run_with_pixel_values(
        task_id="t",
        pixel_values=torch.zeros(1, 3, 4, 4),
        prompt="?",
        template_image=dummy_image,
        max_steps=3,
    )
    assert traj.tool_sequence() == ["query_guidelines"]
    assert traj.final_answer.startswith("Final answer")
    assert "hit_max_steps" not in traj.metadata


def test_run_with_pixel_values_hits_max_steps(dummy_image) -> None:
    """A VLM that keeps emitting tool calls forever flips the max-steps flag."""
    import torch

    vlm = _scripted_vlm(
        ['{"name": "query_guidelines", "arguments": {"condition": "x", "query": "y"}}']
    )
    agent = _make_agent(vlm)
    traj = agent.run_with_pixel_values(
        task_id="t",
        pixel_values=torch.zeros(1, 3, 4, 4),
        prompt="?",
        template_image=dummy_image,
        max_steps=2,
    )
    assert traj.metadata.get("hit_max_steps") is True
    assert len(traj.tool_calls) == 2


def test_run_hits_max_steps(dummy_image) -> None:
    """Same max-steps semantics on the PIL-image ``run`` path (line 86)."""
    vlm = _scripted_vlm(
        ['{"name": "query_guidelines", "arguments": {"condition": "x", "query": "y"}}']
    )
    agent = _make_agent(vlm)
    traj = agent.run(task_id="t", image=dummy_image, prompt="?", max_steps=2)
    assert traj.metadata.get("hit_max_steps") is True
    assert len(traj.tool_calls) == 2


def test_targeted_tool_pgd_does_not_clobber_existing_attack_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``setdefault`` must keep an attack name already present (e.g., when
    PGDAttack tags itself first)."""
    from adversarial_reasoning.attacks import targeted_tool
    from adversarial_reasoning.attacks.base import AttackResult

    def fake_run(self, **_: Any) -> AttackResult:
        r = AttackResult(
            perturbed_image=torch.zeros(1, 3, 4, 4),
            delta=torch.zeros(1, 3, 4, 4),
            loss_final=0.0,
            iterations=1,
        )
        r.metadata["attack"] = "pre_set_by_pgd"
        return r

    monkeypatch.setattr(targeted_tool.PGDAttack, "run", fake_run)

    attack = targeted_tool.TargetedToolPGD(epsilon=0.01, steps=1)
    out = attack.run(
        vlm=object(),
        image=torch.zeros(1, 3, 4, 4),
        prompt_tokens=torch.zeros(1, 1, dtype=torch.long),
        target=torch.zeros(1, 1, dtype=torch.long),
    )
    assert out.metadata["attack"] == "pre_set_by_pgd"
