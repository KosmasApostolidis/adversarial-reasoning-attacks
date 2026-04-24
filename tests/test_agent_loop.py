"""Agent loop smoke test using a scripted stub VLM.

Verifies the ReAct-style loop:
- parses tool-call JSON out of model output
- dispatches correctly to the registry
- records tool args, results, and terminal final-answer
- terminates when the VLM stops emitting tool-call JSON
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from adversarial_reasoning.agents.medical_agent import MedicalAgent
from adversarial_reasoning.models.base import VLMBase, VLMGenerateResult
from adversarial_reasoning.tools import default_registry


@dataclass
class _ScriptedVLM(VLMBase):
    """Returns a fixed script of responses, one per generate() call."""

    family: str = "scripted"
    model_id: str = "stub/scripted"
    supports_gradients: bool = False

    def __post_init__(self) -> None:
        self._script = [
            '{"tool": "query_guidelines", "args": {"condition": "prostate_cancer", '
            '"query": "pi_rads_4"}}',
            '{"tool": "calculate_risk_score", "args": {"name": "pi_rads", '
            '"features": {"psa": 6.0, "volume_cc": 40.0, "lesion_grade": 4}}}',
            '{"tool": "draft_report", "args": {"modality": "MRI", '
            '"findings": ["PI-RADS 4 lesion in peripheral zone."]}}',
            "Final answer: recommend targeted biopsy referral.",
        ]
        self._step = 0

    def generate(self, image, prompt, **kwargs) -> VLMGenerateResult:
        text = self._script[self._step]
        self._step += 1
        return VLMGenerateResult(text=text)


def test_agent_runs_three_tools_then_stops():
    vlm = _ScriptedVLM()
    registry = default_registry()
    agent = MedicalAgent(vlm=vlm, tools=registry)
    image = Image.new("RGB", (32, 32), color=(128, 128, 128))

    trajectory = agent.run(
        task_id="test_case_001",
        image=image,
        prompt="Evaluate this prostate MRI.",
        seed=0,
        max_steps=6,
    )

    assert trajectory.tool_sequence() == [
        "query_guidelines",
        "calculate_risk_score",
        "draft_report",
    ]
    assert "biopsy" in trajectory.final_answer.lower()
    assert all(c.error is None for c in trajectory.tool_calls)


def test_agent_parses_qwen_hermes_tool_call_wrapping():
    """Qwen2.5-VL emits <tool_call>{"name": ..., "arguments": ...}</tool_call>.

    Parser must accept both `name`/`arguments` (Hermes) and `tool`/`args`
    (scaffolded) variants without ambiguity.
    """
    hermes_script = [
        "I'll check the guidelines first.\n"
        "<tool_call>\n"
        '{"name": "query_guidelines", "arguments": {"condition": '
        '"prostate_cancer", "query": "pi_rads"}}\n'
        "</tool_call>",
        "Done.",
    ]

    class _HermesVLM(_ScriptedVLM):
        def __post_init__(self) -> None:
            self._script = hermes_script
            self._step = 0

    vlm = _HermesVLM()
    agent = MedicalAgent(vlm=vlm, tools=default_registry())
    traj = agent.run(
        task_id="hermes_probe",
        image=Image.new("RGB", (16, 16)),
        prompt="Evaluate.",
    )
    assert traj.tool_sequence() == ["query_guidelines"]
    assert traj.tool_calls[0].error is None
    assert traj.tool_calls[0].args["condition"] == "prostate_cancer"


def test_agent_records_tool_error_on_bad_args():
    bad_script = [
        # Invalid urgency should surface as a ValueError from the handler,
        # captured as ToolCall.error rather than crashing the loop.
        '{"tool": "request_followup", "args": {"test_name": "psa", "urgency": "maybe"}}',
        "Done.",
    ]

    class _BadScriptedVLM(_ScriptedVLM):
        def __post_init__(self) -> None:
            self._script = bad_script
            self._step = 0

    vlm = _BadScriptedVLM()
    agent = MedicalAgent(vlm=vlm, tools=default_registry())
    image = Image.new("RGB", (16, 16))

    traj = agent.run(task_id="t", image=image, prompt="go")
    assert traj.tool_calls[0].name == "request_followup"
    assert traj.tool_calls[0].error is not None
