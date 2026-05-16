"""Coverage tests for runner.attacks — perturb dispatch + build_attack table.

run_gradient_attack itself is exercised by integration tests on GPU; here
we only unit-test the parts that don't need a real VLM forward:
  * perturb() noise dispatch + range/shape invariants
  * perturb() rejecting gradient modes (PIL-domain mismatch)
  * build_attack() dispatch table — every mode returns the right class
  * run_gradient_attack() rejects unknown mode + missing prepare_attack_inputs
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from adversarial_reasoning.attacks.apgd import APGDAttack
from adversarial_reasoning.attacks.pgd import PGDAttack
from adversarial_reasoning.attacks.targeted_tool import TargetedToolPGD
from adversarial_reasoning.attacks.trajectory_drift import TrajectoryDriftPGD
from adversarial_reasoning.runner.attacks import (
    build_attack,
    perturb,
    perturb_noise,
    run_gradient_attack,
)


def test_perturb_noise_shape_and_range(dummy_image: Image.Image) -> None:
    out = perturb_noise(dummy_image, epsilon=0.05, seed=0)
    arr = np.asarray(out)
    assert arr.shape == np.asarray(dummy_image).shape
    assert arr.dtype == np.uint8
    assert arr.min() >= 0
    assert arr.max() <= 255


def test_perturb_noise_deterministic(dummy_image: Image.Image) -> None:
    a = np.asarray(perturb_noise(dummy_image, epsilon=0.05, seed=42))
    b = np.asarray(perturb_noise(dummy_image, epsilon=0.05, seed=42))
    assert np.array_equal(a, b)


def test_perturb_dispatches_noise(dummy_image: Image.Image) -> None:
    out = perturb("noise", dummy_image, 0.01, 0)
    assert isinstance(out, Image.Image)


@pytest.mark.parametrize("mode", ["pgd", "apgd", "targeted_tool", "trajectory_drift"])
def test_perturb_rejects_gradient_modes(mode: str, dummy_image: Image.Image) -> None:
    with pytest.raises(NotImplementedError, match=mode):
        perturb(mode, dummy_image, 0.01, 0)


def test_perturb_rejects_unknown_mode(dummy_image: Image.Image) -> None:
    with pytest.raises(ValueError, match="Unknown attack mode"):
        perturb("nonsense", dummy_image, 0.01, 0)


@pytest.mark.parametrize(
    "mode,cls",
    [
        ("pgd", PGDAttack),
        ("apgd", APGDAttack),
        ("targeted_tool", TargetedToolPGD),
        ("trajectory_drift", TrajectoryDriftPGD),
    ],
)
def test_build_attack_dispatch(mode: str, cls: type) -> None:
    a = build_attack(mode, epsilon=0.01, steps=3)
    assert isinstance(a, cls)
    assert a.epsilon == 0.01
    assert a.steps == 3


def test_build_attack_targeted_tool_passes_target() -> None:
    a = build_attack("targeted_tool", epsilon=0.02, steps=5, target_tool="foo", target_step_k=2)
    assert isinstance(a, TargetedToolPGD)
    assert a.target_tool == "foo"
    assert a.target_step_k == 2


def test_build_attack_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown gradient attack mode"):
        build_attack("not_a_mode", epsilon=0.01, steps=3)


def test_run_gradient_attack_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown gradient attack mode"):
        run_gradient_attack(
            mode="nonsense",
            vlm=object(),
            agent=object(),  # type: ignore[arg-type]
            sample=object(),
            benign=object(),  # type: ignore[arg-type]
            epsilon=0.01,
            steps=1,
            seed=0,
            max_steps=1,
            task_id="t",
        )


def test_run_gradient_attack_requires_prepare_attack_inputs() -> None:
    class StubVLM:
        pass

    with pytest.raises(NotImplementedError, match="prepare_attack_inputs"):
        run_gradient_attack(
            mode="pgd",
            vlm=StubVLM(),
            agent=object(),  # type: ignore[arg-type]
            sample=object(),
            benign=object(),  # type: ignore[arg-type]
            epsilon=0.01,
            steps=1,
            seed=0,
            max_steps=1,
            task_id="t",
        )


@pytest.mark.parametrize(
    "mode,target_fn",
    [
        ("pgd", "target_from_benign"),
        ("apgd", "target_from_benign"),
        ("targeted_tool", "target_from_tool"),
        ("trajectory_drift", "target_from_trajectory"),
    ],
)
def test_build_attack_target_dispatch(
    mode: str, target_fn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify _build_attack_target picks the right target builder per mode."""
    from adversarial_reasoning.runner import attacks as runner_attacks

    called = {}

    def fake(*args, **kwargs):
        called["fn"] = target_fn
        return "T"

    monkeypatch.setattr(runner_attacks, target_fn, fake)
    out = runner_attacks._build_attack_target(
        mode=mode,
        vlm=object(),
        benign=object(),  # type: ignore[arg-type]
        prompt_input_ids=type("D", (), {"device": "cpu"})(),
        target_tool="esc",
    )
    assert out == "T"
    assert called["fn"] == target_fn


def _make_targeted_tool_stubs(captured, pv, input_ids, attn):
    """Build (StubVLM, StubAgent, StubAttack) tuple for targeted-tool path."""
    import torch

    from adversarial_reasoning.agents.base import Trajectory
    from adversarial_reasoning.attacks.base import AttackResult

    class StubVLM:
        def prepare_attack_inputs(self, image, prompt):
            return {
                "pixel_values": pv,
                "input_ids": input_ids,
                "attention_mask": attn,
                "image_grid_thw": torch.zeros(1, 3),  # extra Qwen-style kwarg
            }

    class StubAgent:
        def run_with_pixel_values(
            self, *, task_id, pixel_values, prompt, template_image, seed, max_steps, gen_kwargs
        ):
            captured["pv_shape"] = tuple(pixel_values.shape)
            captured["pv_dtype"] = pixel_values.dtype
            captured["gen_kwargs"] = dict(gen_kwargs)
            return Trajectory(task_id=task_id, model_id="m", seed=seed)

    class StubAttack:
        def run(self, *, vlm, image, prompt_tokens, target, forward_kwargs):
            captured["fwd_kwargs"] = dict(forward_kwargs)
            return AttackResult(
                perturbed_image=torch.zeros(3, 8, 8),  # ndim mismatch — exercises unsqueeze
                delta=torch.zeros(3, 8, 8),
                loss_final=0.5,
                iterations=2,
            )

    return StubVLM, StubAgent, StubAttack


def test_run_gradient_attack_full_path(
    monkeypatch: pytest.MonkeyPatch, dummy_image: Image.Image
) -> None:
    """Exercise the full run_gradient_attack body with stubbed torch/VLM/agent.

    Validates that:
      - prepare_attack_inputs is called and its tensors flow into the attack
      - target builder is selected by mode
      - perturbed pixel_values are reshaped/recasted and forwarded to the agent
      - mode-specific metadata is written onto the attacked Trajectory
    """
    import torch

    from adversarial_reasoning.agents.base import Trajectory
    from adversarial_reasoning.runner import attacks as runner_attacks

    pv = torch.zeros(1, 3, 8, 8)
    input_ids = torch.zeros(1, 4, dtype=torch.long)
    attn = torch.ones(1, 4, dtype=torch.long)
    captured: dict = {}
    StubVLM, StubAgent, StubAttack = _make_targeted_tool_stubs(captured, pv, input_ids, attn)
    monkeypatch.setattr(runner_attacks, "build_attack", lambda *a, **kw: StubAttack())
    target_zero = torch.zeros(1, 2, dtype=torch.long)
    monkeypatch.setattr(runner_attacks, "target_from_tool", lambda *a, **kw: target_zero)

    out = runner_attacks.run_gradient_attack(
        mode="targeted_tool",
        vlm=StubVLM(),
        agent=StubAgent(),  # type: ignore[arg-type]
        sample=type("S", (), {"image": dummy_image, "prompt": "?"})(),
        benign=Trajectory(task_id="t", model_id="m", seed=0),
        epsilon=0.01,
        steps=2,
        seed=7,
        max_steps=1,
        task_id="t",
        target_tool="escalate",
    )
    assert captured["pv_shape"] == (1, 3, 8, 8)
    assert captured["pv_dtype"] == pv.dtype
    # image_grid_thw must flow through gen_kwargs (model-family extras path)
    assert "image_grid_thw" in captured["gen_kwargs"]
    # attention mask was concatenated with target in forward_kwargs
    assert captured["fwd_kwargs"]["attention_mask"].shape == (1, 6)
    # mode metadata
    assert out.metadata["targeted_tool_loss_final"] == pytest.approx(0.5)
    assert out.metadata["targeted_tool_steps"] == 2
    assert out.metadata["target_tool"] == "escalate"
    assert out.metadata["target_step_k"] == 0
    assert out.metadata["targeted_hit"] == 0  # stub agent has empty tool_sequence


def _make_reshape_stubs(captured, pv, input_ids):
    """Build (StubVLM, StubAgent, StubAttack) tuple for reshape-fallback path."""
    import torch

    from adversarial_reasoning.agents.base import Trajectory
    from adversarial_reasoning.attacks.base import AttackResult

    class StubVLM:
        def prepare_attack_inputs(self, image, prompt):
            return {"pixel_values": pv, "input_ids": input_ids}

    class StubAgent:
        def run_with_pixel_values(self, *, pixel_values, **kw):
            captured["pv_shape"] = tuple(pixel_values.shape)
            return Trajectory(task_id="t", model_id="m", seed=0)

    class StubAttack:
        def run(self, **_):
            return AttackResult(
                # Same ndim (4) but different shape — exercises the reshape
                # fallback (line 202) instead of the unsqueeze branch.
                perturbed_image=torch.zeros(1, 8, 8, 3),
                delta=torch.zeros(1, 8, 8, 3),
                loss_final=0.0,
                iterations=1,
            )

    return StubVLM, StubAgent, StubAttack


def test_run_gradient_attack_reshape_fallback(
    monkeypatch: pytest.MonkeyPatch, dummy_image: Image.Image
) -> None:
    """When perturbed_pv has the same ndim as pv but a different shape (e.g.,
    the attack loop returned a flat or transposed view), run_gradient_attack
    must reshape it back to pv.shape rather than crash."""
    import torch

    from adversarial_reasoning.agents.base import Trajectory
    from adversarial_reasoning.runner import attacks as runner_attacks

    pv = torch.zeros(1, 3, 8, 8)
    input_ids = torch.zeros(1, 4, dtype=torch.long)
    captured: dict = {}
    StubVLM, StubAgent, StubAttack = _make_reshape_stubs(captured, pv, input_ids)

    monkeypatch.setattr(runner_attacks, "build_attack", lambda *a, **kw: StubAttack())
    monkeypatch.setattr(
        runner_attacks, "target_from_benign", lambda *a, **kw: torch.zeros(1, 2, dtype=torch.long)
    )

    runner_attacks.run_gradient_attack(
        mode="pgd",
        vlm=StubVLM(),
        agent=StubAgent(),  # type: ignore[arg-type]
        sample=type("S", (), {"image": dummy_image, "prompt": "?"})(),
        benign=Trajectory(task_id="t", model_id="m", seed=0),
        epsilon=0.01,
        steps=1,
        seed=0,
        max_steps=1,
        task_id="t",
    )
    # Reshape brought it back to pv.shape (1,3,8,8), not the (1,8,8,3) stub.
    assert captured["pv_shape"] == (1, 3, 8, 8)
