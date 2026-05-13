"""Phase-0 end-to-end probe gate.

Exercise the full chain against a real VLM so that broken contracts (missing
`image_grid_thw`, wrong tool-call wrapping, non-differentiable code paths)
surface before Phase 1 scaling. Intentionally pared down — this is a wiring
test, not a benchmark.

Procedure
---------
1. Load one HF VLM (fp16) + one real image + the default tool registry.
2. Run a benign generation → parse tool trajectory.
3. Build PGD attack inputs via `vlm.prepare_attack_inputs(...)` and run
   a tiny PGD (5 steps, ε=4/255) against a one-token target.
4. Verify (a) no exception was raised, (b) perturbation ‖δ‖∞ respects ε,
   (c) regenerating on the perturbed image still yields a finite trajectory.

Report fields are written to JSON for manifest inclusion. Runtime budget:
~2 minutes on H200 for the 7B anchor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ..agents.medical_agent import MedicalAgent
from ..attacks._epsilon import _LINF_EPSILON_4
from ..attacks.pgd import PGDAttack
from ..tools.registry import default_registry


@dataclass
class E2EProbeResult:
    model_id: str
    benign_tool_sequence: list[str] = field(default_factory=list)
    attacked_tool_sequence: list[str] = field(default_factory=list)
    pgd_linf: float = 0.0
    pgd_budget: float = 0.0
    pgd_steps: int = 0
    pgd_loss_final: float = 0.0
    exception: str | None = None
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "benign_tool_sequence": self.benign_tool_sequence,
            "attacked_tool_sequence": self.attacked_tool_sequence,
            "pgd_linf": self.pgd_linf,
            "pgd_budget": self.pgd_budget,
            "pgd_steps": self.pgd_steps,
            "pgd_loss_final": self.pgd_loss_final,
            "exception": self.exception,
            "passed": self.passed,
        }


def run_e2e_probe(
    vlm: Any,
    *,
    image: Image.Image,
    prompt: str,
    task_id: str = "e2e_probe",
    epsilon: float = _LINF_EPSILON_4,
    steps: int = 5,
) -> E2EProbeResult:
    """Run the minimal wiring probe.

    Any raised exception is captured into `result.exception` rather than
    propagated — the gate's job is to *report* integration-layer breakage,
    not fail the whole Phase-0 pipeline.
    """
    result = E2EProbeResult(
        model_id=getattr(vlm, "model_id", vlm.__class__.__name__),
        pgd_budget=epsilon,
        pgd_steps=steps,
    )
    registry = default_registry()
    agent = MedicalAgent(vlm=vlm, tools=registry)

    try:
        benign = agent.run(task_id=task_id, image=image, prompt=prompt, seed=0)
        result.benign_tool_sequence = benign.tool_sequence()

        attack_inputs = vlm.prepare_attack_inputs(
            image=image,
            prompt=prompt,
            tools_schema=registry.schemas(),
        )
        pixel_values = attack_inputs.pop("pixel_values")
        input_ids = attack_inputs.pop("input_ids")

        # One-token untargeted target = first prompt token (placeholder — the
        # real benchmark uses the first tool-call token id from `benign`).
        target = input_ids[:, :1].clone()

        attack = PGDAttack(
            epsilon=epsilon,
            steps=steps,
            random_restarts=1,
            targeted=False,
        )
        ar = attack.run(
            vlm=vlm,
            image=pixel_values,
            prompt_tokens=input_ids,
            target=target,
            forward_kwargs=attack_inputs,
        )
        result.pgd_linf = float(torch.as_tensor(ar.delta).abs().max().item())
        result.pgd_loss_final = ar.loss_final

        # Regenerate on the perturbed pixel tensor by round-tripping through
        # PIL. Pixel-space re-load exercises the HF→deployment preprocessing
        # path without requiring Ollama for this gate.
        perturbed_pil = _tensor_to_pil(ar.perturbed_image)
        attacked = agent.run(task_id=task_id, image=perturbed_pil, prompt=prompt, seed=0)
        result.attacked_tool_sequence = attacked.tool_sequence()
        result.passed = result.pgd_linf <= epsilon + 1e-5
    except Exception as exc:
        result.exception = f"{type(exc).__name__}: {exc}"
        result.passed = False

    return result


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert a (C, H, W) or (B, C, H, W) tensor in [0, 1] → uint8 PIL image."""
    if t.ndim == 4:
        t = t[0]
    arr = (t.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
    return Image.fromarray(arr.permute(1, 2, 0).numpy())


def write_probe_report(result: E2EProbeResult, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
