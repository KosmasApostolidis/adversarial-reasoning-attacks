"""Perturbation primitives + gradient-attack dispatch."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ..agents.base import Trajectory
from ..agents.medical_agent import MedicalAgent
from ..attacks.apgd import APGDAttack
from ..attacks.base import AttackBase
from ..attacks.pgd import PGDAttack
from ..attacks.targeted_tool import TargetedToolPGD
from ..attacks.targets import (
    target_from_benign,
    target_from_tool,
    target_from_trajectory,
)
from ..attacks.trajectory_drift import TrajectoryDriftPGD

_ATK_CLIP_BOUND: float = 3.0  # pixel-value clipping for normalized tensors
from .config import GRADIENT_MODES


def perturb_noise(image: Image.Image, epsilon: float, seed: int) -> Image.Image:
    """Deterministic uniform-noise perturbation in pixel domain."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-epsilon, epsilon, arr.shape).astype(np.float32)
    adv = np.clip(arr + delta, 0.0, 1.0)
    return Image.fromarray((adv * 255.0).astype(np.uint8))


def perturb(mode: str, image: Image.Image, epsilon: float, seed: int, **_: Any) -> Image.Image:
    if mode == "noise":
        return perturb_noise(image, epsilon, seed)
    if mode in GRADIENT_MODES:
        raise NotImplementedError(
            f"{mode} operates on pre-normalised pixel tensors (not PIL); "
            "use run_gradient_attack in main()."
        )
    raise ValueError(f"Unknown attack mode: {mode}")


def build_attack(
    mode: str,
    *,
    epsilon: float,
    steps: int,
    target_tool: str = "escalate_to_specialist",
    target_step_k: int = 0,
    clip_min: float = -_ATK_CLIP_BOUND,
    clip_max: float = _ATK_CLIP_BOUND,
) -> AttackBase:
    """Construct an attack instance given the runner ``--mode`` value."""
    if mode == "pgd":
        return PGDAttack(
            epsilon=epsilon,
            steps=steps,
            random_restarts=1,
            targeted=False,
            clip_min=clip_min,
            clip_max=clip_max,
        )
    if mode == "apgd":
        return APGDAttack(
            epsilon=epsilon,
            steps=steps,
            random_restarts=1,
            targeted=False,
            clip_min=clip_min,
            clip_max=clip_max,
        )
    if mode == "targeted_tool":
        return TargetedToolPGD(
            epsilon=epsilon,
            steps=steps,
            random_restarts=1,
            target_tool=target_tool,
            target_step_k=target_step_k,
            clip_min=clip_min,
            clip_max=clip_max,
        )
    if mode == "trajectory_drift":
        return TrajectoryDriftPGD(
            epsilon=epsilon,
            steps=steps,
            random_restarts=1,
            targeted=False,
            clip_min=clip_min,
            clip_max=clip_max,
        )
    raise ValueError(f"Unknown gradient attack mode: {mode}")


_NON_GEN_KEYS = frozenset({"pixel_values", "input_ids", "attention_mask"})


def _build_attack_target(
    *,
    mode: str,
    vlm: Any,
    benign: Trajectory,
    prompt_input_ids: Any,
    target_tool: str,
) -> Any:
    """Pick the right target-token builder for ``mode``."""
    if mode == "targeted_tool":
        return target_from_tool(vlm, target_tool, device=prompt_input_ids.device)
    if mode == "trajectory_drift":
        return target_from_trajectory(vlm, benign, prompt_input_ids)
    return target_from_benign(vlm, benign, prompt_input_ids)


def _reshape_and_reinfer(
    *,
    res: Any,  # AttackResult
    pixel_values: Any,  # torch.Tensor
    agent: Any,
    sample: Any,
    task_id: str,
    seed: int,
    max_steps: int,
    model_kwargs: dict[str, Any],
    mode: str,
    target_tool: str,
    target_step_k: int,
) -> Any:  # Trajectory
    """Reshape perturbed tensor, re-run agent, attach attack metadata."""
    import torch

    perturbed_pv = res.perturbed_image
    if perturbed_pv.ndim == pixel_values.ndim - 1:
        perturbed_pv = perturbed_pv.unsqueeze(0)
    if perturbed_pv.shape != pixel_values.shape:
        perturbed_pv = perturbed_pv.reshape(pixel_values.shape)

    attacked = agent.run_with_pixel_values(
        task_id=task_id,
        pixel_values=perturbed_pv.to(pixel_values.dtype),
        prompt=sample.prompt,
        template_image=sample.image,
        seed=seed,
        max_steps=max_steps,
        gen_kwargs=dict(model_kwargs),
    )
    attacked.metadata[f"{mode}_loss_final"] = float(res.loss_final)
    attacked.metadata[f"{mode}_steps"] = int(res.iterations)
    if mode == "targeted_tool":
        attacked.metadata["target_tool"] = target_tool
        attacked.metadata["target_step_k"] = int(target_step_k)
        attacked.metadata["targeted_hit"] = int(target_tool in attacked.tool_sequence())
    return attacked


def run_gradient_attack(
    *,
    mode: str,
    vlm: Any,
    agent: MedicalAgent,
    sample: Any,
    benign: Trajectory,
    epsilon: float,
    steps: int,
    seed: int,
    max_steps: int,
    task_id: str,
    target_tool: str = "escalate_to_specialist",
    target_step_k: int = 0,
) -> Trajectory:
    """Run a gradient-based attack on the model's normalised pixel tensor,
    then re-run the agent on the perturbed pixels.

    Dispatches by ``mode`` to :mod:`adversarial_reasoning.attacks.targets`
    for target-token construction:

      - ``pgd`` / ``apgd``: benign first tool-call block — drop benign
        likelihood.
      - ``targeted_tool``: forced ``target_tool`` tool-call block — push
        attacker-chosen likelihood.
      - ``trajectory_drift``: concat of all benign tool-call blocks —
        KL ascent on the full benign trajectory.

    Notes
    -----
    - ε is applied in the *processor-normalised* pixel domain (CLIP std ≈
      0.27), so a pixel-domain ε of 8/255 corresponds to ~0.116 here.
      Keeping it normalised matches the noise mode apples-to-apples.
    - Clip bounds [-3, 3] generously cover the normalised range.
    - Model-family extras (``image_grid_thw`` for Qwen, ``image_sizes``
      for LLaVA-Next, etc.) flow through ``prepare_attack_inputs`` ⇒
      ``forward_kwargs`` ⇒ ``gen_kwargs`` without per-model branching.
    """
    import torch

    if mode not in GRADIENT_MODES:
        raise ValueError(f"Unknown gradient attack mode: {mode}")
    if not hasattr(vlm, "prepare_attack_inputs"):
        raise NotImplementedError(
            f"{type(vlm).__name__}.prepare_attack_inputs missing — needed for {mode}."
        )

    attack_in = vlm.prepare_attack_inputs(sample.image, sample.prompt)
    pixel_values = attack_in["pixel_values"]
    prompt_input_ids = attack_in["input_ids"]
    prompt_attn = attack_in.get("attention_mask")
    model_kwargs = {k: v for k, v in attack_in.items() if k not in _NON_GEN_KEYS}

    target_ids = _build_attack_target(
        mode=mode,
        vlm=vlm,
        benign=benign,
        prompt_input_ids=prompt_input_ids,
        target_tool=target_tool,
    )

    fwd_kwargs: dict[str, Any] = dict(model_kwargs)
    if prompt_attn is not None:
        fwd_kwargs["attention_mask"] = torch.cat([prompt_attn, torch.ones_like(target_ids)], dim=-1)

    attack = build_attack(
        mode,
        epsilon=epsilon,
        steps=steps,
        target_tool=target_tool,
        target_step_k=target_step_k,
    )
    res = attack.run(
        vlm=vlm,
        image=pixel_values,
        prompt_tokens=prompt_input_ids,
        target=target_ids,
        forward_kwargs=fwd_kwargs,
    )

    return _reshape_and_reinfer(
        res=res,
        pixel_values=pixel_values,
        agent=agent,
        sample=sample,
        task_id=task_id,
        seed=seed,
        max_steps=max_steps,
        model_kwargs=model_kwargs,
        mode=mode,
        target_tool=target_tool,
        target_step_k=target_step_k,
    )
