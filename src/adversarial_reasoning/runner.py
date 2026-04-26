"""Experiment runner.

Loads ``configs/<exp>.yaml``, iterates over
``models x tasks x attacks x epsilons x seeds x samples``, and records one
JSONL row per (benign, attacked) pair.

Attack modes
------------
- ``noise`` (default, smoke): uniform perturbation in ``[-ε, ε]`` on pixels.
  No gradient, but exercises the full harness end-to-end.
- ``pgd``: full PGD-L∞ via :mod:`adversarial_reasoning.attacks.pgd`. Requires
  a model-family-specific ``prepare_attack_inputs`` path and target-token
  construction; currently implemented on a best-effort basis and may raise
  ``NotImplementedError`` for VLMs whose prepare-attack path is incomplete.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from .agents.base import Trajectory
from .agents.medical_agent import MedicalAgent
from .attacks.apgd import APGDAttack
from .attacks.base import AttackBase
from .attacks.pgd import PGDAttack
from .attacks.targeted_tool import TargetedToolPGD
from .attacks.targets import (
    target_from_benign,
    target_from_tool,
    target_from_trajectory,
)
from .attacks.trajectory_drift import TrajectoryDriftPGD
from .metrics.trajectory import trajectory_edit_distance
from .models.loader import load_hf_vlm
from .tasks.loader import load_task
from .tools.registry import default_registry

GRADIENT_MODES = {"pgd", "apgd", "targeted_tool", "trajectory_drift"}


# ----------------------------- config loading -----------------------------


@dataclass
class RunnerConfig:
    name: str
    phase: str
    output_dir: Path
    seeds: list[int]
    models: list[str]
    tasks: list[str]
    attacks: list[str]
    task_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    attack_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    epsilons_linf: list[float] = field(default_factory=list)
    split: str = "dev"


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})


def load_runner_config(exp_path: str | Path) -> RunnerConfig:
    raw = _load_yaml(exp_path)["experiment"]
    return RunnerConfig(
        name=raw["name"],
        phase=str(raw.get("phase", "0")),
        output_dir=Path(raw.get("output_dir", "runs") + ""),
        seeds=list(raw.get("seeds", [0])),
        models=list(raw["models"]),
        tasks=list(raw["tasks"]),
        attacks=list(raw["attacks"]),
        task_overrides=dict(raw.get("task_overrides", {})),
        attack_overrides=dict(raw.get("attack_overrides", {})),
        epsilons_linf=list(raw.get("epsilons_linf", [])),
    )


def resolve_epsilons(cfg: RunnerConfig, attack_name: str, attacks_yaml: dict) -> list[float]:
    overrides = cfg.attack_overrides.get(attack_name, {})
    if "epsilons" in overrides:
        return list(overrides["epsilons"])
    attack_cfg = attacks_yaml["attacks"].get(attack_name, {})
    eps = attack_cfg.get("epsilons")
    return list(cfg.epsilons_linf) if eps in (None, []) else list(eps)


# ----------------------------- perturbation -----------------------------


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
    clip_min: float = -3.0,
    clip_max: float = 3.0,
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
    # Everything other than pixel_values/input_ids/attention_mask is a
    # model-family-specific tensor that must flow through both the gradient
    # forward (forward_kwargs) and the post-attack generation (gen_kwargs)
    # untouched. New backends just add new keys; no per-model branching.
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

    perturbed_pv = res.perturbed_image
    if perturbed_pv.ndim == pixel_values.ndim - 1:
        perturbed_pv = perturbed_pv.unsqueeze(0)
    if perturbed_pv.shape != pixel_values.shape:
        perturbed_pv = perturbed_pv.view(pixel_values.shape)

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
        seq = attacked.tool_sequence()
        attacked.metadata["targeted_hit"] = int(target_tool in seq)
    return attacked


# ----------------------------- record schema ----------------------------


def trajectory_record(t: Trajectory) -> dict:
    return {
        "task_id": t.task_id,
        "model_id": t.model_id,
        "seed": t.seed,
        "tool_sequence": t.tool_sequence(),
        "tool_calls": [c.to_dict() for c in t.tool_calls],
        "final_answer": t.final_answer,
        "metadata": t.metadata,
    }


def pair_record(
    *,
    model_key: str,
    task_id: str,
    sample_id: str,
    attack_name: str,
    attack_mode: str,
    epsilon: float,
    seed: int,
    benign: Trajectory,
    attacked: Trajectory,
    edit_distance: float,
    elapsed_s: float,
) -> dict:
    return {
        "model_key": model_key,
        "task_id": task_id,
        "sample_id": sample_id,
        "attack_name": attack_name,
        "attack_mode": attack_mode,
        "epsilon": epsilon,
        "seed": seed,
        "benign": trajectory_record(benign),
        "attacked": trajectory_record(attacked),
        "edit_distance_norm": edit_distance,
        "elapsed_s": elapsed_s,
    }


# --------------------------------- main ---------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Adversarial reasoning runner")
    p.add_argument("--config", required=True, help="Experiment YAML (e.g. configs/smoke.yaml)")
    p.add_argument("--attacks-config", default="configs/attacks.yaml")
    p.add_argument(
        "--mode",
        choices=["noise", "pgd", "apgd", "targeted_tool", "trajectory_drift"],
        default="noise",
    )
    p.add_argument(
        "--synthetic", action="store_true", help="Skip disk lookup, use synthetic images"
    )
    p.add_argument("--split", default="dev")
    p.add_argument("--out", default=None, help="Override output_dir")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--pgd-steps", type=int, default=20, help="Gradient-attack inner steps")
    p.add_argument("--target-tool", default="escalate_to_specialist", help="targeted_tool target")
    p.add_argument("--target-step-k", type=int, default=0, help="targeted_tool step index")
    args = p.parse_args(argv)

    cfg = load_runner_config(args.config)
    attacks_yaml = _load_yaml(args.attacks_config)

    out_dir = Path(args.out) if args.out else cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    tools = default_registry()
    n_records = 0
    t0 = time.time()

    with records_path.open("w", encoding="utf-8") as f_out:
        for model_key in cfg.models:
            print(f"[runner] loading model: {model_key}")
            vlm = load_hf_vlm(model_key)
            agent = MedicalAgent(vlm=vlm, tools=tools)

            for task_id in cfg.tasks:
                t_override = cfg.task_overrides.get(task_id, {})
                n_samples = int(t_override.get("dataset_split", {}).get(args.split, 0)) or None
                samples = list(
                    load_task(
                        task_id,
                        split=args.split,
                        n=n_samples,
                        synthetic=args.synthetic,
                    )
                )
                if not samples:
                    print(f"[runner] WARN no samples for {task_id}/{args.split}")
                    continue

                for sample in samples:
                    for seed in cfg.seeds:
                        benign = agent.run(
                            task_id=task_id,
                            image=sample.image,
                            prompt=sample.prompt,
                            seed=seed,
                            max_steps=args.max_steps,
                        )

                        for attack_name in cfg.attacks:
                            eps_list = resolve_epsilons(cfg, attack_name, attacks_yaml)
                            for eps in eps_list:
                                t_start = time.time()
                                if args.mode in GRADIENT_MODES:
                                    attacked = run_gradient_attack(
                                        mode=args.mode,
                                        vlm=vlm,
                                        agent=agent,
                                        sample=sample,
                                        benign=benign,
                                        epsilon=float(eps),
                                        steps=args.pgd_steps,
                                        seed=seed,
                                        max_steps=args.max_steps,
                                        task_id=task_id,
                                        target_tool=args.target_tool,
                                        target_step_k=args.target_step_k,
                                    )
                                else:
                                    adv_img = perturb(args.mode, sample.image, eps, seed)
                                    attacked = agent.run(
                                        task_id=task_id,
                                        image=adv_img,
                                        prompt=sample.prompt,
                                        seed=seed,
                                        max_steps=args.max_steps,
                                    )
                                ed = trajectory_edit_distance(
                                    benign.tool_sequence(),
                                    attacked.tool_sequence(),
                                    normalize=True,
                                )
                                rec = pair_record(
                                    model_key=model_key,
                                    task_id=task_id,
                                    sample_id=sample.sample_id,
                                    attack_name=attack_name,
                                    attack_mode=args.mode,
                                    epsilon=float(eps),
                                    seed=seed,
                                    benign=benign,
                                    attacked=attacked,
                                    edit_distance=ed,
                                    elapsed_s=time.time() - t_start,
                                )
                                f_out.write(json.dumps(rec, default=str) + "\n")
                                n_records += 1

    elapsed = time.time() - t0
    summary = {
        "experiment": cfg.name,
        "records": n_records,
        "elapsed_s": elapsed,
        "records_path": str(records_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[runner] done. {n_records} records in {elapsed:.1f}s → {records_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
