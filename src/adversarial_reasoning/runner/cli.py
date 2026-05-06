"""Runner CLI: argparse, model/task/attack iteration, JSONL writer."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from ..agents.medical_agent import MedicalAgent
from ..metrics.trajectory import trajectory_edit_distance
from ..models.loader import load_hf_vlm
from ..tasks.loader import load_task
from ..tools.registry import default_registry
from .attacks import perturb, run_gradient_attack
from .config import GRADIENT_MODES, _load_yaml, load_runner_config, resolve_epsilons
from .records import pair_record


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
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing records.jsonl (default: abort if exists)",
    )
    args = p.parse_args(argv)

    cfg = load_runner_config(args.config)
    attacks_yaml = _load_yaml(args.attacks_config)

    out_dir = Path(args.out) if args.out else cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    if records_path.exists() and not args.overwrite:
        print(
            f"[runner] ERROR {records_path} exists. Pass --overwrite to replace, "
            f"or use a different --out directory.",
            file=sys.stderr,
        )
        return 1

    tools = default_registry()
    n_records = 0
    n_errors = 0
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
                        try:
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
                        except Exception:
                            n_errors += 1
                            print(
                                f"[runner] ERROR model={model_key} task={task_id} "
                                f"sample={sample.sample_id} seed={seed} — skipping",
                                file=sys.stderr,
                            )
                            traceback.print_exc()

    elapsed = time.time() - t0
    summary = {
        "experiment": cfg.name,
        "records": n_records,
        "errors": n_errors,
        "elapsed_s": elapsed,
        "records_path": str(records_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[runner] done. {n_records} record(s)"
        + (f", {n_errors} error(s)" if n_errors else "")
        + f" in {elapsed:.1f}s → {records_path}"
    )
    return 0 if n_errors == 0 else 2
