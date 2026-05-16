"""Runner CLI: argparse, model/task/attack iteration, JSONL writer."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from itertools import product
from pathlib import Path
from typing import IO, Any

try:
    import torch as _torch  # local alias so module import is cheap

    _FATAL_EXC: tuple[type[BaseException], ...] = (
        _torch.cuda.OutOfMemoryError,  # type: ignore[attr-defined]
        MemoryError,
    )
except (ImportError, AttributeError):
    _FATAL_EXC = (MemoryError,)

from ..agents.medical_agent import MedicalAgent
from ..metrics.trajectory import trajectory_edit_distance
from ..models.loader import load_hf_vlm
from ..tasks.loader import TaskSample, load_task
from ..tools.registry import default_registry
from .attacks import perturb, run_gradient_attack
from .config import (
    GRADIENT_MODES,
    RunnerConfig,
    _load_yaml,
    load_runner_config,
    resolve_epsilons,
)
from .records import pair_record


def _build_arg_parser() -> argparse.ArgumentParser:
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
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config, print RunnerConfig, exit before any model load.",
    )
    return p


def _resolve_records_path(args: argparse.Namespace, cfg: RunnerConfig) -> Path | None:
    """Resolve output dir + records.jsonl. Returns None on overwrite conflict."""
    out_dir = Path(args.out) if args.out else cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    if records_path.exists() and not args.overwrite:
        print(
            f"[runner] ERROR {records_path} exists. Pass --overwrite to replace, "
            f"or use a different --out directory.",
            file=sys.stderr,
        )
        return None
    return records_path


def _load_samples(cfg: RunnerConfig, task_id: str, args: argparse.Namespace) -> list[TaskSample]:
    """Load task samples honouring per-task dataset_split overrides."""
    t_override = cfg.task_overrides.get(task_id, {})
    n_samples = int(t_override.get("dataset_split", {}).get(args.split, 0)) or None
    return list(load_task(task_id, split=args.split, n=n_samples, synthetic=args.synthetic))


def _invoke_attack(
    *,
    mode: str,
    vlm: Any,
    agent: MedicalAgent,
    sample: TaskSample,
    benign: Any,
    epsilon: float,
    seed: int,
    args: argparse.Namespace,
    task_id: str,
) -> Any:
    """Dispatch by mode → gradient attack or noise perturbation. Returns Trajectory."""
    if mode in GRADIENT_MODES:
        return run_gradient_attack(
            mode=mode,
            vlm=vlm,
            agent=agent,
            sample=sample,
            benign=benign,
            epsilon=epsilon,
            steps=args.pgd_steps,
            seed=seed,
            max_steps=args.max_steps,
            task_id=task_id,
            target_tool=args.target_tool,
            target_step_k=args.target_step_k,
        )
    adv_img = perturb(mode, sample.image, epsilon, seed)
    return agent.run(
        task_id=task_id,
        image=adv_img,
        prompt=sample.prompt,
        seed=seed,
        max_steps=args.max_steps,
    )


def _run_one_attack(
    *,
    mode: str,
    vlm: Any,
    agent: MedicalAgent,
    sample: TaskSample,
    benign: Any,
    attack_name: str,
    epsilon: float,
    seed: int,
    args: argparse.Namespace,
    model_key: str,
    task_id: str,
) -> dict[str, Any]:
    """Run one (attack_name, eps) probe against ``sample`` and return its record dict."""
    t_start = time.time()
    attacked = _invoke_attack(
        mode=mode,
        vlm=vlm,
        agent=agent,
        sample=sample,
        benign=benign,
        epsilon=epsilon,
        seed=seed,
        args=args,
        task_id=task_id,
    )
    ed = trajectory_edit_distance(benign.tool_sequence(), attacked.tool_sequence(), normalize=True)
    return pair_record(
        model_key=model_key,
        task_id=task_id,
        sample_id=sample.sample_id,
        attack_name=attack_name,
        attack_mode=mode,
        epsilon=epsilon,
        seed=seed,
        benign=benign,
        attacked=attacked,
        edit_distance=ed,
        elapsed_s=time.time() - t_start,
    )


def _process_sample(
    *,
    vlm: Any,
    agent: MedicalAgent,
    args: argparse.Namespace,
    cfg: RunnerConfig,
    attacks_yaml: dict[str, Any],
    model_key: str,
    task_id: str,
    sample: TaskSample,
    seed: int,
    f_out: IO[str],
) -> int:
    """Run benign + every (attack, eps) for one sample/seed. Returns records written."""
    benign = agent.run(
        task_id=task_id,
        image=sample.image,
        prompt=sample.prompt,
        seed=seed,
        max_steps=args.max_steps,
    )
    n_written = 0
    for attack_name in cfg.attacks:
        for eps in resolve_epsilons(cfg, attack_name, attacks_yaml):
            rec = _run_one_attack(
                mode=args.mode,
                vlm=vlm,
                agent=agent,
                sample=sample,
                benign=benign,
                attack_name=attack_name,
                epsilon=float(eps),
                seed=seed,
                args=args,
                model_key=model_key,
                task_id=task_id,
            )
            f_out.write(json.dumps(rec, default=str) + "\n")
            # Flush + fsync per record so an OS-kill (OOM, power loss) cannot
            # truncate the trailing samples of a long sweep. Page-cache loss
            # otherwise loses minutes of GPU compute silently.
            f_out.flush()
            try:
                os.fsync(f_out.fileno())
            except OSError:
                # Some filesystems (procfs, tmpfs subdirs) refuse fsync. The
                # flush above still gets the line into kernel buffers; we'd
                # rather continue the sweep than abort.
                pass
            n_written += 1
    return n_written


def _log_sample_error(
    exc: Exception, *, model_key: str, task_id: str, sample_id: str, seed: int
) -> None:
    """Print structured ERROR line + traceback for one failed sample/seed."""
    print(
        f"[runner] ERROR {type(exc).__name__} model={model_key} "
        f"task={task_id} sample={sample_id} seed={seed} — skipping",
        file=sys.stderr,
    )
    traceback.print_exc()


def _iterate_records(
    *,
    cfg: RunnerConfig,
    args: argparse.Namespace,
    attacks_yaml: dict[str, Any],
    tools: Any,
    f_out: IO[str],
) -> tuple[int, int]:
    """Sweep models × tasks × samples × seeds. Returns (n_records, n_errors)."""
    import gc

    try:
        import torch as _torch_for_cleanup
    except ImportError:  # pragma: no cover — torch is a hard dep
        _torch_for_cleanup = None  # type: ignore[assignment]

    n_records = 0
    n_errors = 0
    for model_key in cfg.models:
        print(f"[runner] loading model: {model_key}")
        vlm = load_hf_vlm(model_key)
        agent = MedicalAgent(vlm=vlm, tools=tools)

        for task_id in cfg.tasks:
            samples = _load_samples(cfg, task_id, args)
            if not samples:
                print(f"[runner] WARN no samples for {task_id}/{args.split}")
                continue

            for sample, seed in product(samples, cfg.seeds):
                try:
                    n_records += _process_sample(
                        vlm=vlm,
                        agent=agent,
                        args=args,
                        cfg=cfg,
                        attacks_yaml=attacks_yaml,
                        model_key=model_key,
                        task_id=task_id,
                        sample=sample,
                        seed=seed,
                        f_out=f_out,
                    )
                except _FATAL_EXC as exc:  # noqa: F821 — defined at module top
                    # CUDA OOM / device-side asserts leave the allocator in an
                    # unrecoverable state; continuing would silently produce
                    # bogus tensors. Abort the sweep — caller can resume from
                    # the JSONL records already on disk.
                    print(
                        f"[runner] FATAL {type(exc).__name__} — aborting "
                        f"sweep at model={model_key} task={task_id} "
                        f"sample={sample.sample_id}",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise
                except Exception as exc:
                    n_errors += 1
                    _log_sample_error(
                        exc,
                        model_key=model_key,
                        task_id=task_id,
                        sample_id=sample.sample_id,
                        seed=seed,
                    )
        # Free VRAM before loading the next surrogate. Without this a 4-model
        # sweep on a 48 GB A6000 hits the allocator ceiling around model 3.
        del agent
        del vlm
        gc.collect()
        if _torch_for_cleanup is not None and _torch_for_cleanup.cuda.is_available():
            _torch_for_cleanup.cuda.empty_cache()
    return n_records, n_errors


def _write_summary(
    out_dir: Path,
    cfg: RunnerConfig,
    n_records: int,
    n_errors: int,
    t0: float,
    records_path: Path,
) -> None:
    """Write summary.json and print final status line."""
    elapsed = time.time() - t0
    summary = {
        "experiment": cfg.name,
        "records": n_records,
        "errors": n_errors,
        "elapsed_s": elapsed,
        "records_path": str(records_path),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[runner] done. {n_records} record(s)"
        + (f", {n_errors} error(s)" if n_errors else "")
        + f" in {elapsed:.1f}s → {records_path}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg = load_runner_config(args.config)
    if args.dry_run:
        print(repr(cfg))
        return 0

    records_path = _resolve_records_path(args, cfg)
    if records_path is None:
        return 1

    attacks_yaml = _load_yaml(args.attacks_config)
    tools = default_registry()
    t0 = time.time()

    with records_path.open("w", encoding="utf-8") as f_out:
        n_records, n_errors = _iterate_records(
            cfg=cfg,
            args=args,
            attacks_yaml=attacks_yaml,
            tools=tools,
            f_out=f_out,
        )

    _write_summary(records_path.parent, cfg, n_records, n_errors, t0, records_path)
    return 0 if n_errors == 0 else 2
