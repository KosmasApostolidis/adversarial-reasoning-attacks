"""Phase-0 gate: per-model tool-call noise floor at T=0.

Procedure
---------
1. Pick one clean image and one task prompt.
2. Run the agent at T=0 with 5 different seeds.
3. Record the trajectory from each run.
4. Compute pairwise trajectory edit distances among the 5.
5. Report the median. Any attack effect must exceed this median by
   ≥2× to count as real signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from ..agents.base import AgentBase
from ..metrics.trajectory import trajectory_edit_distance


@dataclass
class NoiseFloorResult:
    model_name: str
    seeds: list[int]
    pairwise_distances: list[float] = field(default_factory=list)
    median_distance: float = 0.0
    max_distance: float = 0.0
    threshold_for_signal: float = 0.0   # median × 2.0 by convention

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "seeds": self.seeds,
            "pairwise_distances": self.pairwise_distances,
            "median_distance": self.median_distance,
            "max_distance": self.max_distance,
            "threshold_for_signal": self.threshold_for_signal,
        }


def run_noise_floor(
    agent: AgentBase,
    *,
    task_id: str,
    image: Image.Image,
    prompt: str,
    seeds: list[int] | None = None,
    signal_multiplier: float = 2.0,
) -> NoiseFloorResult:
    """Run the noise-floor gate for one agent (= one VLM + tool registry)."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    trajectories = [
        agent.run(task_id=task_id, image=image, prompt=prompt, seed=s) for s in seeds
    ]

    distances: list[float] = []
    for traj_a, traj_b in combinations(trajectories, 2):
        d = trajectory_edit_distance(
            traj_a.tool_sequence(),
            traj_b.tool_sequence(),
            normalize=True,
        )
        distances.append(d)

    arr = np.asarray(distances, dtype=float) if distances else np.zeros(0, dtype=float)
    median = float(np.median(arr)) if arr.size else 0.0
    max_d = float(np.max(arr)) if arr.size else 0.0
    return NoiseFloorResult(
        model_name=getattr(agent, "vlm", None).__class__.__name__ if hasattr(agent, "vlm") else "unknown",
        seeds=seeds,
        pairwise_distances=[float(x) for x in arr],
        median_distance=median,
        max_distance=max_d,
        threshold_for_signal=signal_multiplier * median,
    )


def write_gate_report(result: NoiseFloorResult, out_path: str | Path) -> None:
    import json

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)


def _cli() -> int:
    import argparse

    from ..agents.medical_agent import MedicalAgent
    from ..models.loader import load_hf_vlm
    from ..tasks.loader import load_task_sample
    from ..tools.registry import default_registry

    p = argparse.ArgumentParser(description="Phase-0 noise-floor gate")
    p.add_argument("--model", required=True, help="Model key from configs/models.yaml")
    p.add_argument("--task", default="prostate_mri_workup")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--signal-multiplier", type=float, default=2.0)
    p.add_argument("--synthetic", action="store_true", help="Use synthetic image")
    p.add_argument("--out", default="runs/gates/noise_floor.json")
    args = p.parse_args()

    vlm = load_hf_vlm(args.model)
    tools = default_registry()
    agent = MedicalAgent(vlm=vlm, tools=tools)

    sample = load_task_sample(args.task, index=0, synthetic=args.synthetic)
    result = run_noise_floor(
        agent,
        task_id=args.task,
        image=sample.image,
        prompt=sample.prompt,
        seeds=args.seeds,
        signal_multiplier=args.signal_multiplier,
    )
    write_gate_report(result, args.out)
    print(
        f"[gate:noise_floor] model={result.model_name} seeds={result.seeds} "
        f"median={result.median_distance:.4f} max={result.max_distance:.4f} "
        f"signal_threshold={result.threshold_for_signal:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
