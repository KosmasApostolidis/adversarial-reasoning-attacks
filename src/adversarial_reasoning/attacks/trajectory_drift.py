"""Trajectory-Drift PGD — untargeted, custom.

Loss: ``-KL(p_attack ‖ p_benign)`` computed over the tool-name token
positions along the full benign trajectory (not just step 0). Objective:
maximise divergence of the attacked distribution from the benign one, at
the exact token positions where tool selection happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .base import AttackBase, AttackResult


@dataclass
class TrajectoryDriftPGD(AttackBase):
    name: str = "trajectory_drift_pgd"
    epsilon: float = 8.0 / 255.0
    steps: int = 40
    random_restarts: int = 1
    targeted: bool = False
    clip_min: float = 0.0
    clip_max: float = 1.0

    def run(
        self,
        vlm: Any,
        image: torch.Tensor,
        prompt_tokens: torch.Tensor,
        target: Any,
        **_: Any,
    ) -> AttackResult:
        raise NotImplementedError(
            "TrajectoryDriftPGD is a planned implementation. Needs benign "
            "trajectory distribution capture + KL loss hook; see "
            "configs/attacks.yaml."
        )
