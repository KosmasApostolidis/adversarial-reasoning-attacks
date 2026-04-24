"""Deterministic risk-score calculators used by the medical agent.

Important: these are *shape-compatible* stubs for benchmark purposes, not
clinically validated implementations. They return stable numeric outputs
suitable for use as tool-argument targets in adversarial studies.
"""

from __future__ import annotations

from .registry import Tool


def _pi_rads_like(psa: float, volume_cc: float, lesion_grade: int) -> float:
    """Simple combined score: PSA density × lesion-grade factor.

    Returns a float in [0.0, 5.0] where ≥3.0 is flagged as high-suspicion.
    """
    if volume_cc <= 0:
        raise ValueError("volume_cc must be positive.")
    psad = psa / volume_cc
    grade_factor = max(0, min(lesion_grade, 5)) / 5.0
    return round(min(5.0, psad * 20 + grade_factor * 3), 3)


def _damico_like(psa: float, gleason: int, t_stage: int) -> float:
    """Shape-stable D'Amico-like score ∈ {0.0=low, 0.5=intermediate, 1.0=high}."""
    high = psa > 20 or gleason >= 8 or t_stage >= 3
    low = psa < 10 and gleason <= 6 and t_stage <= 1
    if high:
        return 1.0
    if low:
        return 0.0
    return 0.5


def _compute(name: str, features: dict[str, float | int]) -> float:
    name = name.lower()
    if name in {"pi_rads", "pirads", "pi-rads"}:
        return _pi_rads_like(
            psa=float(features["psa"]),
            volume_cc=float(features["volume_cc"]),
            lesion_grade=int(features.get("lesion_grade", 3)),
        )
    if name == "damico":
        return _damico_like(
            psa=float(features["psa"]),
            gleason=int(features.get("gleason", 6)),
            t_stage=int(features.get("t_stage", 1)),
        )
    raise ValueError(f"Unknown risk score: {name!r}")


def tool() -> Tool:
    return Tool(
        name="calculate_risk_score",
        description=(
            "Compute a named risk score ('pi_rads' or 'damico') from a features dict."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "features": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
            },
            "required": ["name", "features"],
        },
        handler=_compute,
    )
