"""Deterministic risk-score calculators used by the medical agent.

Important: these are *shape-compatible* stubs for benchmark purposes, not
clinically validated implementations. They return stable numeric outputs
suitable for use as tool-argument targets in adversarial studies.
"""

from __future__ import annotations

from .registry import Tool

# PI-RADS v2.1 — Prostate Imaging Reporting & Data System.
# https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/PI-RADS
_PI_RADS_GRADE_CAP: int = 5        # ISUP grade group 1-5
_PI_RADS_PSAD_WEIGHT: float = 20.0  # PSA-density scaling factor
_PI_RADS_GRADE_WEIGHT: float = 3.0  # lesion-grade contribution

# D'Amico risk stratification (JAMA 1998;280:969-974).
# https://doi.org/10.1001/jama.280.11.969
_DAMICO_HIGH_PSA: float = 20.0       # PSA > 20 ng/mL → high risk
_DAMICO_HIGH_GLEASON: int = 8         # Gleason ≥ 8 → high risk
_DAMICO_HIGH_T_STAGE: int = 3         # T-stage ≥ 3 → high risk
_DAMICO_LOW_PSA: float = 10.0        # PSA < 10 ng/mL (strict)
_DAMICO_LOW_GLEASON: int = 6          # Gleason ≤ 6 → low risk
_DAMICO_LOW_T_STAGE: int = 1          # T-stage ≤ 1 → low risk

# Default feature values used when the agent omits optional keys.
_DEFAULT_GLEASON: int = 6
_DEFAULT_T_STAGE: int = 1


def _pi_rads_like(psa: float, volume_cc: float, lesion_grade: int) -> float:
    """Simple combined score: PSA density × lesion-grade factor.

    Returns a float in [0.0, 5.0] where ≥3.0 is flagged as high-suspicion.

    Edge cases (intentional, documented for adversarial-input audits):
      - ``volume_cc <= 0`` raises ``ValueError``. Callers that may receive
        adversarial / corrupted inputs (e.g. perturbed-image agent runs)
        should validate ``volume_cc > 0`` before invoking this tool.
        ``ValueError`` propagates up so it shows up as a tool-call failure
        in the trajectory rather than a silent default.
    """
    if volume_cc <= 0:
        raise ValueError("volume_cc must be positive.")
    psad = psa / volume_cc
    grade_factor = max(0, min(lesion_grade, _PI_RADS_GRADE_CAP)) / float(_PI_RADS_GRADE_CAP)
    return round(min(5.0, psad * _PI_RADS_PSAD_WEIGHT + grade_factor * _PI_RADS_GRADE_WEIGHT), 3)


def _damico_like(psa: float, gleason: int, t_stage: int) -> float:
    """Shape-stable D'Amico-like score ∈ {0.0=low, 0.5=intermediate, 1.0=high}.

    Boundary convention (strict inequalities, intentional):
      - ``psa == 10.0`` → intermediate (low needs ``psa < 10``)
      - ``psa == 20.0`` → intermediate (high needs ``psa > 20``)
      - ``gleason == 7`` → intermediate (low needs ``≤ 6``, high needs ``≥ 8``)
      - ``t_stage == 2`` → intermediate (low needs ``≤ 1``, high needs ``≥ 3``)
    Boundary patients always land in the ``0.5`` (intermediate) bucket.
    Any callers comparing to clinical guidelines must apply their own
    boundary policy on top of this stub.
    """
    high = psa > _DAMICO_HIGH_PSA or gleason >= _DAMICO_HIGH_GLEASON or t_stage >= _DAMICO_HIGH_T_STAGE
    low = psa < _DAMICO_LOW_PSA and gleason <= _DAMICO_LOW_GLEASON and t_stage <= _DAMICO_LOW_T_STAGE
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
            gleason=int(features.get("gleason", _DEFAULT_GLEASON)),
            t_stage=int(features.get("t_stage", _DEFAULT_T_STAGE)),
        )
    raise ValueError(f"Unknown risk score: {name!r}")


def tool() -> Tool:
    return Tool(
        name="calculate_risk_score",
        description=("Compute a named risk score ('pi_rads' or 'damico') from a features dict."),
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
