"""Stubbed clinical guideline database. Deterministic, offline, safe."""

from __future__ import annotations

from .registry import Tool

_GUIDELINES = {
    ("prostate_cancer", "pi_rads_3"): (
        "For PI-RADS 3 lesions, AUA/NCCN suggest MRI-targeted biopsy in men with "
        "elevated PSA or abnormal DRE. Consider PSAD ≥0.15 ng/mL/cc as supportive."
    ),
    ("prostate_cancer", "pi_rads_4"): (
        "PI-RADS 4 lesions have a high likelihood of clinically significant prostate "
        "cancer. NCCN recommends targeted biopsy with systematic sampling."
    ),
    ("prostate_cancer", "pi_rads_5"): (
        "PI-RADS 5 lesions carry the highest likelihood of csPCa. Immediate targeted "
        "biopsy plus systematic sampling is recommended."
    ),
    ("prostate_cancer", "followup_pi_rads_2"): (
        "PI-RADS 1-2 lesions: low suspicion for csPCa. Consider clinical surveillance "
        "with repeat PSA and MRI at 12-24 months unless risk factors dictate sooner."
    ),
}


def _query(condition: str, query: str) -> str:
    key = (condition.lower(), query.lower())
    if key in _GUIDELINES:
        return _GUIDELINES[key]
    return (
        f"No guideline text found for condition={condition!r} query={query!r}. "
        "Consider consulting the live guideline source."
    )


def tool() -> Tool:
    return Tool(
        name="query_guidelines",
        description="Return clinical guideline text for a (condition, query) pair.",
        parameters_schema={
            "type": "object",
            "properties": {
                "condition": {"type": "string", "description": "e.g. 'prostate_cancer'"},
                "query": {"type": "string", "description": "e.g. 'pi_rads_4'"},
            },
            "required": ["condition", "query"],
        },
        handler=_query,
    )
