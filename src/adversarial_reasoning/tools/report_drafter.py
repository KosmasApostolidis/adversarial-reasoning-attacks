"""Minimal report drafter. Concatenates findings into a stable structured string."""

from __future__ import annotations

from .registry import Tool


def _draft(modality: str, findings: list[str]) -> str:
    bullets = "\n".join(f"- {f}" for f in findings)
    return (
        f"[DRAFT REPORT — {modality.upper()}]\n"
        f"Findings:\n{bullets}\n"
        "Assessment: Deterministic stub. Not for clinical use."
    )


def tool() -> Tool:
    return Tool(
        name="draft_report",
        description="Produce a structured-text draft report from a list of findings.",
        parameters_schema={
            "type": "object",
            "properties": {
                "modality": {"type": "string", "description": "e.g. 'MRI', 'CXR'"},
                "findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Bulleted findings observed on the image.",
                },
            },
            "required": ["modality", "findings"],
        },
        handler=_draft,
    )
