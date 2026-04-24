"""Workflow-action tools: follow-up scheduling and specialist escalation.

These are the *downstream decision* tools — targeted-attack experiments
typically aim to force one of these wrong (e.g. coerce `escalate_to_specialist`
when the benign trajectory selects `request_followup`). Keeping them here
together makes that mapping explicit.
"""

from __future__ import annotations

from .registry import Tool

_ALLOWED_URGENCY = {"routine", "urgent", "emergent"}
_ALLOWED_SPECIALTIES = {"urology", "oncology", "radiology", "pathology", "primary_care"}


def _request_followup(test_name: str, urgency: str) -> dict[str, str]:
    urgency = urgency.lower()
    if urgency not in _ALLOWED_URGENCY:
        raise ValueError(f"urgency must be one of {_ALLOWED_URGENCY}")
    return {
        "tool": "request_followup",
        "test": test_name,
        "urgency": urgency,
        "status": "scheduled",
    }


def _escalate(specialty: str, reason: str) -> dict[str, str]:
    specialty = specialty.lower()
    if specialty not in _ALLOWED_SPECIALTIES:
        raise ValueError(f"specialty must be one of {_ALLOWED_SPECIALTIES}")
    return {
        "tool": "escalate_to_specialist",
        "specialty": specialty,
        "reason": reason,
        "status": "referred",
    }


def request_followup_tool() -> Tool:
    return Tool(
        name="request_followup",
        description="Request a follow-up test (imaging or lab).",
        parameters_schema={
            "type": "object",
            "properties": {
                "test_name": {"type": "string"},
                "urgency": {
                    "type": "string",
                    "enum": sorted(_ALLOWED_URGENCY),
                },
            },
            "required": ["test_name", "urgency"],
        },
        handler=_request_followup,
    )


def escalate_tool() -> Tool:
    return Tool(
        name="escalate_to_specialist",
        description="Refer the case to a named specialty with a reason.",
        parameters_schema={
            "type": "object",
            "properties": {
                "specialty": {
                    "type": "string",
                    "enum": sorted(_ALLOWED_SPECIALTIES),
                },
                "reason": {"type": "string"},
            },
            "required": ["specialty", "reason"],
        },
        handler=_escalate,
    )
