"""Smoke tests for the sandboxed medical tool registry."""

from __future__ import annotations

import pytest

from adversarial_reasoning.tools import default_registry


def test_default_registry_contains_six_tools():
    reg = default_registry()
    assert reg.names() == [
        "query_guidelines",
        "lookup_pubmed",
        "calculate_risk_score",
        "draft_report",
        "request_followup",
        "escalate_to_specialist",
    ]


def test_calculate_risk_score_pi_rads_like():
    reg = default_registry()
    tool = reg.get("calculate_risk_score")
    result = tool.handler(
        name="pi_rads",
        features={"psa": 5.0, "volume_cc": 40.0, "lesion_grade": 4},
    )
    assert isinstance(result, float)
    assert 0.0 <= result <= 5.0


def test_request_followup_rejects_bad_urgency():
    reg = default_registry()
    tool = reg.get("request_followup")
    with pytest.raises(ValueError):
        tool.handler(test_name="psa_recheck", urgency="maybe")


def test_escalate_accepts_known_specialty():
    reg = default_registry()
    tool = reg.get("escalate_to_specialist")
    out = tool.handler(specialty="urology", reason="PI-RADS 5 lesion")
    assert out["tool"] == "escalate_to_specialist"
    assert out["specialty"] == "urology"


def test_tool_schemas_are_json_serialisable():
    import json

    reg = default_registry()
    schemas = reg.schemas()
    # Will raise TypeError if any non-JSON-serialisable content is present.
    json.dumps(schemas)
    assert len(schemas) == 6


def test_registry_subset():
    reg = default_registry()
    sub = reg.subset(["query_guidelines", "draft_report"])
    assert sub.names() == ["query_guidelines", "draft_report"]
