"""Shared theme constants for figure-generation scripts.

Holds only values that are byte-identical across multiple ``_common.py``
modules. Family-specific palettes (``C_BENIGN`` etc.) intentionally diverge
between figure styles and stay local to each subpackage.

Currently consolidated:
- ``SHORT``: tool short-name dict
  (``graph_figures``, ``reasoning_flow``, ``comprehensive``)
- ``DARK_BG``, ``DARK_AX``, ``DARK_FG``: byte-identical dark-theme constants
  (``reasoning_flow``, ``comprehensive``).
"""

from __future__ import annotations

SHORT: dict[str, str] = {
    "lookup_pubmed": "PubMed",
    "query_guidelines": "Guidelines",
    "calculate_risk_score": "Risk Score",
    "draft_report": "Draft Report",
    "request_followup": "Followup",
    "escalate_to_specialist": "Escalate",
    "describe_region": "Describe",
}

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
DARK_FG = "#e6edf3"

__all__ = ["DARK_AX", "DARK_BG", "DARK_FG", "SHORT"]
