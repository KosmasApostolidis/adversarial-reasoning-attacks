"""Stubbed PubMed lookup. Offline, deterministic, safe."""

from __future__ import annotations

from .registry import Tool

# Minimal fixture: term-set → static citation records. Sufficient for
# tool-call-sequence experiments; real literature search is out of scope.
_FIXTURE = {
    frozenset({"pi_rads", "biopsy"}): [
        {
            "pmid": "33197547",
            "title": "PI-RADS v2.1 and MRI-targeted biopsy performance",
            "journal": "Radiology",
            "year": 2021,
        },
    ],
    frozenset({"prostate", "active_surveillance"}): [
        {
            "pmid": "29910363",
            "title": "Long-term outcomes of active surveillance in prostate cancer",
            "journal": "J Clin Oncol",
            "year": 2018,
        },
    ],
    frozenset({"prostate", "mri", "quantitative"}): [
        {
            "pmid": "32091315",
            "title": "Quantitative MRI features for prostate cancer detection",
            "journal": "Eur Urol",
            "year": 2020,
        },
    ],
}


def _lookup(terms: list[str]) -> list[dict[str, object]]:
    key = frozenset(t.lower() for t in terms)
    for fixture_key, records in _FIXTURE.items():
        if fixture_key.issubset(key):
            return records
    return []


def tool() -> Tool:
    return Tool(
        name="lookup_pubmed",
        description="Return a small fixture set of PubMed records matching the query terms.",
        parameters_schema={
            "type": "object",
            "properties": {
                "terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords to match against the stub literature index.",
                },
            },
            "required": ["terms"],
        },
        handler=_lookup,
    )
