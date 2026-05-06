"""Shared theme, palette, helpers, and IO for graph figure modules."""

from __future__ import annotations

import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

import matplotlib as mpl
import networkx as nx

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _plotlib import load_records as _load  # noqa: E402

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#0d1117",
        "text.color": "#e6edf3",
        "axes.labelcolor": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "axes.edgecolor": "#30363d",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#0d1117",
        "figure.dpi": 120,
    }
)

C_BENIGN = "#58a6ff"  # blue
C_NOISE = "#3fb950"  # green
C_PGD = "#f85149"  # red
C_NODE = "#e6edf3"

SHORT = {
    "lookup_pubmed": "PubMed",
    "query_guidelines": "Guidelines",
    "calculate_risk_score": "Risk Score",
    "draft_report": "Draft Report",
    "request_followup": "Followup",
    "escalate_to_specialist": "Escalate",
    "describe_region": "Describe",
}

OUT = Path("paper/figures/graphs")
OUT.mkdir(parents=True, exist_ok=True)


def _short(t: str) -> str:
    return SHORT.get(t, t.replace("_", "\n"))


def _build_transition_graph(records: list[dict], key: str) -> nx.DiGraph:
    G = nx.DiGraph()
    for r in records:
        seq = r[key]["tool_sequence"]
        for t in seq:
            G.add_node(t)
            G.nodes[t]["count"] = G.nodes[t].get("count", 0) + 1
        for a, b in pairwise(seq):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)
    return G


__all__ = [
    "C_BENIGN",
    "C_NODE",
    "C_NOISE",
    "C_PGD",
    "Counter",
    "OUT",
    "SHORT",
    "_build_transition_graph",
    "_load",
    "_short",
    "pairwise",
]
