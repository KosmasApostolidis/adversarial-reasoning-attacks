"""Comprehensive figures entrypoint — dispatches to scripts/comprehensive/ package.

Statistical (light background, paper-ready):
  stat1_overview.png            — 4-panel: violin+box, scatter, correlation, grouped bars
  stat2_epsilon_sweep.png       — ε dose-response with per-sample scatter + regression
  stat3_trajectory_lengths.png  — trajectory length distributions + CDF
  stat4_step_heatmap.png        — tool occupancy per step position (benign vs PGD)

Graph / reasoning-flow (dark background, visually striking):
  graph6_bipartite.png          — bipartite benign↔attacked alignment per patient
  graph7_divergence.png         — per-patient divergence tree (where reasoning splits)
  graph8_tool_influence.png     — tool-node influence graph (glow = PGD sensitivity)
  graph9_layered_flow.png       — layered step graph: benign path vs PGD path
  graph10_step_occupancy.png    — heatmap: which tools appear at each step under attack
"""

from __future__ import annotations

from comprehensive import (
    graph6_bipartite,
    graph7_divergence,
    graph8_tool_influence,
    graph9_layered_flow,
    graph10_step_occupancy,
    stat1_overview,
    stat2_epsilon_sweep,
    stat3_trajectory_lengths,
    stat4_step_heatmap,
)
from comprehensive._common import GRAPH_OUT, STAT_OUT


def main() -> int:
    stat1_overview()
    stat2_epsilon_sweep()
    stat3_trajectory_lengths()
    stat4_step_heatmap()
    graph6_bipartite()
    graph7_divergence()
    graph8_tool_influence()
    graph9_layered_flow()
    graph10_step_occupancy()
    print(f"\nStats → {STAT_OUT}/")
    print(f"Graphs → {GRAPH_OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
