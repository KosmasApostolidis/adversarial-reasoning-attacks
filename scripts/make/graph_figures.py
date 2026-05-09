"""Graph figures entrypoint — dispatches to scripts/graph_figures/ package.

Outputs (paper/figures/graphs/):
  graph1_transition_network.png  — directed tool-transition graph: benign vs PGD
  graph2_rewiring.png            — side-by-side agent decision flow before/after attack
  graph3_radial_trajectories.png — polar chart: each patient's tool sequence as radial arcs
  graph4_sankey.png              — Sankey-style flow: what happened to each benign tool
  graph5_similarity_matrix.png   — pairwise trajectory edit-distance heatmap (all conditions)
"""

from __future__ import annotations

from scripts.graph_figures import (
    graph1_transition_network,
    graph2_rewiring,
    graph3_radial_trajectories,
    graph4_sankey,
    graph5_similarity_matrix,
)
from scripts.graph_figures._common import OUT


def main() -> int:
    graph1_transition_network()
    graph2_rewiring()
    graph3_radial_trajectories()
    graph4_sankey()
    graph5_similarity_matrix()
    print(f"\nAll graph figures → {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
