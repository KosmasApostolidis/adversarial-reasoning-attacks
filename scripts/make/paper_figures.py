"""Paper figures entrypoint — dispatches to scripts/paper_figures/ package.

Outputs (all in paper/figures/paper/):
  fig1_main_result.png      — 3-panel: noise vs PGD boxplot | per-sample bars | ε-sweep
  fig2_trajectories.png     — Gantt-style benign / noise / PGD sequences (3 samples)
  fig3_tool_heatmap.png     — Tool-substitution matrix under PGD
  fig4_cross_model.png      — Qwen vs LLaVA under uniform noise
  fig5_attack_landscape.png — Violin distributions: noise, PGD across models
"""

from __future__ import annotations

from scripts.paper_figures import (
    fig1_main_result,
    fig2_trajectories,
    fig3_tool_heatmap,
    fig4_cross_model,
    fig5_attack_landscape,
)
from scripts.paper_figures._common import OUT


def main() -> int:
    fig1_main_result()
    fig2_trajectories()
    fig3_tool_heatmap()
    fig4_cross_model()
    fig5_attack_landscape()
    print(f"\nAll figures written to {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
