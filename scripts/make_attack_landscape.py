"""Attack-landscape figures entrypoint — dispatches to scripts/attack_landscape/ package.

Outputs (paper/figures/attack_landscape/):
  fig1_landscape_overview.png   — 2×2 composite (box + bar + ε-curve + flip rate)
  fig2_eps_curves_ci.png        — per-attack ε vs edit-distance with 95% bootstrap CI bands
  fig3_attack_radar.png         — radar chart of attacks across 5 metrics
  fig4_tool_substitution.png    — heatmap matrix of benign→attacked tool flips
  fig5_violin_grid.png          — violin distributions per (attack, ε) cell
"""

from __future__ import annotations

from pathlib import Path

from attack_landscape import (
    fig_eps_curves,
    fig_landscape_overview,
    fig_radar,
    fig_tool_substitution,
    fig_violin_grid,
)
from attack_landscape._common import edits, flip_rate, load_records


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    by_attack = {
        "noise": load_records(root / "runs/main/noise/records.jsonl"),
        "pgd": load_records(root / "runs/main/pgd/records.jsonl"),
        "apgd": load_records(root / "runs/main/apgd/records.jsonl"),
        "targeted_tool": load_records(root / "runs/main/targeted_tool/records.jsonl"),
        "trajectory_drift": load_records(root / "runs/main/trajectory_drift/records.jsonl"),
    }

    out_dir = root / "paper" / "figures" / "attack_landscape"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_landscape_overview(by_attack, out_dir / "fig1_landscape_overview.png")
    fig_eps_curves(by_attack, out_dir / "fig2_eps_curves_ci.png")
    fig_radar(by_attack, out_dir / "fig3_attack_radar.png")
    fig_tool_substitution(by_attack, out_dir / "fig4_tool_substitution.png")
    fig_violin_grid(by_attack, out_dir / "fig5_violin_grid.png")

    print(f"[make_attack_landscape] wrote 5 figures → {out_dir}")
    for name, recs in by_attack.items():
        eds = edits(recs)
        print(f"  {name:18s} n={eds.size:3d}  μ={eds.mean():.3f}  flip={flip_rate(recs):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
