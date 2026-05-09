"""Hero figures entrypoint — dispatches to scripts/hero/ package.

See `scripts/hero/` for figure implementations:
  fig1_beeswarm.png         — full-bleed beeswarm with stat-card inset
  fig2_ridgeline.png        — joy-plot of edit-distance distributions per attack
  fig3_heatmap_attack_eps.png — attack × ε heatmap with annotated cells
  fig4_radial_profile.png   — circular radial bars per attack across 5 metrics
  fig5_bento.png            — 6-panel magazine composite (hero stat cards + charts)
"""

from __future__ import annotations

from pathlib import Path

from hero import (
    fig_beeswarm,
    fig_bento,
    fig_cot_overlay,
    fig_heatmap,
    fig_heatmap_drift,
    fig_heatmap_faith,
    fig_radial,
    fig_ridgeline,
)
from hero._common import ATTACK_ORDER, edits, gather, has_cot, step1_flip_rate


def main() -> int:
    by_attack = gather()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "paper" / "figures" / "hero"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_beeswarm(by_attack, out_dir / "fig1_beeswarm.png")
    fig_ridgeline(by_attack, out_dir / "fig2_ridgeline.png")
    fig_heatmap(by_attack, out_dir / "fig3_heatmap_attack_eps.png")
    fig_radial(by_attack, out_dir / "fig4_radial_profile.png")
    fig_bento(by_attack, out_dir / "fig5_bento.png")

    n_extra = 0
    if has_cot(by_attack):
        fig_cot_overlay(by_attack, out_dir / "fig6_cot_overlay.png")
        fig_heatmap_drift(by_attack, out_dir / "fig3b_heatmap_drift.png")
        fig_heatmap_faith(by_attack, out_dir / "fig3c_heatmap_faith.png")
        n_extra = 3

    print(f"[make_hero_figures] wrote {5 + n_extra} figures → {out_dir}")
    for name in ATTACK_ORDER:
        eds = edits(by_attack[name])
        print(
            f"  {name:18s} n={eds.size:3d}  μ={eds.mean():.3f}  flip={step1_flip_rate(by_attack[name]):.0%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
