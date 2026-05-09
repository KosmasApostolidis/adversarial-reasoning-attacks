"""Hero figure package — split from monolithic make_hero_figures.py."""

from .beeswarm import fig_beeswarm
from .bento import fig_bento
from .cot_overlay import fig_cot_overlay
from .heatmap import fig_heatmap, fig_heatmap_drift, fig_heatmap_faith
from .radial import fig_radial
from .ridgeline import fig_ridgeline

__all__ = [
    "fig_beeswarm",
    "fig_bento",
    "fig_cot_overlay",
    "fig_heatmap",
    "fig_heatmap_drift",
    "fig_heatmap_faith",
    "fig_radial",
    "fig_ridgeline",
]
