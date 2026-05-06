"""Hero figure package — split from monolithic make_hero_figures.py."""

from .beeswarm import fig_beeswarm
from .bento import fig_bento
from .heatmap import fig_heatmap
from .radial import fig_radial
from .ridgeline import fig_ridgeline

__all__ = [
    "fig_beeswarm",
    "fig_bento",
    "fig_heatmap",
    "fig_radial",
    "fig_ridgeline",
]
