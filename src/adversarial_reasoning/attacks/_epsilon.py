"""Canonical L∞ epsilon values in [0, 1] pixel space (k/255 convention).

Single source of truth for ε constants used across attacks and gates.
Import from here instead of duplicating ``k / 255.0`` expressions.
"""

_LINF_EPSILON_2: float = 2.0 / 255.0
_LINF_EPSILON_4: float = 4.0 / 255.0
_LINF_EPSILON_8: float = 8.0 / 255.0
_LINF_EPSILON_16: float = 16.0 / 255.0
