"""Primary trajectory-divergence metrics for untargeted + targeted attacks.

This module is the single source of truth for the *outcome-side* metrics
in :mod:`adversarial_reasoning.runner` and the figure scripts under
``scripts/``. None of these helpers care how a perturbation was produced
— they operate purely on the resulting tool-call sequences.

Public functions
----------------
- :func:`trajectory_edit_distance` — normalised Levenshtein on tool-name
  lists; the headline untargeted-attack metric used in
  ``records.jsonl`` rows (``edit_distance_norm`` field).
- :func:`flip_rate_at_step` — pointwise mismatch at step ``k``; useful
  for evaluating forced-step targeted attacks.
- :func:`targeted_hit_rate` — global or step-locked hit-rate; the
  headline targeted-attack metric used in the cross-model figures.
- :func:`param_l1_distance` — perturbation budget on tool *arguments*,
  not just tool names; complements edit-distance when an attack flips
  args without flipping the tool itself.

The optional ``python-Levenshtein`` import accelerates pairwise edit
distance ~10× on long sequences; the pure-numpy fallback in
:func:`_levenshtein_dp` matches behaviour exactly so output is
deterministic across both code paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

try:
    import Levenshtein  # python-Levenshtein binding
except ImportError:  # pragma: no cover
    Levenshtein = None  # type: ignore[assignment]


def trajectory_edit_distance(
    seq_a: Sequence[str],
    seq_b: Sequence[str],
    *,
    normalize: bool = True,
) -> float:
    """Levenshtein distance between two tool-name sequences.

    Args:
        seq_a, seq_b: ordered tool-name lists
        normalize: if True, divide by max(len(seq_a), len(seq_b)). Returns 0
            when both sequences are empty.
    """
    if not seq_a and not seq_b:
        return 0.0
    if Levenshtein is not None:
        distance = Levenshtein.distance(list(seq_a), list(seq_b))
    else:
        distance = _levenshtein_dp(seq_a, seq_b)
    if not normalize:
        return float(distance)
    denom = max(len(seq_a), len(seq_b))
    return float(distance) / denom if denom else 0.0


def flip_rate_at_step(
    benign_batch: Sequence[Sequence[str]],
    attack_batch: Sequence[Sequence[str]],
    step_k: int,
) -> float:
    """Fraction of trajectory pairs whose tool selection at step k differs.

    Pairs with either trajectory shorter than step_k+1 contribute to the
    'different' count — a truncated trajectory is definitionally not the
    same as one that calls a tool at step k. Two trajectories that are
    both shorter than ``step_k+1`` resolve to ``None == None`` and do not
    flip.
    """
    if step_k < 0:
        raise ValueError("step_k must be >= 0")
    if len(benign_batch) != len(attack_batch):
        raise ValueError("benign_batch and attack_batch must have the same length.")
    if not benign_batch:
        return 0.0
    flipped = 0
    for b, a in zip(benign_batch, attack_batch, strict=True):
        b_k = b[step_k] if len(b) > step_k else None
        a_k = a[step_k] if len(a) > step_k else None
        if b_k != a_k:
            flipped += 1
    return flipped / len(benign_batch)


def targeted_hit_rate(
    attack_batch: Sequence[Sequence[str]],
    target_tool: str,
    *,
    step_k: int | None = None,
) -> float:
    """P(target_tool appears in attack trajectory).

    If `step_k` is given, require the target tool at exactly that step;
    otherwise count any appearance anywhere in the trajectory.
    """
    if step_k is not None and step_k < 0:
        raise ValueError("step_k must be >= 0")
    if not attack_batch:
        return 0.0
    hits = 0
    for seq in attack_batch:
        if step_k is None:
            hits += int(target_tool in seq)
        else:
            hits += int(len(seq) > step_k and seq[step_k] == target_tool)
    return hits / len(attack_batch)


def param_l1_distance(
    benign_args: dict[str, Any],
    attack_args: dict[str, Any],
    *,
    numeric_only: bool = True,
) -> float:
    """L1 distance on keys numeric in both inputs.

    Iterates the union of keys but only contributes a numeric L1 term when
    a key is present and numeric (int/float) in *both* dicts. Non-numeric
    values are compared for equality when ``numeric_only=False`` (equality
    contributes 0, inequality contributes 1).
    """
    total = 0.0
    keys = set(benign_args) | set(attack_args)
    for key in keys:
        b = benign_args.get(key)
        a = attack_args.get(key)
        if isinstance(b, int | float) and isinstance(a, int | float):
            total += abs(float(a) - float(b))
        elif not numeric_only:
            total += 0.0 if a == b else 1.0
    return total


def _levenshtein_dp(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    """Fallback DP implementation if python-Levenshtein is unavailable."""
    m, n = len(seq_a), len(seq_b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = np.zeros((m + 1, n + 1), dtype=np.int64)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[m, n])
