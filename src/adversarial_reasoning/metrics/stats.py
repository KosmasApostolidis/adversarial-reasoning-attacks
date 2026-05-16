"""Statistical testing + multiple-comparison correction.

These helpers wrap ``scipy.stats`` with consistent return shapes
(immutable dataclasses) and add a dependency-free Benjamini-Hochberg
implementation. The figure scripts call them when comparing benign vs
attacked metrics across paired samples.

Public functions
----------------
- :func:`wilcoxon_signed_rank` — paired test on (benign, attacked)
  metric arrays; the per-attack significance test used in the
  comparison figures.
- :func:`bootstrap_ci` — percentile bootstrap CI on mean / median /
  std; used in the ε-sweep error-band plots.
- :func:`benjamini_hochberg` — step-up FDR control across a family of
  p-values (one per (attack, eps) combination).

All returns are :class:`dataclasses.dataclass(frozen=True)` so they can
be hashed, cached, and serialised without surprises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    pvalue: float
    n: int


@dataclass(frozen=True)
class BootstrapCIResult:
    mean: float
    lower: float
    upper: float
    ci_level: float
    n_resamples: int


def wilcoxon_signed_rank(
    benign: np.ndarray,
    attacked: np.ndarray,
    *,
    alternative: str = "two-sided",
) -> WilcoxonResult:
    """Paired Wilcoxon signed-rank. Zero-difference pairs dropped via 'wilcox' method."""
    b = np.asarray(benign, dtype=float)
    a = np.asarray(attacked, dtype=float)
    if b.shape != a.shape:
        raise ValueError(f"Shape mismatch: benign={b.shape}, attacked={a.shape}")
    result = stats.wilcoxon(
        a - b,
        alternative=alternative,
        zero_method="wilcox",
        correction=False,
    )
    return WilcoxonResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        n=int(b.size),
    )


def bootstrap_ci(
    sample: np.ndarray,
    *,
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    statistic: str = "mean",
    rng_seed: int | None = None,
) -> BootstrapCIResult:
    """Percentile bootstrap CI on a univariate statistic."""
    arr = np.asarray(sample, dtype=float)
    if arr.ndim != 1:
        raise ValueError("sample must be 1-D")
    if arr.size == 0:
        raise ValueError("sample must be non-empty")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    rng = np.random.default_rng(rng_seed)
    stat_fn: Any = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std,
    }[statistic]
    n = arr.size
    # Vectorized resampling: one (n_resamples, n) index draw + one axis=1
    # reduction beats the Python loop by ~30× for n_resamples=10_000.
    # Memory cost is n_resamples*n*8 bytes; cap at ~80 MB for n=1000.
    idx = rng.integers(0, n, size=(n_resamples, n))
    resamples = arr[idx]
    stats_resampled = stat_fn(resamples, axis=1)

    alpha = (1.0 - ci_level) / 2.0
    lower, upper = np.quantile(stats_resampled, [alpha, 1.0 - alpha])
    return BootstrapCIResult(
        mean=float(stat_fn(arr)),
        lower=float(lower),
        upper=float(upper),
        ci_level=ci_level,
        n_resamples=n_resamples,
    )


def benjamini_hochberg(
    pvalues: np.ndarray,
    q: float = 0.05,
) -> np.ndarray:
    """BH step-up FDR control. Returns bool array of rejections at level q.

    Matches the behaviour of statsmodels.stats.multitest.multipletests(method='fdr_bh')
    but avoids the dependency for this core function.
    """
    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1)")
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvalues must be 1-D")
    m = p.size
    if m == 0:
        return np.array([], dtype=bool)

    order = np.argsort(p)
    p_sorted = p[order]
    thresholds = (np.arange(1, m + 1) / m) * q
    below = p_sorted <= thresholds
    if not below.any():
        rejected_sorted = np.zeros(m, dtype=bool)
    else:
        k = np.max(np.where(below)[0])
        rejected_sorted = np.zeros(m, dtype=bool)
        rejected_sorted[: k + 1] = True

    rejected = np.empty(m, dtype=bool)
    rejected[order] = rejected_sorted
    return rejected
