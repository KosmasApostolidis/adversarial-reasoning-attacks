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
- :func:`paired_delta` — joins stock-vs-defended records on the
  natural keys, returns the per-pair Δ array + Wilcoxon + bootstrap
  CI; used by the T3 defense-delta figures.

All returns are :class:`dataclasses.dataclass(frozen=True)` so they can
be hashed, cached, and serialised without surprises.
"""

from __future__ import annotations

from collections.abc import Iterable
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


@dataclass(frozen=True)
class PairedDeltaResult:
    """Result of joining stock-vs-defended records on natural keys.

    ``delta`` is ``defended - stock`` per pair, so a *negative* delta
    means the defended checkpoint reduced the metric under attack
    (the desired direction for ``edit_distance_norm``).
    """

    stock: np.ndarray
    defended: np.ndarray
    delta: np.ndarray
    n_pairs: int
    wilcoxon: WilcoxonResult
    bootstrap: BootstrapCIResult


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
    stats_resampled = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats_resampled[i] = stat_fn(arr[idx])

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


_DEFAULT_PAIRED_JOIN_KEYS: tuple[str, ...] = (
    "task_id",
    "sample_id",
    "attack_name",
    "epsilon",
    "seed",
)


def paired_delta(
    records: Iterable[dict[str, Any]],
    *,
    stock_model_key: str,
    defended_model_key: str,
    metric: str = "edit_distance_norm",
    join_keys: tuple[str, ...] = _DEFAULT_PAIRED_JOIN_KEYS,
    alternative: str = "two-sided",
    bootstrap_resamples: int = 10000,
    ci_level: float = 0.95,
    rng_seed: int | None = None,
) -> PairedDeltaResult:
    """Join stock-vs-defended records on ``join_keys`` and run a paired test on ``metric``.

    The runner emits records with ``model_key`` plus the join keys; this
    function partitions them into stock / defended buckets, joins on the
    join keys (so each pair shares an attacked image cell), and returns
    the Δ array (``defended - stock``) along with a Wilcoxon signed-rank
    test and a bootstrap CI on the mean Δ.

    Raises
    ------
    ValueError
        If no matching pairs are found, or if either bucket contains
        duplicate records under the same join-key tuple, or if some
        join-key entry contains an unhashable value.
    KeyError
        If a record is missing ``metric``, ``model_key``, or any of
        ``join_keys``.
    """
    stock_index: dict[tuple, float] = {}
    defended_index: dict[tuple, float] = {}
    for rec in records:
        mk = rec.get("model_key")
        if mk not in (stock_model_key, defended_model_key):
            continue
        try:
            key = tuple(rec[k] for k in join_keys)
            value = float(rec[metric])
        except KeyError as exc:
            missing = exc.args[0] if exc.args else "unknown"
            raise KeyError(
                f"record missing required field {missing!r}; needs "
                f"{metric!r} and join keys {list(join_keys)!r}"
            ) from exc
        bucket = stock_index if mk == stock_model_key else defended_index
        if key in bucket:
            raise ValueError(
                f"duplicate record for model_key={mk!r} at join keys "
                f"{dict(zip(join_keys, key, strict=True))!r}"
            )
        bucket[key] = value

    matched_keys = sorted(set(stock_index) & set(defended_index))
    if not matched_keys:
        raise ValueError(
            f"no matching (stock={stock_model_key!r}, "
            f"defended={defended_model_key!r}) pairs across join keys "
            f"{list(join_keys)!r}"
        )
    stock = np.array([stock_index[k] for k in matched_keys], dtype=float)
    defended = np.array([defended_index[k] for k in matched_keys], dtype=float)
    delta = defended - stock

    wilcoxon = wilcoxon_signed_rank(stock, defended, alternative=alternative)
    bootstrap = bootstrap_ci(
        delta,
        n_resamples=bootstrap_resamples,
        ci_level=ci_level,
        statistic="mean",
        rng_seed=rng_seed,
    )
    return PairedDeltaResult(
        stock=stock,
        defended=defended,
        delta=delta,
        n_pairs=delta.size,
        wilcoxon=wilcoxon,
        bootstrap=bootstrap,
    )
