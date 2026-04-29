"""Statistical tests: Wilcoxon + bootstrap + BH FDR."""

from __future__ import annotations

import numpy as np

from adversarial_reasoning.metrics.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    wilcoxon_signed_rank,
)


def test_wilcoxon_detects_consistent_shift():
    rng = np.random.default_rng(0)
    benign = rng.normal(0.0, 1.0, size=50)
    attacked = benign + 0.8  # consistent positive shift
    r = wilcoxon_signed_rank(benign, attacked)
    assert r.pvalue < 0.01


def test_wilcoxon_no_effect_under_null():
    rng = np.random.default_rng(1)
    benign = rng.normal(0.0, 1.0, size=50)
    attacked = rng.normal(0.0, 1.0, size=50)  # independent draw, same dist
    r = wilcoxon_signed_rank(benign, attacked)
    assert r.pvalue > 0.01


def test_bootstrap_ci_converges_to_population_mean():
    rng = np.random.default_rng(2)
    sample = rng.normal(loc=3.3, scale=0.5, size=500)
    r = bootstrap_ci(sample, n_resamples=500, rng_seed=2)
    assert abs(r.mean - 3.3) < 0.1
    assert r.lower < r.mean < r.upper


def test_bh_controls_fdr_under_global_null():
    rng = np.random.default_rng(3)
    pvalues = rng.uniform(0.0, 1.0, size=200)
    rejected = benjamini_hochberg(pvalues, q=0.05)
    # Under the global null the expected number of rejections is very small.
    assert rejected.sum() < 20


def test_bootstrap_ci_rejects_empty_sample():
    import pytest

    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_ci(np.array([]), n_resamples=10)


def test_bootstrap_ci_rejects_nonpositive_resamples():
    import pytest

    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_ci(np.array([1.0, 2.0]), n_resamples=0)


def test_benjamini_hochberg_rejects_q_out_of_range():
    import pytest

    p = np.array([0.01, 0.02, 0.5])
    with pytest.raises(ValueError, match=r"q must be in"):
        benjamini_hochberg(p, q=0.0)
    with pytest.raises(ValueError, match=r"q must be in"):
        benjamini_hochberg(p, q=1.0)
    with pytest.raises(ValueError, match=r"q must be in"):
        benjamini_hochberg(p, q=-0.1)
    with pytest.raises(ValueError, match=r"q must be in"):
        benjamini_hochberg(p, q=1.5)
