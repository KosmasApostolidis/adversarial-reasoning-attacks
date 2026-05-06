"""Statistical tests: Wilcoxon + bootstrap + BH FDR."""

from __future__ import annotations

import numpy as np

from adversarial_reasoning.metrics.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    paired_delta,
    wilcoxon_signed_rank,
)


def _fake_records(
    *,
    stock_model: str = "qwen2_5_vl_7b",
    defended_model: str = "defended_qwen2_5_vl_7b",
    metric_stock: list[float] | None = None,
    metric_defended: list[float] | None = None,
    epsilons: tuple[float, ...] = (0.0314,),
    seeds: tuple[int, ...] = (0,),
    n_samples: int = 5,
) -> list[dict]:
    """Build a synthetic records.jsonl-shaped list with paired model_key entries."""
    if metric_stock is None:
        metric_stock = [0.40] * (n_samples * len(epsilons) * len(seeds))
    if metric_defended is None:
        metric_defended = [0.20] * (n_samples * len(epsilons) * len(seeds))
    out: list[dict] = []
    i = 0
    for s in range(n_samples):
        for eps in epsilons:
            for sd in seeds:
                base = {
                    "task_id": "prostate_mri_workup",
                    "sample_id": f"p{s:02d}",
                    "attack_name": "apgd_linf",
                    "epsilon": eps,
                    "seed": sd,
                }
                out.append(
                    {**base, "model_key": stock_model, "edit_distance_norm": metric_stock[i]}
                )
                out.append(
                    {**base, "model_key": defended_model, "edit_distance_norm": metric_defended[i]}
                )
                i += 1
    return out


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


def test_paired_delta_joins_stock_vs_defended_and_signs_correctly():
    rng = np.random.default_rng(7)
    n = 30
    stock = rng.uniform(0.30, 0.50, size=n).tolist()
    # Defended has consistently lower edit-distance — defense works.
    defended = [s - 0.15 for s in stock]
    records = _fake_records(metric_stock=stock, metric_defended=defended, n_samples=n)
    result = paired_delta(
        records,
        stock_model_key="qwen2_5_vl_7b",
        defended_model_key="defended_qwen2_5_vl_7b",
        bootstrap_resamples=500,
        rng_seed=7,
    )
    assert result.n_pairs == n
    # delta = defended - stock should be uniformly ≈ -0.15.
    assert np.allclose(result.delta, np.array(defended) - np.array(stock))
    assert result.bootstrap.upper < 0.0  # CI excludes zero on the negative side
    assert result.wilcoxon.pvalue < 0.001


def test_paired_delta_under_null_does_not_reject():
    rng = np.random.default_rng(8)
    n = 40
    stock = rng.uniform(0.30, 0.50, size=n).tolist()
    defended = rng.uniform(0.30, 0.50, size=n).tolist()  # independent draw, same dist
    records = _fake_records(metric_stock=stock, metric_defended=defended, n_samples=n)
    result = paired_delta(
        records,
        stock_model_key="qwen2_5_vl_7b",
        defended_model_key="defended_qwen2_5_vl_7b",
        bootstrap_resamples=500,
        rng_seed=8,
    )
    assert result.wilcoxon.pvalue > 0.05


def test_paired_delta_ignores_records_with_other_model_keys():
    records = _fake_records(n_samples=4)
    # Inject noise: a record under a third model_key that should be ignored.
    records.append(
        {
            "task_id": "prostate_mri_workup",
            "sample_id": "p99",
            "attack_name": "apgd_linf",
            "epsilon": 0.0314,
            "seed": 0,
            "model_key": "llava_v1_6_mistral_7b",
            "edit_distance_norm": 0.99,
        }
    )
    result = paired_delta(
        records,
        stock_model_key="qwen2_5_vl_7b",
        defended_model_key="defended_qwen2_5_vl_7b",
        bootstrap_resamples=200,
    )
    assert result.n_pairs == 4


def test_paired_delta_raises_when_no_pairs_match():
    import pytest

    records = _fake_records(n_samples=3)
    # Same defended_model_key as the records, but mismatched stock_model_key
    # → no overlap on join keys.
    with pytest.raises(ValueError, match="no matching"):
        paired_delta(
            records,
            stock_model_key="some_other_stock_model",
            defended_model_key="defended_qwen2_5_vl_7b",
        )


def test_paired_delta_raises_on_duplicate_records():
    import pytest

    records = _fake_records(n_samples=2)
    # Duplicate one stock record.
    dup = dict(records[0])
    records.append(dup)
    with pytest.raises(ValueError, match="duplicate record"):
        paired_delta(
            records,
            stock_model_key="qwen2_5_vl_7b",
            defended_model_key="defended_qwen2_5_vl_7b",
        )


def test_paired_delta_raises_on_missing_metric():
    import pytest

    records = _fake_records(n_samples=2)
    records[0].pop("edit_distance_norm")
    with pytest.raises(KeyError, match="edit_distance_norm"):
        paired_delta(
            records,
            stock_model_key="qwen2_5_vl_7b",
            defended_model_key="defended_qwen2_5_vl_7b",
        )


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
