"""Smoke tests for trajectory edit distance, flip rate, and hit rate."""

from __future__ import annotations

import numpy as np

from adversarial_reasoning.metrics import (
    benjamini_hochberg,
    bootstrap_ci,
    flip_rate_at_step,
    param_l1_distance,
    targeted_hit_rate,
    trajectory_edit_distance,
    wilcoxon_signed_rank,
)


class TestTrajectoryEditDistance:
    def test_identical_sequences(self):
        seq = ["a", "b", "c"]
        assert trajectory_edit_distance(seq, seq) == 0.0

    def test_completely_different(self):
        assert trajectory_edit_distance(["a"], ["b"], normalize=True) == 1.0

    def test_partial_overlap(self, dummy_trajectory_pair):
        benign, attack = dummy_trajectory_pair
        d = trajectory_edit_distance(benign, attack, normalize=False)
        # benign has 4 tools, attack has 3 with divergence at step 3 -> distance >=2
        assert d >= 2.0

    def test_empty_pair_is_zero(self):
        assert trajectory_edit_distance([], []) == 0.0


class TestFlipRate:
    def test_step_flip_detected(self):
        benign = [["a", "b", "c"], ["a", "b", "d"]]
        attack = [["a", "x", "c"], ["a", "b", "d"]]
        assert flip_rate_at_step(benign, attack, step_k=1) == 0.5

    def test_truncated_trajectory_counts_as_flipped(self):
        benign = [["a", "b", "c"]]
        attack = [["a", "b"]]
        assert flip_rate_at_step(benign, attack, step_k=2) == 1.0

    def test_negative_step_k_raises(self):
        import pytest

        with pytest.raises(ValueError, match="step_k must be >= 0"):
            flip_rate_at_step([["a"]], [["b"]], step_k=-1)


class TestTargetedHitRate:
    def test_hit_anywhere(self):
        batch = [["a", "b"], ["a", "x"], ["a", "b"]]
        assert targeted_hit_rate(batch, "x") == 1 / 3

    def test_hit_at_step(self):
        batch = [["a", "b"], ["a", "x"], ["a", "b"]]
        assert targeted_hit_rate(batch, "x", step_k=1) == 1 / 3

    def test_miss_at_step(self):
        batch = [["a", "x"], ["a", "x"]]
        assert targeted_hit_rate(batch, "x", step_k=0) == 0.0

    def test_negative_step_k_raises(self):
        import pytest

        with pytest.raises(ValueError, match="step_k must be >= 0"):
            targeted_hit_rate([["a"]], "a", step_k=-1)


class TestLevenshteinParity:
    """Verify the Levenshtein C-binding matches the pure-numpy DP path on
    list-of-string inputs (multi-character tokens). Both paths must agree
    so output is deterministic regardless of which import resolved."""

    def test_multi_char_tokens(self):
        from adversarial_reasoning.metrics.trajectory import _levenshtein_dp

        a = ["a", "bb", "c"]
        b = ["a", "cc"]
        # Distance computed via the public function (uses C binding when present).
        public = trajectory_edit_distance(a, b, normalize=False)
        # Pure-numpy DP fallback.
        dp = _levenshtein_dp(a, b)
        assert public == float(dp) == 2.0


class TestParamL1:
    def test_numeric_distance(self):
        assert param_l1_distance({"x": 1.0}, {"x": 2.5}) == 1.5

    def test_non_numeric_ignored_by_default(self):
        assert param_l1_distance({"a": "foo"}, {"a": "bar"}) == 0.0

    def test_non_numeric_counted_when_enabled(self):
        assert param_l1_distance({"a": "foo"}, {"a": "bar"}, numeric_only=False) == 1.0


class TestStats:
    def test_wilcoxon_shape_mismatch_raises(self):
        import pytest

        with pytest.raises(ValueError):
            wilcoxon_signed_rank(np.array([1.0, 2.0]), np.array([1.0]))

    def test_bootstrap_mean_ci_contains_true_mean(self):
        rng = np.random.default_rng(42)
        sample = rng.normal(loc=5.0, scale=1.0, size=200)
        result = bootstrap_ci(sample, n_resamples=1000, rng_seed=42)
        assert result.lower < 5.0 < result.upper

    def test_benjamini_hochberg_known_reference(self):
        # BH at q=0.05 with m=4: thresholds = (i/m)*q = [0.0125, 0.025, 0.0375, 0.05].
        # sorted p = [0.001, 0.008, 0.03, 0.06]; compare to thresholds:
        #   0.001 <= 0.0125 ✓, 0.008 <= 0.025 ✓, 0.03 <= 0.0375 ✓, 0.06 <= 0.05 ✗
        # Largest k passing is 3, so reject first three sorted p-values.
        p = np.array([0.001, 0.008, 0.03, 0.06])
        rejected = benjamini_hochberg(p, q=0.05)
        assert rejected.tolist() == [True, True, True, False]

    def test_benjamini_hochberg_empty(self):
        assert benjamini_hochberg(np.array([]), q=0.05).size == 0
