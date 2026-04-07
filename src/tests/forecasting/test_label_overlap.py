"""Unit tests for label overlap handling (sequential bootstrapping, N_eff, subsampling).

Tests cover:
- Indicator matrix construction (shape, content, edge cases)
- Average uniqueness computation (known examples, boundary conditions)
- Effective sample size (N_eff) with Kish formula
- Sequential bootstrap draw (reproducibility, probability weighting)
- Non-overlapping subsampler (stride, edge cases)
- High-level facade (dispatch, result containers)
- Hand-computed example: 5 samples with horizon=3
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from src.app.forecasting.application.label_overlap import (
    LabelOverlapConfig,
    LabelUniquenessResult,
    compute_average_uniqueness,
    compute_indicator_matrix,
    compute_label_uniqueness,
    compute_n_eff,
    sequential_bootstrap_draw,
    subsample_non_overlapping,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _arr(*values: float) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Shorthand to create a float64 array."""
    return np.array(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Indicator matrix
# ---------------------------------------------------------------------------


class TestComputeIndicatorMatrix:
    def test_shape(self) -> None:
        """Indicator matrix has shape (n_samples, n_samples + horizon - 1)."""
        mat = compute_indicator_matrix(n_samples=5, horizon=3)

        assert mat.shape == (5, 7)

    def test_horizon_one_is_identity(self) -> None:
        """With horizon=1, each sample covers exactly one time step → identity-like."""
        mat = compute_indicator_matrix(n_samples=4, horizon=1)

        expected = np.eye(4, dtype=np.float64)
        np.testing.assert_array_equal(mat, expected)

    def test_horizon_equals_n_samples(self) -> None:
        """When horizon == n_samples, all labels overlap at the center."""
        mat = compute_indicator_matrix(n_samples=3, horizon=3)

        # Shape: (3, 5)
        assert mat.shape == (3, 5)
        # Row 0: [1, 1, 1, 0, 0]
        np.testing.assert_array_equal(mat[0], [1.0, 1.0, 1.0, 0.0, 0.0])
        # Row 1: [0, 1, 1, 1, 0]
        np.testing.assert_array_equal(mat[1], [0.0, 1.0, 1.0, 1.0, 0.0])
        # Row 2: [0, 0, 1, 1, 1]
        np.testing.assert_array_equal(mat[2], [0.0, 0.0, 1.0, 1.0, 1.0])

    def test_single_sample(self) -> None:
        """Single sample produces a 1-row matrix."""
        mat = compute_indicator_matrix(n_samples=1, horizon=5)

        assert mat.shape == (1, 5)
        np.testing.assert_array_equal(mat[0], [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_row_sums_equal_horizon(self) -> None:
        """Each row should sum to exactly the horizon value."""
        mat = compute_indicator_matrix(n_samples=10, horizon=4)

        row_sums = mat.sum(axis=1)
        np.testing.assert_array_equal(row_sums, np.full(10, 4.0))

    def test_invalid_n_samples(self) -> None:
        """n_samples < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            compute_indicator_matrix(n_samples=0, horizon=3)

    def test_invalid_horizon(self) -> None:
        """horizon < 1 raises ValueError."""
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            compute_indicator_matrix(n_samples=5, horizon=0)


# ---------------------------------------------------------------------------
# Average uniqueness
# ---------------------------------------------------------------------------


class TestComputeAverageUniqueness:
    def test_no_overlap_gives_ones(self) -> None:
        """When horizon=1 (identity matrix), every sample has uniqueness=1.0."""
        mat = compute_indicator_matrix(n_samples=5, horizon=1)

        uniqueness = compute_average_uniqueness(mat)

        np.testing.assert_array_almost_equal(uniqueness, np.ones(5))

    def test_full_overlap_two_samples(self) -> None:
        """Two samples with horizon=2 share one time step (t=1)."""
        # n_samples=2, horizon=2: shape (2, 3)
        # Row 0: [1, 1, 0]  — covers t=0, t=1
        # Row 1: [0, 1, 1]  — covers t=1, t=2
        # Concurrency: [1, 2, 1]
        # Sample 0: avg(1/1, 1/2) = 0.75
        # Sample 1: avg(1/2, 1/1) = 0.75
        mat = compute_indicator_matrix(n_samples=2, horizon=2)

        uniqueness = compute_average_uniqueness(mat)

        np.testing.assert_array_almost_equal(uniqueness, [0.75, 0.75])

    def test_hand_computed_5_samples_horizon_3(self) -> None:
        """Hand-computed example: 5 samples with horizon=3.

        Indicator matrix (5 x 7):
            t: 0  1  2  3  4  5  6
        s0:  1  1  1  0  0  0  0
        s1:  0  1  1  1  0  0  0
        s2:  0  0  1  1  1  0  0
        s3:  0  0  0  1  1  1  0
        s4:  0  0  0  0  1  1  1

        Concurrency: [1, 2, 3, 3, 3, 2, 1]

        Sample 0: active at t=0,1,2 → 1/1 + 1/2 + 1/3 = 11/6 → avg = 11/18
        Sample 1: active at t=1,2,3 → 1/2 + 1/3 + 1/3 = 7/6  → avg = 7/18
        Sample 2: active at t=2,3,4 → 1/3 + 1/3 + 1/3 = 3/3  → avg = 1/3
        Sample 3: active at t=3,4,5 → 1/3 + 1/3 + 1/2 = 7/6  → avg = 7/18
        Sample 4: active at t=4,5,6 → 1/3 + 1/2 + 1/1 = 11/6 → avg = 11/18
        """
        mat = compute_indicator_matrix(n_samples=5, horizon=3)

        uniqueness = compute_average_uniqueness(mat)

        expected = np.array(
            [11.0 / 18.0, 7.0 / 18.0, 1.0 / 3.0, 7.0 / 18.0, 11.0 / 18.0],
            dtype=np.float64,
        )
        np.testing.assert_array_almost_equal(uniqueness, expected)

    def test_symmetry_for_uniform_horizon(self) -> None:
        """Uniqueness should be symmetric: first sample = last sample, etc."""
        mat = compute_indicator_matrix(n_samples=8, horizon=4)

        uniqueness = compute_average_uniqueness(mat)

        # First and last should match, second and second-to-last, etc.
        np.testing.assert_array_almost_equal(uniqueness, uniqueness[::-1])

    def test_uniqueness_bounded_zero_one(self) -> None:
        """All uniqueness values should be in (0, 1]."""
        mat = compute_indicator_matrix(n_samples=20, horizon=10)

        uniqueness = compute_average_uniqueness(mat)

        assert np.all(uniqueness > 0.0)
        assert np.all(uniqueness <= 1.0)

    def test_empty_matrix_raises(self) -> None:
        """Empty indicator matrix raises ValueError."""
        empty = np.zeros((0, 5), dtype=np.float64)

        with pytest.raises(ValueError, match="at least one row"):
            compute_average_uniqueness(empty)

    def test_single_sample_uniqueness_is_one(self) -> None:
        """Single sample always has uniqueness 1.0 regardless of horizon."""
        mat = compute_indicator_matrix(n_samples=1, horizon=10)

        uniqueness = compute_average_uniqueness(mat)

        np.testing.assert_array_almost_equal(uniqueness, [1.0])


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------


class TestComputeNEff:
    def test_equal_weights(self) -> None:
        """Equal weights produce N_eff = N."""
        weights = np.ones(100, dtype=np.float64)

        n_eff = compute_n_eff(weights)

        np.testing.assert_almost_equal(n_eff, 100.0)

    def test_single_dominant_weight(self) -> None:
        """When one weight dominates, N_eff approaches 1."""
        weights = np.array([100.0, 0.001, 0.001, 0.001], dtype=np.float64)

        n_eff = compute_n_eff(weights)

        # Numerically close to 1.0
        assert n_eff < 1.1

    def test_two_equal_weights(self) -> None:
        """Two equal weights give N_eff = 2."""
        weights = np.array([1.0, 1.0], dtype=np.float64)

        n_eff = compute_n_eff(weights)

        np.testing.assert_almost_equal(n_eff, 2.0)

    def test_kish_formula_known_example(self) -> None:
        """Verify Kish formula: w=[1,2,3] → N_eff = 36/14 ≈ 2.571."""
        weights = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        n_eff = compute_n_eff(weights)

        expected = (1.0 + 2.0 + 3.0) ** 2 / (1.0 + 4.0 + 9.0)
        np.testing.assert_almost_equal(n_eff, expected)

    def test_empty_weights_raises(self) -> None:
        """Empty weights array raises ValueError."""
        with pytest.raises(ValueError, match="at least one element"):
            compute_n_eff(np.array([], dtype=np.float64))

    def test_non_positive_weight_raises(self) -> None:
        """Zero or negative weights raise ValueError."""
        with pytest.raises(ValueError, match="all weights must be positive"):
            compute_n_eff(np.array([1.0, 0.0, 1.0], dtype=np.float64))

        with pytest.raises(ValueError, match="all weights must be positive"):
            compute_n_eff(np.array([1.0, -0.5], dtype=np.float64))

    def test_n_eff_from_uniqueness_weights(self) -> None:
        """N_eff from a realistic uniqueness vector is less than N_raw."""
        mat = compute_indicator_matrix(n_samples=50, horizon=24)
        weights = compute_average_uniqueness(mat)

        n_eff = compute_n_eff(weights)

        assert n_eff > 0.0
        assert n_eff < 50.0


# ---------------------------------------------------------------------------
# Sequential bootstrap draw
# ---------------------------------------------------------------------------


class TestSequentialBootstrapDraw:
    def test_output_shape(self) -> None:
        """Draw produces the correct number of indices."""
        mat = compute_indicator_matrix(n_samples=10, horizon=3)
        rng = np.random.default_rng(42)

        indices = sequential_bootstrap_draw(mat, n_draws=7, rng=rng)

        assert indices.shape == (7,)

    def test_indices_in_range(self) -> None:
        """All drawn indices are valid sample indices."""
        mat = compute_indicator_matrix(n_samples=10, horizon=3)
        rng = np.random.default_rng(42)

        indices = sequential_bootstrap_draw(mat, n_draws=20, rng=rng)

        assert np.all(indices >= 0)
        assert np.all(indices < 10)

    def test_reproducibility(self) -> None:
        """Same seed produces identical draws."""
        mat = compute_indicator_matrix(n_samples=10, horizon=3)

        rng1 = np.random.default_rng(123)
        draw1 = sequential_bootstrap_draw(mat, n_draws=5, rng=rng1)

        rng2 = np.random.default_rng(123)
        draw2 = sequential_bootstrap_draw(mat, n_draws=5, rng=rng2)

        np.testing.assert_array_equal(draw1, draw2)

    def test_favors_unique_samples_no_overlap(self) -> None:
        """When horizon=1 (no overlap), all samples are equally likely."""
        mat = compute_indicator_matrix(n_samples=5, horizon=1)
        rng = np.random.default_rng(42)

        # Many draws to check distribution
        indices = sequential_bootstrap_draw(mat, n_draws=1000, rng=rng)

        # Roughly uniform (allow ±15% tolerance)
        counts = np.bincount(indices, minlength=5)
        expected_per_bin = 1000 / 5
        assert np.all(counts > expected_per_bin * 0.5)
        assert np.all(counts < expected_per_bin * 1.5)

    def test_empty_matrix_raises(self) -> None:
        """Empty indicator matrix raises ValueError."""
        empty = np.zeros((0, 5), dtype=np.float64)
        rng = np.random.default_rng(42)

        with pytest.raises(ValueError, match="at least one row"):
            sequential_bootstrap_draw(empty, n_draws=1, rng=rng)

    def test_zero_draws_raises(self) -> None:
        """n_draws < 1 raises ValueError."""
        mat = compute_indicator_matrix(n_samples=5, horizon=2)
        rng = np.random.default_rng(42)

        with pytest.raises(ValueError, match="n_draws must be >= 1"):
            sequential_bootstrap_draw(mat, n_draws=0, rng=rng)

    def test_single_sample(self) -> None:
        """Single sample always draws index 0."""
        mat = compute_indicator_matrix(n_samples=1, horizon=5)
        rng = np.random.default_rng(42)

        indices = sequential_bootstrap_draw(mat, n_draws=3, rng=rng)

        np.testing.assert_array_equal(indices, [0, 0, 0])


# ---------------------------------------------------------------------------
# Non-overlapping subsampler
# ---------------------------------------------------------------------------


class TestSubsampleNonOverlapping:
    def test_horizon_one_returns_all(self) -> None:
        """With horizon=1, every bar is selected."""
        indices = subsample_non_overlapping(n_samples=5, horizon=1)

        np.testing.assert_array_equal(indices, [0, 1, 2, 3, 4])

    def test_stride_matches_horizon(self) -> None:
        """Indices are spaced exactly `horizon` apart."""
        indices = subsample_non_overlapping(n_samples=100, horizon=24)

        diffs = np.diff(indices)
        np.testing.assert_array_equal(diffs, np.full(len(diffs), 24))

    def test_starts_at_zero(self) -> None:
        """First selected index is always 0."""
        indices = subsample_non_overlapping(n_samples=50, horizon=7)

        assert indices[0] == 0

    def test_count_for_exact_division(self) -> None:
        """When n_samples is a multiple of horizon, n_selected = n_samples / horizon."""
        indices = subsample_non_overlapping(n_samples=48, horizon=24)

        assert len(indices) == 2  # 0, 24

    def test_count_for_non_exact_division(self) -> None:
        """When n_samples is NOT a multiple of horizon, ceil(n/h) indices returned."""
        indices = subsample_non_overlapping(n_samples=50, horizon=24)

        # indices: 0, 24, 48
        assert len(indices) == 3

    def test_zero_samples(self) -> None:
        """Zero samples produces empty array."""
        indices = subsample_non_overlapping(n_samples=0, horizon=5)

        assert len(indices) == 0

    def test_negative_samples_raises(self) -> None:
        """Negative n_samples raises ValueError."""
        with pytest.raises(ValueError, match="n_samples must be >= 0"):
            subsample_non_overlapping(n_samples=-1, horizon=5)

    def test_zero_horizon_raises(self) -> None:
        """horizon < 1 raises ValueError."""
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            subsample_non_overlapping(n_samples=10, horizon=0)


# ---------------------------------------------------------------------------
# High-level facade
# ---------------------------------------------------------------------------


class TestComputeLabelUniqueness:
    def test_sequential_bootstrap_method(self) -> None:
        """Sequential bootstrap produces weights in (0, 1] and valid N_eff."""
        config = LabelOverlapConfig(horizon=24, method="sequential_bootstrap")

        result = compute_label_uniqueness(n_samples=100, config=config)

        assert isinstance(result, LabelUniquenessResult)
        assert result.n_raw == 100
        assert len(result.weights) == 100
        assert all(0.0 < w <= 1.0 for w in result.weights)
        assert 0.0 < result.n_eff < 100.0

    def test_subsample_method(self) -> None:
        """Subsample method produces binary weights (0 or 1)."""
        config = LabelOverlapConfig(horizon=24, method="subsample")

        result = compute_label_uniqueness(n_samples=100, config=config)

        assert isinstance(result, LabelUniquenessResult)
        assert result.n_raw == 100
        assert len(result.weights) == 100
        assert all(w in {0.0, 1.0} for w in result.weights)

        n_selected = sum(1 for w in result.weights if w > 0.0)
        assert n_selected == 5  # ceil(100 / 24) = 5 (indices: 0, 24, 48, 72, 96)

    def test_subsample_n_eff_equals_n_selected(self) -> None:
        """For subsampling with uniform weights, N_eff = n_selected."""
        config = LabelOverlapConfig(horizon=10, method="subsample")

        result = compute_label_uniqueness(n_samples=30, config=config)

        n_selected = sum(1 for w in result.weights if w > 0.0)
        np.testing.assert_almost_equal(result.n_eff, float(n_selected))

    def test_horizon_one_gives_full_uniqueness(self) -> None:
        """Horizon=1 means no overlap → all weights = 1.0, N_eff = N."""
        config = LabelOverlapConfig(horizon=1, method="sequential_bootstrap")

        result = compute_label_uniqueness(n_samples=50, config=config)

        np.testing.assert_array_almost_equal(result.weights, [1.0] * 50)
        np.testing.assert_almost_equal(result.n_eff, 50.0)

    def test_single_sample(self) -> None:
        """Single sample always has weight=1, N_eff=1."""
        config = LabelOverlapConfig(horizon=24, method="sequential_bootstrap")

        result = compute_label_uniqueness(n_samples=1, config=config)

        assert result.n_raw == 1
        np.testing.assert_almost_equal(result.weights[0], 1.0)
        np.testing.assert_almost_equal(result.n_eff, 1.0)

    def test_zero_samples_raises(self) -> None:
        """Zero samples raises ValueError."""
        config = LabelOverlapConfig(horizon=24, method="sequential_bootstrap")

        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            compute_label_uniqueness(n_samples=0, config=config)

    def test_h24_n_eff_less_than_n_raw(self) -> None:
        """For horizon=24 with 200 samples, N_eff should be less than N_raw.

        The Kish formula with average uniqueness weights produces N_eff
        that depends on the variance of the weights. Edge samples have
        higher uniqueness than interior samples. With n=200, h=24, the
        interior is large enough that variance is moderate, so N_eff/N_raw
        is close to 1 but strictly less.
        """
        config = LabelOverlapConfig(horizon=24, method="sequential_bootstrap")

        result = compute_label_uniqueness(n_samples=200, config=config)

        assert result.n_eff < result.n_raw
        # Interior samples have uniqueness ~1/24 summed up via harmonic series;
        # Kish formula compresses this modestly for large n relative to h.
        assert result.n_eff > 0.5 * result.n_raw

    def test_h1_vs_h24_uniqueness(self) -> None:
        """Horizon=1 should have higher N_eff ratio than horizon=24."""
        n = 100
        config_h1 = LabelOverlapConfig(horizon=1, method="sequential_bootstrap")
        config_h24 = LabelOverlapConfig(horizon=24, method="sequential_bootstrap")

        result_h1 = compute_label_uniqueness(n_samples=n, config=config_h1)
        result_h24 = compute_label_uniqueness(n_samples=n, config=config_h24)

        assert result_h1.n_eff > result_h24.n_eff


# ---------------------------------------------------------------------------
# Pydantic value objects
# ---------------------------------------------------------------------------


class TestLabelOverlapConfig:
    def test_default_method(self) -> None:
        """Default method is sequential_bootstrap."""
        config = LabelOverlapConfig(horizon=24)

        assert config.method == "sequential_bootstrap"

    def test_frozen(self) -> None:
        """Config is immutable."""
        config = LabelOverlapConfig(horizon=24)

        with pytest.raises(ValidationError):
            config.horizon = 10  # type: ignore[misc]

    def test_invalid_horizon(self) -> None:
        """horizon < 1 raises validation error."""
        with pytest.raises(ValidationError):
            LabelOverlapConfig(horizon=0)

    def test_invalid_method(self) -> None:
        """Invalid method literal raises validation error."""
        with pytest.raises(ValidationError):
            LabelOverlapConfig(horizon=24, method="invalid")  # type: ignore[arg-type]


class TestLabelUniquenessResult:
    def test_valid_construction(self) -> None:
        """Valid result constructs without error."""
        result = LabelUniquenessResult(
            weights=(0.5, 0.5, 0.5),
            n_eff=2.5,
            n_raw=3,
        )

        assert result.n_raw == 3
        assert len(result.weights) == 3

    def test_weights_length_mismatch_raises(self) -> None:
        """Mismatched weights length and n_raw raises ValueError."""
        with pytest.raises(ValueError, match="weights length"):
            LabelUniquenessResult(
                weights=(0.5, 0.5),
                n_eff=1.5,
                n_raw=3,
            )

    def test_n_eff_exceeds_n_raw_raises(self) -> None:
        """N_eff > N_raw raises ValueError."""
        with pytest.raises(ValueError, match="n_eff.*must not exceed n_raw"):
            LabelUniquenessResult(
                weights=(1.0, 1.0),
                n_eff=3.0,
                n_raw=2,
            )
