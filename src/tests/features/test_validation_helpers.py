"""Unit tests for pure validation helper functions in validation.py.

Tests each stateless helper function in isolation using known inputs,
verifying correctness of MI computation, empirical p-values, BH correction,
directional accuracy, DC-MAE, Ridge evaluation, and feature group classification.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.app.features.application.validation import (
    apply_bh_correction,
    classify_feature_group,
    compute_dc_mae,
    compute_directional_accuracy,
    compute_empirical_pvalue,
    compute_mi_score,
    evaluate_single_feature_ridge,
)


_SEED: int = 42


class TestComputeMIScore:
    """Tests for the compute_mi_score function."""

    def test_mi_score_nonnegative(self) -> None:
        """MI is always >= 0."""
        rng: np.random.Generator = np.random.default_rng(_SEED)
        feature: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(200)
        target: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(200)
        mi: float = compute_mi_score(feature, target, random_seed=_SEED)
        assert mi >= 0.0

    def test_mi_score_identical_feature_and_target_is_high(self) -> None:
        """MI of identical feature and target should be relatively high."""
        rng: np.random.Generator = np.random.default_rng(_SEED + 1)
        x: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(500)
        mi_identical: float = compute_mi_score(x, x, random_seed=_SEED)
        mi_independent: float = compute_mi_score(x, rng.standard_normal(500), random_seed=_SEED)
        # MI with itself should be much larger than with random noise
        assert mi_identical > mi_independent

    def test_mi_score_independent_arrays_near_zero(self) -> None:
        """MI of statistically independent arrays should be near 0."""
        rng: np.random.Generator = np.random.default_rng(_SEED + 2)
        feature: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(300)
        target: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(300)
        mi: float = compute_mi_score(feature, target, random_seed=_SEED)
        # With n=300, MI with independent arrays should be small (< 0.2 nats typically)
        assert mi < 0.3

    def test_mi_score_deterministic_with_same_seed(self) -> None:
        """Same inputs and seed must produce the same MI value."""
        rng: np.random.Generator = np.random.default_rng(_SEED)
        feature: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(200)
        target: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(200)
        mi_1: float = compute_mi_score(feature, target, random_seed=10)
        mi_2: float = compute_mi_score(feature, target, random_seed=10)
        assert mi_1 == pytest.approx(mi_2)


class TestComputeEmpiricalPvalue:
    """Tests for the compute_empirical_pvalue function."""

    def test_empirical_pvalue_in_range_zero_one(self) -> None:
        """Empirical p-value must always be in (0, 1]."""
        null_dist: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        for observed_val in [0.0, 0.25, 0.5, 1.0, 10.0]:
            p: float = compute_empirical_pvalue(observed_val, null_dist)
            assert 0.0 < p <= 1.0, f"p={p} not in (0, 1] for observed={observed_val}"

    def test_empirical_pvalue_very_high_observed_gives_low_p(self) -> None:
        """When observed >> all null values, p-value should be small."""
        null_dist: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        observed: float = 100.0  # much larger than all null values
        p: float = compute_empirical_pvalue(observed, null_dist)
        # count(null >= 100) = 0, so p = (0+1)/(5+1) = 1/6 ≈ 0.167
        expected: float = 1.0 / 6.0
        assert p == pytest.approx(expected, rel=1e-6)

    def test_empirical_pvalue_phipson_smyth_formula(self) -> None:
        """Verify exact Phipson & Smyth formula: p = (count(null >= obs) + 1) / (n + 1)."""
        null_dist: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # observed = 3.0 → count(null >= 3) = 3 (values 3, 4, 5) → p = (3+1)/(5+1) = 4/6
        p: float = compute_empirical_pvalue(3.0, null_dist)
        assert p == pytest.approx(4.0 / 6.0, rel=1e-6)

    def test_empirical_pvalue_all_null_above_gives_max(self) -> None:
        """When all null values exceed observed, p = n / (n + 1)."""
        n: int = 99
        null_dist: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(n) * 10.0
        p: float = compute_empirical_pvalue(0.0, null_dist)
        # count(null >= 0) = 99 → p = (99+1)/(99+1) = 1.0
        assert p == pytest.approx(1.0, rel=1e-6)


class TestApplyBHCorrection:
    """Tests for the apply_bh_correction function."""

    def test_bh_correction_returns_correct_shapes(self) -> None:
        """BH correction returns reject mask and corrected p-values of same length."""
        pvalues: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.01, 0.05, 0.1, 0.5, 0.9])
        reject, corrected = apply_bh_correction(pvalues, alpha=0.05)
        assert len(reject) == len(pvalues)
        assert len(corrected) == len(pvalues)

    def test_bh_correction_small_pvalues_rejected(self) -> None:
        """Very small p-values should be rejected after BH correction."""
        pvalues: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.0001, 0.0002, 0.0003, 0.9, 0.95])
        reject, _ = apply_bh_correction(pvalues, alpha=0.05)
        # The first three very small p-values should be rejected
        assert reject[0]
        assert reject[1]
        assert reject[2]

    def test_bh_correction_large_pvalues_not_rejected(self) -> None:
        """Large p-values should not be rejected after BH correction."""
        pvalues: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.8, 0.9, 0.95, 0.99])
        reject, _ = apply_bh_correction(pvalues, alpha=0.05)
        assert not any(reject)

    def test_bh_corrected_pvalues_ge_raw(self) -> None:
        """Corrected p-values must be >= the original p-values (BH inflates them)."""
        pvalues: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.01, 0.03, 0.05, 0.1, 0.5])
        _, corrected = apply_bh_correction(pvalues, alpha=0.05)
        for raw, corr in zip(pvalues.tolist(), corrected.tolist(), strict=True):
            assert corr >= raw - 1e-10, f"Corrected p={corr} < raw p={raw}"


class TestComputeDirectionalAccuracy:
    """Tests for the compute_directional_accuracy function."""

    def test_da_perfect_prediction(self) -> None:
        """When y_pred matches y_true in sign exactly, DA = 1.0."""
        y_true: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([1.0, -1.0, 2.0, -3.0])
        y_pred: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.5, -0.5, 3.0, -0.1])
        da: float = compute_directional_accuracy(y_true, y_pred)
        assert da == pytest.approx(1.0)

    def test_da_opposite_signs(self) -> None:
        """When all predictions have the opposite sign, DA = 0.0."""
        y_true: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([1.0, 2.0, 3.0])
        y_pred: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([-1.0, -2.0, -3.0])
        da: float = compute_directional_accuracy(y_true, y_pred)
        assert da == pytest.approx(0.0)

    def test_da_all_zeros_returns_half(self) -> None:
        """When all true or predicted values are zero, DA defaults to 0.5."""
        y_true: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(5)
        y_pred: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        da: float = compute_directional_accuracy(y_true, y_pred)
        assert da == pytest.approx(0.5)

    def test_da_partial_match(self) -> None:
        """Partial sign matches should give a fractional DA."""
        y_true: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([1.0, -1.0, 1.0, -1.0])
        y_pred: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([1.0, 1.0, 1.0, 1.0])
        da: float = compute_directional_accuracy(y_true, y_pred)
        assert da == pytest.approx(0.5)

    def test_da_in_range_zero_one(self) -> None:
        """DA must always be in [0, 1]."""
        rng: np.random.Generator = np.random.default_rng(_SEED)
        y_true: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(100)
        y_pred: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(100)
        da: float = compute_directional_accuracy(y_true, y_pred)
        assert 0.0 <= da <= 1.0


class TestComputeDCMAE:
    """Tests for the compute_dc_mae function."""

    def test_dc_mae_perfect_prediction(self) -> None:
        """When predictions exactly match true values, DC-MAE = 0."""
        y_true: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([1.0, -2.0, 3.0])
        y_pred: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([1.0, -2.0, 3.0])
        dc_mae: float = compute_dc_mae(y_true, y_pred)
        assert dc_mae == pytest.approx(0.0)

    def test_dc_mae_no_correct_direction_is_inf(self) -> None:
        """When all predictions have wrong direction, DC-MAE = inf."""
        y_true: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([1.0, 2.0, 3.0])
        y_pred: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([-1.0, -2.0, -3.0])
        dc_mae: float = compute_dc_mae(y_true, y_pred)
        assert dc_mae == float("inf")

    def test_dc_mae_nonneg_when_some_correct(self) -> None:
        """DC-MAE must be non-negative when some correct-direction predictions exist."""
        y_true: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([1.0, -1.0, 2.0, -2.0])
        y_pred: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.5, -2.0, -1.0, -3.0])
        # Correct direction: index 0 (+/+), index 1 (-/-), index 3 (-/-) = 3 correct
        dc_mae: float = compute_dc_mae(y_true, y_pred)
        assert dc_mae >= 0.0

    def test_dc_mae_known_value(self) -> None:
        """Verify DC-MAE on known values.

        y_true = [2.0, -1.0], y_pred = [1.0, -3.0] — both correct direction.
        MAE on correct = mean(|2-1|, |-1-(-3)|) = mean(1, 2) = 1.5.
        """
        y_true: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([2.0, -1.0])
        y_pred: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([1.0, -3.0])
        dc_mae: float = compute_dc_mae(y_true, y_pred)
        assert dc_mae == pytest.approx(1.5)


class TestEvaluateSingleFeatureRidge:
    """Tests for the evaluate_single_feature_ridge function."""

    def test_ridge_returns_two_floats(self) -> None:
        """evaluate_single_feature_ridge must return a tuple of two floats."""
        rng: np.random.Generator = np.random.default_rng(_SEED)
        feature: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(200)
        target: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(200)
        result: tuple[float, float] = evaluate_single_feature_ridge(feature, target, ridge_alpha=1.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        da: float = result[0]
        dc: float = result[1]
        assert isinstance(da, float)
        assert isinstance(dc, float)

    def test_ridge_da_in_range(self) -> None:
        """DA from Ridge evaluation must be in [0, 1]."""
        rng: np.random.Generator = np.random.default_rng(_SEED + 1)
        feature: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(200)
        target: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(200)
        da, _ = evaluate_single_feature_ridge(feature, target, ridge_alpha=1.0)
        assert 0.0 <= da <= 1.0

    def test_ridge_perfect_linear_signal_high_da(self) -> None:
        """A perfect linear signal should produce high directional accuracy."""
        n: int = 300
        rng: np.random.Generator = np.random.default_rng(_SEED + 2)
        feature: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(n)
        # Target is feature + small noise → strong linear relationship
        target: np.ndarray[tuple[int], np.dtype[np.float64]] = feature + 0.01 * rng.standard_normal(n)
        da, _ = evaluate_single_feature_ridge(feature, target, ridge_alpha=0.001, train_fraction=0.7)
        assert da > 0.8


class TestClassifyFeatureGroup:
    """Tests for the classify_feature_group function."""

    def test_classify_returns_group(self) -> None:
        """'logret_1' should classify into the 'returns' group."""
        from src.app.features.domain.value_objects import _DEFAULT_FEATURE_GROUPS

        result: str = classify_feature_group("logret_1", dict(_DEFAULT_FEATURE_GROUPS))
        assert result == "returns"

    def test_classify_volatility_prefix(self) -> None:
        """'rv_12' should classify into the 'volatility' group."""
        from src.app.features.domain.value_objects import _DEFAULT_FEATURE_GROUPS

        result: str = classify_feature_group("rv_12", dict(_DEFAULT_FEATURE_GROUPS))
        assert result == "volatility"

    def test_classify_momentum_prefix(self) -> None:
        """'rsi_14' should classify into the 'momentum' group."""
        from src.app.features.domain.value_objects import _DEFAULT_FEATURE_GROUPS

        result: str = classify_feature_group("rsi_14", dict(_DEFAULT_FEATURE_GROUPS))
        assert result == "momentum"

    def test_classify_unknown_returns_other(self) -> None:
        """An unrecognised feature name should return 'other'."""
        from src.app.features.domain.value_objects import _DEFAULT_FEATURE_GROUPS

        result: str = classify_feature_group("unknown_feature_xyz", dict(_DEFAULT_FEATURE_GROUPS))
        assert result == "other"

    def test_classify_custom_groups(self) -> None:
        """Custom group mapping should work correctly."""
        custom_groups: dict[str, tuple[str, ...]] = {"custom": ("my_feat_",)}
        result: str = classify_feature_group("my_feat_123", custom_groups)
        assert result == "custom"

    def test_classify_empty_groups_returns_other(self) -> None:
        """Empty feature_groups dict always returns 'other'."""
        result: str = classify_feature_group("anything", {})
        assert result == "other"
