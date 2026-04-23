"""Unit tests for regression metrics (standalone and volatility).

Tests cover MAE, RMSE, R-squared, implicit DA, CRPS (Gaussian closed-form),
winsorization, QLIKE, Mincer-Zarnowitz, log-vol MAE, and pipeline stubs.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.app.forecasting.application.regression_metrics import (
    CRPSResult,
    RegressionMetrics,
    VolatilityMetrics,
    WinsorizeConfig,
    compute_crps_gaussian,
    compute_dc_mae,
    compute_dc_rmse,
    compute_economic_sharpe,
    compute_pdr,
    compute_regression_metrics,
    compute_volatility_metrics,
    compute_wdl,
    winsorize_predictions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _arr(*values: float) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Shorthand to create a float64 array."""
    return np.array(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Winsorization
# ---------------------------------------------------------------------------


class TestWinsorizeConfig:
    def test_default_bounds(self) -> None:
        config = WinsorizeConfig()
        assert config.lower_percentile == 1.0
        assert config.upper_percentile == 99.0

    def test_custom_bounds(self) -> None:
        config = WinsorizeConfig(lower_percentile=5.0, upper_percentile=95.0)
        assert config.lower_percentile == 5.0
        assert config.upper_percentile == 95.0


class TestWinsorize:
    def test_outliers_clipped(self) -> None:
        """Predictions outside percentile bounds get clipped."""
        preds = np.array([-100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 200.0], dtype=np.float64)
        config = WinsorizeConfig(lower_percentile=10.0, upper_percentile=90.0)
        result = winsorize_predictions(preds, config)

        lower_bound = float(np.percentile(preds, 10.0))
        upper_bound = float(np.percentile(preds, 90.0))
        assert float(result[0]) >= lower_bound
        assert float(result[-1]) <= upper_bound
        # Interior values should be unchanged
        np.testing.assert_almost_equal(result[1], 1.0)
        np.testing.assert_almost_equal(result[2], 2.0)

    def test_no_clipping_at_0_100(self) -> None:
        """Percentiles 0-100 should not clip anything."""
        preds = _arr(-10.0, 0.0, 10.0)
        config = WinsorizeConfig(lower_percentile=0.0, upper_percentile=100.0)
        result = winsorize_predictions(preds, config)
        np.testing.assert_array_equal(result, preds)

    def test_empty_raises(self) -> None:
        config = WinsorizeConfig()
        with pytest.raises(ValueError, match="at least one sample"):
            winsorize_predictions(np.array([], dtype=np.float64), config)

    def test_single_sample(self) -> None:
        config = WinsorizeConfig(lower_percentile=5.0, upper_percentile=95.0)
        result = winsorize_predictions(_arr(42.0), config)
        np.testing.assert_almost_equal(result[0], 42.0)


# ---------------------------------------------------------------------------
# Core regression metrics
# ---------------------------------------------------------------------------


class TestRegressionMetrics:
    def test_mae_rmse_known_values(self) -> None:
        """MAE and RMSE on small known data should match hand-computed values."""
        actuals = _arr(1.0, 2.0, 3.0)
        preds = _arr(1.1, 2.2, 2.8)
        result = compute_regression_metrics(actuals, preds)

        assert isinstance(result, RegressionMetrics)
        assert result.n_samples == 3

        # MAE = mean(|1.0-1.1|, |2.0-2.2|, |3.0-2.8|) = mean(0.1, 0.2, 0.2) = 0.1667
        expected_mae = (0.1 + 0.2 + 0.2) / 3.0
        np.testing.assert_almost_equal(result.mae, expected_mae, decimal=6)

        # RMSE = sqrt(mean(0.01, 0.04, 0.04)) = sqrt(0.03) = 0.1732
        expected_rmse = np.sqrt((0.01 + 0.04 + 0.04) / 3.0)
        np.testing.assert_almost_equal(result.rmse, expected_rmse, decimal=6)

    def test_r_squared_perfect(self) -> None:
        """R-squared = 1.0 for perfect predictions."""
        actuals = _arr(1.0, 2.0, 3.0, 4.0, 5.0)
        result = compute_regression_metrics(actuals, actuals.copy())
        np.testing.assert_almost_equal(result.r_squared, 1.0, decimal=10)

    def test_r_squared_constant_prediction(self) -> None:
        """R-squared = 0.0 when predicting the mean (constant)."""
        actuals = _arr(1.0, 2.0, 3.0, 4.0, 5.0)
        mean_pred = np.full(5, np.mean(actuals), dtype=np.float64)
        result = compute_regression_metrics(actuals, mean_pred)
        np.testing.assert_almost_equal(result.r_squared, 0.0, decimal=10)

    def test_r_squared_negative(self) -> None:
        """R-squared can be negative when predictions are worse than the mean."""
        actuals = _arr(1.0, 2.0, 3.0)
        preds = _arr(10.0, 20.0, 30.0)
        result = compute_regression_metrics(actuals, preds)
        assert result.r_squared < 0.0

    def test_implicit_da_known(self) -> None:
        """Implicit DA: known sign agreements."""
        actuals = _arr(1.0, -1.0, 1.0, -1.0)
        preds = _arr(0.5, -0.5, -0.5, 0.5)
        result = compute_regression_metrics(actuals, preds)
        # First two match sign, last two don't → DA = 2/4 = 0.5
        np.testing.assert_almost_equal(result.implicit_da, 0.5)

    def test_implicit_da_perfect(self) -> None:
        """DA = 1.0 when all signs match."""
        actuals = _arr(1.0, -2.0, 3.0)
        preds = _arr(0.1, -0.1, 0.1)
        result = compute_regression_metrics(actuals, preds)
        np.testing.assert_almost_equal(result.implicit_da, 1.0)

    def test_implicit_da_zeros(self) -> None:
        """Zeros are treated via np.sign convention (sign(0)=0)."""
        actuals = _arr(0.0, 1.0)
        preds = _arr(0.0, 1.0)
        result = compute_regression_metrics(actuals, preds)
        # sign(0) == sign(0), sign(1) == sign(1) → DA = 1.0
        np.testing.assert_almost_equal(result.implicit_da, 1.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one sample"):
            compute_regression_metrics(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="predictions length"):
            compute_regression_metrics(_arr(1.0, 2.0), _arr(1.0))

    def test_single_sample(self) -> None:
        """Gracefully handle a single sample (R² = 0 due to zero SS_tot)."""
        result = compute_regression_metrics(_arr(3.0), _arr(2.5))
        assert result.n_samples == 1
        np.testing.assert_almost_equal(result.mae, 0.5)
        np.testing.assert_almost_equal(result.rmse, 0.5)
        # SS_tot = 0 for single sample → R² defined as 0.0
        np.testing.assert_almost_equal(result.r_squared, 0.0)

    def test_constant_actuals_r_squared(self) -> None:
        """When actuals are constant, SS_tot ≈ 0 → R² = 0.0."""
        actuals = np.full(10, 5.0, dtype=np.float64)
        preds = np.full(10, 5.1, dtype=np.float64)
        result = compute_regression_metrics(actuals, preds)
        np.testing.assert_almost_equal(result.r_squared, 0.0)


# ---------------------------------------------------------------------------
# CRPS — Gaussian closed-form
# ---------------------------------------------------------------------------


class TestCRPSGaussian:
    def test_unit_normal_known_value(self) -> None:
        """CRPS for y=0 with N(0,1) has a known analytical value.

        CRPS(N(0,1), 0) = sigma * (0*(2*0.5-1) + 2*phi(0) - 1/sqrt(pi))
                        = 1 * (0 + 2*(1/sqrt(2*pi)) - 1/sqrt(pi))
                        = 2/sqrt(2*pi) - 1/sqrt(pi)
                        ≈ 0.7979 - 0.5642 ≈ 0.2337
        """
        actuals = _arr(0.0)
        mean = _arr(0.0)
        std = _arr(1.0)
        result = compute_crps_gaussian(actuals, mean, std)

        assert isinstance(result, CRPSResult)
        expected = 2.0 / np.sqrt(2.0 * np.pi) - 1.0 / np.sqrt(np.pi)
        np.testing.assert_almost_equal(result.mean_crps, expected, decimal=6)
        np.testing.assert_almost_equal(result.per_sample_crps[0], expected, decimal=6)

    def test_degenerate_std_zero(self) -> None:
        """When std=0, CRPS reduces to |y - mean|."""
        actuals = _arr(3.0, -1.0)
        mean = _arr(2.0, 1.0)
        std = _arr(0.0, 0.0)
        result = compute_crps_gaussian(actuals, mean, std)

        np.testing.assert_almost_equal(result.per_sample_crps[0], 1.0)
        np.testing.assert_almost_equal(result.per_sample_crps[1], 2.0)
        np.testing.assert_almost_equal(result.mean_crps, 1.5)

    def test_crps_decreases_with_smaller_std(self) -> None:
        """More confident (smaller std) correct predictions should have lower CRPS."""
        actuals = _arr(0.0, 0.0)
        mean = _arr(0.0, 0.0)
        std_wide = _arr(2.0, 2.0)
        std_narrow = _arr(0.5, 0.5)

        result_wide = compute_crps_gaussian(actuals, mean, std_wide)
        result_narrow = compute_crps_gaussian(actuals, mean, std_narrow)

        assert result_narrow.mean_crps < result_wide.mean_crps

    def test_crps_increases_with_bias(self) -> None:
        """CRPS should increase when the mean is biased away from actual."""
        actuals = _arr(0.0)
        std = _arr(1.0)
        result_unbiased = compute_crps_gaussian(actuals, _arr(0.0), std)
        result_biased = compute_crps_gaussian(actuals, _arr(5.0), std)

        assert result_biased.mean_crps > result_unbiased.mean_crps

    def test_per_sample_shape(self) -> None:
        n = 50
        rng = np.random.default_rng(42)
        actuals = rng.standard_normal(n).astype(np.float64)
        mean = rng.standard_normal(n).astype(np.float64)
        std = np.abs(rng.standard_normal(n)).astype(np.float64) + 0.01
        result = compute_crps_gaussian(actuals, mean, std)
        assert result.per_sample_crps.shape == (n,)
        assert np.all(result.per_sample_crps >= 0.0)

    def test_negative_std_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            compute_crps_gaussian(_arr(0.0), _arr(0.0), _arr(-1.0))

    def test_empty_raises(self) -> None:
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="at least one sample"):
            compute_crps_gaussian(empty, empty, empty)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="array lengths must match"):
            compute_crps_gaussian(_arr(1.0, 2.0), _arr(1.0), _arr(1.0))


# ---------------------------------------------------------------------------
# Volatility metrics
# ---------------------------------------------------------------------------


class TestVolatilityMetrics:
    def test_qlike_perfect(self) -> None:
        """QLIKE = 0 when actual_var == predicted_var."""
        vol = _arr(0.5, 1.0, 1.5, 2.0)
        result = compute_volatility_metrics(vol, vol.copy())

        assert isinstance(result, VolatilityMetrics)
        np.testing.assert_almost_equal(result.qlike, 0.0, decimal=10)

    def test_qlike_asymmetry(self) -> None:
        """Under-prediction (predicted < actual) should be penalised more than over-prediction.

        QLIKE is asymmetric: underestimating variance is costlier because
        ratio = h_actual/h_predicted > 1, and log(ratio) grows slower
        than ratio itself.
        """
        actual_vol = _arr(2.0, 2.0, 2.0, 2.0)

        # Under-prediction: predicted_vol = 1.0 (predicted_var=1, actual_var=4 → ratio=4)
        under_pred = _arr(1.0, 1.0, 1.0, 1.0)
        result_under = compute_volatility_metrics(actual_vol, under_pred)

        # Over-prediction: predicted_vol = 4.0 (predicted_var=16, actual_var=4 → ratio=0.25)
        over_pred = _arr(4.0, 4.0, 4.0, 4.0)
        result_over = compute_volatility_metrics(actual_vol, over_pred)

        assert result_under.qlike > result_over.qlike

    def test_qlike_known_value(self) -> None:
        """QLIKE on known scalar: actual_var=4, predicted_var=1 → ratio=4, QLIKE = 4-ln(4)-1."""
        actual_vol = _arr(2.0)
        predicted_vol = _arr(1.0)
        result = compute_volatility_metrics(actual_vol, predicted_vol)
        expected_qlike = 4.0 - np.log(4.0) - 1.0
        np.testing.assert_almost_equal(result.qlike, expected_qlike, decimal=10)

    def test_mincer_zarnowitz_perfect(self) -> None:
        """Perfect forecast → R²=1, slope≈1, intercept≈0."""
        rng = np.random.default_rng(42)
        vol = (rng.uniform(0.5, 2.0, size=100)).astype(np.float64)
        result = compute_volatility_metrics(vol, vol.copy())

        np.testing.assert_almost_equal(result.mincer_zarnowitz_r2, 1.0, decimal=6)
        np.testing.assert_almost_equal(result.mincer_zarnowitz_slope, 1.0, decimal=6)
        np.testing.assert_almost_equal(result.mincer_zarnowitz_intercept, 0.0, decimal=6)

    def test_mincer_zarnowitz_biased(self) -> None:
        """Systematic bias: predicted_vol = 0.5 * actual_vol → slope ≈ 2, intercept ≈ 0."""
        rng = np.random.default_rng(42)
        actual_vol = (rng.uniform(1.0, 3.0, size=200)).astype(np.float64)
        predicted_vol = (actual_vol * 0.5).astype(np.float64)
        result = compute_volatility_metrics(actual_vol, predicted_vol)

        # actual = alpha + beta * (0.5 * actual) → beta ≈ 2.0, alpha ≈ 0
        np.testing.assert_almost_equal(result.mincer_zarnowitz_slope, 2.0, decimal=2)
        np.testing.assert_almost_equal(result.mincer_zarnowitz_intercept, 0.0, decimal=2)
        # R² should still be 1.0 since the relationship is perfectly linear
        np.testing.assert_almost_equal(result.mincer_zarnowitz_r2, 1.0, decimal=6)

    def test_log_vol_mae_known(self) -> None:
        """Log-vol MAE on simple known inputs."""
        actual_vol = _arr(1.0, np.e)
        predicted_vol = _arr(np.e, 1.0)
        result = compute_volatility_metrics(actual_vol, predicted_vol)

        # |log(1) - log(e)| = 1, |log(e) - log(1)| = 1 → MAE = 1.0
        np.testing.assert_almost_equal(result.log_vol_mae, 1.0, decimal=10)

    def test_log_vol_mae_perfect(self) -> None:
        """Log-vol MAE = 0 for perfect predictions."""
        vol = _arr(0.5, 1.0, 2.0)
        result = compute_volatility_metrics(vol, vol.copy())
        np.testing.assert_almost_equal(result.log_vol_mae, 0.0, decimal=10)

    def test_non_positive_actual_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            compute_volatility_metrics(_arr(0.0, 1.0), _arr(1.0, 1.0))

    def test_non_positive_predicted_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            compute_volatility_metrics(_arr(1.0, 1.0), _arr(-0.5, 1.0))

    def test_empty_raises(self) -> None:
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="at least one sample"):
            compute_volatility_metrics(empty, empty)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="predicted_vol length"):
            compute_volatility_metrics(_arr(1.0, 2.0), _arr(1.0))


# ---------------------------------------------------------------------------
# Pipeline stubs — require Phase 11 classifier
# ---------------------------------------------------------------------------


class TestCRPSInvariants:
    def test_crps_non_negativity(self) -> None:
        """CRPS is a proper scoring rule and must be >= 0 for all inputs."""
        rng = np.random.default_rng(42)
        n = 200
        actuals = rng.standard_normal(n).astype(np.float64)
        mean = rng.standard_normal(n).astype(np.float64)
        std = np.abs(rng.standard_normal(n)).astype(np.float64) + 0.01
        result = compute_crps_gaussian(actuals, mean, std)
        assert np.all(result.per_sample_crps >= 0.0), "CRPS must be non-negative for all samples"

    def test_crps_scale_equivariance(self) -> None:
        """CRPS(a*y, a*mu, a*sigma) = |a| * CRPS(y, mu, sigma).

        This ensures the metric is meaningful under different return scales
        (e.g. raw returns vs percentage returns).
        """
        rng = np.random.default_rng(42)
        n = 100
        actuals = rng.standard_normal(n).astype(np.float64)
        mean = rng.standard_normal(n).astype(np.float64)
        std = np.abs(rng.standard_normal(n)).astype(np.float64) + 0.1

        scale_factor: float = 100.0  # e.g. converting to percentage returns
        result_original = compute_crps_gaussian(actuals, mean, std)
        result_scaled = compute_crps_gaussian(
            actuals * scale_factor,
            mean * scale_factor,
            std * scale_factor,
        )

        np.testing.assert_allclose(
            result_scaled.mean_crps,
            scale_factor * result_original.mean_crps,
            rtol=1e-6,
            err_msg="CRPS must satisfy scale equivariance: CRPS(aY, aMu, aSigma) = |a|*CRPS(Y, Mu, Sigma)",
        )


class TestCRPSMonotonicity:
    def test_crps_perfectly_calibrated_beats_miscalibrated(self) -> None:
        """CRPS for perfectly calibrated Gaussian should be lower than miscalibrated."""
        rng = np.random.default_rng(42)
        n = 500

        # Generate actuals from N(0, 1)
        actuals = rng.standard_normal(n).astype(np.float64)

        # Perfectly calibrated: mean=0, std=1 (matches the true DGP)
        mean_good = np.zeros(n, dtype=np.float64)
        std_good = np.ones(n, dtype=np.float64)

        # Miscalibrated: mean=0, std=3 (too wide)
        std_wide = np.full(n, 3.0, dtype=np.float64)

        # Miscalibrated: mean=2, std=1 (biased)
        mean_biased = np.full(n, 2.0, dtype=np.float64)
        std_biased = np.ones(n, dtype=np.float64)

        crps_good = compute_crps_gaussian(actuals, mean_good, std_good)
        crps_wide = compute_crps_gaussian(actuals, mean_good, std_wide)
        crps_biased = compute_crps_gaussian(actuals, mean_biased, std_biased)

        assert crps_good.mean_crps < crps_wide.mean_crps, "Calibrated should beat too-wide"
        assert crps_good.mean_crps < crps_biased.mean_crps, "Calibrated should beat biased"


class TestQLIKEInvariants:
    def test_qlike_non_negativity(self) -> None:
        """QLIKE = mean(ratio - log(ratio) - 1) >= 0 by Jensen's inequality.

        For any positive ratio x: x - ln(x) - 1 >= 0 (equality only at x=1).
        """
        rng = np.random.default_rng(42)
        n = 200
        actual_vol = rng.uniform(0.1, 3.0, size=n).astype(np.float64)
        predicted_vol = rng.uniform(0.1, 3.0, size=n).astype(np.float64)
        result = compute_volatility_metrics(actual_vol, predicted_vol)
        assert result.qlike >= 0.0, f"QLIKE must be non-negative, got {result.qlike}"

    def test_mincer_zarnowitz_r2_with_constant_shift(self) -> None:
        """If predicted = actual + constant, MZ R² should still be 1.0.

        The slope absorbs the scaling and the intercept absorbs the bias,
        so a constant shift is a perfectly linear relationship.
        """
        rng = np.random.default_rng(42)
        actual_vol = rng.uniform(0.5, 2.0, size=100).astype(np.float64)
        predicted_vol = (actual_vol + 0.5).astype(np.float64)  # constant shift
        result = compute_volatility_metrics(actual_vol, predicted_vol)

        np.testing.assert_almost_equal(result.mincer_zarnowitz_r2, 1.0, decimal=6)
        np.testing.assert_almost_equal(result.mincer_zarnowitz_slope, 1.0, decimal=4)


class TestVolatilityMetricsWithNoise:
    def test_noisy_predictions_lower_r2_higher_qlike(self) -> None:
        """Noisy vol predictions should have lower MZ R^2 and higher QLIKE than perfect."""
        rng = np.random.default_rng(42)
        n = 200
        actual_vol = rng.uniform(0.5, 2.0, size=n).astype(np.float64)

        # Perfect predictions
        result_perfect = compute_volatility_metrics(actual_vol, actual_vol.copy())

        # Noisy predictions: true vol + noise
        noise = rng.normal(0, 0.5, size=n).astype(np.float64)
        noisy_vol = np.maximum(actual_vol + noise, 0.01).astype(np.float64)
        result_noisy = compute_volatility_metrics(actual_vol, noisy_vol)

        assert result_noisy.mincer_zarnowitz_r2 < result_perfect.mincer_zarnowitz_r2
        assert result_noisy.qlike > result_perfect.qlike
        assert result_noisy.log_vol_mae > result_perfect.log_vol_mae


class TestDCMAE:
    def test_known_values(self) -> None:
        """DC-MAE on known data with a direction mask."""
        actuals = _arr(1.0, 2.0, 3.0, 4.0)
        preds = _arr(1.5, 2.5, 3.5, 4.5)
        mask = np.array([True, True, False, False], dtype=np.bool_)
        result = compute_dc_mae(actuals, preds, mask)
        # Only first two samples: |1.0-1.5| + |2.0-2.5| = 0.5+0.5 = 1.0, mean = 0.5
        np.testing.assert_almost_equal(result, 0.5)

    def test_all_correct(self) -> None:
        """DC-MAE equals regular MAE when all directions correct."""
        actuals = _arr(1.0, 2.0, 3.0)
        preds = _arr(1.1, 2.2, 2.8)
        mask = np.array([True, True, True], dtype=np.bool_)
        result = compute_dc_mae(actuals, preds, mask)
        expected = (0.1 + 0.2 + 0.2) / 3.0
        np.testing.assert_almost_equal(result, expected, decimal=6)

    def test_no_correct_raises(self) -> None:
        mask = np.array([False, False], dtype=np.bool_)
        with pytest.raises(ValueError, match="no direction-correct"):
            compute_dc_mae(_arr(1.0, 2.0), _arr(1.0, 2.0), mask)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one sample"):
            compute_dc_mae(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.bool_),
            )


class TestDCRMSE:
    def test_known_values(self) -> None:
        """DC-RMSE on known data with a direction mask."""
        actuals = _arr(1.0, 2.0, 3.0, 4.0)
        preds = _arr(1.5, 2.5, 3.5, 4.5)
        mask = np.array([True, True, False, False], dtype=np.bool_)
        result = compute_dc_rmse(actuals, preds, mask)
        # Only first two: sqrt(mean(0.25, 0.25)) = sqrt(0.25) = 0.5
        np.testing.assert_almost_equal(result, 0.5)

    def test_no_correct_raises(self) -> None:
        mask = np.array([False, False], dtype=np.bool_)
        with pytest.raises(ValueError, match="no direction-correct"):
            compute_dc_rmse(_arr(1.0, 2.0), _arr(1.0, 2.0), mask)


class TestPDR:
    def test_perfect_pdr(self) -> None:
        """PDR=1.0 when all confident predictions are successful."""
        actuals = _arr(0.02, 0.03, 0.015)
        preds = _arr(0.02, 0.025, 0.02)
        mask = np.array([True, True, True], dtype=np.bool_)
        result = compute_pdr(actuals, preds, mask, magnitude_threshold=0.01, realized_threshold=0.005)
        np.testing.assert_almost_equal(result, 1.0)

    def test_no_confident_returns_zero(self) -> None:
        """PDR=0.0 when no predictions exceed magnitude threshold."""
        actuals = _arr(0.02, 0.03)
        preds = _arr(0.001, 0.002)  # below threshold
        mask = np.array([True, True], dtype=np.bool_)
        result = compute_pdr(actuals, preds, mask, magnitude_threshold=0.01)
        np.testing.assert_almost_equal(result, 0.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one sample"):
            compute_pdr(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.bool_),
            )


class TestEconomicSharpe:
    def test_perfect_signal(self) -> None:
        """Positive Sharpe when direction predictions are always correct."""
        actuals = _arr(0.01, -0.01, 0.02, -0.02, 0.01)
        preds = _arr(0.01, 0.01, 0.02, 0.02, 0.01)
        dirs = _arr(1.0, -1.0, 1.0, -1.0, 1.0)
        result = compute_economic_sharpe(actuals, preds, dirs, transaction_cost=0.0)
        assert result > 0.0

    def test_transaction_costs_reduce_sharpe(self) -> None:
        """Higher costs should reduce Sharpe."""
        actuals = _arr(0.01, -0.01, 0.02, -0.02, 0.01)
        dirs = _arr(1.0, -1.0, 1.0, -1.0, 1.0)
        preds = _arr(0.01, 0.01, 0.02, 0.02, 0.01)
        sr_no_cost = compute_economic_sharpe(actuals, preds, dirs, transaction_cost=0.0)
        sr_with_cost = compute_economic_sharpe(actuals, preds, dirs, transaction_cost=0.01)
        assert sr_with_cost < sr_no_cost

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one sample"):
            compute_economic_sharpe(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )


class TestWDLStub:
    """WDL remains a stub — requires backtest engine for realistic cost modeling."""

    def test_wdl_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="Phase 11"):
            compute_wdl(_arr(1.0), _arr(1.0), _arr(1.0))
