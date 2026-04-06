"""Unit tests for the HARRVForecaster (Corsi 2009 HAR-RV model).

Tests feature construction, fit/predict shapes, R-squared on persistent
RV series, and error handling for short series.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.app.forecasting.application.har_rv import HARRVForecaster
from src.app.forecasting.domain.value_objects import HARRVConfig, VolatilityForecast

from src.tests.forecasting.conftest import make_har_config, make_rv_series


class TestHARRVConstructFeatures:
    """Tests for the static construct_har_features method."""

    def test_har_construct_features_shape(
        self,
        rv_series: np.ndarray[tuple[int], np.dtype[np.float64]],
        har_config: HARRVConfig,
    ) -> None:
        """Output shape is (n - monthly_lag + 1, 3)."""
        features: np.ndarray[tuple[int, int], np.dtype[np.float64]] = HARRVForecaster.construct_har_features(
            rv_series, har_config
        )

        expected_rows: int = len(rv_series) - har_config.monthly_lag + 1
        assert features.shape == (expected_rows, 3)

    def test_har_construct_features_daily_is_last_value(
        self,
        har_config: HARRVConfig,
    ) -> None:
        """Daily RV column equals the rv_series value at the corresponding position (when daily_lag=1)."""
        rv: np.ndarray[tuple[int], np.dtype[np.float64]] = np.arange(20, dtype=np.float64) + 1.0
        config: HARRVConfig = make_har_config(daily_lag=1, weekly_lag=3, monthly_lag=5)

        features: np.ndarray[tuple[int, int], np.dtype[np.float64]] = HARRVForecaster.construct_har_features(
            rv, config
        )

        warmup: int = config.monthly_lag - 1  # 4
        # Daily column (col 0) with daily_lag=1 should be rv[warmup], rv[warmup+1], ...
        for i in range(features.shape[0]):
            expected_daily: float = rv[warmup + i]
            np.testing.assert_allclose(features[i, 0], expected_daily, rtol=1e-10)

    def test_har_construct_features_weekly_is_rolling_mean(
        self,
        har_config: HARRVConfig,
    ) -> None:
        """Weekly RV column equals the rolling mean over weekly_lag bars."""
        rv: np.ndarray[tuple[int], np.dtype[np.float64]] = np.arange(20, dtype=np.float64) + 1.0
        config: HARRVConfig = make_har_config(daily_lag=1, weekly_lag=3, monthly_lag=5)

        features: np.ndarray[tuple[int, int], np.dtype[np.float64]] = HARRVForecaster.construct_har_features(
            rv, config
        )

        warmup: int = config.monthly_lag - 1  # 4
        w: int = config.weekly_lag  # 3
        # Weekly column (col 1) at position i = mean(rv[warmup+i - w + 1 : warmup+i + 1])
        for i in range(features.shape[0]):
            idx: int = warmup + i
            expected_weekly: float = float(np.mean(rv[idx - w + 1 : idx + 1]))
            np.testing.assert_allclose(features[i, 1], expected_weekly, rtol=1e-10)


class TestHARRVForecaster:
    """Tests for HARRVForecaster fit/predict behaviour."""

    def test_har_fit_predict_shape(
        self,
        rv_series: np.ndarray[tuple[int], np.dtype[np.float64]],
        har_config: HARRVConfig,
    ) -> None:
        """predict(n_steps=5) returns arrays of length 5."""
        model: HARRVForecaster = HARRVForecaster(har_config)
        model.fit(rv_series)

        n_steps: int = 5
        forecast: VolatilityForecast = model.predict(n_steps)

        assert forecast.predicted_vol.shape == (n_steps,)
        assert forecast.predicted_var.shape == (n_steps,)

    def test_har_fit_predict_with_features(
        self,
        rv_series: np.ndarray[tuple[int], np.dtype[np.float64]],
        har_config: HARRVConfig,
    ) -> None:
        """predict with x_test feature matrix works correctly."""
        model: HARRVForecaster = HARRVForecaster(har_config)
        model.fit(rv_series)

        # Construct features from the same series and use last 5 rows as x_test
        features: np.ndarray[tuple[int, int], np.dtype[np.float64]] = HARRVForecaster.construct_har_features(
            rv_series, har_config
        )
        n_steps: int = 5
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = features[-n_steps:]

        forecast: VolatilityForecast = model.predict(n_steps, x_test=x_test)

        assert forecast.predicted_vol.shape == (n_steps,)
        assert forecast.predicted_var.shape == (n_steps,)
        # Variance should be positive
        assert np.all(forecast.predicted_var > 0)

    def test_har_r_squared_on_persistent_rv(
        self,
        har_config: HARRVConfig,
    ) -> None:
        """On a persistent (autocorrelated) RV series, R-squared > 0.3."""
        rv: np.ndarray[tuple[int], np.dtype[np.float64]] = make_rv_series(n=300, seed=42)

        model: HARRVForecaster = HARRVForecaster(har_config)
        model.fit(rv[:250])

        # Construct features for out-of-sample prediction
        features_oos: np.ndarray[tuple[int, int], np.dtype[np.float64]] = HARRVForecaster.construct_har_features(
            rv[245:], har_config
        )
        # The targets for these features
        warmup: int = har_config.monthly_lag
        actual_rv: np.ndarray[tuple[int], np.dtype[np.float64]] = rv[245 + warmup :]

        n_steps: int = min(features_oos.shape[0], len(actual_rv))
        # Use features[:-1] as input (they predict next-step RV)
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = features_oos[: n_steps - 1]
        y_actual: np.ndarray[tuple[int], np.dtype[np.float64]] = actual_rv[: n_steps - 1]

        forecast: VolatilityForecast = model.predict(x_test.shape[0], x_test=x_test)

        # Compute R-squared
        ss_res: float = float(np.sum((y_actual - forecast.predicted_var) ** 2))
        ss_tot: float = float(np.sum((y_actual - np.mean(y_actual)) ** 2))
        r_squared: float = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        assert r_squared > 0.3, f"R-squared={r_squared:.4f} is too low for persistent RV"

    def test_har_short_series_raises(
        self,
        har_config: HARRVConfig,
    ) -> None:
        """ValueError for series shorter than monthly_lag + 2."""
        min_needed: int = har_config.monthly_lag + 2
        short_rv: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(min_needed - 1, dtype=np.float64)

        model: HARRVForecaster = HARRVForecaster(har_config)

        with pytest.raises(ValueError, match="too short"):
            model.fit(short_rv)

    def test_har_predict_before_fit_raises(
        self,
        har_config: HARRVConfig,
    ) -> None:
        """Calling predict() before fit() raises RuntimeError."""
        model: HARRVForecaster = HARRVForecaster(har_config)

        with pytest.raises(RuntimeError, match="fitted before prediction"):
            model.predict(5)

    def test_har_iterative_vs_features_one_step(
        self,
        rv_series: np.ndarray[tuple[int], np.dtype[np.float64]],
        har_config: HARRVConfig,
    ) -> None:
        """Iterative and feature-based prediction should produce similar 1-step forecast."""
        model: HARRVForecaster = HARRVForecaster(har_config)
        model.fit(rv_series)

        # Iterative 1-step forecast
        forecast_iter: VolatilityForecast = model.predict(1)

        # Feature-based 1-step forecast: construct features from the same series
        features: np.ndarray[tuple[int, int], np.dtype[np.float64]] = HARRVForecaster.construct_har_features(
            rv_series, har_config
        )
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = features[-1:]
        forecast_feat: VolatilityForecast = model.predict(1, x_test=x_test)

        # Both should produce very similar results for the first step
        np.testing.assert_allclose(
            forecast_iter.predicted_var,
            forecast_feat.predicted_var,
            rtol=1e-6,
            err_msg="Iterative and feature-based 1-step forecasts should match",
        )

    def test_har_no_intercept(
        self,
        rv_series: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Config with fit_intercept=False should fit and predict without error."""
        config: HARRVConfig = make_har_config(fit_intercept=False)
        model: HARRVForecaster = HARRVForecaster(config)
        model.fit(rv_series)

        forecast: VolatilityForecast = model.predict(3)

        assert forecast.predicted_vol.shape == (3,)
        assert forecast.predicted_var.shape == (3,)
        assert np.all(forecast.predicted_var > 0)

    def test_har_iterative_prediction_stays_finite(
        self,
        rv_series: np.ndarray[tuple[int], np.dtype[np.float64]],
        har_config: HARRVConfig,
    ) -> None:
        """Multi-step iterative forecasting should not diverge."""
        model: HARRVForecaster = HARRVForecaster(har_config)
        model.fit(rv_series)

        n_steps: int = 50
        forecast: VolatilityForecast = model.predict(n_steps)

        assert np.all(np.isfinite(forecast.predicted_var)), "Iterative predictions must stay finite"
        assert np.all(forecast.predicted_var > 0), "All predicted variances must be positive"
        assert np.all(forecast.predicted_vol > 0), "All predicted volatilities must be positive"

    def test_har_daily_coefficient_dominates_on_persistent_rv(self) -> None:
        """On persistent RV data, the daily coefficient should be the largest.

        The daily (short-term) RV component carries most predictive power
        in Corsi (2009)'s HAR-RV model.  Due to multicollinearity between
        the three components, individual coefficients can be negative, but
        the sum must be positive (overall persistence) and the daily coefficient
        should dominate.
        """
        rv: np.ndarray[tuple[int], np.dtype[np.float64]] = make_rv_series(n=500, seed=42)
        config: HARRVConfig = make_har_config()

        model: HARRVForecaster = HARRVForecaster(config)
        model.fit(rv)

        assert model._coefficients is not None
        start_idx: int = 1 if config.fit_intercept else 0
        slope_coeffs: np.ndarray[tuple[int], np.dtype[np.float64]] = model._coefficients[start_idx:]

        daily_coeff: float = float(slope_coeffs[0])
        coeff_sum: float = float(np.sum(slope_coeffs))

        assert daily_coeff > 0, f"Daily coefficient should be positive, got {daily_coeff}"
        assert coeff_sum > 0, f"Sum of HAR coefficients should be positive (persistence), got {coeff_sum}"
        assert daily_coeff == float(np.max(slope_coeffs)), (
            f"Daily coefficient should be the largest, got {slope_coeffs}"
        )

    def test_har_construct_features_too_short_raises(self) -> None:
        """Series shorter than monthly_lag + 1 raises ValueError in construct_har_features."""
        config: HARRVConfig = make_har_config(monthly_lag=5)
        # Need at least monthly_lag + 1 = 6 observations; provide 5
        short_rv: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(5, dtype=np.float64)

        with pytest.raises(ValueError, match="too short"):
            HARRVForecaster.construct_har_features(short_rv, config)
