"""Unit tests for the ARIMAGARCHForecaster (GARCH conditional variance model).

Tests fit/predict on synthetic GARCH returns, variance positivity,
vol/var consistency, rescale behaviour, and error handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.app.forecasting.application.arima_garch import ARIMAGARCHForecaster
from src.app.forecasting.domain.value_objects import GARCHConfig, VolatilityForecast

from src.tests.forecasting.conftest import make_garch_config


class TestARIMAGARCHForecaster:
    """Tests for GARCH forecaster fit/predict behaviour."""

    def test_garch_fit_on_synthetic_data(
        self,
        garch_returns: np.ndarray[tuple[int], np.dtype[np.float64]],
        garch_config: GARCHConfig,
    ) -> None:
        """Fits without error on synthetic GARCH returns."""
        model: ARIMAGARCHForecaster = ARIMAGARCHForecaster(garch_config)
        # Should not raise
        model.fit(garch_returns)

    def test_garch_predict_shape(
        self,
        garch_returns: np.ndarray[tuple[int], np.dtype[np.float64]],
        garch_config: GARCHConfig,
    ) -> None:
        """predict(n_steps=5) returns arrays of length 5."""
        model: ARIMAGARCHForecaster = ARIMAGARCHForecaster(garch_config)
        model.fit(garch_returns)

        n_steps: int = 5
        forecast: VolatilityForecast = model.predict(n_steps)

        assert forecast.predicted_vol.shape == (n_steps,)
        assert forecast.predicted_var.shape == (n_steps,)

    def test_garch_predicted_var_positive(
        self,
        garch_returns: np.ndarray[tuple[int], np.dtype[np.float64]],
        garch_config: GARCHConfig,
    ) -> None:
        """All predicted variances must be strictly positive."""
        model: ARIMAGARCHForecaster = ARIMAGARCHForecaster(garch_config)
        model.fit(garch_returns)

        forecast: VolatilityForecast = model.predict(10)

        assert np.all(forecast.predicted_var > 0), "All predicted variances should be > 0"
        assert np.all(forecast.predicted_vol > 0), "All predicted volatilities should be > 0"

    def test_garch_predicted_vol_consistent(
        self,
        garch_returns: np.ndarray[tuple[int], np.dtype[np.float64]],
        garch_config: GARCHConfig,
    ) -> None:
        """predicted_vol^2 should approximately equal predicted_var."""
        model: ARIMAGARCHForecaster = ARIMAGARCHForecaster(garch_config)
        model.fit(garch_returns)

        forecast: VolatilityForecast = model.predict(5)

        np.testing.assert_allclose(
            forecast.predicted_vol**2,
            forecast.predicted_var,
            rtol=1e-10,
            err_msg="predicted_vol^2 should equal predicted_var",
        )

    def test_garch_rescale_undo(
        self,
        garch_returns: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """When rescale=True, output is in the original return scale (not percentage returns)."""
        config_rescale: GARCHConfig = make_garch_config(rescale=True)
        config_no_rescale: GARCHConfig = make_garch_config(rescale=False)

        model_rescale: ARIMAGARCHForecaster = ARIMAGARCHForecaster(config_rescale)
        model_rescale.fit(garch_returns)
        forecast_rescale: VolatilityForecast = model_rescale.predict(5)

        model_no_rescale: ARIMAGARCHForecaster = ARIMAGARCHForecaster(config_no_rescale)
        model_no_rescale.fit(garch_returns)
        forecast_no_rescale: VolatilityForecast = model_no_rescale.predict(5)

        # Both should produce variance in the same order of magnitude
        # (rescale should undo the 100x scaling, dividing variance by 100^2=10000)
        ratio: np.ndarray[tuple[int], np.dtype[np.float64]] = (
            forecast_rescale.predicted_var / forecast_no_rescale.predicted_var
        )

        # The variances should be within an order of magnitude of each other
        # since rescaling is just for numerical stability
        assert np.all(ratio > 0.01), "Rescaled variance should be positive"
        assert np.all(ratio < 100), "Rescaled variance should be in a reasonable range"

    def test_garch_predict_before_fit_raises(
        self,
        garch_config: GARCHConfig,
    ) -> None:
        """Calling predict() before fit() raises RuntimeError."""
        model: ARIMAGARCHForecaster = ARIMAGARCHForecaster(garch_config)

        with pytest.raises(RuntimeError, match="fitted before prediction"):
            model.predict(5)

    def test_garch_short_series_raises(
        self,
        garch_config: GARCHConfig,
    ) -> None:
        """ValueError for return series that is too short."""
        # GARCH(1,1) needs at least max(p,q) + 10 = 11 observations
        rng: np.random.Generator = np.random.default_rng(42)
        short_returns: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(5).astype(np.float64)

        model: ARIMAGARCHForecaster = ARIMAGARCHForecaster(garch_config)

        with pytest.raises(ValueError, match="too short"):
            model.fit(short_returns)

    def test_garch_student_t_dist(
        self,
        garch_returns: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Fits with Student-t distribution (default) without error and produces valid forecasts."""
        config: GARCHConfig = make_garch_config(dist="t")
        model: ARIMAGARCHForecaster = ARIMAGARCHForecaster(config)
        model.fit(garch_returns)

        forecast: VolatilityForecast = model.predict(3)

        assert forecast.predicted_vol.shape == (3,)
        assert np.all(forecast.predicted_var > 0)
        assert not np.any(np.isnan(forecast.predicted_var))
