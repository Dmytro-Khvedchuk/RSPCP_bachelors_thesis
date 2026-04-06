"""Unit tests for the GradientBoostingRegressor (LightGBM quantile regressor).

Tests quantile prediction shapes, monotonicity correction, point prediction
convenience, and error handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.app.forecasting.application.gradient_boosting_reg import GradientBoostingRegressor
from src.app.forecasting.domain.value_objects import (
    GradientBoostingConfig,
    PointPrediction,
    QuantilePrediction,
)

from src.tests.forecasting.conftest import make_gb_config


class TestGradientBoostingRegressor:
    """Tests for LightGBM quantile regression fit/predict behaviour."""

    def test_gb_quantile_shapes(
        self,
        linear_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        gb_config: GradientBoostingConfig,
    ) -> None:
        """predict_quantiles output has correct shape (n_samples, 5)."""
        x, y = linear_data
        model: GradientBoostingRegressor = GradientBoostingRegressor(gb_config)
        model.fit(x[:150], y[:150])

        n_test: int = 50
        qpred: QuantilePrediction = model.predict_quantiles(x[150:])

        assert qpred.values.shape == (n_test, 5)
        assert qpred.quantiles == (0.05, 0.25, 0.50, 0.75, 0.95)

    def test_gb_quantile_monotonicity(
        self,
        linear_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
    ) -> None:
        """With isotonic=True, q_0.05 <= q_0.25 <= q_0.50 <= q_0.75 <= q_0.95 for all samples."""
        x, y = linear_data
        config: GradientBoostingConfig = make_gb_config(apply_isotonic=True)
        model: GradientBoostingRegressor = GradientBoostingRegressor(config)
        model.fit(x[:150], y[:150])

        qpred: QuantilePrediction = model.predict_quantiles(x[150:])

        # Check monotonicity: each column should be <= the next
        for col in range(qpred.values.shape[1] - 1):
            diff: np.ndarray[tuple[int], np.dtype[np.float64]] = qpred.values[:, col + 1] - qpred.values[:, col]
            assert np.all(diff >= -1e-10), f"Quantile monotonicity violated between columns {col} and {col + 1}"

    def test_gb_point_prediction_shape(
        self,
        linear_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        gb_config: GradientBoostingConfig,
    ) -> None:
        """predict() returns PointPrediction with correct shapes."""
        x, y = linear_data
        model: GradientBoostingRegressor = GradientBoostingRegressor(gb_config)
        model.fit(x[:150], y[:150])

        n_test: int = 50
        pred: PointPrediction = model.predict(x[150:])

        assert pred.mean.shape == (n_test,)
        assert pred.std.shape == (n_test,)

    def test_gb_iqr_std_nonnegative(
        self,
        linear_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        gb_config: GradientBoostingConfig,
    ) -> None:
        """IQR-based std values are all >= 0."""
        x, y = linear_data
        model: GradientBoostingRegressor = GradientBoostingRegressor(gb_config)
        model.fit(x[:150], y[:150])

        pred: PointPrediction = model.predict(x[150:])

        assert np.all(pred.std >= 0), "All std values should be non-negative"

    def test_gb_predict_before_fit_raises(
        self,
        gb_config: GradientBoostingConfig,
    ) -> None:
        """Calling predict or predict_quantiles before fit() raises RuntimeError."""
        model: GradientBoostingRegressor = GradientBoostingRegressor(gb_config)
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((10, 5), dtype=np.float64)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict_quantiles(x_test)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(x_test)

    def test_gb_empty_input_raises(
        self,
        gb_config: GradientBoostingConfig,
    ) -> None:
        """Empty input arrays raise ValueError on both fit and predict_quantiles."""
        model: GradientBoostingRegressor = GradientBoostingRegressor(gb_config)

        x_empty: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty((0, 5), dtype=np.float64)
        y_empty: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(0, dtype=np.float64)

        with pytest.raises(ValueError, match="at least one sample"):
            model.fit(x_empty, y_empty)

    def test_gb_empty_predict_raises(
        self,
        linear_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        gb_config: GradientBoostingConfig,
    ) -> None:
        """Empty test array raises ValueError on predict_quantiles."""
        x, y = linear_data
        model: GradientBoostingRegressor = GradientBoostingRegressor(gb_config)
        model.fit(x[:150], y[:150])

        x_empty: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty((0, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="at least one sample"):
            model.predict_quantiles(x_empty)
