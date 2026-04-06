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

    def test_gb_null_test_on_noise(self) -> None:
        """On pure noise targets, GBM MAE should not significantly beat the naive predictor."""
        rng: np.random.Generator = np.random.default_rng(42)
        n: int = 300
        n_features: int = 5
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = rng.standard_normal((n, n_features)).astype(np.float64)
        y: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(n).astype(np.float64)

        config: GradientBoostingConfig = make_gb_config()
        model: GradientBoostingRegressor = GradientBoostingRegressor(config)
        model.fit(x[:200], y[:200])

        pred: PointPrediction = model.predict(x[200:])

        mae_model: float = float(np.mean(np.abs(pred.mean - y[200:])))
        mae_naive: float = float(np.mean(np.abs(y[200:])))  # predict zero

        # Model should not do dramatically better than naive on pure noise
        assert mae_model > 0.6 * mae_naive, f"GBM MAE={mae_model:.4f} << naive MAE={mae_naive:.4f} on pure noise"

    def test_gb_on_linear_data_reasonable_mae(
        self,
        linear_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        gb_config: GradientBoostingConfig,
    ) -> None:
        """On linear data, the median quantile should produce reasonable point predictions."""
        x, y = linear_data
        model: GradientBoostingRegressor = GradientBoostingRegressor(gb_config)
        model.fit(x[:150], y[:150])

        pred: PointPrediction = model.predict(x[150:])

        mae: float = float(np.mean(np.abs(pred.mean - y[150:])))
        # Linear data with noise_std=0.1 — GBM should get < 1.0 MAE
        assert mae < 1.0, f"GBM MAE={mae:.4f} too high on linear data"

    def test_gb_quantile_coverage_calibration(self) -> None:
        """The (0.05, 0.95) quantile pair should cover ~90% of actuals on well-specified data."""
        rng: np.random.Generator = np.random.default_rng(42)
        n: int = 500
        n_features: int = 5
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = rng.standard_normal((n, n_features)).astype(np.float64)
        w: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(n_features).astype(np.float64)
        y: np.ndarray[tuple[int], np.dtype[np.float64]] = (x @ w + rng.normal(0, 0.5, n)).astype(np.float64)

        config: GradientBoostingConfig = make_gb_config(n_estimators=50)
        model: GradientBoostingRegressor = GradientBoostingRegressor(config)
        model.fit(x[:350], y[:350])

        qpred: QuantilePrediction = model.predict_quantiles(x[350:])
        y_test: np.ndarray[tuple[int], np.dtype[np.float64]] = y[350:]

        # Find q_0.05 and q_0.95 columns
        q05_idx: int = 0  # first quantile = 0.05
        q95_idx: int = 4  # last quantile = 0.95

        inside: np.ndarray[tuple[int], np.dtype[np.bool_]] = (y_test >= qpred.values[:, q05_idx]) & (
            y_test <= qpred.values[:, q95_idx]
        )
        coverage: float = float(np.mean(inside))

        # On well-specified linear data, 90% PI should cover 70-100% of actuals
        # (allowing margin for finite sample + model approximation)
        assert coverage > 0.65, f"90% PI coverage={coverage:.4f} is too low"

    def test_gb_fallback_std_without_q25_q75(self) -> None:
        """When quantiles lack 0.25/0.75, predict() falls back to full-range spread."""
        rng: np.random.Generator = np.random.default_rng(42)
        n: int = 200
        n_features: int = 5
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = rng.standard_normal((n, n_features)).astype(np.float64)
        y: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(n).astype(np.float64)

        # Config with only 3 quantiles (no 0.25/0.75)
        config: GradientBoostingConfig = make_gb_config(quantiles=(0.10, 0.50, 0.90))
        model: GradientBoostingRegressor = GradientBoostingRegressor(config)
        model.fit(x[:150], y[:150])

        pred: PointPrediction = model.predict(x[150:])

        # Should produce valid output via fallback path
        assert pred.mean.shape == (50,)
        assert pred.std.shape == (50,)
        assert np.all(pred.std >= 0), "Fallback std should be non-negative"

    def test_gb_no_median_quantile_raises(self) -> None:
        """Config without 0.50 in quantiles causes predict() to raise ValueError."""
        config: GradientBoostingConfig = make_gb_config(quantiles=(0.05, 0.25, 0.75, 0.95))
        rng: np.random.Generator = np.random.default_rng(42)
        n: int = 100
        n_features: int = 5
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = rng.standard_normal((n, n_features)).astype(np.float64)
        y: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(n).astype(np.float64)

        model: GradientBoostingRegressor = GradientBoostingRegressor(config)
        model.fit(x[:80], y[:80])

        with pytest.raises(ValueError, match="must include 0.50"):
            model.predict(x[80:])
