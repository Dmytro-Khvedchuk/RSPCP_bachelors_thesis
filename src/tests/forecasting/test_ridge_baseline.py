"""Unit tests for the RidgeBaseline return magnitude regressor.

Tests the Ridge/Huber linear baseline against known linear data,
verifying prediction quality, output shapes, error handling, and
behaviour on pure-noise targets.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.app.forecasting.application.ridge_baseline import RidgeBaseline
from src.app.forecasting.domain.value_objects import PointPrediction, RidgeConfig

from src.tests.forecasting.conftest import make_ridge_config


class TestRidgeBaseline:
    """Tests for RidgeBaseline fit/predict behaviour."""

    def test_ridge_on_linear_data_low_mae(
        self,
        linear_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        ridge_config: RidgeConfig,
    ) -> None:
        """On linear data with noise_std=0.1, MAE should be close to the noise level."""
        x, y = linear_data
        n_train: int = 150
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[:n_train]
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]] = y[:n_train]
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[n_train:]
        y_test: np.ndarray[tuple[int], np.dtype[np.float64]] = y[n_train:]

        model: RidgeBaseline = RidgeBaseline(ridge_config)
        model.fit(x_train, y_train)
        pred: PointPrediction = model.predict(x_test)

        mae: float = float(np.mean(np.abs(pred.mean - y_test)))
        # Noise std is 0.1, so MAE should be well below 0.5
        assert mae < 0.5, f"MAE={mae:.4f} is too high for linear data with noise=0.1"

    def test_ridge_huber_on_linear_data(
        self,
        linear_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
    ) -> None:
        """Huber variant fits and predicts without error on linear data."""
        x, y = linear_data
        config: RidgeConfig = make_ridge_config(use_huber=True)
        model: RidgeBaseline = RidgeBaseline(config)
        model.fit(x[:150], y[:150])
        pred: PointPrediction = model.predict(x[150:])

        mae: float = float(np.mean(np.abs(pred.mean - y[150:])))
        assert mae < 0.5, f"Huber MAE={mae:.4f} is too high"

    def test_ridge_predict_shape(
        self,
        linear_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        ridge_config: RidgeConfig,
    ) -> None:
        """Output mean and std arrays match the number of test samples."""
        x, y = linear_data
        model: RidgeBaseline = RidgeBaseline(ridge_config)
        model.fit(x[:150], y[:150])

        n_test: int = 50
        pred: PointPrediction = model.predict(x[150:])

        assert pred.mean.shape == (n_test,)
        assert pred.std.shape == (n_test,)

    def test_ridge_residual_std_positive(
        self,
        linear_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        ridge_config: RidgeConfig,
    ) -> None:
        """Residual std should be positive on noisy data."""
        x, y = linear_data
        model: RidgeBaseline = RidgeBaseline(ridge_config)
        model.fit(x[:150], y[:150])
        pred: PointPrediction = model.predict(x[150:])

        # All std values should be the same (homoscedastic) and positive
        assert np.all(pred.std > 0), "Residual std should be positive"

    def test_ridge_predict_before_fit_raises(
        self,
        ridge_config: RidgeConfig,
    ) -> None:
        """Calling predict() before fit() raises RuntimeError."""
        model: RidgeBaseline = RidgeBaseline(ridge_config)
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((10, 5), dtype=np.float64)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(x_test)

    def test_ridge_empty_input_raises(
        self,
        ridge_config: RidgeConfig,
    ) -> None:
        """Empty input arrays raise ValueError on both fit and predict."""
        model: RidgeBaseline = RidgeBaseline(ridge_config)

        x_empty: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty((0, 5), dtype=np.float64)
        y_empty: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(0, dtype=np.float64)

        with pytest.raises(ValueError, match="at least one sample"):
            model.fit(x_empty, y_empty)

    def test_ridge_empty_predict_raises(
        self,
        linear_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        ridge_config: RidgeConfig,
    ) -> None:
        """Empty test array raises ValueError on predict."""
        x, y = linear_data
        model: RidgeBaseline = RidgeBaseline(ridge_config)
        model.fit(x[:150], y[:150])

        x_empty: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty((0, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="at least one sample"):
            model.predict(x_empty)

    def test_ridge_on_noise_no_improvement(self) -> None:
        """On pure noise targets, MAE should be close to unconditional std (no significant improvement)."""
        rng: np.random.Generator = np.random.default_rng(42)
        n: int = 200
        n_features: int = 5
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = rng.standard_normal((n, n_features)).astype(np.float64)
        # Pure noise target -- no signal
        y: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(n).astype(np.float64)

        config: RidgeConfig = make_ridge_config()
        model: RidgeBaseline = RidgeBaseline(config)
        model.fit(x[:150], y[:150])
        pred: PointPrediction = model.predict(x[150:])

        mae_model: float = float(np.mean(np.abs(pred.mean - y[150:])))
        mae_naive: float = float(np.mean(np.abs(y[150:])))  # predict zero

        # Model should not do dramatically better than naive on pure noise
        # Allow 30% improvement since regularised model can sometimes edge out
        assert mae_model > 0.7 * mae_naive, f"Model MAE={mae_model:.4f} << naive MAE={mae_naive:.4f} on pure noise"
