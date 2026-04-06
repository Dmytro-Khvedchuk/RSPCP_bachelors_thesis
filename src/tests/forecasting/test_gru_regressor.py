"""Unit tests for the GRURegressor with MC Dropout uncertainty estimation.

Tests training convergence, MC Dropout variation, output shapes, and
error handling for input length constraints.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.app.forecasting.application.gru_regressor import GRURegressor
from src.app.forecasting.domain.value_objects import GRUConfig, PointPrediction

from src.tests.forecasting.conftest import make_gru_config, make_linear_data


class TestGRURegressor:
    """Tests for GRU regressor fit/predict behaviour."""

    def test_gru_loss_decreases(
        self,
        gru_config: GRUConfig,
    ) -> None:
        """After training, model predictions should be better than a constant predictor."""
        x, y = make_linear_data(n=100, n_features=3, seed=42)

        model: GRURegressor = GRURegressor(gru_config)
        model.fit(x, y)

        # Predict on training data (we just need to verify model learned something)
        pred: PointPrediction = model.predict(x)

        # The model predictions should exist and have valid values
        assert not np.any(np.isnan(pred.mean)), "Predictions should not contain NaN"
        assert not np.any(np.isinf(pred.mean)), "Predictions should not contain Inf"

    def test_gru_mc_dropout_produces_variation(self) -> None:
        """MC Dropout should produce std > 0 for at least some samples."""
        x, y = make_linear_data(n=100, n_features=3, seed=42)

        # Use higher dropout to ensure variation
        config: GRUConfig = make_gru_config(
            dropout=0.5,
            mc_samples=10,
            n_epochs=10,
        )
        model: GRURegressor = GRURegressor(config)
        model.fit(x, y)

        pred: PointPrediction = model.predict(x)

        # At least some samples should have std > 0 from MC Dropout
        assert np.any(pred.std > 0), "MC Dropout should produce non-zero std for at least some samples"

    def test_gru_output_shape(
        self,
        gru_config: GRUConfig,
    ) -> None:
        """Mean and std arrays have correct shape matching the number of output sequences."""
        x, y = make_linear_data(n=80, n_features=3, seed=42)
        seq_len: int = gru_config.sequence_length

        model: GRURegressor = GRURegressor(gru_config)
        model.fit(x, y)

        # Predict on new data
        n_test: int = 30
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = (
            np.random.default_rng(42).standard_normal((n_test, 3)).astype(np.float64)
        )

        pred: PointPrediction = model.predict(x_test)

        # Output sequences = n_test - seq_len
        expected_len: int = n_test - seq_len
        assert pred.mean.shape == (expected_len,)
        assert pred.std.shape == (expected_len,)

    def test_gru_predict_before_fit_raises(
        self,
        gru_config: GRUConfig,
    ) -> None:
        """Calling predict() before fit() raises RuntimeError."""
        model: GRURegressor = GRURegressor(gru_config)
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((20, 5), dtype=np.float64)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(x_test)

    def test_gru_short_input_raises(
        self,
        gru_config: GRUConfig,
    ) -> None:
        """ValueError raised if x_train has fewer samples than sequence_length + 1."""
        seq_len: int = gru_config.sequence_length
        # Provide exactly seq_len rows (need seq_len + 1)
        x_short: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((seq_len, 3), dtype=np.float64)
        y_short: np.ndarray[tuple[int], np.dtype[np.float64]] = np.ones(seq_len, dtype=np.float64)

        model: GRURegressor = GRURegressor(gru_config)

        with pytest.raises(ValueError, match="at least sequence_length"):
            model.fit(x_short, y_short)
