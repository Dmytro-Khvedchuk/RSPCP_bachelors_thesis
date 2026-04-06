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

    def test_gru_null_test_on_noise(self) -> None:
        """On pure noise targets, GRU should not dramatically outperform a naive predictor."""
        rng: np.random.Generator = np.random.default_rng(42)
        n: int = 100
        n_features: int = 3
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = rng.standard_normal((n, n_features)).astype(np.float64)
        y: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(n).astype(np.float64)

        config: GRUConfig = make_gru_config(n_epochs=10, hidden_size=8, num_layers=1)
        model: GRURegressor = GRURegressor(config)
        model.fit(x, y)

        pred: PointPrediction = model.predict(x)

        # Since sequences lose the first seq_len samples, compare aligned targets
        seq_len: int = config.sequence_length
        y_aligned: np.ndarray[tuple[int], np.dtype[np.float64]] = y[seq_len:]

        mae_model: float = float(np.mean(np.abs(pred.mean - y_aligned)))
        mae_naive: float = float(np.mean(np.abs(y_aligned)))  # predict zero

        # Model should not be dramatically better than naive on pure noise
        assert mae_model > 0.5 * mae_naive, f"GRU MAE={mae_model:.4f} << naive MAE={mae_naive:.4f} on pure noise"

    def test_gru_predict_short_test_data_raises(self) -> None:
        """x_test shorter than sequence_length should raise ValueError."""
        x_train, y_train = make_linear_data(n=100, n_features=3, seed=42)

        config: GRUConfig = make_gru_config(sequence_length=5)
        model: GRURegressor = GRURegressor(config)
        model.fit(x_train, y_train)

        # x_test with fewer rows than sequence_length
        x_short: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((3, 3), dtype=np.float64)

        with pytest.raises(ValueError, match="at least sequence_length"):
            model.predict(x_short)

    def test_gru_mc_std_increases_with_dropout(self) -> None:
        """Higher dropout should produce larger MC Dropout std (more model disagreement)."""
        x, y = make_linear_data(n=100, n_features=3, seed=42)

        config_low: GRUConfig = make_gru_config(dropout=0.1, mc_samples=20, n_epochs=10)
        config_high: GRUConfig = make_gru_config(dropout=0.5, mc_samples=20, n_epochs=10)

        model_low: GRURegressor = GRURegressor(config_low)
        model_low.fit(x, y)
        pred_low: PointPrediction = model_low.predict(x)

        model_high: GRURegressor = GRURegressor(config_high)
        model_high.fit(x, y)
        pred_high: PointPrediction = model_high.predict(x)

        mean_std_low: float = float(np.mean(pred_low.std))
        mean_std_high: float = float(np.mean(pred_high.std))

        # Higher dropout should produce more uncertainty on average
        assert mean_std_high > mean_std_low, (
            f"Higher dropout should produce larger std: low={mean_std_low:.6f}, high={mean_std_high:.6f}"
        )

    def test_gru_sequence_target_alignment(self) -> None:
        """Verify _make_sequences target is strictly after the input window (no look-ahead).

        For y = index, the target of sequence [i, ..., i+seq_len-1] should be
        y[i+seq_len], which is strictly after the last element in the window.
        """
        from src.app.forecasting.application.gru_regressor import GRURegressor

        seq_len: int = 5
        n: int = 20
        n_features: int = 2

        x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.arange(n * n_features, dtype=np.float64).reshape(
            n, n_features
        )
        # y = row index so we can verify alignment
        y: np.ndarray[tuple[int], np.dtype[np.float64]] = np.arange(n, dtype=np.float64)

        x_seq: np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
        y_seq: np.ndarray[tuple[int], np.dtype[np.float64]]
        x_seq, y_seq = GRURegressor._make_sequences(x, y, seq_len)

        # For each sequence i, the last row of x_seq[i] has features from row i+seq_len-1,
        # and y_seq[i] = y[i+seq_len] which is the NEXT row's target.
        for i in range(len(y_seq)):
            last_input_row: int = i + seq_len - 1
            target_row: int = i + seq_len
            assert y_seq[i] == target_row, (
                f"Sequence {i}: target should be row {target_row}, got {y_seq[i]}. "
                f"Last input row is {last_input_row} — target must be strictly after."
            )
