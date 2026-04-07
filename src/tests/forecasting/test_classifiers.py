"""Unit tests for direction classification models (Logistic, RandomForest, LightGBM, GRU).

Tests fit/predict behaviour, calibration quality, direction values,
horizon propagation, error handling, and model-specific features
(feature importances, Platt scaling, MC Dropout).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.app.forecasting.application.gradient_boosting_clf import GradientBoostingClassifier
from src.app.forecasting.application.gru_classifier import GRUClassifier
from src.app.forecasting.application.logistic_baseline import LogisticBaseline
from src.app.forecasting.application.random_forest_clf import RandomForestClassifier
from src.app.forecasting.domain.value_objects import (
    DirectionForecast,
    ForecastHorizon,
    GradientBoostingClassifierConfig,
    GRUClassifierConfig,
    LogisticConfig,
    RandomForestClassifierConfig,
)

from src.tests.forecasting.conftest import (
    make_gb_clf_config,
    make_gru_clf_config,
    make_logistic_config,
    make_rf_clf_config,
)


# ---------------------------------------------------------------------------
# Helper — linearly separable data
# ---------------------------------------------------------------------------


def _make_separable_data(
    n: int = 200,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    """Generate linearly separable data with a wide margin for perfect classification.

    The first feature determines the class with a large gap (no overlap).

    Args:
        n: Number of samples.
        n_features: Number of features.
        seed: Random seed.

    Returns:
        Tuple of (X, y) where y in {-1, +1} and the problem is trivially separable.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = rng.standard_normal((n, n_features)).astype(np.float64)

    # Class +1: shift first feature to +3.0; class -1: shift to -3.0
    half: int = n // 2
    x[:half, 0] += 3.0
    x[half:, 0] -= 3.0

    y: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(n, dtype=np.float64)
    y[:half] = 1.0
    y[half:] = -1.0

    return x, y


# ===========================================================================
# Logistic Baseline Tests
# ===========================================================================


class TestLogisticBaseline:
    """Tests for LogisticBaseline fit/predict behaviour."""

    def test_logistic_on_separable_data_near_perfect(self) -> None:
        """On linearly separable data, logistic should achieve near-perfect accuracy."""
        x, y = _make_separable_data()
        n_train: int = 150
        config: LogisticConfig = make_logistic_config()
        model: LogisticBaseline = LogisticBaseline(config, ForecastHorizon.H1)
        model.fit(x[:n_train], y[:n_train])

        forecasts: list[DirectionForecast] = model.predict(x[n_train:])
        preds: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(
            [f.predicted_direction for f in forecasts], dtype=np.float64
        )
        acc: float = float(np.mean(preds == y[n_train:]))

        assert acc > 0.95, f"Logistic accuracy={acc:.4f} should be > 0.95 on separable data"

    def test_logistic_fit_predict_smoke(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        logistic_config: LogisticConfig,
    ) -> None:
        """Fit and predict produces correct number of DirectionForecast objects."""
        x, y = classification_data
        n_train: int = 200
        model: LogisticBaseline = LogisticBaseline(logistic_config, ForecastHorizon.H4)
        model.fit(x[:n_train], y[:n_train])

        n_test: int = x.shape[0] - n_train
        forecasts: list[DirectionForecast] = model.predict(x[n_train:])

        assert len(forecasts) == n_test

    def test_logistic_confidence_range(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        logistic_config: LogisticConfig,
    ) -> None:
        """All predicted confidences are in [0.5, 1.0] (max-class probability)."""
        x, y = classification_data
        model: LogisticBaseline = LogisticBaseline(logistic_config, ForecastHorizon.H1)
        model.fit(x[:200], y[:200])

        forecasts: list[DirectionForecast] = model.predict(x[200:])

        for f in forecasts:
            assert 0.0 <= f.confidence <= 1.0, f"Confidence {f.confidence} out of [0, 1]"
            assert f.confidence >= 0.5, f"Max-class confidence {f.confidence} should be >= 0.5"

    def test_logistic_direction_values(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        logistic_config: LogisticConfig,
    ) -> None:
        """All predicted directions are +1 or -1."""
        x, y = classification_data
        model: LogisticBaseline = LogisticBaseline(logistic_config, ForecastHorizon.H1)
        model.fit(x[:200], y[:200])

        forecasts: list[DirectionForecast] = model.predict(x[200:])

        for f in forecasts:
            assert f.predicted_direction in {1, -1}, f"Direction {f.predicted_direction} not in {{+1, -1}}"

    def test_logistic_horizon_propagation(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        logistic_config: LogisticConfig,
    ) -> None:
        """ForecastHorizon is correctly propagated to every forecast."""
        x, y = classification_data
        target_horizon: ForecastHorizon = ForecastHorizon.H24
        model: LogisticBaseline = LogisticBaseline(logistic_config, target_horizon)
        model.fit(x[:200], y[:200])

        forecasts: list[DirectionForecast] = model.predict(x[200:])

        for f in forecasts:
            assert f.horizon == target_horizon

    def test_logistic_predict_before_fit_raises(
        self,
        logistic_config: LogisticConfig,
    ) -> None:
        """Calling predict before fit raises RuntimeError."""
        model: LogisticBaseline = LogisticBaseline(logistic_config, ForecastHorizon.H1)
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((10, 5), dtype=np.float64)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(x_test)

    def test_logistic_empty_fit_raises(
        self,
        logistic_config: LogisticConfig,
    ) -> None:
        """Empty input arrays raise ValueError on fit."""
        model: LogisticBaseline = LogisticBaseline(logistic_config, ForecastHorizon.H1)
        x_empty: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty((0, 5), dtype=np.float64)
        y_empty: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(0, dtype=np.float64)

        with pytest.raises(ValueError, match="at least one sample"):
            model.fit(x_empty, y_empty)

    def test_logistic_empty_predict_raises(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        logistic_config: LogisticConfig,
    ) -> None:
        """Empty test array raises ValueError on predict."""
        x, y = classification_data
        model: LogisticBaseline = LogisticBaseline(logistic_config, ForecastHorizon.H1)
        model.fit(x[:200], y[:200])

        x_empty: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty((0, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="at least one sample"):
            model.predict(x_empty)


# ===========================================================================
# Random Forest Classifier Tests
# ===========================================================================


class TestRandomForestClassifier:
    """Tests for RandomForestClassifier fit/predict behaviour."""

    def test_rf_on_separable_data_near_perfect(self) -> None:
        """On linearly separable data, RF should achieve near-perfect accuracy."""
        x, y = _make_separable_data()
        n_train: int = 150
        config: RandomForestClassifierConfig = make_rf_clf_config()
        model: RandomForestClassifier = RandomForestClassifier(config, ForecastHorizon.H1)
        model.fit(x[:n_train], y[:n_train])

        forecasts: list[DirectionForecast] = model.predict(x[n_train:])
        preds: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(
            [f.predicted_direction for f in forecasts], dtype=np.float64
        )
        acc: float = float(np.mean(preds == y[n_train:]))

        assert acc > 0.95, f"RF accuracy={acc:.4f} should be > 0.95 on separable data"

    def test_rf_fit_predict_smoke(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        rf_clf_config: RandomForestClassifierConfig,
    ) -> None:
        """Fit and predict produces correct number of DirectionForecast objects."""
        x, y = classification_data
        n_train: int = 200
        model: RandomForestClassifier = RandomForestClassifier(rf_clf_config, ForecastHorizon.H4)
        model.fit(x[:n_train], y[:n_train])

        n_test: int = x.shape[0] - n_train
        forecasts: list[DirectionForecast] = model.predict(x[n_train:])

        assert len(forecasts) == n_test

    def test_rf_confidence_range(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        rf_clf_config: RandomForestClassifierConfig,
    ) -> None:
        """All predicted confidences are in [0.5, 1.0]."""
        x, y = classification_data
        model: RandomForestClassifier = RandomForestClassifier(rf_clf_config, ForecastHorizon.H1)
        model.fit(x[:200], y[:200])

        forecasts: list[DirectionForecast] = model.predict(x[200:])

        for f in forecasts:
            assert 0.0 <= f.confidence <= 1.0, f"Confidence {f.confidence} out of [0, 1]"
            assert f.confidence >= 0.5, f"Max-class confidence {f.confidence} should be >= 0.5"

    def test_rf_direction_values(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        rf_clf_config: RandomForestClassifierConfig,
    ) -> None:
        """All predicted directions are +1 or -1."""
        x, y = classification_data
        model: RandomForestClassifier = RandomForestClassifier(rf_clf_config, ForecastHorizon.H1)
        model.fit(x[:200], y[:200])

        forecasts: list[DirectionForecast] = model.predict(x[200:])

        for f in forecasts:
            assert f.predicted_direction in {1, -1}

    def test_rf_horizon_propagation(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        rf_clf_config: RandomForestClassifierConfig,
    ) -> None:
        """ForecastHorizon is correctly propagated to every forecast."""
        x, y = classification_data
        target_horizon: ForecastHorizon = ForecastHorizon.H24
        model: RandomForestClassifier = RandomForestClassifier(rf_clf_config, target_horizon)
        model.fit(x[:200], y[:200])

        forecasts: list[DirectionForecast] = model.predict(x[200:])

        for f in forecasts:
            assert f.horizon == target_horizon

    def test_rf_feature_importances_after_fit(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        rf_clf_config: RandomForestClassifierConfig,
    ) -> None:
        """Feature importances are populated after fitting with correct shape."""
        x, y = classification_data
        n_features: int = x.shape[1]
        model: RandomForestClassifier = RandomForestClassifier(rf_clf_config, ForecastHorizon.H1)

        assert model.feature_importances_ is None, "Should be None before fit"

        model.fit(x[:200], y[:200])

        assert model.feature_importances_ is not None, "Should be set after fit"
        assert model.feature_importances_.shape == (n_features,)
        assert float(np.sum(model.feature_importances_)) == pytest.approx(1.0, abs=1e-6)
        assert np.all(model.feature_importances_ >= 0), "All importances should be non-negative"

    def test_rf_predict_before_fit_raises(
        self,
        rf_clf_config: RandomForestClassifierConfig,
    ) -> None:
        """Calling predict before fit raises RuntimeError."""
        model: RandomForestClassifier = RandomForestClassifier(rf_clf_config, ForecastHorizon.H1)
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((10, 5), dtype=np.float64)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(x_test)

    def test_rf_empty_fit_raises(
        self,
        rf_clf_config: RandomForestClassifierConfig,
    ) -> None:
        """Empty input arrays raise ValueError on fit."""
        model: RandomForestClassifier = RandomForestClassifier(rf_clf_config, ForecastHorizon.H1)
        x_empty: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty((0, 5), dtype=np.float64)
        y_empty: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(0, dtype=np.float64)

        with pytest.raises(ValueError, match="at least one sample"):
            model.fit(x_empty, y_empty)

    def test_rf_empty_predict_raises(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        rf_clf_config: RandomForestClassifierConfig,
    ) -> None:
        """Empty test array raises ValueError on predict."""
        x, y = classification_data
        model: RandomForestClassifier = RandomForestClassifier(rf_clf_config, ForecastHorizon.H1)
        model.fit(x[:200], y[:200])

        x_empty: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty((0, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="at least one sample"):
            model.predict(x_empty)


# ===========================================================================
# Gradient Boosting Classifier Tests
# ===========================================================================


class TestGradientBoostingClassifier:
    """Tests for GradientBoostingClassifier (LightGBM + Platt scaling)."""

    def test_gb_clf_on_separable_data_near_perfect(self) -> None:
        """On linearly separable data, LightGBM classifier should achieve near-perfect accuracy."""
        x, y = _make_separable_data(n=300)
        n_train: int = 200
        config: GradientBoostingClassifierConfig = make_gb_clf_config()
        model: GradientBoostingClassifier = GradientBoostingClassifier(config, ForecastHorizon.H1)
        model.fit(x[:n_train], y[:n_train])

        forecasts: list[DirectionForecast] = model.predict(x[n_train:])
        preds: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(
            [f.predicted_direction for f in forecasts], dtype=np.float64
        )
        acc: float = float(np.mean(preds == y[n_train:]))

        assert acc > 0.95, f"LGBM clf accuracy={acc:.4f} should be > 0.95 on separable data"

    def test_gb_clf_fit_predict_smoke(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        gb_clf_config: GradientBoostingClassifierConfig,
    ) -> None:
        """Fit and predict produces correct number of DirectionForecast objects."""
        x, y = classification_data
        n_train: int = 200
        model: GradientBoostingClassifier = GradientBoostingClassifier(gb_clf_config, ForecastHorizon.H4)
        model.fit(x[:n_train], y[:n_train])

        n_test: int = x.shape[0] - n_train
        forecasts: list[DirectionForecast] = model.predict(x[n_train:])

        assert len(forecasts) == n_test

    def test_gb_clf_confidence_range(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        gb_clf_config: GradientBoostingClassifierConfig,
    ) -> None:
        """All predicted confidences are in [0.5, 1.0]."""
        x, y = classification_data
        model: GradientBoostingClassifier = GradientBoostingClassifier(gb_clf_config, ForecastHorizon.H1)
        model.fit(x[:200], y[:200])

        forecasts: list[DirectionForecast] = model.predict(x[200:])

        for f in forecasts:
            assert 0.0 <= f.confidence <= 1.0, f"Confidence {f.confidence} out of [0, 1]"
            assert f.confidence >= 0.5, f"Max-class confidence {f.confidence} should be >= 0.5"

    def test_gb_clf_direction_values(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        gb_clf_config: GradientBoostingClassifierConfig,
    ) -> None:
        """All predicted directions are +1 or -1."""
        x, y = classification_data
        model: GradientBoostingClassifier = GradientBoostingClassifier(gb_clf_config, ForecastHorizon.H1)
        model.fit(x[:200], y[:200])

        forecasts: list[DirectionForecast] = model.predict(x[200:])

        for f in forecasts:
            assert f.predicted_direction in {1, -1}

    def test_gb_clf_horizon_propagation(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        gb_clf_config: GradientBoostingClassifierConfig,
    ) -> None:
        """ForecastHorizon is correctly propagated to every forecast."""
        x, y = classification_data
        target_horizon: ForecastHorizon = ForecastHorizon.H24
        model: GradientBoostingClassifier = GradientBoostingClassifier(gb_clf_config, target_horizon)
        model.fit(x[:200], y[:200])

        forecasts: list[DirectionForecast] = model.predict(x[200:])

        for f in forecasts:
            assert f.horizon == target_horizon

    def test_gb_clf_uses_calibrated_classifier_cv(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        gb_clf_config: GradientBoostingClassifierConfig,
    ) -> None:
        """After fitting, the calibrated model is a CalibratedClassifierCV instance."""
        from sklearn.calibration import CalibratedClassifierCV

        x, y = classification_data
        model: GradientBoostingClassifier = GradientBoostingClassifier(gb_clf_config, ForecastHorizon.H1)
        model.fit(x[:200], y[:200])

        assert model._calibrated_model is not None
        assert isinstance(model._calibrated_model, CalibratedClassifierCV)

    def test_gb_clf_predict_before_fit_raises(
        self,
        gb_clf_config: GradientBoostingClassifierConfig,
    ) -> None:
        """Calling predict before fit raises RuntimeError."""
        model: GradientBoostingClassifier = GradientBoostingClassifier(gb_clf_config, ForecastHorizon.H1)
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((10, 5), dtype=np.float64)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(x_test)

    def test_gb_clf_empty_fit_raises(
        self,
        gb_clf_config: GradientBoostingClassifierConfig,
    ) -> None:
        """Empty input arrays raise ValueError on fit."""
        model: GradientBoostingClassifier = GradientBoostingClassifier(gb_clf_config, ForecastHorizon.H1)
        x_empty: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty((0, 5), dtype=np.float64)
        y_empty: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(0, dtype=np.float64)

        with pytest.raises(ValueError, match="at least one sample"):
            model.fit(x_empty, y_empty)

    def test_gb_clf_empty_predict_raises(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        gb_clf_config: GradientBoostingClassifierConfig,
    ) -> None:
        """Empty test array raises ValueError on predict."""
        x, y = classification_data
        model: GradientBoostingClassifier = GradientBoostingClassifier(gb_clf_config, ForecastHorizon.H1)
        model.fit(x[:200], y[:200])

        x_empty: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty((0, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="at least one sample"):
            model.predict(x_empty)


# ===========================================================================
# GRU Classifier Tests (Negative-Result Experiment)
# ===========================================================================


class TestGRUClassifier:
    """Tests for GRUClassifier (GRU + MC Dropout for direction classification)."""

    def test_gru_clf_loss_decreases(self) -> None:
        """On separable data, GRU classifier should learn better than chance."""
        x, y = _make_separable_data(n=200, n_features=5, seed=42)
        n_train: int = 150

        config: GRUClassifierConfig = make_gru_clf_config(n_epochs=20, patience=15)
        model: GRUClassifier = GRUClassifier(config, ForecastHorizon.H1)
        model.fit(x[:n_train], y[:n_train])

        forecasts: list[DirectionForecast] = model.predict(x[n_train:])
        preds: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(
            [f.predicted_direction for f in forecasts],
            dtype=np.float64,
        )

        # Align targets: predict returns n_test - sequence_length forecasts
        seq_len: int = config.sequence_length
        y_test_aligned: np.ndarray[tuple[int], np.dtype[np.float64]] = y[n_train + seq_len :]
        acc: float = float(np.mean(preds == y_test_aligned))

        # Should be better than 50% on well-separated data
        assert acc > 0.5, f"GRU clf accuracy={acc:.4f} should be > 0.5 on separable data"

    def test_gru_clf_fit_predict_smoke(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        gru_clf_config: GRUClassifierConfig,
    ) -> None:
        """Fit and predict produces correct number of DirectionForecast objects."""
        x, y = classification_data
        n_train: int = 200
        model: GRUClassifier = GRUClassifier(gru_clf_config, ForecastHorizon.H4)
        model.fit(x[:n_train], y[:n_train])

        n_test: int = x.shape[0] - n_train
        seq_len: int = gru_clf_config.sequence_length
        expected_forecasts: int = n_test - seq_len

        forecasts: list[DirectionForecast] = model.predict(x[n_train:])

        assert len(forecasts) == expected_forecasts

    def test_gru_clf_confidence_range(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        gru_clf_config: GRUClassifierConfig,
    ) -> None:
        """All predicted confidences are in [0.5, 1.0]."""
        x, y = classification_data
        model: GRUClassifier = GRUClassifier(gru_clf_config, ForecastHorizon.H1)
        model.fit(x[:200], y[:200])

        forecasts: list[DirectionForecast] = model.predict(x[200:])

        for f in forecasts:
            assert 0.5 <= f.confidence <= 1.0, f"Confidence {f.confidence} out of [0.5, 1.0]"

    def test_gru_clf_direction_values(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        gru_clf_config: GRUClassifierConfig,
    ) -> None:
        """All predicted directions are +1 or -1."""
        x, y = classification_data
        model: GRUClassifier = GRUClassifier(gru_clf_config, ForecastHorizon.H1)
        model.fit(x[:200], y[:200])

        forecasts: list[DirectionForecast] = model.predict(x[200:])

        for f in forecasts:
            assert f.predicted_direction in {1, -1}, f"Direction {f.predicted_direction} not in {{+1, -1}}"

    def test_gru_clf_horizon_propagation(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        gru_clf_config: GRUClassifierConfig,
    ) -> None:
        """ForecastHorizon is correctly propagated to every forecast."""
        x, y = classification_data
        target_horizon: ForecastHorizon = ForecastHorizon.H24
        model: GRUClassifier = GRUClassifier(gru_clf_config, target_horizon)
        model.fit(x[:200], y[:200])

        forecasts: list[DirectionForecast] = model.predict(x[200:])

        for f in forecasts:
            assert f.horizon == target_horizon

    def test_gru_clf_predict_before_fit_raises(
        self,
        gru_clf_config: GRUClassifierConfig,
    ) -> None:
        """Calling predict before fit raises RuntimeError."""
        model: GRUClassifier = GRUClassifier(gru_clf_config, ForecastHorizon.H1)
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.ones((20, 5), dtype=np.float64)

        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(x_test)

    def test_gru_clf_empty_fit_raises(
        self,
        gru_clf_config: GRUClassifierConfig,
    ) -> None:
        """Empty input arrays raise ValueError on fit."""
        model: GRUClassifier = GRUClassifier(gru_clf_config, ForecastHorizon.H1)
        x_empty: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty((0, 5), dtype=np.float64)
        y_empty: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(0, dtype=np.float64)

        with pytest.raises(ValueError, match="at least one sample"):
            model.fit(x_empty, y_empty)

    def test_gru_clf_empty_predict_raises(
        self,
        classification_data: tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        gru_clf_config: GRUClassifierConfig,
    ) -> None:
        """Empty test array raises ValueError on predict."""
        x, y = classification_data
        model: GRUClassifier = GRUClassifier(gru_clf_config, ForecastHorizon.H1)
        model.fit(x[:200], y[:200])

        x_empty: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty((0, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="at least one sample"):
            model.predict(x_empty)

    def test_gru_clf_mc_dropout_produces_variation(self) -> None:
        """With high dropout, MC Dropout should produce varying probabilities across forward passes."""
        x, y = _make_separable_data(n=200, n_features=5, seed=42)

        config: GRUClassifierConfig = make_gru_clf_config(
            dropout=0.5,
            mc_samples=20,
            n_epochs=10,
            patience=8,
        )
        model: GRUClassifier = GRUClassifier(config, ForecastHorizon.H1)
        model.fit(x[:150], y[:150])

        # Access the internal model to check MC Dropout variation directly
        assert model._model is not None

        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] = x[150:]
        x_scaled: np.ndarray[tuple[int, int], np.dtype[np.float64]] = model._scaler.transform(x_test).astype(
            np.float64,
        )
        dummy_y: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(x_test.shape[0], dtype=np.float64)

        import torch

        seq_len: int = config.sequence_length
        x_seq: np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
        x_seq, _ = GRUClassifier._make_sequences(x_scaled, dummy_y, seq_len)
        x_tensor: torch.Tensor = torch.from_numpy(x_seq).float().to(model._device)

        # Run multiple MC forward passes
        model._model.train()
        mc_results: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []
        with torch.no_grad():
            for _ in range(20):
                probs: torch.Tensor = model._model(x_tensor)
                mc_results.append(probs.cpu().numpy().astype(np.float64))

        stacked: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.stack(mc_results, axis=0)
        mc_std: np.ndarray[tuple[int], np.dtype[np.float64]] = np.std(stacked, axis=0).astype(np.float64)

        # At least some samples should have non-zero std from MC Dropout
        assert np.any(mc_std > 0), "MC Dropout should produce non-zero std for at least some samples"
