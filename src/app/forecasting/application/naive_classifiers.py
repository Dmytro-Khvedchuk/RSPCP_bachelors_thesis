"""Naive baseline classifiers for sanity-checking direction forecasters."""

from __future__ import annotations

import numpy as np
from loguru import logger

from src.app.forecasting.domain.value_objects import (
    DirectionForecast,
    ForecastHorizon,
    MajorityConfig,
    MomentumSignConfig,
    PersistenceConfig,
)


class MajorityClassifier:
    """Always predicts the majority class from training data.

    This is the simplest possible baseline: it memorises the most frequent
    direction label in the training set and predicts it for every test sample.
    Any useful model must beat this.

    Attributes:
        config: Majority classifier configuration.
        horizon: Forecast horizon embedded in every ``DirectionForecast``.
    """

    def __init__(self, config: MajorityConfig, horizon: ForecastHorizon) -> None:
        """Initialise the majority-class classifier.

        Args:
            config: Configuration (contains random_seed for interface parity).
            horizon: Forecast horizon to embed in predictions.
        """
        self.config: MajorityConfig = config
        self.horizon: ForecastHorizon = horizon
        self._majority_direction: int | None = None
        self._majority_frequency: float = 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Learn the majority class from training labels.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
                Ignored — included for protocol compatibility.
            y_train: Direction labels of shape ``(n_samples,)`` with values +1 or -1.

        Raises:
            ValueError: If inputs are empty.
        """
        n_samples: int = x_train.shape[0]
        if n_samples == 0:
            msg: str = "x_train must contain at least one sample"
            raise ValueError(msg)

        n_positive: int = int(np.sum(y_train == 1.0))
        n_negative: int = n_samples - n_positive

        if n_positive >= n_negative:
            self._majority_direction = 1
            self._majority_frequency = n_positive / n_samples
        else:
            self._majority_direction = -1
            self._majority_frequency = n_negative / n_samples

        logger.info(
            "MajorityClassifier fitted on {} samples | majority={:+d} | freq={:.4f}",
            n_samples,
            self._majority_direction,
            self._majority_frequency,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> list[DirectionForecast]:
        """Predict the majority class for every test sample.

        Confidence is the majority class frequency from training data.

        Args:
            x_test: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            List of direction forecasts, one per sample.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``x_test`` is empty.
        """
        if self._majority_direction is None:
            msg: str = "Model has not been fitted — call fit() first"
            raise RuntimeError(msg)

        n_test: int = x_test.shape[0]
        if n_test == 0:
            msg = "x_test must contain at least one sample"
            raise ValueError(msg)

        forecast: DirectionForecast = DirectionForecast(
            predicted_direction=self._majority_direction,
            confidence=self._majority_frequency,
            horizon=self.horizon,
        )
        forecasts: list[DirectionForecast] = [forecast] * n_test
        return forecasts


class PersistenceClassifier:
    """Predicts the last observed direction from training data.

    In a classification context, "persistence" means predicting that the
    most recently observed state persists into the future.  Since the test
    feature matrix does not carry temporal ordering, this classifier stores
    the last training label and predicts it for all test samples.

    Attributes:
        config: Persistence classifier configuration.
        horizon: Forecast horizon embedded in every ``DirectionForecast``.
    """

    def __init__(self, config: PersistenceConfig, horizon: ForecastHorizon) -> None:
        """Initialise the persistence classifier.

        Args:
            config: Configuration (contains random_seed for interface parity).
            horizon: Forecast horizon to embed in predictions.
        """
        self.config: PersistenceConfig = config
        self.horizon: ForecastHorizon = horizon
        self._last_direction: int | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Store the last training label as the persistent direction.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
                Ignored — included for protocol compatibility.
            y_train: Direction labels of shape ``(n_samples,)`` with values +1 or -1.

        Raises:
            ValueError: If inputs are empty.
        """
        n_samples: int = x_train.shape[0]
        if n_samples == 0:
            msg: str = "x_train must contain at least one sample"
            raise ValueError(msg)

        self._last_direction = int(y_train[-1])

        logger.info(
            "PersistenceClassifier fitted on {} samples | last_direction={:+d}",
            n_samples,
            self._last_direction,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> list[DirectionForecast]:
        """Predict the last training direction for every test sample.

        Confidence is fixed at 0.5 since this baseline has no probabilistic
        model — it is purely a heuristic.

        Args:
            x_test: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            List of direction forecasts, one per sample.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``x_test`` is empty.
        """
        if self._last_direction is None:
            msg: str = "Model has not been fitted — call fit() first"
            raise RuntimeError(msg)

        n_test: int = x_test.shape[0]
        if n_test == 0:
            msg = "x_test must contain at least one sample"
            raise ValueError(msg)

        _confidence: float = 0.5
        forecast: DirectionForecast = DirectionForecast(
            predicted_direction=self._last_direction,
            confidence=_confidence,
            horizon=self.horizon,
        )
        forecasts: list[DirectionForecast] = [forecast] * n_test
        return forecasts


class MomentumSignClassifier:
    """Predicts the sign of a trailing momentum feature column.

    Reads a single column from the test feature matrix (specified by
    ``config.momentum_col_idx``) and predicts ``+1`` if the value is
    non-negative, ``-1`` otherwise.

    Attributes:
        config: Momentum-sign classifier configuration.
        horizon: Forecast horizon embedded in every ``DirectionForecast``.
    """

    def __init__(self, config: MomentumSignConfig, horizon: ForecastHorizon) -> None:
        """Initialise the momentum-sign classifier.

        Args:
            config: Configuration with ``momentum_col_idx``.
            horizon: Forecast horizon to embed in predictions.
        """
        self.config: MomentumSignConfig = config
        self.horizon: ForecastHorizon = horizon
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],  # noqa: ARG002
    ) -> None:
        """Validate inputs and mark the classifier as fitted.

        No parameters are learned — the momentum column index is fixed at
        construction time.  This method exists for protocol compatibility.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Direction labels of shape ``(n_samples,)`` with values +1 or -1.

        Raises:
            ValueError: If inputs are empty or ``momentum_col_idx`` is out of bounds.
        """
        n_samples: int = x_train.shape[0]
        if n_samples == 0:
            msg: str = "x_train must contain at least one sample"
            raise ValueError(msg)

        n_features: int = x_train.shape[1]
        if self.config.momentum_col_idx >= n_features:
            msg = (
                f"momentum_col_idx={self.config.momentum_col_idx} is out of bounds "
                f"for feature matrix with {n_features} columns"
            )
            raise ValueError(msg)

        self._fitted = True

        logger.info(
            "MomentumSignClassifier fitted on {} samples | momentum_col_idx={}",
            n_samples,
            self.config.momentum_col_idx,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> list[DirectionForecast]:
        """Predict direction as the sign of the momentum column.

        Values >= 0 map to ``+1``, values < 0 map to ``-1``.  Confidence
        is fixed at 0.5 since this is a simple sign-based heuristic.

        Args:
            x_test: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            List of direction forecasts, one per sample.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``x_test`` is empty.
        """
        if not self._fitted:
            msg: str = "Model has not been fitted — call fit() first"
            raise RuntimeError(msg)

        n_test: int = x_test.shape[0]
        if n_test == 0:
            msg = "x_test must contain at least one sample"
            raise ValueError(msg)

        col_idx: int = self.config.momentum_col_idx
        momentum_values: np.ndarray[tuple[int], np.dtype[np.float64]] = x_test[:, col_idx]
        signs: np.ndarray[tuple[int], np.dtype[np.float64]] = np.where(
            momentum_values >= 0,
            1.0,
            -1.0,
        ).astype(np.float64)

        _confidence: float = 0.5
        forecasts: list[DirectionForecast] = [
            DirectionForecast(
                predicted_direction=int(s),
                confidence=_confidence,
                horizon=self.horizon,
            )
            for s in signs
        ]
        return forecasts
