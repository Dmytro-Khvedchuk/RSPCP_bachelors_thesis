"""Forecasting domain protocols — structural interfaces for regression, classification, and volatility models."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.app.forecasting.domain.value_objects import (
    DirectionForecast,
    PointPrediction,
    QuantilePrediction,
    VolatilityForecast,
)


class IReturnRegressor(Protocol):
    """Structural interface for return magnitude regression models.

    Implementations predict the magnitude of future returns (``fwd_zret_h``).
    Each model must support ``fit`` for training and ``predict`` for
    point-estimate inference.
    """

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Train the model on feature matrix and target vector.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Target vector of shape ``(n_samples,)``.
        """
        ...

    def predict(
        self,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> PointPrediction:
        """Generate point predictions with uncertainty estimates.

        Args:
            x_test: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Point prediction with mean and standard deviation arrays.
        """
        ...


class IQuantileRegressor(Protocol):
    """Structural interface for quantile regression models.

    Extends the return regressor concept with distributional predictions
    at specified quantile levels.
    """

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Train the model on feature matrix and target vector.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Target vector of shape ``(n_samples,)``.
        """
        ...

    def predict(
        self,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> PointPrediction:
        """Generate point predictions with uncertainty estimates.

        The point estimate is the median quantile; uncertainty is derived
        from the interquartile range.

        Args:
            x_test: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Point prediction with mean and standard deviation arrays.
        """
        ...

    def predict_quantiles(
        self,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> QuantilePrediction:
        """Generate predictions at all configured quantile levels.

        Args:
            x_test: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Quantile prediction with values at each quantile level.
        """
        ...


class IVolatilityForecaster(Protocol):
    """Structural interface for volatility forecasting models.

    Implementations predict future realized volatility (``forward_rv_h``).
    """

    def fit(
        self,
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,
    ) -> None:
        """Train the volatility model.

        Args:
            y_train: Realized volatility or return series of shape ``(n_samples,)``.
            x_train: Optional exogenous regressors of shape ``(n_samples, n_features)``.
                Used by HAR-RV (pre-constructed RV regressors); ignored by GARCH.
        """
        ...

    def predict(
        self,
        n_steps: int,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,
    ) -> VolatilityForecast:
        """Forecast volatility for the next ``n_steps`` periods.

        Args:
            n_steps: Number of steps to forecast.
            x_test: Optional exogenous regressors for multi-step ahead forecast.
                Required for HAR-RV; ignored by GARCH.

        Returns:
            Volatility forecast with predicted vol and variance arrays.
        """
        ...


class IDirectionClassifier(Protocol):
    """Structural interface for direction classification models.

    Implementations predict the direction (+1 long / -1 short) of future
    returns.  Each model must support ``fit`` for training and ``predict``
    for inference, returning a list of :class:`DirectionForecast` objects.
    """

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Train the classifier on feature matrix and direction labels.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Direction labels of shape ``(n_samples,)`` with values +1 or -1.
        """
        ...

    def predict(
        self,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> list[DirectionForecast]:
        """Generate direction forecasts with confidence estimates.

        Args:
            x_test: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            List of direction forecasts, one per sample.
        """
        ...
