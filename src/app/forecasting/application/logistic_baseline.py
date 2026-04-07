"""Logistic regression baseline for direction classification."""

from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression

from src.app.forecasting.domain.value_objects import (
    DirectionForecast,
    ForecastHorizon,
    LogisticConfig,
)


class LogisticBaseline:
    """Logistic regression baseline for direction prediction.

    Uses sklearn ``LogisticRegression`` with L2 regularisation.  Natively
    outputs calibrated probabilities via ``predict_proba``, so no post-hoc
    calibration is needed.

    Attributes:
        config: Logistic regression configuration object.
        horizon: Forecast horizon embedded in every ``DirectionForecast``.
    """

    def __init__(self, config: LogisticConfig, horizon: ForecastHorizon) -> None:
        """Initialise the logistic regression baseline.

        Args:
            config: Logistic configuration (C, max_iter, class_weight, seed).
            horizon: Forecast horizon to embed in predictions.
        """
        self.config: LogisticConfig = config
        self.horizon: ForecastHorizon = horizon
        self._model: LogisticRegression | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Train the logistic regression on feature matrix and direction labels.

        Labels are expected as +1 / -1.  Sklearn handles {-1, +1} natively
        for ``LogisticRegression``, so no label remapping is required.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Direction labels of shape ``(n_samples,)`` with values +1 or -1.

        Raises:
            ValueError: If inputs are empty.
        """
        n_samples: int = x_train.shape[0]
        if n_samples == 0:
            msg: str = "x_train must contain at least one sample"
            raise ValueError(msg)

        # sklearn >= 1.8 deprecates explicit penalty='l2'; use l1_ratio=0 instead.
        self._model = LogisticRegression(
            C=self.config.c,
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            random_state=self.config.random_seed,
            solver="lbfgs",
            l1_ratio=0,
        )
        self._model.fit(x_train, y_train)

        train_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = self._model.predict(x_train)
        train_acc: float = float(np.mean(train_preds == y_train))

        logger.info(
            "LogisticRegression fitted on {} samples | train_acc={:.4f} | C={}",
            n_samples,
            train_acc,
            self.config.c,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> list[DirectionForecast]:
        """Generate direction forecasts with confidence estimates.

        Confidence is the predicted probability for the chosen direction,
        obtained from ``predict_proba``.

        Args:
            x_test: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            List of direction forecasts, one per sample.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``x_test`` is empty.
        """
        if self._model is None:
            msg: str = "Model has not been fitted — call fit() first"
            raise RuntimeError(msg)

        n_test: int = x_test.shape[0]
        if n_test == 0:
            msg = "x_test must contain at least one sample"
            raise ValueError(msg)

        # predict_proba returns shape (n_samples, n_classes)
        # classes_ tells us which column corresponds to which label
        proba: np.ndarray[tuple[int, int], np.dtype[np.float64]] = self._model.predict_proba(x_test)
        classes: np.ndarray[tuple[int], np.dtype[np.float64]] = self._model.classes_

        forecasts: list[DirectionForecast] = []
        for i in range(n_test):
            # Find the class with highest probability
            best_idx: int = int(np.argmax(proba[i]))
            direction: int = int(classes[best_idx])
            confidence: float = float(proba[i, best_idx])

            forecast: DirectionForecast = DirectionForecast(
                predicted_direction=direction,
                confidence=confidence,
                horizon=self.horizon,
            )
            forecasts.append(forecast)

        return forecasts
