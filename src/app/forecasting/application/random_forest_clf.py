"""Random Forest classifier for direction prediction."""

from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier

from src.app.forecasting.domain.value_objects import (
    DirectionForecast,
    ForecastHorizon,
    RandomForestClassifierConfig,
)


class RandomForestClassifier:
    """Random Forest classifier for direction prediction.

    Uses sklearn ``RandomForestClassifier`` with configurable tree
    parameters.  Outputs probabilities via ``predict_proba`` and
    exposes feature importances after fitting.

    Attributes:
        config: Random Forest configuration object.
        horizon: Forecast horizon embedded in every ``DirectionForecast``.
        feature_importances_: Gini-based feature importances after fitting.
    """

    def __init__(self, config: RandomForestClassifierConfig, horizon: ForecastHorizon) -> None:
        """Initialise the Random Forest classifier.

        Args:
            config: Random Forest configuration (n_estimators, max_depth, etc.).
            horizon: Forecast horizon to embed in predictions.
        """
        self.config: RandomForestClassifierConfig = config
        self.horizon: ForecastHorizon = horizon
        self._model: SklearnRandomForestClassifier | None = None
        self.feature_importances_: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Train the Random Forest on feature matrix and direction labels.

        After fitting, ``feature_importances_`` is populated with Gini-based
        importances of shape ``(n_features,)``.

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

        self._model = SklearnRandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            class_weight=self.config.class_weight,
            random_state=self.config.random_seed,
            n_jobs=-1,
        )
        self._model.fit(x_train, y_train)

        # Store feature importances for downstream analysis
        self.feature_importances_ = self._model.feature_importances_.astype(np.float64)

        train_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = self._model.predict(x_train)
        train_acc: float = float(np.mean(train_preds == y_train))

        logger.info(
            "RandomForest fitted on {} samples | train_acc={:.4f} | n_estimators={} | top_importance={:.4f}",
            n_samples,
            train_acc,
            self.config.n_estimators,
            float(np.max(self.feature_importances_)),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> list[DirectionForecast]:
        """Generate direction forecasts with confidence estimates.

        Confidence is the predicted probability for the chosen direction
        from the Random Forest's ``predict_proba``.

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

        proba: np.ndarray[tuple[int, int], np.dtype[np.float64]] = self._model.predict_proba(x_test)
        classes: np.ndarray[tuple[int], np.dtype[np.float64]] = self._model.classes_  # type: ignore[assignment] # ty: ignore[invalid-assignment]

        forecasts: list[DirectionForecast] = []
        for i in range(n_test):
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
