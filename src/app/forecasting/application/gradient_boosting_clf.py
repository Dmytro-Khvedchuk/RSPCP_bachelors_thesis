"""LightGBM classifier with Platt scaling for direction prediction."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV

from src.app.forecasting.domain.value_objects import (
    DirectionForecast,
    ForecastHorizon,
    GradientBoostingClassifierConfig,
)


class GradientBoostingClassifier:
    """LightGBM-based direction classifier with post-hoc probability calibration.

    Uses ``LGBMClassifier`` with binary objective, post-hoc calibrated via
    sklearn ``CalibratedClassifierCV`` (Platt scaling by default, or
    isotonic regression).  This addresses the known issue of LightGBM
    outputting poorly calibrated probabilities in imbalanced or
    high-dimensional settings.

    Attributes:
        config: LightGBM classifier configuration object.
        horizon: Forecast horizon embedded in every ``DirectionForecast``.
    """

    def __init__(self, config: GradientBoostingClassifierConfig, horizon: ForecastHorizon) -> None:
        """Initialise the LightGBM classifier with calibration.

        Args:
            config: Gradient boosting classifier configuration.
            horizon: Forecast horizon to embed in predictions.
        """
        self.config: GradientBoostingClassifierConfig = config
        self.horizon: ForecastHorizon = horizon
        self._base_model: lgb.LGBMClassifier | None = None
        self._calibrated_model: CalibratedClassifierCV | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Train the LightGBM classifier with Platt scaling calibration.

        The base LGBMClassifier is wrapped in ``CalibratedClassifierCV``
        which performs internal cross-validation to fit the calibration
        mapping (sigmoid for Platt, isotonic for non-parametric).

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

        self._base_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_child_samples=self.config.min_child_samples,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_seed,
            verbose=-1,
        )

        # Wrap with CalibratedClassifierCV for Platt scaling
        self._calibrated_model = CalibratedClassifierCV(
            estimator=self._base_model,
            method=self.config.calibration_method,
            cv=self.config.calibration_cv,
        )
        self._calibrated_model.fit(x_train, y_train)

        # Log training accuracy using the calibrated model
        train_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = self._calibrated_model.predict(x_train)
        train_acc: float = float(np.mean(train_preds == y_train))

        logger.info(
            "LightGBM classifier fitted on {} samples | train_acc={:.4f} | n_estimators={} | calibration={}",
            n_samples,
            train_acc,
            self.config.n_estimators,
            self.config.calibration_method,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> list[DirectionForecast]:
        """Generate direction forecasts with calibrated confidence estimates.

        Uses the calibrated model's ``predict_proba`` for well-calibrated
        probability estimates.

        Args:
            x_test: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            List of direction forecasts, one per sample.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``x_test`` is empty.
        """
        if self._calibrated_model is None:
            msg: str = "Model has not been fitted — call fit() first"
            raise RuntimeError(msg)

        n_test: int = x_test.shape[0]
        if n_test == 0:
            msg = "x_test must contain at least one sample"
            raise ValueError(msg)

        proba: np.ndarray[tuple[int, int], np.dtype[np.float64]] = self._calibrated_model.predict_proba(x_test)
        classes: np.ndarray[tuple[int], np.dtype[np.float64]] = self._calibrated_model.classes_

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
