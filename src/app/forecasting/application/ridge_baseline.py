"""Ridge / Huber regression baseline for return magnitude prediction."""

from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.linear_model import HuberRegressor, Ridge

from src.app.forecasting.domain.value_objects import PointPrediction, RidgeConfig


class RidgeBaseline:
    """Ridge (or Huber) linear regression baseline.

    Provides a simple, interpretable baseline for return magnitude
    prediction.  When ``config.use_huber=True`` the model switches to
    Huber regression for robustness against fat-tailed crypto returns.

    Attributes:
        config: Ridge/Huber configuration object.
    """

    def __init__(self, config: RidgeConfig) -> None:
        """Initialise the Ridge/Huber baseline.

        Args:
            config: Ridge configuration (alpha, huber settings, seed).
        """
        self.config: RidgeConfig = config
        self._model: Ridge | HuberRegressor | None = None
        self._residual_std: float = 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Train the linear model and estimate residual standard deviation.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Target vector of shape ``(n_samples,)``.

        Raises:
            ValueError: If inputs are empty.
        """
        n_samples: int = x_train.shape[0]
        if n_samples == 0:
            msg: str = "x_train must contain at least one sample"
            raise ValueError(msg)

        if self.config.use_huber:
            self._model = HuberRegressor(
                epsilon=self.config.huber_epsilon,
                alpha=self.config.alpha,
                max_iter=200,
            )
            model_name: str = "HuberRegressor"
        else:
            self._model = Ridge(alpha=self.config.alpha)
            model_name = "Ridge"

        self._model.fit(x_train, y_train)

        # Compute residual std from training data
        train_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = self._model.predict(x_train)
        residuals: np.ndarray[tuple[int], np.dtype[np.float64]] = y_train - train_preds
        self._residual_std = float(np.std(residuals, ddof=1)) if n_samples > 1 else 0.0

        logger.info(
            "{} fitted on {} samples | residual_std={:.6f}",
            model_name,
            n_samples,
            self._residual_std,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> PointPrediction:
        """Generate point predictions with constant residual-std uncertainty.

        Args:
            x_test: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Point prediction with mean from the linear model and std
            equal to the training residual standard deviation (homoscedastic).

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

        mean: np.ndarray[tuple[int], np.dtype[np.float64]] = self._model.predict(x_test).astype(np.float64)
        std: np.ndarray[tuple[int], np.dtype[np.float64]] = np.full(n_test, self._residual_std, dtype=np.float64)

        return PointPrediction(mean=mean, std=std)
