"""LightGBM quantile regressor for distributional return prediction."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from loguru import logger

from src.app.forecasting.domain.value_objects import (
    GradientBoostingConfig,
    PointPrediction,
    QuantilePrediction,
)

# IQR-to-std conversion constant: std = IQR / 1.349 for a normal distribution.
_IQR_TO_STD_FACTOR: float = 1.349

# Named quantile levels for locating median / Q25 / Q75 columns.
_Q_MEDIAN: float = 0.50
_Q_25: float = 0.25
_Q_75: float = 0.75

# Floating-point tolerance for quantile level matching.
_QUANTILE_TOLERANCE: float = 1e-9


class GradientBoostingRegressor:
    """LightGBM-based quantile regressor with optional isotonic correction.

    Trains one LightGBM model per quantile level using the ``quantile``
    objective.  At inference, quantile monotonicity is optionally enforced
    via isotonic regression (Chernozhukov et al. 2010) so that
    ``q_0.05 <= q_0.25 <= ... <= q_0.95`` for every sample.

    The ``predict()`` convenience method returns a point estimate (median)
    with IQR-based uncertainty for compatibility with :class:`IReturnRegressor`.

    Attributes:
        config: LightGBM quantile configuration object.
    """

    def __init__(self, config: GradientBoostingConfig) -> None:
        """Initialise the LightGBM quantile regressor.

        Args:
            config: Gradient boosting configuration (quantiles, tree params, seed).
        """
        self.config: GradientBoostingConfig = config
        self._models: dict[float, lgb.LGBMRegressor] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Train one LightGBM model per quantile level.

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

        self._models.clear()

        for q in self.config.quantiles:
            model: lgb.LGBMRegressor = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
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
            model.fit(x_train, y_train)
            self._models[q] = model

        logger.info(
            "LightGBM quantile models fitted on {} samples | quantiles={}",
            n_samples,
            self.config.quantiles,
        )

    # ------------------------------------------------------------------
    # Quantile inference
    # ------------------------------------------------------------------

    def predict_quantiles(
        self,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> QuantilePrediction:
        """Generate predictions at all configured quantile levels.

        When ``config.apply_isotonic=True``, the raw quantile predictions
        are sorted per-sample so that monotonicity is guaranteed
        (Chernozhukov et al. 2010 rearrangement).

        Args:
            x_test: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Quantile prediction with values at each quantile level.

        Raises:
            RuntimeError: If the models have not been fitted.
            ValueError: If ``x_test`` is empty.
        """
        if not self._models:
            msg: str = "Models have not been fitted — call fit() first"
            raise RuntimeError(msg)

        n_test: int = x_test.shape[0]
        if n_test == 0:
            msg = "x_test must contain at least one sample"
            raise ValueError(msg)

        n_quantiles: int = len(self.config.quantiles)
        raw_values: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty(
            (n_test, n_quantiles), dtype=np.float64
        )

        for col_idx, q in enumerate(self.config.quantiles):
            preds: np.ndarray[tuple[int], np.dtype[np.float64]] = self._models[q].predict(x_test).astype(np.float64)
            raw_values[:, col_idx] = preds

        # Isotonic monotonicity correction (Chernozhukov et al. 2010)
        if self.config.apply_isotonic:
            raw_values = self._apply_isotonic_correction(raw_values)

        return QuantilePrediction(
            quantiles=self.config.quantiles,
            values=raw_values,
        )

    # ------------------------------------------------------------------
    # Point-estimate convenience (IReturnRegressor compat)
    # ------------------------------------------------------------------

    def predict(
        self,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> PointPrediction:
        """Generate point predictions from the median quantile.

        Uncertainty is estimated from the interquartile range:
        ``std = (q_0.75 - q_0.25) / 1.349`` (Gaussian IQR relationship).

        Falls back to the full range when only the median is configured.

        Args:
            x_test: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Point prediction with mean = median, std = IQR-based estimate.

        Raises:
            ValueError: If the configured quantiles do not include a
                median (0.50).
        """
        qpred: QuantilePrediction = self.predict_quantiles(x_test)

        # Locate median column
        median_idx: int | None = None
        q25_idx: int | None = None
        q75_idx: int | None = None

        for idx, q in enumerate(self.config.quantiles):
            if abs(q - _Q_MEDIAN) < _QUANTILE_TOLERANCE:
                median_idx = idx
            if abs(q - _Q_25) < _QUANTILE_TOLERANCE:
                q25_idx = idx
            if abs(q - _Q_75) < _QUANTILE_TOLERANCE:
                q75_idx = idx

        if median_idx is None:
            msg: str = "Configured quantiles must include 0.50 for point prediction"
            raise ValueError(msg)

        mean: np.ndarray[tuple[int], np.dtype[np.float64]] = qpred.values[:, median_idx]

        # IQR-based std estimate
        if q25_idx is not None and q75_idx is not None:
            iqr: np.ndarray[tuple[int], np.dtype[np.float64]] = qpred.values[:, q75_idx] - qpred.values[:, q25_idx]
            std: np.ndarray[tuple[int], np.dtype[np.float64]] = np.maximum(iqr / _IQR_TO_STD_FACTOR, 0.0).astype(
                np.float64
            )
        else:
            # Fallback: use full range of available quantiles
            spread: np.ndarray[tuple[int], np.dtype[np.float64]] = qpred.values[:, -1] - qpred.values[:, 0]
            std = np.maximum(spread / _IQR_TO_STD_FACTOR, 0.0).astype(np.float64)

        return PointPrediction(mean=mean, std=std)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_isotonic_correction(
        values: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Enforce quantile monotonicity via per-sample sorting.

        For each sample row, sorts quantile predictions so that lower
        quantiles do not exceed higher ones.  This is the simplified
        rearrangement approach from Chernozhukov, Fernandez-Val, and
        Galichon (2010).

        Args:
            values: Raw quantile predictions of shape
                ``(n_samples, n_quantiles)``.

        Returns:
            Monotonicity-corrected array (same shape, sorted per row).
        """
        corrected: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.sort(values, axis=1)
        return corrected
