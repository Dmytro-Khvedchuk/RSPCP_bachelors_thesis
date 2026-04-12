"""LightGBM gradient boosting recommender predicting strategy return per asset."""

from __future__ import annotations

from typing import Annotated

import lightgbm as lgb
import numpy as np
from loguru import logger
from pydantic import BaseModel
from pydantic import Field as PydanticField

from src.app.recommendation.domain.value_objects import Recommendation

# ---------------------------------------------------------------------------
# Minimum volatility floor to prevent division-by-zero in position sizing.
# ---------------------------------------------------------------------------
_MIN_SIGMA: float = 1e-8

# Default asset label used when asset names are not explicitly provided.
_DEFAULT_ASSET: str = "UNKNOWN"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class GradientBoostingRecommenderConfig(BaseModel, frozen=True):
    """Configuration for the LightGBM gradient boosting recommender.

    All LightGBM hyperparameters and position-sizing parameters are
    centralised here. No magic numbers in the model class.

    Attributes:
        n_estimators: Number of boosting rounds.
        learning_rate: Step-size shrinkage.
        max_depth: Maximum tree depth (``-1`` for unlimited).
        min_child_samples: Minimum samples in a leaf node.
        reg_alpha: L1 regularisation.
        reg_lambda: L2 regularisation.
        subsample: Row subsampling ratio per tree.
        colsample_bytree: Column subsampling ratio per tree.
        random_seed: Reproducibility seed.
        min_threshold: Minimum predicted return for deployment.
        position_size_cap: Maximum position size after Kelly scaling.
    """

    n_estimators: Annotated[
        int,
        PydanticField(default=100, gt=0, description="Number of boosting rounds"),
    ]

    learning_rate: Annotated[
        float,
        PydanticField(default=0.05, gt=0.0, description="Step-size shrinkage"),
    ]

    max_depth: Annotated[
        int,
        PydanticField(default=5, ge=-1, description="Max tree depth (-1 unlimited)"),
    ]

    min_child_samples: Annotated[
        int,
        PydanticField(default=20, gt=0, description="Min samples per leaf"),
    ]

    reg_alpha: Annotated[
        float,
        PydanticField(default=0.1, ge=0.0, description="L1 regularisation"),
    ]

    reg_lambda: Annotated[
        float,
        PydanticField(default=1.0, ge=0.0, description="L2 regularisation"),
    ]

    subsample: Annotated[
        float,
        PydanticField(default=0.8, gt=0.0, le=1.0, description="Row subsampling ratio"),
    ]

    colsample_bytree: Annotated[
        float,
        PydanticField(default=0.8, gt=0.0, le=1.0, description="Column subsampling ratio"),
    ]

    random_seed: Annotated[
        int,
        PydanticField(default=42, ge=0, description="Reproducibility seed"),
    ]

    min_threshold: Annotated[
        float,
        PydanticField(default=0.0, ge=0.0, description="Min predicted return for deployment"),
    ]

    position_size_cap: Annotated[
        float,
        PydanticField(default=1.0, gt=0.0, le=1.0, description="Max position size"),
    ]


# ---------------------------------------------------------------------------
# GradientBoostingRecommender
# ---------------------------------------------------------------------------


class GradientBoostingRecommender:
    """LightGBM regressor predicting expected strategy return per asset.

    This is the primary recommender model. It learns the mapping from
    assembled feature vectors (market state, classifier/regressor outputs,
    regime indicators) to realised strategy returns (continuous generalised
    meta-labels).

    Position sizing follows a Kelly-adjacent rule::

        size = max(r_hat - threshold, 0) / sigma

    clamped to ``[0.0, position_size_cap]``.

    Feature importance is computed via LightGBM's built-in ``gain``
    importance after fitting, exposed through :attr:`feature_importances`.

    Attributes:
        config: Frozen configuration object.
    """

    def __init__(
        self,
        config: GradientBoostingRecommenderConfig | None = None,
        *,
        asset_names: list[str] | None = None,
    ) -> None:
        """Initialise the gradient boosting recommender.

        Args:
            config: Model configuration. Uses defaults if not provided.
            asset_names: Optional asset labels for each sample row in
                ``predict()``. When ``None``, defaults to ``"UNKNOWN"``
                for every row.
        """
        self._config: GradientBoostingRecommenderConfig = (
            config or GradientBoostingRecommenderConfig()  # ty: ignore[missing-argument]
        )
        self._asset_names: list[str] | None = asset_names
        self._model: lgb.LGBMRegressor | None = None
        self._train_std: float = 0.0
        self._feature_importances: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None

    @property
    def config(self) -> GradientBoostingRecommenderConfig:
        """Return the frozen configuration.

        Returns:
            Configuration object.
        """
        return self._config

    @property
    def feature_importances(self) -> np.ndarray[tuple[int], np.dtype[np.float64]] | None:
        """Return LightGBM gain-based feature importances after fitting.

        Returns:
            Array of shape ``(n_features,)`` with importance scores,
            or ``None`` if not yet fitted.
        """
        return self._feature_importances

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted.

        Returns:
            ``True`` if ``fit()`` has been called.
        """
        return self._model is not None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Train the LightGBM model on feature matrix and realised returns.

        After training, the standard deviation of ``y_train`` is stored
        for Kelly-adjacent position sizing, and feature importances are
        extracted.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Realised strategy returns of shape ``(n_samples,)``.

        Raises:
            ValueError: If inputs have fewer than 2 samples (LightGBM minimum)
                or shapes are inconsistent.
        """
        n_samples: int = x_train.shape[0]
        if n_samples < 2:  # noqa: PLR2004
            msg: str = "x_train must contain at least 2 samples (LightGBM minimum)"
            raise ValueError(msg)
        if y_train.shape[0] != n_samples:
            msg = f"x_train and y_train must have the same number of samples, got {n_samples} and {y_train.shape[0]}"
            raise ValueError(msg)

        cfg: GradientBoostingRecommenderConfig = self._config

        self._model = lgb.LGBMRegressor(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            min_child_samples=cfg.min_child_samples,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            random_state=cfg.random_seed,
            verbose=-1,
        )
        self._model.fit(x_train, y_train)

        # Store training target std for Kelly position sizing
        y_std: float = float(np.std(y_train, ddof=1)) if n_samples > 1 else 0.0
        self._train_std = max(y_std, _MIN_SIGMA)

        # Extract gain-based feature importance
        raw_importance: np.ndarray[tuple[int], np.dtype[np.float64]] = self._model.feature_importances_.astype(
            np.float64
        )
        self._feature_importances = raw_importance

        logger.info(
            "GradientBoostingRecommender fitted on {} samples | n_features={} | y_std={:.6f} | threshold={:.6f}",
            n_samples,
            x_train.shape[1],
            self._train_std,
            cfg.min_threshold,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> list[Recommendation]:
        """Predict recommendations for input features.

        For each sample row:

        1. Predict expected strategy return ``r_hat``.
        2. Direction = ``sign(r_hat)`` (``+1`` if ``r_hat >= 0``, else ``-1``).
        3. Magnitude = ``|r_hat|``.
        4. Deploy = ``r_hat > min_threshold``.
        5. Position size = ``max(r_hat - threshold, 0) / sigma``,
           clamped to ``[0.0, position_size_cap]``.
        6. Confidence = scaled prediction magnitude relative to training
           target dispersion.

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            List of recommendations, one per sample.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``x`` is empty.
        """
        if self._model is None:
            msg: str = "Model has not been fitted — call fit() first"
            raise RuntimeError(msg)

        n_samples: int = x.shape[0]
        if n_samples == 0:
            msg = "x must contain at least one sample"
            raise ValueError(msg)

        raw_predictions: np.ndarray[tuple[int], np.dtype[np.float64]] = self._model.predict(x).astype(np.float64)

        recommendations: list[Recommendation] = []
        sigma: float = self._train_std
        threshold: float = self._config.min_threshold
        cap: float = self._config.position_size_cap

        for i in range(n_samples):
            r_hat: float = float(raw_predictions[i])
            direction: int = 1 if r_hat >= 0.0 else -1
            magnitude: float = abs(r_hat)
            deploy: bool = r_hat > threshold

            # Kelly-adjacent position sizing: size ∝ max(r_hat - threshold, 0) / sigma
            position_size: float = min(max(r_hat - threshold, 0.0) / sigma, cap)

            # Confidence: magnitude relative to training dispersion, clamped to [0, 1]
            confidence: float = min(magnitude / sigma, 1.0) if sigma > _MIN_SIGMA else 0.0

            recommendations.append(
                Recommendation(
                    asset=_get_asset_name(self._asset_names, i),
                    predicted_strategy_return=r_hat,
                    confidence=confidence,
                    deploy=deploy,
                    predicted_direction=direction,
                    predicted_magnitude=magnitude,
                    position_size=position_size,
                )
            )

        logger.debug(
            "GradientBoostingRecommender predicted {} recommendations | deployed={}",
            n_samples,
            sum(1 for r in recommendations if r.deploy),
        )

        return recommendations

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importances(
        self,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Return feature importances as a name-to-score mapping.

        Uses LightGBM's gain-based importance. When ``feature_names``
        are not provided, features are labelled ``f_0``, ``f_1``, etc.

        Args:
            feature_names: Optional list of feature names matching
                the training feature matrix columns.

        Returns:
            Dictionary mapping feature name to importance score,
            sorted descending by importance.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self._feature_importances is None:
            msg: str = "Model has not been fitted — call fit() first"
            raise RuntimeError(msg)

        n_features: int = len(self._feature_importances)
        names: list[str] = (
            feature_names
            if feature_names is not None and len(feature_names) == n_features
            else [f"f_{i}" for i in range(n_features)]
        )

        importance_dict: dict[str, float] = {names[i]: float(self._feature_importances[i]) for i in range(n_features)}
        sorted_dict: dict[str, float] = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        return sorted_dict


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _get_asset_name(asset_names: list[str] | None, idx: int) -> str:
    """Retrieve asset name by index with fallback.

    Args:
        asset_names: Optional asset name list.
        idx: Row index.

    Returns:
        Asset name or default placeholder.
    """
    if asset_names is not None and idx < len(asset_names):
        return asset_names[idx]
    return _DEFAULT_ASSET
