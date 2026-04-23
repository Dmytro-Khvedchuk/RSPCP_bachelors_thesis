"""Baseline recommenders for comparison against the primary LightGBM model."""

from __future__ import annotations

from typing import Annotated

import numpy as np
from loguru import logger
from pydantic import BaseModel
from pydantic import Field as PydanticField

from src.app.recommendation.domain.value_objects import Recommendation

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_DEFAULT_ASSET: str = "UNKNOWN"

# Minimum sigma floor for position sizing division.
_MIN_SIGMA: float = 1e-8


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


class RandomRecommenderConfig(BaseModel, frozen=True):
    """Configuration for the random recommender baseline.

    Attributes:
        random_seed: Reproducibility seed.
        deploy_probability: Probability that any given sample is deployed.
    """

    random_seed: Annotated[
        int,
        PydanticField(default=42, ge=0, description="Reproducibility seed"),
    ]

    deploy_probability: Annotated[
        float,
        PydanticField(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Probability of deploying each sample",
        ),
    ]


class ColumnIndexConfig(BaseModel, frozen=True):
    """Configuration for recommenders that read a specific feature column.

    Attributes:
        col_idx: Column index in the feature matrix.
        min_threshold: Minimum value for deployment.
        random_seed: Reproducibility seed (for interface parity).
    """

    col_idx: Annotated[
        int,
        PydanticField(ge=0, description="Column index in feature matrix"),
    ]

    min_threshold: Annotated[
        float,
        PydanticField(default=0.0, ge=0.0, description="Min value for deployment"),
    ]

    random_seed: Annotated[
        int,
        PydanticField(default=42, ge=0, description="Reproducibility seed"),
    ]


# ---------------------------------------------------------------------------
# RandomRecommender
# ---------------------------------------------------------------------------


class RandomRecommender:
    """Randomly select assets for deployment (null hypothesis baseline).

    Direction is random (+1/-1), magnitude is drawn from U(0, 0.01),
    position size is uniform U(0, 1) for deployed assets and 0 otherwise.
    This is the simplest possible baseline: any useful recommender must
    beat this.

    Attributes:
        config: Random recommender configuration.
    """

    def __init__(
        self,
        config: RandomRecommenderConfig | None = None,
        *,
        asset_names: list[str] | None = None,
    ) -> None:
        """Initialise the random recommender.

        Args:
            config: Configuration. Uses defaults if not provided.
            asset_names: Optional asset labels per sample row.
        """
        self._config: RandomRecommenderConfig = config or RandomRecommenderConfig()  # ty: ignore[missing-argument]
        self._asset_names: list[str] | None = asset_names
        self._rng: np.random.Generator = np.random.default_rng(self._config.random_seed)
        self._fitted: bool = False

    @property
    def config(self) -> RandomRecommenderConfig:
        """Return the configuration.

        Returns:
            Frozen configuration object.
        """
        return self._config

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],  # noqa: ARG002
    ) -> None:
        """Validate inputs and mark as fitted. No parameters are learned.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Realised strategy returns (ignored).

        Raises:
            ValueError: If inputs are empty.
        """
        n_samples: int = x_train.shape[0]
        if n_samples == 0:
            msg: str = "x_train must contain at least one sample"
            raise ValueError(msg)

        self._fitted = True
        logger.info("RandomRecommender fitted on {} samples (no parameters learned)", n_samples)

    def predict(
        self,
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> list[Recommendation]:
        """Generate random recommendations.

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            List of random recommendations, one per sample.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``x`` is empty.
        """
        if not self._fitted:
            msg: str = "Model has not been fitted — call fit() first"
            raise RuntimeError(msg)

        n_samples: int = x.shape[0]
        if n_samples == 0:
            msg = "x must contain at least one sample"
            raise ValueError(msg)

        # Draw random values
        deploy_prob: float = self._config.deploy_probability
        deploy_draws: np.ndarray[tuple[int], np.dtype[np.float64]] = self._rng.uniform(0.0, 1.0, n_samples)
        directions: np.ndarray[tuple[int], np.dtype[np.int64]] = self._rng.choice(
            np.array([1, -1], dtype=np.int64), size=n_samples
        )
        magnitudes: np.ndarray[tuple[int], np.dtype[np.float64]] = self._rng.uniform(0.0, 0.01, n_samples).astype(
            np.float64
        )
        sizes: np.ndarray[tuple[int], np.dtype[np.float64]] = self._rng.uniform(0.0, 1.0, n_samples).astype(np.float64)

        recommendations: list[Recommendation] = []
        for i in range(n_samples):
            deploy: bool = bool(deploy_draws[i] < deploy_prob)
            direction: int = int(directions[i])
            magnitude: float = float(magnitudes[i])
            r_hat: float = direction * magnitude
            asset: str = _get_asset_name(self._asset_names, i)

            rec: Recommendation = Recommendation(
                asset=asset,
                predicted_strategy_return=r_hat,
                confidence=0.5,
                deploy=deploy,
                predicted_direction=direction,
                predicted_magnitude=magnitude,
                position_size=float(sizes[i]) if deploy else 0.0,
            )
            recommendations.append(rec)

        return recommendations


# ---------------------------------------------------------------------------
# AllAssetsRecommender
# ---------------------------------------------------------------------------


class AllAssetsRecommender:
    """Deploy strategy on all assets unconditionally (unfiltered baseline).

    Always deploys with equal weight (position_size = 1.0), direction and
    magnitude from training mean. This represents the "no filtering" approach.

    Attributes:
        config: No configuration needed; uses defaults for interface parity.
    """

    def __init__(
        self,
        *,
        asset_names: list[str] | None = None,
    ) -> None:
        """Initialise the all-assets recommender.

        Args:
            asset_names: Optional asset labels per sample row.
        """
        self._asset_names: list[str] | None = asset_names
        self._fitted: bool = False
        self._mean_return: float = 0.0

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Store training mean return for direction and magnitude.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Realised strategy returns of shape ``(n_samples,)``.

        Raises:
            ValueError: If inputs are empty.
        """
        n_samples: int = x_train.shape[0]
        if n_samples == 0:
            msg: str = "x_train must contain at least one sample"
            raise ValueError(msg)

        self._mean_return = float(np.mean(y_train))
        self._fitted = True

        logger.info(
            "AllAssetsRecommender fitted on {} samples | mean_return={:.6f}",
            n_samples,
            self._mean_return,
        )

    def predict(
        self,
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> list[Recommendation]:
        """Deploy on all assets with full position size.

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            List of recommendations, all deployed.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``x`` is empty.
        """
        if not self._fitted:
            msg: str = "Model has not been fitted — call fit() first"
            raise RuntimeError(msg)

        n_samples: int = x.shape[0]
        if n_samples == 0:
            msg = "x must contain at least one sample"
            raise ValueError(msg)

        direction: int = 1 if self._mean_return >= 0.0 else -1
        magnitude: float = abs(self._mean_return)

        recommendations: list[Recommendation] = []
        for i in range(n_samples):
            asset: str = _get_asset_name(self._asset_names, i)
            rec: Recommendation = Recommendation(
                asset=asset,
                predicted_strategy_return=self._mean_return,
                confidence=0.5,
                deploy=True,
                predicted_direction=direction,
                predicted_magnitude=magnitude,
                position_size=1.0,
            )
            recommendations.append(rec)

        return recommendations


# ---------------------------------------------------------------------------
# ClassifierOnlyRecommender
# ---------------------------------------------------------------------------


class ClassifierOnlyRecommender:
    """Deploy based on classifier confidence alone.

    Reads the classifier confidence from a specified column in the
    feature matrix and deploys when it exceeds the threshold.
    Direction is determined by the sign of the confidence column
    (assumed signed: positive => long, negative => short).

    Attributes:
        config: Column index configuration.
    """

    def __init__(
        self,
        config: ColumnIndexConfig,
        *,
        asset_names: list[str] | None = None,
    ) -> None:
        """Initialise the classifier-only recommender.

        Args:
            config: Column index and threshold configuration.
            asset_names: Optional asset labels per sample row.
        """
        self._config: ColumnIndexConfig = config
        self._asset_names: list[str] | None = asset_names
        self._fitted: bool = False
        self._train_std: float = 0.0

    @property
    def config(self) -> ColumnIndexConfig:
        """Return the configuration.

        Returns:
            Frozen configuration object.
        """
        return self._config

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Validate inputs and store training target dispersion.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Realised strategy returns of shape ``(n_samples,)``.

        Raises:
            ValueError: If inputs are empty or column index is out of bounds.
        """
        n_samples: int = x_train.shape[0]
        if n_samples == 0:
            msg: str = "x_train must contain at least one sample"
            raise ValueError(msg)

        n_features: int = x_train.shape[1]
        if self._config.col_idx >= n_features:
            msg = f"col_idx={self._config.col_idx} is out of bounds for feature matrix with {n_features} columns"
            raise ValueError(msg)

        y_std: float = float(np.std(y_train, ddof=1)) if n_samples > 1 else 0.0
        self._train_std = max(y_std, _MIN_SIGMA)
        self._fitted = True

        logger.info(
            "ClassifierOnlyRecommender fitted on {} samples | clf_col_idx={}",
            n_samples,
            self._config.col_idx,
        )

    def predict(
        self,
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> list[Recommendation]:
        """Predict based on classifier confidence column.

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            List of recommendations.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``x`` is empty.
        """
        if not self._fitted:
            msg: str = "Model has not been fitted — call fit() first"
            raise RuntimeError(msg)

        n_samples: int = x.shape[0]
        if n_samples == 0:
            msg = "x must contain at least one sample"
            raise ValueError(msg)

        col_idx: int = self._config.col_idx
        confidence_values: np.ndarray[tuple[int], np.dtype[np.float64]] = x[:, col_idx]
        threshold: float = self._config.min_threshold

        recommendations: list[Recommendation] = []
        for i in range(n_samples):
            clf_conf: float = float(confidence_values[i])
            magnitude: float = abs(clf_conf)
            direction: int = 1 if clf_conf >= 0.0 else -1
            deploy: bool = magnitude > threshold
            position_size: float = min(max(magnitude / self._train_std, 0.0), 1.0) if deploy else 0.0
            confidence: float = min(magnitude / self._train_std, 1.0) if self._train_std > _MIN_SIGMA else 0.0
            asset: str = _get_asset_name(self._asset_names, i)

            rec: Recommendation = Recommendation(
                asset=asset,
                predicted_strategy_return=clf_conf,
                confidence=confidence,
                deploy=deploy,
                predicted_direction=direction,
                predicted_magnitude=magnitude,
                position_size=position_size,
            )
            recommendations.append(rec)

        return recommendations


# ---------------------------------------------------------------------------
# RegressorOnlyRecommender
# ---------------------------------------------------------------------------


class RegressorOnlyRecommender:
    """Deploy based on predicted return magnitude alone.

    Reads the regressor predicted return from a specified column in the
    feature matrix. Deploys when the predicted return exceeds the threshold.

    Attributes:
        config: Column index configuration.
    """

    def __init__(
        self,
        config: ColumnIndexConfig,
        *,
        asset_names: list[str] | None = None,
    ) -> None:
        """Initialise the regressor-only recommender.

        Args:
            config: Column index and threshold configuration.
            asset_names: Optional asset labels per sample row.
        """
        self._config: ColumnIndexConfig = config
        self._asset_names: list[str] | None = asset_names
        self._fitted: bool = False
        self._train_std: float = 0.0

    @property
    def config(self) -> ColumnIndexConfig:
        """Return the configuration.

        Returns:
            Frozen configuration object.
        """
        return self._config

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Validate inputs and store training target dispersion.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Realised strategy returns of shape ``(n_samples,)``.

        Raises:
            ValueError: If inputs are empty or column index is out of bounds.
        """
        n_samples: int = x_train.shape[0]
        if n_samples == 0:
            msg: str = "x_train must contain at least one sample"
            raise ValueError(msg)

        n_features: int = x_train.shape[1]
        if self._config.col_idx >= n_features:
            msg = f"col_idx={self._config.col_idx} is out of bounds for feature matrix with {n_features} columns"
            raise ValueError(msg)

        y_std: float = float(np.std(y_train, ddof=1)) if n_samples > 1 else 0.0
        self._train_std = max(y_std, _MIN_SIGMA)
        self._fitted = True

        logger.info(
            "RegressorOnlyRecommender fitted on {} samples | reg_col_idx={}",
            n_samples,
            self._config.col_idx,
        )

    def predict(
        self,
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> list[Recommendation]:
        """Predict based on regressor return column.

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            List of recommendations.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``x`` is empty.
        """
        if not self._fitted:
            msg: str = "Model has not been fitted — call fit() first"
            raise RuntimeError(msg)

        n_samples: int = x.shape[0]
        if n_samples == 0:
            msg = "x must contain at least one sample"
            raise ValueError(msg)

        col_idx: int = self._config.col_idx
        return_values: np.ndarray[tuple[int], np.dtype[np.float64]] = x[:, col_idx]
        threshold: float = self._config.min_threshold

        recommendations: list[Recommendation] = []
        for i in range(n_samples):
            r_hat: float = float(return_values[i])
            magnitude: float = abs(r_hat)
            direction: int = 1 if r_hat >= 0.0 else -1
            deploy: bool = r_hat > threshold

            raw_size: float = max(r_hat - threshold, 0.0) / self._train_std
            position_size: float = min(max(raw_size, 0.0), 1.0) if deploy else 0.0
            confidence: float = min(magnitude / self._train_std, 1.0) if self._train_std > _MIN_SIGMA else 0.0
            asset: str = _get_asset_name(self._asset_names, i)

            rec: Recommendation = Recommendation(
                asset=asset,
                predicted_strategy_return=r_hat,
                confidence=confidence,
                deploy=deploy,
                predicted_direction=direction,
                predicted_magnitude=magnitude,
                position_size=position_size,
            )
            recommendations.append(rec)

        return recommendations


# ---------------------------------------------------------------------------
# EqualWeightRecommender
# ---------------------------------------------------------------------------


class EqualWeightRecommender:
    """Equal weight to all assets with positive forecast.

    Reads the predicted return from a specified column. All assets
    with a positive forecast receive equal position size (1 / n_positive).

    Attributes:
        config: Column index configuration.
    """

    def __init__(
        self,
        config: ColumnIndexConfig,
        *,
        asset_names: list[str] | None = None,
    ) -> None:
        """Initialise the equal weight recommender.

        Args:
            config: Column index and threshold configuration.
            asset_names: Optional asset labels per sample row.
        """
        self._config: ColumnIndexConfig = config
        self._asset_names: list[str] | None = asset_names
        self._fitted: bool = False

    @property
    def config(self) -> ColumnIndexConfig:
        """Return the configuration.

        Returns:
            Frozen configuration object.
        """
        return self._config

    def fit(
        self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int], np.dtype[np.float64]],  # noqa: ARG002
    ) -> None:
        """Validate inputs and mark as fitted.

        Args:
            x_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Realised strategy returns (ignored).

        Raises:
            ValueError: If inputs are empty or column index is out of bounds.
        """
        n_samples: int = x_train.shape[0]
        if n_samples == 0:
            msg: str = "x_train must contain at least one sample"
            raise ValueError(msg)

        n_features: int = x_train.shape[1]
        if self._config.col_idx >= n_features:
            msg = f"col_idx={self._config.col_idx} is out of bounds for feature matrix with {n_features} columns"
            raise ValueError(msg)

        self._fitted = True

        logger.info(
            "EqualWeightRecommender fitted on {} samples | return_col_idx={}",
            n_samples,
            self._config.col_idx,
        )

    def predict(
        self,
        x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> list[Recommendation]:
        """Predict with equal weight for all positive-forecast assets.

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            List of recommendations.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``x`` is empty.
        """
        if not self._fitted:
            msg: str = "Model has not been fitted — call fit() first"
            raise RuntimeError(msg)

        n_samples: int = x.shape[0]
        if n_samples == 0:
            msg = "x must contain at least one sample"
            raise ValueError(msg)

        col_idx: int = self._config.col_idx
        return_values: np.ndarray[tuple[int], np.dtype[np.float64]] = x[:, col_idx]

        # Count positive forecasts to determine equal weight
        n_positive: int = int(np.sum(return_values > 0.0))
        equal_weight: float = 1.0 / n_positive if n_positive > 0 else 0.0
        # Clamp to [0.0, 1.0] — always true by construction but explicit
        equal_weight = min(equal_weight, 1.0)

        recommendations: list[Recommendation] = []
        for i in range(n_samples):
            r_hat: float = float(return_values[i])
            magnitude: float = abs(r_hat)
            direction: int = 1 if r_hat >= 0.0 else -1
            deploy: bool = r_hat > 0.0
            position_size: float = equal_weight if deploy else 0.0
            asset: str = _get_asset_name(self._asset_names, i)

            rec: Recommendation = Recommendation(
                asset=asset,
                predicted_strategy_return=r_hat,
                confidence=0.5,
                deploy=deploy,
                predicted_direction=direction,
                predicted_magnitude=magnitude,
                position_size=position_size,
            )
            recommendations.append(rec)

        return recommendations


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
