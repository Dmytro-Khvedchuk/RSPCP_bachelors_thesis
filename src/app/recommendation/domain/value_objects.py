"""Recommendation domain value objects — input, output, and configuration for the recommender."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from pydantic import model_validator


# ---------------------------------------------------------------------------
# RecommendationInput
# ---------------------------------------------------------------------------


class RecommendationInput(BaseModel, frozen=True):
    """Input to the recommender at a single decision point.

    Bundles the assembled feature vector with the upstream classifier
    direction forecast and regressor return forecast for a given asset
    and timestamp.

    Attributes:
        asset: Asset symbol (e.g. ``"BTCUSDT"``).
        timestamp: Decision-point timestamp.
        feature_vector: Assembled feature vector of shape ``(n_features,)``
            produced by :class:`RecommenderFeatureBuilder`.
        direction_forecast: Predicted direction from the classifier (+1 long / -1 short).
        return_forecast: Predicted return magnitude from the regressor.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    asset: Annotated[
        str,
        PydanticField(min_length=1, description="Asset symbol (e.g. 'BTCUSDT')"),
    ]
    """Asset symbol."""

    timestamp: datetime
    """Decision-point timestamp."""

    feature_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    """Shape ``(n_features,)`` — assembled features from RecommenderFeatureBuilder."""

    direction_forecast: Annotated[
        int,
        PydanticField(description="Predicted direction: +1 (long) or -1 (short)"),
    ]
    """Predicted direction from the classifier (+1 long / -1 short)."""

    return_forecast: float
    """Predicted return magnitude from the regressor."""

    @model_validator(mode="after")
    def _direction_valid(self) -> Self:
        """Ensure direction_forecast is +1 or -1.

        Returns:
            Validated instance.

        Raises:
            ValueError: If direction_forecast is not +1 or -1.
        """
        if self.direction_forecast not in {1, -1}:
            msg: str = f"direction_forecast must be +1 or -1, got {self.direction_forecast}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _feature_vector_1d(self) -> Self:
        """Ensure feature_vector is one-dimensional.

        Returns:
            Validated instance.

        Raises:
            ValueError: If feature_vector is not 1-D.
        """
        if self.feature_vector.ndim != 1:
            msg: str = f"feature_vector must be 1-D, got ndim={self.feature_vector.ndim}"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


class Recommendation(BaseModel, frozen=True):
    """Output from the recommender — a deploy/no-deploy decision with sizing.

    Represents the recommender's prediction of expected strategy return
    (continuous generalised meta-label), a confidence score, and the
    resulting deployment decision and position size.

    Attributes:
        asset: Asset symbol.
        predicted_strategy_return: Predicted strategy return (continuous,
            generalised meta-label).
        confidence: Model confidence in ``[0, 1]``.
        deploy: Whether to deploy the strategy on this asset.
        predicted_direction: +1 (long) or -1 (short).
        predicted_magnitude: Predicted return magnitude (absolute).
        position_size: Position size from generalised meta-label, in ``[0.0, 1.0]``.
            This is the key field that enables continuous meta-labeling —
            not just binary bet / no-bet.
    """

    asset: Annotated[
        str,
        PydanticField(min_length=1, description="Asset symbol"),
    ]
    """Asset symbol."""

    predicted_strategy_return: float
    """Predicted strategy return (continuous, generalised meta-label)."""

    confidence: Annotated[
        float,
        PydanticField(ge=0.0, le=1.0, description="Model confidence in [0, 1]"),
    ]
    """Model confidence in [0, 1]."""

    deploy: bool
    """Whether to deploy the strategy on this asset."""

    predicted_direction: Annotated[
        int,
        PydanticField(description="Predicted direction: +1 (long) or -1 (short)"),
    ]
    """Predicted direction: +1 (long) or -1 (short)."""

    predicted_magnitude: Annotated[
        float,
        PydanticField(ge=0.0, description="Predicted return magnitude (absolute)"),
    ]
    """Predicted return magnitude (absolute)."""

    position_size: Annotated[
        float,
        PydanticField(
            ge=0.0,
            le=1.0,
            description="Position size from generalised meta-label [0.0, 1.0]",
        ),
    ]
    """Position size from generalised meta-label, in [0.0, 1.0]."""

    @model_validator(mode="after")
    def _direction_valid(self) -> Self:
        """Ensure predicted_direction is +1 or -1.

        Returns:
            Validated instance.

        Raises:
            ValueError: If predicted_direction is not +1 or -1.
        """
        if self.predicted_direction not in {1, -1}:
            msg: str = f"predicted_direction must be +1 or -1, got {self.predicted_direction}"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# RecommenderConfig
# ---------------------------------------------------------------------------


class RecommenderConfig(BaseModel, frozen=True):
    """Configuration for the recommender model.

    Controls model selection, training schedule, deployment threshold,
    and label horizon consistency.

    Attributes:
        model_type: Model identifier (e.g. ``"lightgbm"``, ``"random"``, ``"all_assets"``).
        train_window: Number of samples in the training window.
        retrain_frequency: How often to retrain (in number of new samples).
        min_threshold: Minimum predicted return for deployment.
        label_horizon: Horizon for labels (consistent with
            :class:`~src.app.recommendation.application.label_builder.LabelConfig`).
    """

    model_type: Annotated[
        str,
        PydanticField(
            default="lightgbm",
            min_length=1,
            description="Model identifier (e.g. 'lightgbm', 'random', 'all_assets')",
        ),
    ]
    """Model identifier."""

    train_window: Annotated[
        int,
        PydanticField(
            default=500,
            gt=0,
            description="Number of samples in the training window",
        ),
    ]
    """Number of samples in the training window."""

    retrain_frequency: Annotated[
        int,
        PydanticField(
            default=50,
            gt=0,
            description="Retrain after this many new samples",
        ),
    ]
    """How often to retrain (in number of new samples)."""

    min_threshold: Annotated[
        float,
        PydanticField(
            default=0.0,
            ge=0.0,
            description="Minimum predicted return for deployment",
        ),
    ]
    """Minimum predicted return for deployment."""

    label_horizon: Annotated[
        int,
        PydanticField(
            default=7,
            gt=0,
            description="Horizon for labels (consistent with LabelConfig)",
        ),
    ]
    """Horizon for labels (consistent with LabelConfig)."""
