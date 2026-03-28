"""Volatility targeting strategy — always-long with inverse-vol position sizing."""

from __future__ import annotations

from typing import Annotated

import polars as pl
from pydantic import BaseModel
from pydantic import Field as PydanticField

from src.app.features.domain.value_objects import FeatureSet


_EPSILON: float = 1e-12
"""Division safety constant to prevent zero-division in inverse-vol calculation."""


class VolatilityTargetingConfig(BaseModel, frozen=True):
    """Configuration for the volatility targeting strategy.

    Controls the annualised target volatility level and the pre-computed
    realised volatility column.  The strength signal is inversely
    proportional to current realised vol, encoding a "trade smaller when
    vol is high" heuristic.

    Attributes:
        target_vol: Target volatility level (must be on the same scale
            as the ``rv_column`` values — typically annualised or
            per-bar, depending on the feature pipeline).
        rv_column: Name of the pre-computed realised volatility column
            in the feature set.
    """

    target_vol: Annotated[
        float,
        PydanticField(
            default=0.15,
            gt=0.0,
            le=1.0,
            description="Target volatility level (same scale as rv column)",
        ),
    ]

    rv_column: str = "rv_24"


class VolatilityTargeting(BaseModel, frozen=True):
    """Always-long strategy with inverse-volatility signal strength.

    Every bar is marked ``"long"``.  Signal strength equals
    ``target_vol / realised_vol``, clipped to ``[0, 1]``.  When
    realised vol is high relative to the target, strength drops,
    signalling the downstream position sizer to reduce exposure.

    Attributes:
        config: Strategy configuration parameters.
    """

    config: VolatilityTargetingConfig = PydanticField(default_factory=VolatilityTargetingConfig)

    @property
    def name(self) -> str:
        """Return strategy identifier.

        Returns:
            Strategy name string.
        """
        return "volatility_targeting"

    def generate_signals(self, feature_set: FeatureSet) -> pl.DataFrame:
        """Produce always-long signals with inverse-vol strength scaling.

        Strength is ``target_vol / rv``, clipped to ``[0, 1]``.

        Args:
            feature_set: Structured output from the feature matrix builder.

        Returns:
            Polars DataFrame with ``timestamp``, ``side``, and ``strength``
            columns.
        """
        df: pl.DataFrame = feature_set.df
        cfg: VolatilityTargetingConfig = self.config

        signals: pl.DataFrame = df.select(
            pl.col("timestamp"),
            pl.lit("long").alias("side"),
            (pl.lit(cfg.target_vol) / (pl.col(cfg.rv_column) + pl.lit(_EPSILON))).clip(0.0, 1.0).alias("strength"),
        )
        return signals
