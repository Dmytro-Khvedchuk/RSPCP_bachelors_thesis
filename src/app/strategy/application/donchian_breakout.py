"""Donchian breakout strategy — long-only channel breakout with ATR-scaled strength."""

from __future__ import annotations

from typing import Annotated

import polars as pl
from pydantic import BaseModel
from pydantic import Field as PydanticField

from src.app.features.domain.value_objects import FeatureSet


_EPSILON: float = 1e-12
"""Division safety constant to prevent zero-division in strength calculation."""


class DonchianBreakoutConfig(BaseModel, frozen=True):
    """Configuration for the Donchian breakout strategy.

    Controls the channel lookback period, the pre-computed ATR column
    used for strength normalisation, and an ATR multiplier reserved for
    trailing-stop logic in the downstream adapter layer.

    Attributes:
        channel_period: Number of bars for the rolling high channel.
        atr_column: Name of the pre-computed ATR column in the feature set.
        atr_multiplier: ATR multiplier for trailing stop (reserved for the
            adapter layer that bridges to the backtest engine).
    """

    channel_period: Annotated[
        int,
        PydanticField(
            default=20,
            ge=5,
            description="Lookback period for Donchian channel",
        ),
    ]

    atr_column: str = "atr_14"

    atr_multiplier: Annotated[
        float,
        PydanticField(
            default=2.0,
            gt=0.0,
            description="ATR multiplier for trailing stop (reserved for adapter layer)",
        ),
    ]


class DonchianBreakout(BaseModel, frozen=True):
    """Long-only Donchian channel breakout strategy.

    Generates ``"long"`` when the close price exceeds the upper Donchian
    channel (rolling max of the *previous* bars' highs), and ``"flat"``
    otherwise.  Signal strength is proportional to the breakout distance
    normalised by ATR.

    The upper channel is computed with ``shift(1)`` before
    ``rolling_max`` to prevent look-ahead bias — the channel for bar
    *t* uses only bars up to *t-1*.

    Attributes:
        config: Strategy configuration parameters.
    """

    config: DonchianBreakoutConfig = PydanticField(default_factory=DonchianBreakoutConfig)

    @property
    def name(self) -> str:
        """Return strategy identifier.

        Returns:
            Strategy name string.
        """
        return "donchian_breakout"

    def generate_signals(self, feature_set: FeatureSet) -> pl.DataFrame:
        """Produce long-only breakout signals from the Donchian channel.

        Long when ``close > upper_channel``, flat otherwise.  Strength
        equals the ATR-normalised breakout distance clipped to ``[0, 1]``.

        Args:
            feature_set: Structured output from the feature matrix builder.

        Returns:
            Polars DataFrame with ``timestamp``, ``side``, and ``strength``
            columns.
        """
        df: pl.DataFrame = feature_set.df
        period: int = self.config.channel_period
        atr_col: str = self.config.atr_column

        upper: pl.Expr = (
            pl.col("high")
            .shift(1)
            .rolling_max(
                window_size=period,
                min_samples=period,
            )
        )
        enriched: pl.DataFrame = df.with_columns(upper.alias("_dc_upper"))

        signals: pl.DataFrame = enriched.select(
            pl.col("timestamp"),
            pl.when(pl.col("close") > pl.col("_dc_upper"))
            .then(pl.lit("long"))
            .otherwise(pl.lit("flat"))
            .alias("side"),
            pl.when(pl.col("close") > pl.col("_dc_upper"))
            .then(
                ((pl.col("close") - pl.col("_dc_upper")) / (pl.col(atr_col) + _EPSILON)).clip(
                    0.0,
                    1.0,
                ),
            )
            .otherwise(pl.lit(0.0))
            .alias("strength"),
        )
        return signals
