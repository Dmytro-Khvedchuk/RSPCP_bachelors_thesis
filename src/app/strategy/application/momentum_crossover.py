"""Momentum crossover strategy — EMA crossover signal with configurable threshold."""

from __future__ import annotations

from typing import Annotated

import polars as pl
from pydantic import BaseModel
from pydantic import Field as PydanticField

from src.app.features.domain.value_objects import FeatureSet


class MomentumCrossoverConfig(BaseModel, frozen=True):
    """Configuration for the momentum crossover strategy.

    Controls which feature column carries the EMA crossover signal,
    the minimum absolute crossover magnitude to generate a directional
    signal, and ATR-based stop-loss / take-profit multipliers reserved
    for the downstream adapter layer (not used in signal generation).

    Attributes:
        xover_column: Name of the EMA crossover column in the feature set.
        signal_threshold: Minimum ``|ema_xover|`` to trigger a directional
            signal.  Values below this threshold produce ``"flat"``.
        atr_multiplier_sl: ATR multiplier for stop-loss distance (reserved
            for the adapter layer that bridges to the backtest engine).
        atr_multiplier_tp: ATR multiplier for take-profit distance (reserved
            for the adapter layer that bridges to the backtest engine).
    """

    xover_column: str = "ema_xover_8_21"

    signal_threshold: Annotated[
        float,
        PydanticField(
            default=0.0,
            ge=0.0,
            description="Minimum |ema_xover| to generate a directional signal",
        ),
    ]

    atr_multiplier_sl: Annotated[
        float,
        PydanticField(
            default=2.0,
            gt=0.0,
            description="ATR multiplier for stop-loss (reserved for adapter layer)",
        ),
    ]

    atr_multiplier_tp: Annotated[
        float,
        PydanticField(
            default=3.0,
            gt=0.0,
            description="ATR multiplier for take-profit (reserved for adapter layer)",
        ),
    ]


class MomentumCrossover(BaseModel, frozen=True):
    """EMA crossover strategy producing directional signals.

    Generates ``"long"`` when the crossover column exceeds the positive
    threshold, ``"short"`` when it falls below the negative threshold,
    and ``"flat"`` otherwise.  Signal strength equals the clipped
    absolute crossover value.

    Attributes:
        config: Strategy configuration parameters.
    """

    config: MomentumCrossoverConfig = PydanticField(default_factory=MomentumCrossoverConfig)

    @property
    def name(self) -> str:
        """Return strategy identifier.

        Returns:
            Strategy name string.
        """
        return "momentum_crossover"

    def generate_signals(self, feature_set: FeatureSet) -> pl.DataFrame:
        """Produce directional signals from the EMA crossover feature.

        Long when ``ema_xover > threshold``, short when
        ``ema_xover < -threshold``, flat otherwise.  Strength is the
        clipped absolute crossover value in ``[0, 1]``.

        Args:
            feature_set: Structured output from the feature matrix builder.

        Returns:
            Polars DataFrame with ``timestamp``, ``side``, and ``strength``
            columns.
        """
        df: pl.DataFrame = feature_set.df
        xover: str = self.config.xover_column
        thr: float = self.config.signal_threshold

        signals: pl.DataFrame = df.select(
            pl.col("timestamp"),
            pl.when(pl.col(xover) > thr)
            .then(pl.lit("long"))
            .when(pl.col(xover) < -thr)
            .then(pl.lit("short"))
            .otherwise(pl.lit("flat"))
            .alias("side"),
            pl.when(pl.col(xover).abs() > thr)
            .then(pl.col(xover).abs().clip(0.0, 1.0))
            .otherwise(pl.lit(0.0))
            .alias("strength"),
        )
        return signals
