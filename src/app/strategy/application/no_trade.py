"""No-trade strategy — always flat with avoidance-confidence strength signal."""

from __future__ import annotations

from typing import Annotated

import polars as pl
from pydantic import BaseModel
from pydantic import Field as PydanticField

from src.app.features.domain.value_objects import FeatureSet


class NoTradeConfig(BaseModel, frozen=True):
    """Configuration for the no-trade strategy.

    Controls the global permutation entropy gate and the per-bar
    low-volatility filter.  When the market exhibits near-random
    behaviour (high PE), all bars are flat with maximum conviction.
    Otherwise, individual bars with realised vol below the threshold
    are flagged as untradeable.

    Attributes:
        pe_threshold: Permutation entropy above this value triggers the
            global "market is random" gate (all bars flat, strength 1.0).
        pe_value: Pre-computed global permutation entropy from the
            statistical profiling module.
        rv_column: Name of the pre-computed realised volatility column.
        low_vol_threshold: Realised vol below this value makes a bar
            untradeable (strength 1.0 for that bar).
    """

    pe_threshold: Annotated[
        float,
        PydanticField(
            default=0.98,
            gt=0.0,
            le=1.0,
            description="PE above this -> all bars flat",
        ),
    ]

    pe_value: Annotated[
        float,
        PydanticField(
            default=1.0,
            ge=0.0,
            le=1.0,
            description="Global permutation entropy from profiling (pre-computed)",
        ),
    ]

    rv_column: str = "rv_24"

    low_vol_threshold: Annotated[
        float,
        PydanticField(
            default=0.0,
            ge=0.0,
            description="RV below this -> flat for that bar",
        ),
    ]


class NoTrade(BaseModel, frozen=True):
    """Always-flat strategy encoding the decision not to trade.

    Produces ``"flat"`` for every bar.  The ``strength`` column encodes
    avoidance confidence: ``1.0`` means "definitely do not trade this
    bar", ``0.0`` means "no particular reason to avoid".

    Two filters are applied hierarchically:

    1. **Global PE gate** — if the pre-computed permutation entropy
       exceeds ``pe_threshold``, the entire series is deemed random
       and all bars receive ``strength = 1.0``.
    2. **Per-bar low-vol filter** — bars whose realised vol falls below
       ``low_vol_threshold`` receive ``strength = 1.0``.

    Attributes:
        config: Strategy configuration parameters.
    """

    config: NoTradeConfig = PydanticField(default_factory=NoTradeConfig)

    @property
    def name(self) -> str:
        """Return strategy identifier.

        Returns:
            Strategy name string.
        """
        return "no_trade"

    def generate_signals(self, feature_set: FeatureSet) -> pl.DataFrame:
        """Produce always-flat signals with avoidance-confidence strength.

        When the global PE gate is active, all bars get ``strength=1.0``.
        Otherwise, low-vol bars get ``strength=1.0`` and the rest get
        ``strength=0.0``.

        Args:
            feature_set: Structured output from the feature matrix builder.

        Returns:
            Polars DataFrame with ``timestamp``, ``side``, and ``strength``
            columns.
        """
        df: pl.DataFrame = feature_set.df
        cfg: NoTradeConfig = self.config

        # Global PE check — if market is near-random, all bars flat with max strength
        if cfg.pe_value > cfg.pe_threshold:
            signals: pl.DataFrame = df.select(
                pl.col("timestamp"),
                pl.lit("flat").alias("side"),
                pl.lit(1.0).alias("strength"),
            )
            return signals

        # Per-bar: low-vol bars get strength=1.0, others get strength=0.0
        return df.select(
            pl.col("timestamp"),
            pl.lit("flat").alias("side"),
            pl.when(pl.col(cfg.rv_column) < cfg.low_vol_threshold)
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
            .alias("strength"),
        )
