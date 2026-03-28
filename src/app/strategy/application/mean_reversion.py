"""Mean reversion strategy — Bollinger Band fade with Hurst exponent regime filter."""

from __future__ import annotations

from typing import Annotated

import polars as pl
from pydantic import BaseModel
from pydantic import Field as PydanticField

from src.app.features.domain.value_objects import FeatureSet


_EPSILON: float = 1e-12
"""Division safety constant to prevent zero-division in bandwidth normalisation."""


class MeanReversionConfig(BaseModel, frozen=True):
    """Configuration for the mean reversion strategy.

    Controls the Bollinger Band parameters (window, number of standard
    deviations) and the Hurst exponent regime filter that gates all
    signals.  A wider band (2.5 sigma) is the default because crypto
    return distributions exhibit excess kurtosis.

    Attributes:
        bb_window: Rolling window for Bollinger Bands mid and std.
        bb_num_std: Number of standard deviations for the upper and lower
            bands.  Default 2.5 accounts for crypto fat tails.
        hurst_column: Name of the pre-computed Hurst exponent column.
        hurst_threshold: Signal only when ``Hurst < threshold`` (indicating
            a mean-reverting regime).
    """

    bb_window: Annotated[
        int,
        PydanticField(
            default=20,
            ge=5,
            description="Bollinger Bands rolling window",
        ),
    ]

    bb_num_std: Annotated[
        float,
        PydanticField(
            default=2.5,
            gt=0.0,
            description="Number of std deviations for bands (2.5 for crypto kurtosis)",
        ),
    ]

    hurst_column: str = "hurst_100"

    hurst_threshold: Annotated[
        float,
        PydanticField(
            default=0.5,
            gt=0.0,
            le=1.0,
            description="Signal only when Hurst < threshold (mean-reverting regime)",
        ),
    ]


class MeanReversion(BaseModel, frozen=True):
    """Bollinger Band mean-reversion strategy with Hurst regime gate.

    Generates ``"long"`` when the close falls below the lower Bollinger
    Band and the Hurst exponent indicates a mean-reverting regime
    (``Hurst < threshold``), ``"short"`` when the close exceeds the
    upper band under the same regime filter, and ``"flat"`` otherwise.
    Signal strength is the distance from the breached band normalised
    by bandwidth.

    Attributes:
        config: Strategy configuration parameters.
    """

    config: MeanReversionConfig = PydanticField(default_factory=MeanReversionConfig)

    @property
    def name(self) -> str:
        """Return strategy identifier.

        Returns:
            Strategy name string.
        """
        return "mean_reversion"

    def generate_signals(self, feature_set: FeatureSet) -> pl.DataFrame:
        """Produce mean-reversion signals gated by Hurst regime filter.

        Long when ``close < lower_band`` and ``Hurst < threshold``,
        short when ``close > upper_band`` and ``Hurst < threshold``,
        flat otherwise.  Strength is proportional to the band-normalised
        distance.

        Args:
            feature_set: Structured output from the feature matrix builder.

        Returns:
            Polars DataFrame with ``timestamp``, ``side``, and ``strength``
            columns.
        """
        df: pl.DataFrame = feature_set.df
        cfg: MeanReversionConfig = self.config
        close: str = "close"

        mid: pl.Expr = pl.col(close).rolling_mean(
            window_size=cfg.bb_window,
            min_samples=cfg.bb_window,
        )
        std: pl.Expr = pl.col(close).rolling_std(
            window_size=cfg.bb_window,
            min_samples=cfg.bb_window,
        )
        upper: pl.Expr = mid + std * cfg.bb_num_std
        lower: pl.Expr = mid - std * cfg.bb_num_std
        width: pl.Expr = upper - lower + pl.lit(_EPSILON)
        hurst_ok: pl.Expr = pl.col(cfg.hurst_column) < cfg.hurst_threshold

        enriched: pl.DataFrame = df.with_columns(
            upper.alias("_bb_upper"),
            lower.alias("_bb_lower"),
            width.alias("_bb_width"),
            hurst_ok.alias("_hurst_ok"),
        )

        signals: pl.DataFrame = enriched.select(
            pl.col("timestamp"),
            pl.when(pl.col("_hurst_ok") & (pl.col(close) < pl.col("_bb_lower")))
            .then(pl.lit("long"))
            .when(pl.col("_hurst_ok") & (pl.col(close) > pl.col("_bb_upper")))
            .then(pl.lit("short"))
            .otherwise(pl.lit("flat"))
            .alias("side"),
            pl.when(pl.col("_hurst_ok") & (pl.col(close) < pl.col("_bb_lower")))
            .then(((pl.col("_bb_lower") - pl.col(close)) / pl.col("_bb_width")).clip(0.0, 1.0))
            .when(pl.col("_hurst_ok") & (pl.col(close) > pl.col("_bb_upper")))
            .then(((pl.col(close) - pl.col("_bb_upper")) / pl.col("_bb_width")).clip(0.0, 1.0))
            .otherwise(pl.lit(0.0))
            .alias("strength"),
        )
        return signals
