"""BTC-lagged cross-asset features for altcoin models.

RC2 Section 5.4 confirms BTC Granger-causes all altcoins at lag 1
(p < 0.05).  This module constructs BTC-lagged features for altcoin
models via asof join + temporal shift, ensuring no future leakage.

Feature naming convention:
    ``btc_logret_lag1``    -- BTC log return at t-1.
    ``btc_rv_lag1``        -- BTC realized volatility at t-1.
    ``btc_direction_lag1`` -- sign(BTC return) at t-1 (+1 or -1).
"""

from __future__ import annotations

from typing import Annotated, Final

import polars as pl
from pydantic import BaseModel
from pydantic import Field as PydanticField

from src.app.features.application.indicators import log_return, realized_vol


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BTC_ASSET: Final[str] = "BTCUSDT"
"""Asset identifier for BTC — used to guard against self-referencing."""

_TIMESTAMP_COL: Final[str] = "timestamp"
"""Default timestamp column for asof join."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class CrossAssetConfig(BaseModel, frozen=True):
    """Configuration for BTC-lagged cross-asset features.

    Controls the realized volatility window and column naming for
    BTC-lagged features added to altcoin DataFrames.

    Attributes:
        rv_window: Rolling window for BTC realized volatility (default 24,
            matching the project's ``rv_24`` indicator).
        timestamp_col: Column name used for the asof join.
        close_col: BTC close price column name.
    """

    rv_window: Annotated[
        int,
        PydanticField(
            default=24,
            ge=2,
            description="Rolling window for BTC realized volatility",
        ),
    ]

    timestamp_col: str = "timestamp"
    """Column name used for temporal alignment (asof join)."""

    close_col: str = "close"
    """BTC close price column name."""


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

_BTC_LOGRET_LAG1: Final[str] = "btc_logret_lag1"
_BTC_RV_LAG1: Final[str] = "btc_rv_lag1"
_BTC_DIRECTION_LAG1: Final[str] = "btc_direction_lag1"


def get_cross_asset_column_names() -> list[str]:
    """Return the sorted list of BTC-lagged cross-asset feature column names.

    Returns:
        Sorted list of cross-asset feature column names.
    """
    return sorted([_BTC_LOGRET_LAG1, _BTC_RV_LAG1, _BTC_DIRECTION_LAG1])


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def is_btc_asset(asset: str) -> bool:
    """Check whether the given asset identifier refers to BTC.

    BTC models must NOT include BTC-lagged features (self-referencing).

    Args:
        asset: Asset ticker string (e.g. ``"BTCUSDT"``).

    Returns:
        True if the asset is BTC.
    """
    return asset.upper() == _BTC_ASSET


def _prepare_btc_features(
    btc_df: pl.DataFrame,
    config: CrossAssetConfig,
) -> pl.DataFrame:
    """Compute BTC features (log return, realized vol, direction) without lag.

    These are the *contemporaneous* BTC features.  The temporal lag is
    applied after the asof join to ensure correct alignment.

    Args:
        btc_df: BTC OHLCV DataFrame with at least ``timestamp`` and
            ``close`` columns.
        config: Cross-asset feature configuration.

    Returns:
        DataFrame with columns: ``timestamp``, ``_btc_logret``,
        ``_btc_rv``, ``_btc_direction``.
    """
    close_expr: pl.Expr = pl.col(config.close_col)
    logret_expr: pl.Expr = log_return(close_expr, periods=1)
    rv_expr: pl.Expr = realized_vol(logret_expr, window=config.rv_window)

    # Direction: sign of 1-bar log return, zero → +1
    direction_expr: pl.Expr = (
        pl.when(logret_expr > 0.0)
        .then(pl.lit(1, dtype=pl.Int8))
        .when(logret_expr < 0.0)
        .then(pl.lit(-1, dtype=pl.Int8))
        .when(logret_expr.is_null())
        .then(pl.lit(None, dtype=pl.Int8))
        .otherwise(pl.lit(1, dtype=pl.Int8))  # zero return → +1
    )

    btc_features: pl.DataFrame = btc_df.select(
        pl.col(config.timestamp_col),
        logret_expr.alias("_btc_logret"),
        rv_expr.alias("_btc_rv"),
        direction_expr.alias("_btc_direction"),
    )
    return btc_features


def add_btc_lagged_features(
    altcoin_df: pl.DataFrame,
    btc_df: pl.DataFrame,
    config: CrossAssetConfig | None = None,
) -> pl.DataFrame:
    """Add BTC-lagged cross-asset features to an altcoin DataFrame.

    Methodology:
        1. Compute BTC log return, realized volatility, and direction
           from the BTC DataFrame.
        2. Asof join on timestamp to align BTC features with the altcoin
           bar schedule (forward strategy = pick latest BTC bar <= altcoin
           timestamp).
        3. Shift all BTC features by 1 bar **backward** (``shift(1)``) to
           ensure no temporal leakage — the altcoin bar at time *t* sees
           BTC features from time *t-1* at most.

    Leakage Safety:
        The asof join with ``strategy="backward"`` ensures we never look
        ahead in BTC data.  The additional ``shift(1)`` guarantees that
        even if BTC and altcoin bars are perfectly synchronised, we use
        only strictly past BTC information.

    Args:
        altcoin_df: Altcoin OHLCV DataFrame with a ``timestamp`` column.
        btc_df: BTC OHLCV DataFrame with ``timestamp`` and ``close``.
        config: Cross-asset feature configuration (uses defaults if None).

    Returns:
        Altcoin DataFrame enriched with BTC-lagged feature columns:
        ``btc_logret_lag1``, ``btc_rv_lag1``, ``btc_direction_lag1``.
    """
    if config is None:
        config = CrossAssetConfig()  # ty: ignore[missing-argument]

    ts_col: str = config.timestamp_col

    # Step 1: compute contemporaneous BTC features
    btc_features: pl.DataFrame = _prepare_btc_features(btc_df, config)

    # Step 2: asof join — backward strategy picks latest BTC bar <= altcoin ts
    joined: pl.DataFrame = altcoin_df.join_asof(
        btc_features.sort(ts_col),
        on=ts_col,
        strategy="backward",
    )

    # Step 3: shift BTC features by 1 bar to prevent leakage, then rename
    result: pl.DataFrame = joined.with_columns(
        pl.col("_btc_logret").shift(1).alias(_BTC_LOGRET_LAG1),
        pl.col("_btc_rv").shift(1).alias(_BTC_RV_LAG1),
        pl.col("_btc_direction").shift(1).alias(_BTC_DIRECTION_LAG1),
    ).drop(["_btc_logret", "_btc_rv", "_btc_direction"])

    return result
