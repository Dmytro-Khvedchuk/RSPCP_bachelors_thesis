"""Shared aggregation logic for standard bar construction.

Provides a vectorised Polars-based aggregation pipeline used by tick,
volume, and dollar bar aggregators.  The only difference between the three
is the *metric expression* that drives bar boundaries.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import polars as pl

from src.app.bars.domain.entities import AggregatedBar
from src.app.bars.domain.value_objects import BarType
from src.app.ohlcv.domain.value_objects import Asset


_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"timestamp", "open", "high", "low", "close", "volume"},
)


def _validate_input(trades: pl.DataFrame) -> None:
    """Check that *trades* contains every required column.

    Args:
        trades: Input DataFrame to validate.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing: frozenset[str] = _REQUIRED_COLUMNS - set(trades.columns)
    if missing:
        msg: str = f"Input DataFrame is missing required columns: {sorted(missing)}"
        raise ValueError(msg)


def _infer_candle_period(trades: pl.DataFrame) -> timedelta:
    """Infer the candle duration from the first two timestamps.

    Falls back to one minute when the duration cannot be determined
    (e.g. single-row input or duplicate timestamps).

    Args:
        trades: Sorted input DataFrame.

    Returns:
        Estimated candle period.
    """
    _min_rows_for_inference: int = 2
    if len(trades) >= _min_rows_for_inference:
        ts_col: pl.Series = trades["timestamp"].sort()
        delta: timedelta = ts_col[1] - ts_col[0]
        if isinstance(delta, timedelta) and delta > timedelta(0):
            return delta
    return timedelta(minutes=1)


def aggregate_by_metric(
    trades: pl.DataFrame,
    *,
    asset: Asset,
    bar_type: BarType,
    threshold: float,
    metric_expr: pl.Expr,
) -> list[AggregatedBar]:
    """Aggregate OHLCV rows into bars based on a cumulative metric.

    Groups consecutive rows until the cumulative value of *metric_expr*
    reaches *threshold*, then starts a new bar.  Buy / sell volume is
    estimated from the close position within the high–low range.

    Args:
        trades: Polars DataFrame with columns ``timestamp``, ``open``,
            ``high``, ``low``, ``close``, ``volume``.
        asset: Trading-pair symbol for the resulting bars.
        bar_type: The bar aggregation type tag.
        threshold: Cumulative metric value at which a new bar begins.
        metric_expr: Polars expression computing the per-row metric
            (e.g. ``pl.lit(1)`` for tick bars, ``pl.col("volume")``
            for volume bars).

    Returns:
        List of aggregated bars ordered by ``start_ts``.
    """
    _validate_input(trades)

    if trades.is_empty():
        return []

    candle_period: timedelta = _infer_candle_period(trades)
    df: pl.DataFrame = trades.sort("timestamp")

    # ── per-row metric & cumulative sum ──────────────────────────────
    df = df.with_columns(metric_expr.alias("_metric"))
    df = df.with_columns(pl.col("_metric").cum_sum().alias("_cumsum"))

    # Bar ID is based on cumulative sum *before* this row's contribution
    # so the row that crosses the threshold is the LAST row of its bar.
    df = df.with_columns(
        ((pl.col("_cumsum") - pl.col("_metric")) / threshold).floor().cast(pl.Int64).alias("_bar_id"),
    )

    # ── buy / sell volume estimation (close position heuristic) ──────
    hl_range: pl.Expr = pl.col("high").cast(pl.Float64) - pl.col("low").cast(pl.Float64)
    buy_frac: pl.Expr = (
        pl.when(hl_range > 0)
        .then((pl.col("close").cast(pl.Float64) - pl.col("low").cast(pl.Float64)) / hl_range)
        .otherwise(0.5)
    )
    typical_price: pl.Expr = (
        pl.col("high").cast(pl.Float64) + pl.col("low").cast(pl.Float64) + pl.col("close").cast(pl.Float64)
    ) / 3.0

    df = df.with_columns(
        (pl.col("volume") * buy_frac).alias("_buy_vol"),
        (pl.col("volume") * (pl.lit(1.0) - buy_frac)).alias("_sell_vol"),
        (typical_price * pl.col("volume")).alias("_dollar_val"),
    )

    # ── group by bar_id → aggregate OHLCV ────────────────────────────
    bars_df: pl.DataFrame = (
        df.group_by("_bar_id", maintain_order=True)
        .agg(
            pl.col("timestamp").first().alias("start_ts"),
            pl.col("timestamp").last().alias("end_ts"),
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.len().alias("tick_count"),
            pl.col("_buy_vol").sum().alias("buy_volume"),
            pl.col("_sell_vol").sum().alias("sell_volume"),
            pl.col("_dollar_val").sum().alias("_total_dollar_val"),
        )
        .sort("_bar_id")
    )

    # VWAP = Σ(typical_price × volume) / Σ(volume); fallback to close
    bars_df = bars_df.with_columns(
        pl.when(pl.col("volume") > 0)
        .then(pl.col("_total_dollar_val") / pl.col("volume"))
        .otherwise(pl.col("close").cast(pl.Float64))
        .alias("vwap"),
    )

    # end_ts should represent the *end* of the last candle, not its start
    bars_df = bars_df.with_columns((pl.col("end_ts") + candle_period).alias("end_ts"))

    # ── convert to domain entities ───────────────────────────────────
    results: list[AggregatedBar] = []
    for row in bars_df.iter_rows(named=True):
        bar: AggregatedBar = AggregatedBar(
            asset=asset,
            bar_type=bar_type,
            start_ts=row["start_ts"],
            end_ts=row["end_ts"],
            open=Decimal(str(row["open"])),
            high=Decimal(str(row["high"])),
            low=Decimal(str(row["low"])),
            close=Decimal(str(row["close"])),
            volume=float(row["volume"]),
            tick_count=int(row["tick_count"]),
            buy_volume=float(row["buy_volume"]),
            sell_volume=float(row["sell_volume"]),
            vwap=Decimal(str(row["vwap"])),
        )
        results.append(bar)

    return results
