"""Typer CLI for bar aggregation — constructs Lopez de Prado alternative bars from OHLCV data.

Run via ``python -m src.app.bars.cli aggregate --help`` or the ``just bars``
recipe defined in ``justfile``.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Annotated, Any

import polars as pl
import typer
from loguru import logger
from pydantic import ValidationError

from src.app.bars.application.dollar_bars import DollarBarAggregator
from src.app.bars.application.imbalance_bars import ImbalanceBarAggregator
from src.app.bars.application.run_bars import RunBarAggregator
from src.app.bars.application.tick_bars import TickBarAggregator
from src.app.bars.application.volume_bars import VolumeBarAggregator
from src.app.bars.domain.entities import AggregatedBar
from src.app.bars.domain.protocols import IBarAggregator
from src.app.bars.domain.value_objects import BarConfig, BarType
from src.app.bars.infrastructure.duckdb_repository import DuckDBBarRepository
from src.app.ohlcv.domain.value_objects import Asset, DateRange, Timeframe
from src.app.ohlcv.infrastructure.duckdb_repository import DuckDBOHLCVRepository
from src.app.system.database.connection import ConnectionManager
from src.app.system.database.settings import DatabaseSettings
from src.app.system.logging import setup_logging


app: typer.Typer = typer.Typer(
    name="bars",
    help="Aggregate OHLCV candles into Lopez de Prado alternative bars.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Aggregator registry
# ---------------------------------------------------------------------------

_AGGREGATORS: dict[BarType, type[IBarAggregator]] = {
    BarType.TICK: TickBarAggregator,
    BarType.VOLUME: VolumeBarAggregator,
    BarType.DOLLAR: DollarBarAggregator,
    BarType.TICK_IMBALANCE: ImbalanceBarAggregator,
    BarType.VOLUME_IMBALANCE: ImbalanceBarAggregator,
    BarType.DOLLAR_IMBALANCE: ImbalanceBarAggregator,
    BarType.TICK_RUN: RunBarAggregator,
    BarType.VOLUME_RUN: RunBarAggregator,
    BarType.DOLLAR_RUN: RunBarAggregator,
}

# Default thresholds per bar type (sensible starting points for BTC hourly data).
_DEFAULT_THRESHOLDS: dict[BarType, float] = {
    BarType.TICK: 1000.0,
    BarType.VOLUME: 50_000.0,
    BarType.DOLLAR: 1_000_000_000.0,
    BarType.TICK_IMBALANCE: 500.0,
    BarType.VOLUME_IMBALANCE: 25_000.0,
    BarType.DOLLAR_IMBALANCE: 500_000_000.0,
    BarType.TICK_RUN: 500.0,
    BarType.VOLUME_RUN: 25_000.0,
    BarType.DOLLAR_RUN: 500_000_000.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_assets(raw: str) -> list[Asset]:
    """Parse comma-separated asset symbols.

    Args:
        raw: Comma-separated symbols (e.g. ``"BTCUSDT,ETHUSDT"``).

    Returns:
        Validated Asset list.

    Raises:
        typer.BadParameter: If no symbols provided or a symbol is invalid.
    """
    symbols: list[str] = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not symbols:
        raise typer.BadParameter("At least one asset symbol is required.")

    assets: list[Asset] = []
    for symbol in symbols:
        try:
            assets.append(Asset(symbol=symbol))
        except ValidationError as exc:
            raise typer.BadParameter(f"Invalid asset symbol '{symbol}': {exc}") from exc
    return assets


def _parse_bar_types(raw: str) -> list[BarType]:
    """Parse comma-separated bar type names.

    Args:
        raw: Comma-separated bar types (e.g. ``"tick,dollar,tick_imbalance"``).

    Returns:
        Validated BarType list.

    Raises:
        typer.BadParameter: If no types provided or a type name is invalid.
    """
    valid_values: dict[str, BarType] = {bt.value: bt for bt in BarType}
    names: list[str] = [s.strip().lower() for s in raw.split(",") if s.strip()]
    if not names:
        raise typer.BadParameter("At least one bar type is required.")

    bar_types: list[BarType] = []
    for name in names:
        if name not in valid_values:
            raise typer.BadParameter(f"Invalid bar type '{name}'. Valid values: {sorted(valid_values.keys())}")
        bar_types.append(valid_values[name])
    return bar_types


def _parse_date(raw: str) -> datetime:
    """Parse an ISO-8601 date string into a UTC-aware datetime.

    Args:
        raw: ISO-8601 string (e.g. ``"2020-01-01"``).

    Returns:
        UTC-aware datetime.

    Raises:
        typer.BadParameter: If the string is not valid ISO-8601.
    """
    try:
        parsed: datetime = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise typer.BadParameter(f"Cannot parse date '{raw}' — expected ISO-8601 format.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _aggregate_single_bar_type(  # noqa: PLR0913, PLR0917
    trades_df: pl.DataFrame,
    asset: Asset,
    bar_type: BarType,
    threshold: float,
    ewm_span: int,
    warmup_period: int,
    bar_repo: DuckDBBarRepository,
) -> int:
    """Aggregate one bar type for a single asset and persist results.

    Args:
        trades_df: OHLCV candles as Polars DataFrame.
        asset: Trading pair.
        bar_type: Bar aggregation type.
        threshold: Sampling threshold (0 uses built-in default).
        ewm_span: EWM span for adaptive threshold estimation.
        warmup_period: Warmup period before adaptive thresholds.
        bar_repo: Repository for persisting bars.

    Returns:
        Number of rows written.
    """
    bar_threshold: float = threshold if threshold > 0 else _DEFAULT_THRESHOLDS.get(bar_type, 1000.0)

    config: BarConfig = BarConfig(
        bar_type=bar_type,
        threshold=bar_threshold,
        ewm_span=ewm_span,
        warmup_period=warmup_period,
    )

    aggregator_cls: type[IBarAggregator] | None = _AGGREGATORS.get(bar_type)
    if aggregator_cls is None:
        logger.warning("No aggregator for bar type '{}' — skipping.", bar_type.value)
        return 0

    aggregator: IBarAggregator = aggregator_cls()

    t1: float = time.perf_counter()
    bars: list[AggregatedBar] = aggregator.aggregate(trades_df, asset=asset, config=config)
    agg_elapsed: float = time.perf_counter() - t1

    if not bars:
        logger.warning(
            "  {} / {} -> 0 bars (threshold={} may be too high).",
            asset.symbol,
            bar_type.value,
            bar_threshold,
        )
        return 0

    config_hash: str = config.config_hash
    written: int = bar_repo.ingest(bars, config_hash=config_hash)

    logger.info(
        "  {} / {} -> {} bars aggregated in {:.2f}s, {} written (hash={})",
        asset.symbol,
        bar_type.value,
        len(bars),
        agg_elapsed,
        written,
        config_hash,
    )
    return written


def _load_ohlcv_as_polars(
    ohlcv_repo: DuckDBOHLCVRepository,
    asset: Asset,
    timeframe: Timeframe,
    date_range: DateRange,
) -> pl.DataFrame:
    """Load OHLCV candles and convert to the Polars DataFrame expected by aggregators.

    Args:
        ohlcv_repo: Repository for OHLCV data.
        asset: Trading pair.
        timeframe: Candle interval.
        date_range: Date boundaries.

    Returns:
        Polars DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    candles: list[Any] = ohlcv_repo.query(asset, timeframe, date_range)
    if not candles:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )

    rows: list[dict[str, Any]] = [
        {
            "timestamp": c.timestamp.astimezone(UTC),
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": c.volume,
        }
        for c in candles
    ]
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@app.command()
def aggregate(  # noqa: PLR0913, PLR0917
    assets: Annotated[
        str,
        typer.Option("--assets", "-a", help="Comma-separated trading-pair symbols (e.g. BTCUSDT,ETHUSDT)."),
    ],
    bar_types: Annotated[
        str,
        typer.Option(
            "--bar-types",
            "-b",
            help="Comma-separated bar types (e.g. tick,volume,dollar,tick_imbalance,tick_run).",
        ),
    ],
    timeframe: Annotated[
        str,
        typer.Option("--timeframe", "-t", help="Source OHLCV timeframe (1h, 4h, or 1d). Defaults to 1h."),
    ] = "1h",
    start: Annotated[
        str,
        typer.Option("--start", "-s", help="Start date in ISO-8601 format (e.g. 2020-01-01)."),
    ] = "2020-01-01",
    end: Annotated[
        str,
        typer.Option("--end", "-e", help="End date in ISO-8601 format. Defaults to now (UTC)."),
    ] = "",
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="Override the default threshold for all bar types."),
    ] = 0.0,
    ewm_span: Annotated[
        int,
        typer.Option("--ewm-span", help="EWM span for information-driven bars. Defaults to 100."),
    ] = 100,
    warmup_period: Annotated[
        int,
        typer.Option("--warmup", help="Warmup period for information-driven bars. Defaults to 100."),
    ] = 100,
    log_level: Annotated[
        str,
        typer.Option("--log-level", "-l", help="Minimum log level (DEBUG, INFO, WARNING, ERROR)."),
    ] = "INFO",
) -> None:
    r"""Aggregate OHLCV candles into alternative bars and store in DuckDB.

    Reads OHLCV data for each ``--assets x --bar-types`` combination, runs the
    corresponding Lopez de Prado bar aggregator, and persists the results via
    ``INSERT OR IGNORE`` (duplicates are skipped).

    Examples::

        # Standard bars with default thresholds
        just bars --assets BTCUSDT,ETHUSDT --bar-types tick,volume,dollar

        # Information-driven bars with custom threshold
        just bars --assets BTCUSDT --bar-types tick_imbalance,tick_run --threshold 500

        # All 9 bar types for BTC
        just bars --assets BTCUSDT \
            --bar-types tick,volume,dollar,tick_imbalance,volume_imbalance,\
            dollar_imbalance,tick_run,volume_run,dollar_run

    Args:
        assets: Comma-separated trading-pair symbols.
        bar_types: Comma-separated bar type names.
        timeframe: Source OHLCV timeframe. Defaults to ``"1h"``.
        start: Aggregation start date. Defaults to ``"2020-01-01"``.
        end: Aggregation end date. Defaults to now (UTC).
        threshold: Custom threshold for all bar types. When 0, uses built-in defaults.
        ewm_span: EWM span for adaptive threshold estimation. Defaults to 100.
        warmup_period: Warmup period before adaptive thresholds. Defaults to 100.
        log_level: Minimum log level for the session. Defaults to ``"INFO"``.
    """
    setup_logging(level=log_level)

    parsed_assets: list[Asset] = _parse_assets(assets)
    parsed_bar_types: list[BarType] = _parse_bar_types(bar_types)
    start_dt: datetime = _parse_date(start)
    end_dt: datetime = datetime.now(UTC) if end == "" else _parse_date(end)
    parsed_timeframe: Timeframe = Timeframe(timeframe)
    date_range: DateRange = DateRange(start=start_dt, end=end_dt)

    logger.info(
        "Starting bar aggregation | assets={} bar_types={} timeframe={} start={} end={}",
        [a.symbol for a in parsed_assets],
        [bt.value for bt in parsed_bar_types],
        parsed_timeframe.value,
        start_dt.isoformat(),
        end_dt.isoformat(),
    )

    grand_total: int = 0

    with ConnectionManager(DatabaseSettings()) as cm:
        ohlcv_repo: DuckDBOHLCVRepository = DuckDBOHLCVRepository(cm)
        bar_repo: DuckDBBarRepository = DuckDBBarRepository(cm)

        for asset in parsed_assets:
            # Load OHLCV once per asset, reuse for all bar types.
            t0: float = time.perf_counter()
            trades_df: pl.DataFrame = _load_ohlcv_as_polars(ohlcv_repo, asset, parsed_timeframe, date_range)
            load_elapsed: float = time.perf_counter() - t0

            if trades_df.is_empty():
                logger.warning("No OHLCV data for {} / {} in range — skipping.", asset.symbol, parsed_timeframe.value)
                continue

            logger.info(
                "Loaded {} OHLCV candles for {} in {:.2f}s",
                len(trades_df),
                asset.symbol,
                load_elapsed,
            )

            for bar_type in parsed_bar_types:
                grand_total += _aggregate_single_bar_type(
                    trades_df=trades_df,
                    asset=asset,
                    bar_type=bar_type,
                    threshold=threshold,
                    ewm_span=ewm_span,
                    warmup_period=warmup_period,
                    bar_repo=bar_repo,
                )

    logger.success("Bar aggregation complete | total_rows_written={}", grand_total)


if __name__ == "__main__":
    app()
