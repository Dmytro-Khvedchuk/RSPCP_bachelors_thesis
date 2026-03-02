"""Typer CLI entry point for OHLCV data ingestion from Binance into DuckDB.

Run via ``python -m src.app.ingestion.cli ingest --help`` or the ``just ingest``
recipe defined in ``justfile``.
"""

from __future__ import annotations

from datetime import datetime, UTC
from typing import Annotated

import typer
from loguru import logger
from pydantic import ValidationError

from src.app.ingestion.application.commands import IngestUniverseCommand
from src.app.ingestion.application.services import IngestionService
from src.app.ingestion.infrastructure.binance_fetcher import BinanceFetcher
from src.app.ingestion.infrastructure.settings import BinanceSettings
from src.app.ohlcv.domain.value_objects import Asset, DateRange, Timeframe
from src.app.ohlcv.infrastructure.duckdb_repository import DuckDBOHLCVRepository
from src.app.system.database.connection import ConnectionManager
from src.app.system.database.settings import DatabaseSettings
from src.app.system.logging import setup_logging


app: typer.Typer = typer.Typer(
    name="ingest",
    help="OHLCV data ingestion from Binance into DuckDB.",
    no_args_is_help=True,
)


def _parse_assets(raw: str) -> list[Asset]:
    """Parse a comma-separated string of trading-pair symbols into ``Asset`` objects.

    Args:
        raw: Comma-separated symbols, e.g. ``"BTCUSDT,ETHUSDT"``.

    Returns:
        A list of validated :class:`Asset` value objects.

    Raises:
        typer.BadParameter: If any symbol fails ``Asset`` validation.
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


def _parse_timeframes(raw: str) -> list[Timeframe]:
    """Parse a comma-separated string of interval values into ``Timeframe`` enums.

    Valid values are the string values of :class:`Timeframe`: ``1h``, ``4h``, ``1d``.

    Args:
        raw: Comma-separated intervals, e.g. ``"1h,4h"``.

    Returns:
        A list of validated :class:`Timeframe` enum members.

    Raises:
        typer.BadParameter: If any interval is not a valid ``Timeframe`` value.
    """
    valid_values: set[str] = {tf.value for tf in Timeframe}
    intervals: list[str] = [s.strip() for s in raw.split(",") if s.strip()]

    if not intervals:
        raise typer.BadParameter("At least one timeframe interval is required.")

    timeframes: list[Timeframe] = []
    for interval in intervals:
        if interval not in valid_values:
            raise typer.BadParameter(f"Invalid timeframe '{interval}'. Valid values: {sorted(valid_values)}")
        timeframes.append(Timeframe(interval))

    return timeframes


def _parse_date(raw: str) -> datetime:
    """Parse an ISO-8601 date string into a UTC-aware :class:`datetime`.

    Naive datetimes (no timezone offset) are assumed to be UTC.

    Args:
        raw: ISO-8601 string, e.g. ``"2020-01-01"`` or ``"2020-01-01T00:00:00"``.

    Returns:
        A timezone-aware :class:`datetime` in UTC.

    Raises:
        typer.BadParameter: If the string cannot be parsed as ISO-8601.
    """
    try:
        parsed: datetime = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise typer.BadParameter(f"Cannot parse date '{raw}' — expected ISO-8601 format (e.g. 2020-01-01).") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)

    return parsed


def _build_service(cm: ConnectionManager) -> IngestionService:
    """Wire up the dependency injection graph and return a ready :class:`IngestionService`.

    Args:
        cm: An initialised :class:`ConnectionManager` used to build the repository.

    Returns:
        A fully wired :class:`IngestionService` ready to execute ingestion commands.
    """
    settings: BinanceSettings = BinanceSettings()  # type: ignore[call-arg]  # populated from env
    fetcher: BinanceFetcher = BinanceFetcher(settings)
    repository: DuckDBOHLCVRepository = DuckDBOHLCVRepository(cm)
    return IngestionService(fetcher=fetcher, repository=repository)


@app.command()
def ingest(  # noqa: PLR0913, PLR0917
    assets: Annotated[
        str,
        typer.Option("--assets", "-a", help="Comma-separated trading-pair symbols (e.g. BTCUSDT,ETHUSDT)."),
    ],
    timeframes: Annotated[
        str,
        typer.Option("--timeframes", "-t", help="Comma-separated timeframe intervals (e.g. 1h,4h,1d)."),
    ],
    start: Annotated[
        str,
        typer.Option("--start", "-s", help="Ingestion start date in ISO-8601 format (e.g. 2020-01-01)."),
    ],
    end: Annotated[
        str,
        typer.Option("--end", "-e", help="Ingestion end date in ISO-8601 format. Defaults to now (UTC)."),
    ] = "",
    log_level: Annotated[
        str,
        typer.Option("--log-level", "-l", help="Minimum log level (DEBUG, INFO, WARNING, ERROR)."),
    ] = "INFO",
    incremental: Annotated[
        bool,
        typer.Option("--incremental", "-i", help="Only fetch data after the last stored candle."),
    ] = False,
) -> None:
    """Ingest OHLCV candlestick data from Binance into DuckDB.

    Fetches the Cartesian product of ``--assets x --timeframes`` over the
    given date range.  Use ``--incremental`` to skip already-stored data and
    only download what is missing.

    Examples:

    .. code-block:: bash

        # Full historical ingest
        just ingest --assets BTCUSDT,ETHUSDT --timeframes 1h,4h --start 2020-01-01

        # Incremental top-up from last stored candle
        just ingest --assets BTCUSDT --timeframes 1h --start 2020-01-01 --incremental

    Args:
        assets: Comma-separated trading-pair symbols (e.g. ``BTCUSDT,ETHUSDT``).
        timeframes: Comma-separated timeframe intervals (e.g. ``1h,4h,1d``).
        start: Ingestion start date in ISO-8601 format.
        end: Ingestion end date in ISO-8601 format. Defaults to now (UTC) when empty.
        log_level: Minimum log level for the session. Defaults to ``INFO``.
        incremental: When *True*, skip already-stored candles and fetch only missing data.
    """
    setup_logging(level=log_level)

    parsed_assets: list[Asset] = _parse_assets(assets)
    parsed_timeframes: list[Timeframe] = _parse_timeframes(timeframes)
    start_dt: datetime = _parse_date(start)
    end_dt: datetime = datetime.now(UTC) if end == "" else _parse_date(end)

    date_range: DateRange = DateRange(start=start_dt, end=end_dt)

    logger.info(
        "Starting ingestion | assets={} timeframes={} start={} end={} incremental={}",
        [a.symbol for a in parsed_assets],
        [tf.value for tf in parsed_timeframes],
        start_dt.isoformat(),
        end_dt.isoformat(),
        incremental,
    )

    with ConnectionManager(DatabaseSettings()) as cm:
        service: IngestionService = _build_service(cm)

        if incremental:
            total: int = 0
            for asset in parsed_assets:
                for timeframe in parsed_timeframes:
                    written: int = service.ingest_incremental(asset, timeframe, date_range)
                    total += written
            logger.success(
                "Incremental ingestion complete | total_rows={}",
                total,
            )
        else:
            command: IngestUniverseCommand = IngestUniverseCommand(
                assets=parsed_assets,
                timeframes=parsed_timeframes,
                date_range=date_range,
            )
            results: dict[str, int] = service.ingest_universe(command)

            for pair, rows_written in results.items():
                logger.info("  {} -> {} rows", pair, rows_written)

            grand_total: int = sum(results.values())
            logger.success(
                "Universe ingestion complete | pairs={} total_rows={}",
                len(results),
                grand_total,
            )


if __name__ == "__main__":
    app()
