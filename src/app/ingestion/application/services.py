"""Ingestion application service — orchestrates fetch-and-store workflows."""

from __future__ import annotations

from datetime import timedelta

from loguru import logger

from src.app.ingestion.application.commands import IngestAssetCommand, IngestUniverseCommand
from src.app.ingestion.domain.protocols import IMarketDataFetcher
from src.app.ingestion.domain.value_objects import FetchRequest
from src.app.ohlcv.domain.entities import OHLCVCandle
from src.app.ohlcv.domain.repositories import IOHLCVRepository
from src.app.ohlcv.domain.value_objects import Asset, DateRange, Timeframe


class IngestionService:
    """Orchestrates fetching OHLCV data from an exchange and persisting it.

    Depends only on domain protocols — the concrete fetcher and repository
    implementations are injected at construction time.
    """

    def __init__(
        self,
        fetcher: IMarketDataFetcher,
        repository: IOHLCVRepository,
    ) -> None:
        """Initialise the service with a fetcher and repository.

        Args:
            fetcher: Any object satisfying the :class:`IMarketDataFetcher` protocol.
            repository: Any object satisfying the :class:`IOHLCVRepository` protocol.
        """
        self._fetcher: IMarketDataFetcher = fetcher
        self._repo: IOHLCVRepository = repository

    def ingest_asset(self, command: IngestAssetCommand) -> int:
        """Fetch and store OHLCV data for a single asset and timeframe.

        Args:
            command: Describes which asset, timeframe, and date range to ingest.

        Returns:
            Number of rows actually written (duplicates are skipped).
        """
        logger.info(
            "Ingesting {} {} [{} -> {}]",
            command.asset.symbol,
            command.timeframe.value,
            command.date_range.start.isoformat(),
            command.date_range.end.isoformat(),
        )

        request: FetchRequest = FetchRequest(
            asset=command.asset,
            timeframe=command.timeframe,
            date_range=command.date_range,
        )
        candles: list[OHLCVCandle] = self._fetcher.fetch_ohlcv(request)

        if not candles:
            logger.warning(
                "No candles fetched for {} {}",
                command.asset.symbol,
                command.timeframe.value,
            )
            return 0

        written: int = self._repo.ingest(candles)
        logger.info(
            "Stored {}/{} candles for {} {}",
            written,
            len(candles),
            command.asset.symbol,
            command.timeframe.value,
        )
        return written

    def ingest_universe(self, command: IngestUniverseCommand) -> dict[str, int]:
        """Fetch and store OHLCV data for multiple assets and timeframes.

        Iterates the Cartesian product of ``assets x timeframes`` over the
        given date range.

        Args:
            command: Describes which assets, timeframes, and date range to ingest.

        Returns:
            Mapping from ``"SYMBOL/TIMEFRAME"`` to rows written.
        """
        results: dict[str, int] = {}

        for asset in command.assets:
            for timeframe in command.timeframes:
                key: str = f"{asset.symbol}/{timeframe.value}"
                asset_command: IngestAssetCommand = IngestAssetCommand(
                    asset=asset,
                    timeframe=timeframe,
                    date_range=command.date_range,
                )
                written: int = self.ingest_asset(asset_command)
                results[key] = written

        total: int = sum(results.values())
        logger.info("Universe ingestion complete: {} total rows across {} pairs", total, len(results))
        return results

    def ingest_incremental(self, asset: Asset, timeframe: Timeframe, date_range: DateRange) -> int:
        """Fetch only missing data by advancing the start past existing records.

        Queries the repository for the latest stored timestamp for the given
        asset and timeframe.  If data already exists, the fetch starts one
        interval after the last stored candle — avoiding re-fetching known
        data while still being idempotent (``INSERT OR IGNORE`` handles
        overlap at the boundary).

        Args:
            asset: The trading pair to fetch.
            timeframe: The candlestick interval.
            date_range: The full desired date range (end bound is kept as-is).

        Returns:
            Number of rows actually written.
        """
        existing_range: DateRange | None = self._repo.get_date_range(asset, timeframe)

        if existing_range is not None:
            incremental_start = existing_range.end - timedelta(seconds=1)
            if incremental_start >= date_range.end:
                logger.info(
                    "Already up-to-date for {} {} (stored through {})",
                    asset.symbol,
                    timeframe.value,
                    existing_range.end.isoformat(),
                )
                return 0

            if incremental_start > date_range.start:
                logger.info(
                    "Incremental fetch for {} {}: advancing start {} -> {}",
                    asset.symbol,
                    timeframe.value,
                    date_range.start.isoformat(),
                    incremental_start.isoformat(),
                )
                date_range = DateRange(start=incremental_start, end=date_range.end)

        command: IngestAssetCommand = IngestAssetCommand(
            asset=asset,
            timeframe=timeframe,
            date_range=date_range,
        )
        return self.ingest_asset(command)
