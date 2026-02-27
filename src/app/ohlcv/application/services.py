"""OHLCV application service — thin orchestration over the repository protocol."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.app.ohlcv.domain.entities import OHLCVCandle
from src.app.ohlcv.domain.repositories import IOHLCVRepository
from src.app.ohlcv.domain.value_objects import Asset, DateRange, TemporalSplit, Timeframe


class OHLCVService:
    """Use-case layer for OHLCV data.

    Depends only on the :class:`IOHLCVRepository` protocol — the concrete
    implementation is injected at construction time.
    """

    def __init__(self, repository: IOHLCVRepository) -> None:
        """Initialise the service with a repository implementation.

        Args:
            repository: Any object satisfying the :class:`IOHLCVRepository` protocol.
        """
        self._repo = repository

    # -- ingestion -----------------------------------------------------------

    def ingest(self, candles: list[OHLCVCandle]) -> int:
        """Persist *candles* and return the number of rows written.

        Args:
            candles: OHLCV candle entities to store.

        Returns:
            Number of rows actually written (duplicates are skipped).
        """
        logger.info("Ingesting {} candles", len(candles))
        count = self._repo.ingest(candles)
        logger.info("Ingested {} rows", count)
        return count

    def ingest_from_parquet(self, path: Path, asset: Asset, timeframe: Timeframe) -> int:
        """Bulk-load a Parquet file and return the number of rows written.

        Args:
            path: Filesystem path to the Parquet file.
            asset: Trading-pair symbol to tag the data with.
            timeframe: Candlestick interval for the data.

        Returns:
            Number of rows written.
        """
        logger.info("Ingesting from parquet: {} (asset={}, tf={})", path, asset, timeframe)
        count = self._repo.ingest_from_parquet(path, asset, timeframe)
        logger.info("Ingested {} rows from parquet", count)
        return count

    # -- queries -------------------------------------------------------------

    def query(self, asset: Asset, timeframe: Timeframe, date_range: DateRange) -> list[OHLCVCandle]:
        """Return candles matching the filter.

        Args:
            asset: Trading-pair symbol.
            timeframe: Candlestick interval.
            date_range: Inclusive UTC date boundaries.

        Returns:
            Candles ordered by timestamp.
        """
        logger.debug("Querying {} {} [{} → {}]", asset, timeframe, date_range.start, date_range.end)
        results = self._repo.query(asset, timeframe, date_range)
        logger.debug("Returned {} candles", len(results))
        return results

    def query_split(
        self,
        asset: Asset,
        timeframe: Timeframe,
        split: TemporalSplit,
        partition: str,
    ) -> list[OHLCVCandle]:
        """Return candles for a single partition of a temporal split.

        Args:
            asset: Trading-pair symbol.
            timeframe: Candlestick interval.
            split: Temporal split defining train/validation/test ranges.
            partition: One of ``"train"``, ``"validation"``, or ``"test"``.

        Returns:
            Candles for the requested partition, ordered by timestamp.
        """
        logger.debug("Split query: {} {} partition={}", asset, timeframe, partition)
        results = self._repo.query_split(asset, timeframe, split, partition)
        logger.debug("Returned {} candles for '{}' partition", len(results), partition)
        return results

    def query_cross_asset(
        self,
        assets: list[Asset],
        timeframe: Timeframe,
        date_range: DateRange,
    ) -> dict[str, list[OHLCVCandle]]:
        """Return candles for multiple assets grouped by symbol.

        Args:
            assets: Trading-pair symbols to query.
            timeframe: Candlestick interval.
            date_range: Inclusive UTC date boundaries.

        Returns:
            Mapping from asset symbol to its list of candles.
        """
        symbols = [str(a) for a in assets]
        logger.debug("Cross-asset query: {} tf={}", symbols, timeframe)
        results = self._repo.query_cross_asset(assets, timeframe, date_range)
        for sym, rows in results.items():
            logger.debug("  {} → {} candles", sym, len(rows))
        return results

    # -- metadata ------------------------------------------------------------

    def get_available_assets(self) -> list[str]:
        """Return distinct asset symbols.

        Returns:
            Sorted list of unique asset symbol strings.
        """
        assets = self._repo.get_available_assets()
        logger.debug("Available assets: {}", assets)
        return assets

    def get_date_range(self, asset: Asset, timeframe: Timeframe) -> DateRange | None:
        """Return the min/max date range for an asset+timeframe.

        Args:
            asset: Trading-pair symbol.
            timeframe: Candlestick interval.

        Returns:
            The date range or *None* if no data exists.
        """
        dr = self._repo.get_date_range(asset, timeframe)
        logger.debug("Date range for {} {}: {}", asset, timeframe, dr)
        return dr

    def count(self) -> int:
        """Return the total row count.

        Returns:
            Number of OHLCV rows stored.
        """
        n = self._repo.count()
        logger.debug("Total OHLCV rows: {}", n)
        return n
