"""Concrete DuckDB-backed OHLCV repository."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta, UTC
from decimal import Decimal
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import CursorResult, Row, text, TextClause

from src.app.ohlcv.domain.entities import OHLCVCandle
from src.app.ohlcv.domain.value_objects import Asset, DateRange, TemporalSplit, Timeframe
from src.app.system.database.exceptions import QueryError
from src.app.system.database.repository import BaseRepository


class DuckDBOHLCVRepository(BaseRepository[OHLCVCandle]):
    """DuckDB implementation of :class:`IOHLCVRepository`.

    Satisfies the protocol via structural subtyping — no explicit ``implements``
    is required.
    """

    TABLE_NAME: str = "ohlcv"

    # -- ingestion -----------------------------------------------------------

    def ingest(self, candles: list[OHLCVCandle]) -> int:
        """Bulk ``INSERT OR IGNORE`` candles, returning rows written.

        Args:
            candles: OHLCV candle entities to persist.

        Returns:
            Number of rows actually inserted (duplicates are skipped).

        Raises:
            QueryError: If the SQL insert fails.
        """
        if not candles:
            return 0

        t0: float = time.perf_counter()
        sql: TextClause = text(
            f"INSERT OR IGNORE INTO {self.TABLE_NAME} "  # noqa: S608
            "(asset, timeframe, timestamp, open, high, low, close, volume) "
            "VALUES (:asset, :timeframe, :timestamp, :open, :high, :low, :close, :volume)"
        )
        params: list[dict[str, Any]] = [
            {
                "asset": c.asset.symbol,
                "timeframe": c.timeframe.value,
                "timestamp": c.timestamp,
                "open": str(c.open),
                "high": str(c.high),
                "low": str(c.low),
                "close": str(c.close),
                "volume": c.volume,
            }
            for c in candles
        ]

        try:
            with self._get_connection() as conn:
                result: CursorResult[Any] = conn.execute(sql, params)
                conn.commit()
                written: int = result.rowcount if result.rowcount >= 0 else len(candles)
        except Exception as exc:
            raise QueryError(f"Failed to ingest candles: {exc}") from exc

        elapsed: float = time.perf_counter() - t0
        logger.info("Ingested {} rows in {:.3f}s", written, elapsed)
        return written

    def ingest_from_parquet(self, path: Path, asset: Asset, timeframe: Timeframe) -> int:
        """Load a Parquet file via DuckDB's native ``read_parquet()``.

        Args:
            path: Filesystem path to the Parquet file.
            asset: Trading-pair symbol to tag each row with.
            timeframe: Candlestick interval for the data.

        Returns:
            Number of rows written.

        Raises:
            QueryError: If the SQL insert fails.
        """
        t0: float = time.perf_counter()
        resolved: str = str(path.resolve())
        sql: TextClause = text(
            f"INSERT OR IGNORE INTO {self.TABLE_NAME} "  # noqa: S608
            "SELECT "
            "  :asset AS asset, "
            "  :timeframe AS timeframe, "
            "  timestamp, open, high, low, close, volume "
            f"FROM read_parquet('{resolved}')"
        )

        try:
            with self._get_connection() as conn:
                result: CursorResult[Any] = conn.execute(
                    sql, {"asset": asset.symbol, "timeframe": timeframe.value}
                )
                conn.commit()
                written: int = result.rowcount if result.rowcount >= 0 else 0
        except Exception as exc:
            raise QueryError(f"Failed to ingest from parquet {path}: {exc}") from exc

        elapsed: float = time.perf_counter() - t0
        logger.info("Ingested {} rows from {} in {:.3f}s", written, path.name, elapsed)
        return written

    # -- queries -------------------------------------------------------------

    def query(self, asset: Asset, timeframe: Timeframe, date_range: DateRange) -> list[OHLCVCandle]:
        """Return candles matching the filter, ordered by timestamp.

        Args:
            asset: Trading-pair symbol.
            timeframe: Candlestick interval.
            date_range: Inclusive UTC date boundaries.

        Returns:
            Candles ordered chronologically.

        Raises:
            QueryError: If the SQL query fails.
        """
        t0: float = time.perf_counter()
        sql: TextClause = text(
            f"SELECT asset, timeframe, timestamp, open, high, low, close, volume "  # noqa: S608
            f"FROM {self.TABLE_NAME} "
            "WHERE asset = :asset AND timeframe = :timeframe "
            "  AND timestamp >= :start AND timestamp < :end "
            "ORDER BY timestamp"
        )

        try:
            with self._get_connection() as conn:
                rows: Sequence[Row[Any]] = conn.execute(
                    sql,
                    {
                        "asset": asset.symbol,
                        "timeframe": timeframe.value,
                        "start": date_range.start,
                        "end": date_range.end,
                    },
                ).fetchall()
        except Exception as exc:
            raise QueryError(f"Query failed: {exc}") from exc

        candles: list[OHLCVCandle] = [self._row_to_entity(r) for r in rows]
        elapsed: float = time.perf_counter() - t0
        logger.debug("Queried {} candles in {:.3f}s", len(candles), elapsed)
        return candles

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
        date_range: DateRange = split.get_range(partition)
        return self.query(asset, timeframe, date_range)

    def query_cross_asset(
        self,
        assets: list[Asset],
        timeframe: Timeframe,
        date_range: DateRange,
    ) -> dict[str, list[OHLCVCandle]]:
        """Return candles for multiple assets, grouped by symbol.

        Args:
            assets: Trading-pair symbols to query.
            timeframe: Candlestick interval.
            date_range: Inclusive UTC date boundaries.

        Returns:
            Mapping from asset symbol to its list of candles.

        Raises:
            QueryError: If the SQL query fails.
        """
        t0: float = time.perf_counter()
        symbols: list[str] = [a.symbol for a in assets]
        placeholders: str = ", ".join(f"'{s}'" for s in symbols)
        sql: TextClause = text(
            f"SELECT asset, timeframe, timestamp, open, high, low, close, volume "  # noqa: S608
            f"FROM {self.TABLE_NAME} "
            f"WHERE asset IN ({placeholders}) AND timeframe = :timeframe "
            "  AND timestamp >= :start AND timestamp < :end "
            "ORDER BY asset, timestamp"
        )

        try:
            with self._get_connection() as conn:
                rows: Sequence[Row[Any]] = conn.execute(
                    sql,
                    {
                        "timeframe": timeframe.value,
                        "start": date_range.start,
                        "end": date_range.end,
                    },
                ).fetchall()
        except Exception as exc:
            raise QueryError(f"Cross-asset query failed: {exc}") from exc

        grouped: dict[str, list[OHLCVCandle]] = defaultdict(list)
        for r in rows:
            candle: OHLCVCandle = self._row_to_entity(r)
            grouped[candle.asset.symbol].append(candle)

        elapsed: float = time.perf_counter() - t0
        total: int = sum(len(v) for v in grouped.values())
        logger.debug("Cross-asset query: {} assets, {} candles in {:.3f}s", len(grouped), total, elapsed)
        return dict(grouped)

    # -- metadata ------------------------------------------------------------

    def get_available_assets(self) -> list[str]:
        """Return distinct asset symbols.

        Returns:
            Sorted list of unique asset symbol strings.
        """
        sql: TextClause = text(f"SELECT DISTINCT asset FROM {self.TABLE_NAME} ORDER BY asset")  # noqa: S608
        with self._get_connection() as conn:
            rows: Sequence[Row[Any]] = conn.execute(sql).fetchall()
        return [r[0] for r in rows]

    def get_date_range(self, asset: Asset, timeframe: Timeframe) -> DateRange | None:
        """Return the min/max timestamp range, or *None* if no data exists.

        Args:
            asset: Trading-pair symbol.
            timeframe: Candlestick interval.

        Returns:
            The date range or *None* when the store has no matching rows.
        """
        sql: TextClause = text(
            f"SELECT MIN(timestamp), MAX(timestamp) FROM {self.TABLE_NAME} "  # noqa: S608
            "WHERE asset = :asset AND timeframe = :timeframe"
        )
        with self._get_connection() as conn:
            row: Row[Any] | None = conn.execute(
                sql, {"asset": asset.symbol, "timeframe": timeframe.value}
            ).fetchone()

        if row is None or row[0] is None:
            return None

        start: datetime = row[0] if row[0].tzinfo else row[0].replace(tzinfo=UTC)
        end: datetime = row[1] if row[1].tzinfo else row[1].replace(tzinfo=UTC)
        # Nudge end slightly so DateRange start < end is satisfied even for a single row.
        if start == end:
            end += timedelta(seconds=1)
        return DateRange(start=start, end=end)

    def count(self) -> int:
        """Return total rows in the ohlcv table.

        Returns:
            Row count.
        """
        sql: TextClause = text(f"SELECT COUNT(*) FROM {self.TABLE_NAME}")  # noqa: S608
        with self._get_connection() as conn:
            result: int | None = conn.execute(sql).scalar()
        return int(result) if result else 0

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _row_to_entity(row: Row[Any]) -> OHLCVCandle:
        """Map a database row to an :class:`OHLCVCandle` with Decimal precision.

        Args:
            row: A SQLAlchemy result row with columns in schema order.

        Returns:
            Hydrated domain entity.
        """
        ts: datetime = row[2] if row[2].tzinfo else row[2].replace(tzinfo=UTC)
        return OHLCVCandle(
            asset=Asset(symbol=row[0]),
            timeframe=Timeframe(row[1]),
            timestamp=ts,
            open=Decimal(str(row[3])),
            high=Decimal(str(row[4])),
            low=Decimal(str(row[5])),
            close=Decimal(str(row[6])),
            volume=float(row[7]),
        )
