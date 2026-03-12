"""Concrete DuckDB-backed repository for aggregated bars."""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime, timedelta, UTC
from decimal import Decimal
from typing import Any

from loguru import logger
from sqlalchemy import CursorResult, Row, text, TextClause

from src.app.bars.domain.entities import AggregatedBar
from src.app.bars.domain.value_objects import BarType
from src.app.ohlcv.domain.value_objects import Asset, DateRange
from src.app.system.database.exceptions import QueryError
from src.app.system.database.repository import BaseRepository


def _to_utc(dt: datetime) -> datetime:
    """Normalise a datetime to UTC.

    DuckDB ``TIMESTAMPTZ`` columns may be returned in the system's local
    timezone.  This helper converts any aware datetime to UTC, and
    assumes UTC for naive datetimes.

    Args:
        dt: Datetime value (possibly non-UTC aware, or naive).

    Returns:
        Timezone-aware datetime in UTC.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class DuckDBBarRepository(BaseRepository[AggregatedBar]):
    """DuckDB implementation of :class:`IBarRepository`.

    Satisfies the protocol via structural subtyping — no explicit ``implements``
    is required.  The composite primary key
    ``(asset, bar_type, bar_config_hash, start_ts)`` guarantees uniqueness.
    """

    TABLE_NAME: str = "aggregated_bars"

    # -- ingestion -----------------------------------------------------------

    def ingest(self, bars: list[AggregatedBar], *, config_hash: str) -> int:
        """Bulk ``INSERT OR IGNORE`` bars, returning rows written.

        Args:
            bars: Aggregated bar entities to persist.
            config_hash: Hex digest identifying the :class:`BarConfig` that
                produced these bars.

        Returns:
            Number of rows actually inserted (duplicates are skipped).

        Raises:
            QueryError: If the SQL insert fails.
        """
        if not bars:
            return 0

        t0: float = time.perf_counter()
        sql: TextClause = text(
            f"INSERT OR IGNORE INTO {self.TABLE_NAME} "  # noqa: S608
            "(asset, bar_type, bar_config_hash, start_ts, end_ts, "
            "open, high, low, close, volume, tick_count, "
            "buy_volume, sell_volume, vwap) "
            "VALUES (:asset, :bar_type, :bar_config_hash, :start_ts, :end_ts, "
            ":open, :high, :low, :close, :volume, :tick_count, "
            ":buy_volume, :sell_volume, :vwap)"
        )
        params: list[dict[str, Any]] = [
            {
                "asset": b.asset.symbol,
                "bar_type": b.bar_type.value,
                "bar_config_hash": config_hash,
                "start_ts": b.start_ts,
                "end_ts": b.end_ts,
                "open": str(b.open),
                "high": str(b.high),
                "low": str(b.low),
                "close": str(b.close),
                "volume": b.volume,
                "tick_count": b.tick_count,
                "buy_volume": b.buy_volume,
                "sell_volume": b.sell_volume,
                "vwap": str(b.vwap),
            }
            for b in bars
        ]

        try:
            with self._get_connection() as conn:
                result: CursorResult[Any] = conn.execute(sql, params)
                conn.commit()
                written: int = result.rowcount if result.rowcount >= 0 else len(bars)
        except Exception as exc:
            raise QueryError(f"Failed to ingest bars: {exc}") from exc

        elapsed: float = time.perf_counter() - t0
        logger.info("Ingested {} bar rows in {:.3f}s", written, elapsed)
        return written

    # -- queries -------------------------------------------------------------

    def query(
        self,
        asset: Asset,
        bar_type: BarType,
        config_hash: str,
        date_range: DateRange,
    ) -> list[AggregatedBar]:
        """Return bars matching the filter, ordered by ``start_ts``.

        Args:
            asset: Trading-pair symbol.
            bar_type: Bar aggregation type.
            config_hash: Configuration hash to filter by.
            date_range: UTC date boundaries (``start`` inclusive, ``end`` exclusive).

        Returns:
            Bars ordered chronologically by ``start_ts``.

        Raises:
            QueryError: If the SQL query fails.
        """
        t0: float = time.perf_counter()
        sql: TextClause = text(
            f"SELECT asset, bar_type, bar_config_hash, start_ts, end_ts, "  # noqa: S608
            f"open, high, low, close, volume, tick_count, "
            f"buy_volume, sell_volume, vwap "
            f"FROM {self.TABLE_NAME} "
            "WHERE asset = :asset AND bar_type = :bar_type "
            "  AND bar_config_hash = :config_hash "
            "  AND start_ts >= :start AND start_ts < :end "
            "ORDER BY start_ts"
        )

        try:
            with self._get_connection() as conn:
                rows: Sequence[Row[Any]] = conn.execute(
                    sql,
                    {
                        "asset": asset.symbol,
                        "bar_type": bar_type.value,
                        "config_hash": config_hash,
                        "start": date_range.start,
                        "end": date_range.end,
                    },
                ).fetchall()
        except Exception as exc:
            raise QueryError(f"Bar query failed: {exc}") from exc

        bars: list[AggregatedBar] = [self._row_to_entity(r) for r in rows]
        elapsed: float = time.perf_counter() - t0
        logger.debug("Queried {} bars in {:.3f}s", len(bars), elapsed)
        return bars

    # -- metadata ------------------------------------------------------------

    def get_available_configs(self, asset: Asset) -> list[tuple[str, str]]:
        """Return distinct ``(bar_type, config_hash)`` pairs for an asset.

        Args:
            asset: Trading-pair symbol.

        Returns:
            List of ``(bar_type, config_hash)`` tuples.

        Raises:
            QueryError: If the SQL query fails.
        """
        sql: TextClause = text(
            f"SELECT DISTINCT bar_type, bar_config_hash "  # noqa: S608
            f"FROM {self.TABLE_NAME} "
            "WHERE asset = :asset "
            "ORDER BY bar_type, bar_config_hash"
        )
        try:
            with self._get_connection() as conn:
                rows: Sequence[Row[Any]] = conn.execute(sql, {"asset": asset.symbol}).fetchall()
        except Exception as exc:
            raise QueryError(f"Failed to query available configs: {exc}") from exc

        return [(str(r[0]), str(r[1])) for r in rows]

    def get_date_range(
        self,
        asset: Asset,
        bar_type: BarType,
        config_hash: str,
    ) -> DateRange | None:
        """Return the min/max ``start_ts`` range, or *None* if no data exists.

        Args:
            asset: Trading-pair symbol.
            bar_type: Bar aggregation type.
            config_hash: Configuration hash to filter by.

        Returns:
            The date range or *None* when the store has no matching rows.
        """
        sql: TextClause = text(
            f"SELECT MIN(start_ts), MAX(start_ts) FROM {self.TABLE_NAME} "  # noqa: S608
            "WHERE asset = :asset AND bar_type = :bar_type "
            "  AND bar_config_hash = :config_hash"
        )
        with self._get_connection() as conn:
            row: Row[Any] | None = conn.execute(
                sql,
                {
                    "asset": asset.symbol,
                    "bar_type": bar_type.value,
                    "config_hash": config_hash,
                },
            ).fetchone()

        if row is None or row[0] is None:
            return None

        start: datetime = _to_utc(row[0])
        end: datetime = _to_utc(row[1])
        # Nudge end slightly so DateRange start < end is satisfied even for a single row.
        if start == end:
            end += timedelta(seconds=1)
        return DateRange(start=start, end=end)

    def get_latest_end_ts(
        self,
        asset: Asset,
        bar_type: BarType,
        config_hash: str,
    ) -> datetime | None:
        """Return the latest ``end_ts`` for incremental ingestion, or *None*.

        Useful to determine where to resume bar construction when
        appending new data.

        Args:
            asset: Trading-pair symbol.
            bar_type: Bar aggregation type.
            config_hash: Configuration hash to filter by.

        Returns:
            The most recent ``end_ts`` or *None* when no bars exist.
        """
        sql: TextClause = text(
            f"SELECT MAX(end_ts) FROM {self.TABLE_NAME} "  # noqa: S608
            "WHERE asset = :asset AND bar_type = :bar_type "
            "  AND bar_config_hash = :config_hash"
        )
        with self._get_connection() as conn:
            scalar: datetime | None = conn.execute(
                sql,
                {
                    "asset": asset.symbol,
                    "bar_type": bar_type.value,
                    "config_hash": config_hash,
                },
            ).scalar()

        if scalar is None:
            return None
        return _to_utc(scalar)

    def count(self) -> int:
        """Return total rows in the aggregated_bars table.

        Returns:
            Row count.
        """
        sql: TextClause = text(f"SELECT COUNT(*) FROM {self.TABLE_NAME}")  # noqa: S608
        with self._get_connection() as conn:
            result: int | None = conn.execute(sql).scalar()
        return int(result) if result else 0

    def count_by_config(
        self,
        asset: Asset,
        bar_type: BarType,
        config_hash: str,
    ) -> int:
        """Return the number of bars for a specific asset, type, and config.

        Args:
            asset: Trading-pair symbol.
            bar_type: Bar aggregation type.
            config_hash: Configuration hash to filter by.

        Returns:
            Row count for the specified filter.
        """
        sql: TextClause = text(
            f"SELECT COUNT(*) FROM {self.TABLE_NAME} "  # noqa: S608
            "WHERE asset = :asset AND bar_type = :bar_type "
            "  AND bar_config_hash = :config_hash"
        )
        with self._get_connection() as conn:
            result: int | None = conn.execute(
                sql,
                {
                    "asset": asset.symbol,
                    "bar_type": bar_type.value,
                    "config_hash": config_hash,
                },
            ).scalar()
        return int(result) if result else 0

    def delete(
        self,
        asset: Asset,
        bar_type: BarType,
        config_hash: str,
    ) -> int:
        """Delete all bars for a given asset, bar type, and config hash.

        Useful for re-computing bars after parameter changes.

        The DuckDB driver does not reliably report ``rowcount`` for DELETE
        statements, so this method queries the count before deleting.

        Args:
            asset: Trading-pair symbol.
            bar_type: Bar aggregation type.
            config_hash: Configuration hash identifying the bars to delete.

        Returns:
            Number of rows deleted.

        Raises:
            QueryError: If the SQL delete fails.
        """
        # DuckDB's SQLAlchemy driver returns -1 for DELETE rowcount,
        # so we query the count before the delete to report accurately.
        before_count: int = self.count_by_config(asset, bar_type, config_hash)
        if before_count == 0:
            return 0

        sql: TextClause = text(
            f"DELETE FROM {self.TABLE_NAME} "  # noqa: S608
            "WHERE asset = :asset AND bar_type = :bar_type "
            "  AND bar_config_hash = :config_hash"
        )
        try:
            with self._get_connection() as conn:
                conn.execute(
                    sql,
                    {
                        "asset": asset.symbol,
                        "bar_type": bar_type.value,
                        "config_hash": config_hash,
                    },
                )
                conn.commit()
        except Exception as exc:
            raise QueryError(f"Failed to delete bars: {exc}") from exc

        logger.info(
            "Deleted {} bars for {} / {} / {}",
            before_count,
            asset.symbol,
            bar_type.value,
            config_hash,
        )
        return before_count

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _row_to_entity(row: Row[Any]) -> AggregatedBar:
        """Map a database row to an :class:`AggregatedBar` with Decimal precision.

        Row column order matches the SELECT in :meth:`query`:
        ``asset, bar_type, bar_config_hash, start_ts, end_ts,
        open, high, low, close, volume, tick_count,
        buy_volume, sell_volume, vwap``.

        Args:
            row: A SQLAlchemy result row with columns in schema order.

        Returns:
            Hydrated domain entity.
        """
        start_ts: datetime = _to_utc(row[3])
        end_ts: datetime = _to_utc(row[4])
        return AggregatedBar(
            asset=Asset(symbol=row[0]),
            bar_type=BarType(row[1]),
            start_ts=start_ts,
            end_ts=end_ts,
            open=Decimal(str(row[5])),
            high=Decimal(str(row[6])),
            low=Decimal(str(row[7])),
            close=Decimal(str(row[8])),
            volume=float(row[9]),
            tick_count=int(row[10]),
            buy_volume=float(row[11]),
            sell_volume=float(row[12]),
            vwap=Decimal(str(row[13])),
        )
