"""Data loading service for research analysis — bridges DuckDB to Pandas."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import Row, TextClause, text

from src.app.system.database.connection import ConnectionManager

# ---------------------------------------------------------------------------
# Constants — OHLCV and bar column names
# ---------------------------------------------------------------------------

_OHLCV_COLUMNS: list[str] = ["timestamp", "open", "high", "low", "close", "volume"]
_OHLCV_NUMERIC_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]

_BAR_COLUMNS: list[str] = [
    "start_ts",
    "end_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "tick_count",
    "buy_volume",
    "sell_volume",
    "vwap",
]
_BAR_NUMERIC_COLUMNS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "tick_count",
    "buy_volume",
    "sell_volume",
    "vwap",
]


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


class DataLoader:
    """Loads OHLCV candles and aggregated bars from DuckDB into Pandas DataFrames.

    This service provides a thin SQL-to-DataFrame bridge for the research
    module.  It queries DuckDB directly via SQLAlchemy (bypassing domain
    repositories) so that results arrive as Pandas DataFrames suitable for
    statistical analysis.

    All DECIMAL(18,8) columns are converted to ``float64`` for numerical
    convenience.
    """

    def __init__(self, connection_manager: ConnectionManager) -> None:
        """Initialise the loader with a database connection manager.

        Args:
            connection_manager: An initialised :class:`ConnectionManager`
                pointing at the DuckDB database.
        """
        self._connection_manager: ConnectionManager = connection_manager

    # -- OHLCV ---------------------------------------------------------------

    def load_ohlcv(
        self,
        asset: str,
        timeframe: str,
        date_range: tuple[datetime, datetime] | None = None,
    ) -> pd.DataFrame:
        """Load OHLCV candles for a single asset and timeframe.

        Args:
            asset: Trading pair symbol (e.g. ``"BTCUSDT"``).
            timeframe: Candlestick interval (e.g. ``"1h"``).
            date_range: Optional ``(start, end)`` UTC boundaries.
                When provided, only candles with ``start <= timestamp < end``
                are returned.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
            Numeric columns are ``float64``.  Empty DataFrame when no data
            matches.
        """
        where_clause: str = "WHERE asset = :asset AND timeframe = :timeframe"
        params: dict[str, Any] = {"asset": asset, "timeframe": timeframe}

        if date_range is not None:
            where_clause += " AND timestamp >= :start AND timestamp < :end"
            params["start"] = date_range[0]
            params["end"] = date_range[1]

        sql: TextClause = text(
            "SELECT timestamp, open, high, low, close, volume "  # noqa: S608
            f"FROM ohlcv {where_clause} "
            "ORDER BY timestamp"
        )

        with self._connection_manager.connect() as conn:
            rows: Sequence[Row[Any]] = conn.execute(sql, params).fetchall()

        if not rows:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        df: pd.DataFrame = pd.DataFrame(rows, columns=_OHLCV_COLUMNS)
        # Convert DuckDB DECIMAL / Decimal objects to float64.
        for col in _OHLCV_NUMERIC_COLUMNS:
            df[col] = df[col].astype(float)
        # Ensure timestamps are UTC-aware.
        df["timestamp"] = df["timestamp"].apply(_to_utc)
        return df

    # -- Aggregated bars -----------------------------------------------------

    def load_bars(
        self,
        asset: str,
        bar_type: str,
        config_hash: str,
        date_range: tuple[datetime, datetime] | None = None,
    ) -> pd.DataFrame:
        """Load aggregated bars for a single asset, bar type, and config.

        Args:
            asset: Trading pair symbol (e.g. ``"BTCUSDT"``).
            bar_type: Bar aggregation type (e.g. ``"dollar"``).
            config_hash: Hex digest identifying the bar configuration.
            date_range: Optional ``(start, end)`` UTC boundaries.
                When provided, only bars with ``start <= start_ts < end``
                are returned.

        Returns:
            DataFrame with columns: start_ts, end_ts, open, high, low, close,
            volume, tick_count, buy_volume, sell_volume, vwap.  Numeric columns
            are ``float64``.  Empty DataFrame when no data matches.
        """
        where_clause: str = "WHERE asset = :asset AND bar_type = :bar_type AND bar_config_hash = :config_hash"
        params: dict[str, Any] = {
            "asset": asset,
            "bar_type": bar_type,
            "config_hash": config_hash,
        }

        if date_range is not None:
            where_clause += " AND start_ts >= :start AND start_ts < :end"
            params["start"] = date_range[0]
            params["end"] = date_range[1]

        sql: TextClause = text(
            "SELECT start_ts, end_ts, open, high, low, close, volume, "  # noqa: S608
            "tick_count, buy_volume, sell_volume, vwap "
            f"FROM aggregated_bars {where_clause} "
            "ORDER BY start_ts"
        )

        with self._connection_manager.connect() as conn:
            rows: Sequence[Row[Any]] = conn.execute(sql, params).fetchall()

        if not rows:
            return pd.DataFrame(columns=_BAR_COLUMNS)

        df: pd.DataFrame = pd.DataFrame(rows, columns=_BAR_COLUMNS)
        # Convert DuckDB DECIMAL / Decimal objects to float64.
        for col in _BAR_NUMERIC_COLUMNS:
            df[col] = df[col].astype(float)
        # Ensure timestamps are UTC-aware.
        df["start_ts"] = df["start_ts"].apply(_to_utc)
        df["end_ts"] = df["end_ts"].apply(_to_utc)
        return df

    # -- metadata queries ----------------------------------------------------

    def get_available_assets(self) -> list[str]:
        """Return distinct asset symbols present in the OHLCV table.

        Returns:
            Sorted list of unique asset symbol strings.
        """
        sql: TextClause = text(
            "SELECT DISTINCT asset FROM ohlcv ORDER BY asset"  # noqa: S608
        )
        with self._connection_manager.connect() as conn:
            rows: Sequence[Row[Any]] = conn.execute(sql).fetchall()
        return [str(r[0]) for r in rows]

    def get_available_bar_configs(self, asset: str) -> list[tuple[str, str]]:
        """Return distinct ``(bar_type, config_hash)`` pairs for an asset.

        Args:
            asset: Trading pair symbol.

        Returns:
            List of ``(bar_type, config_hash)`` tuples, ordered by bar_type
            then config_hash.
        """
        sql: TextClause = text(
            "SELECT DISTINCT bar_type, bar_config_hash "  # noqa: S608
            "FROM aggregated_bars "
            "WHERE asset = :asset "
            "ORDER BY bar_type, bar_config_hash"
        )
        with self._connection_manager.connect() as conn:
            rows: Sequence[Row[Any]] = conn.execute(sql, {"asset": asset}).fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]

    def get_ohlcv_date_range(
        self,
        asset: str,
        timeframe: str,
    ) -> tuple[datetime, datetime] | None:
        """Return the earliest and latest timestamps for an (asset, timeframe) pair.

        Args:
            asset: Trading pair symbol.
            timeframe: Candlestick interval.

        Returns:
            ``(min_timestamp, max_timestamp)`` both UTC-aware, or *None* when
            no data exists.
        """
        sql: TextClause = text(
            "SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv "  # noqa: S608
            "WHERE asset = :asset AND timeframe = :timeframe"
        )
        with self._connection_manager.connect() as conn:
            row: Row[Any] | None = conn.execute(
                sql,
                {"asset": asset, "timeframe": timeframe},
            ).fetchone()

        if row is None or row[0] is None:
            return None

        start: datetime = _to_utc(row[0])
        end: datetime = _to_utc(row[1])
        return (start, end)
