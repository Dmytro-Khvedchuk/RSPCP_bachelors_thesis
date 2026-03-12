"""Unit tests for the DataLoader service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
from sqlalchemy import text

from src.app.research.application.data_loader import DataLoader
from src.app.system.database.connection import ConnectionManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)


def _insert_ohlcv_rows(
    cm: ConnectionManager,
    asset: str,
    timeframe: str,
    n: int,
    *,
    start: datetime = _BASE_TS,
    interval_hours: float = 1.0,
) -> None:
    """Insert *n* synthetic OHLCV rows into the in-memory database.

    Args:
        cm: Connection manager with an initialised in-memory DuckDB.
        asset: Asset symbol to use.
        timeframe: Timeframe string to use.
        n: Number of rows to insert.
        start: Starting timestamp. Defaults to ``_BASE_TS``.
        interval_hours: Hours between consecutive rows. Defaults to ``1.0``.
    """
    with cm.connect() as conn:
        for i in range(n):
            ts: datetime = start + timedelta(hours=interval_hours * i)
            price: float = 42000.0 + i * 10.0
            conn.execute(
                text(
                    "INSERT INTO ohlcv (asset, timeframe, timestamp, open, high, low, close, volume) "
                    "VALUES (:asset, :tf, :ts, :o, :h, :l, :c, :v)"
                ),
                {
                    "asset": asset,
                    "tf": timeframe,
                    "ts": ts,
                    "o": price,
                    "h": price + 50.0,
                    "l": price - 50.0,
                    "c": price + 5.0,
                    "v": 100.0 + i,
                },
            )
        conn.commit()


def _insert_bar_rows(
    cm: ConnectionManager,
    asset: str,
    bar_type: str,
    config_hash: str,
    n: int,
) -> None:
    """Insert *n* synthetic aggregated bar rows.

    Args:
        cm: Connection manager with an initialised in-memory DuckDB.
        asset: Asset symbol.
        bar_type: Bar type string.
        config_hash: Configuration hash string.
        n: Number of rows to insert.
    """
    with cm.connect() as conn:
        for i in range(n):
            start_ts: datetime = _BASE_TS + timedelta(hours=i)
            end_ts: datetime = start_ts + timedelta(hours=1)
            price: float = 42000.0 + i * 10.0
            conn.execute(
                text(
                    "INSERT INTO aggregated_bars "
                    "(asset, bar_type, bar_config_hash, start_ts, end_ts, "
                    "open, high, low, close, volume, tick_count, "
                    "buy_volume, sell_volume, vwap) "
                    "VALUES (:asset, :bt, :bch, :sts, :ets, "
                    ":o, :h, :l, :c, :v, :tc, :bv, :sv, :vwap)"
                ),
                {
                    "asset": asset,
                    "bt": bar_type,
                    "bch": config_hash,
                    "sts": start_ts,
                    "ets": end_ts,
                    "o": price,
                    "h": price + 50.0,
                    "l": price - 50.0,
                    "c": price + 5.0,
                    "v": 100.0,
                    "tc": 42,
                    "bv": 60.0,
                    "sv": 40.0,
                    "vwap": price + 2.0,
                },
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDataLoader:
    """Tests for :class:`DataLoader`."""

    def test_load_ohlcv_returns_correct_columns(
        self,
        in_memory_connection_manager: ConnectionManager,
    ) -> None:
        """Loaded OHLCV DataFrame has the expected column names."""
        _insert_ohlcv_rows(in_memory_connection_manager, "BTCUSDT", "1h", 5)
        loader: DataLoader = DataLoader(in_memory_connection_manager)

        df: pd.DataFrame = loader.load_ohlcv("BTCUSDT", "1h")

        expected_cols: list[str] = ["timestamp", "open", "high", "low", "close", "volume"]
        assert list(df.columns) == expected_cols

    def test_load_ohlcv_converts_decimal_to_float(
        self,
        in_memory_connection_manager: ConnectionManager,
    ) -> None:
        """Numeric columns are float64 (not Decimal)."""
        _insert_ohlcv_rows(in_memory_connection_manager, "BTCUSDT", "1h", 3)
        loader: DataLoader = DataLoader(in_memory_connection_manager)

        df: pd.DataFrame = loader.load_ohlcv("BTCUSDT", "1h")

        for col in ["open", "high", "low", "close", "volume"]:
            assert df[col].dtype == "float64", f"Column {col} should be float64"

    def test_load_ohlcv_empty_returns_empty_df(
        self,
        in_memory_connection_manager: ConnectionManager,
    ) -> None:
        """Loading from empty table returns an empty DataFrame with correct columns."""
        loader: DataLoader = DataLoader(in_memory_connection_manager)

        df: pd.DataFrame = loader.load_ohlcv("BTCUSDT", "1h")

        assert df.empty
        expected_cols: list[str] = ["timestamp", "open", "high", "low", "close", "volume"]
        assert list(df.columns) == expected_cols

    def test_load_ohlcv_with_date_range(
        self,
        in_memory_connection_manager: ConnectionManager,
    ) -> None:
        """Date range filter correctly limits returned rows."""
        _insert_ohlcv_rows(in_memory_connection_manager, "BTCUSDT", "1h", 10)
        loader: DataLoader = DataLoader(in_memory_connection_manager)

        start: datetime = _BASE_TS + timedelta(hours=2)
        end: datetime = _BASE_TS + timedelta(hours=5)
        df: pd.DataFrame = loader.load_ohlcv("BTCUSDT", "1h", date_range=(start, end))

        # hours 2, 3, 4 → 3 rows (start inclusive, end exclusive)
        assert len(df) == 3
        assert df["timestamp"].iloc[0] == start

    def test_load_bars_returns_correct_columns(
        self,
        in_memory_connection_manager: ConnectionManager,
    ) -> None:
        """Loaded bars DataFrame has the expected column names."""
        _insert_bar_rows(in_memory_connection_manager, "BTCUSDT", "dollar", "abc123", 5)
        loader: DataLoader = DataLoader(in_memory_connection_manager)

        df: pd.DataFrame = loader.load_bars("BTCUSDT", "dollar", "abc123")

        expected_cols: list[str] = [
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
        assert list(df.columns) == expected_cols

    def test_get_available_assets(
        self,
        in_memory_connection_manager: ConnectionManager,
    ) -> None:
        """Distinct assets are returned in sorted order."""
        _insert_ohlcv_rows(in_memory_connection_manager, "ETHUSDT", "1h", 2)
        _insert_ohlcv_rows(in_memory_connection_manager, "BTCUSDT", "1h", 2)
        loader: DataLoader = DataLoader(in_memory_connection_manager)

        assets: list[str] = loader.get_available_assets()

        assert assets == ["BTCUSDT", "ETHUSDT"]

    def test_get_available_bar_configs(
        self,
        in_memory_connection_manager: ConnectionManager,
    ) -> None:
        """Distinct (bar_type, config_hash) pairs are returned correctly."""
        _insert_bar_rows(in_memory_connection_manager, "BTCUSDT", "dollar", "hash1", 2)
        _insert_bar_rows(
            in_memory_connection_manager,
            "BTCUSDT",
            "volume",
            "hash2",
            2,
        )
        loader: DataLoader = DataLoader(in_memory_connection_manager)

        configs: list[tuple[str, str]] = loader.get_available_bar_configs("BTCUSDT")

        assert len(configs) == 2
        assert ("dollar", "hash1") in configs
        assert ("volume", "hash2") in configs

    def test_get_ohlcv_date_range(
        self,
        in_memory_connection_manager: ConnectionManager,
    ) -> None:
        """Min and max timestamps are returned as a UTC-aware tuple."""
        _insert_ohlcv_rows(in_memory_connection_manager, "BTCUSDT", "1h", 10)
        loader: DataLoader = DataLoader(in_memory_connection_manager)

        result: tuple[datetime, datetime] | None = loader.get_ohlcv_date_range("BTCUSDT", "1h")

        assert result is not None
        min_ts: datetime = result[0]
        max_ts: datetime = result[1]
        assert min_ts == _BASE_TS
        assert max_ts == _BASE_TS + timedelta(hours=9)

    def test_get_ohlcv_date_range_empty(
        self,
        in_memory_connection_manager: ConnectionManager,
    ) -> None:
        """None is returned when no data exists for the query."""
        loader: DataLoader = DataLoader(in_memory_connection_manager)

        result: tuple[datetime, datetime] | None = loader.get_ohlcv_date_range("BTCUSDT", "1h")

        assert result is None
