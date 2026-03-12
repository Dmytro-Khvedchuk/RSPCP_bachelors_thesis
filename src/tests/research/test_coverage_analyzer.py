"""Unit tests for the CoverageAnalyzer service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import text

from src.app.research.application.coverage_analyzer import CoverageAnalyzer
from src.app.research.application.data_loader import DataLoader
from src.app.research.domain.value_objects import AssetFilterResult, CoverageRecord, GapRecord
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
    skip_indices: set[int] | None = None,
) -> None:
    """Insert *n* synthetic OHLCV rows, optionally skipping some indices to create gaps.

    Args:
        cm: Connection manager with an initialised in-memory DuckDB.
        asset: Asset symbol.
        timeframe: Timeframe string.
        n: Number of time slots to iterate over.
        start: Starting timestamp.
        interval_hours: Hours between consecutive slots.
        skip_indices: Set of slot indices to skip (creating gaps).
    """
    actual_skip: set[int] = skip_indices or set()
    with cm.connect() as conn:
        for i in range(n):
            if i in actual_skip:
                continue
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCoverageAnalyzer:
    """Tests for :class:`CoverageAnalyzer`."""

    def test_compute_coverage_correct_count(
        self,
        in_memory_connection_manager: ConnectionManager,
    ) -> None:
        """Total bars count matches the number of inserted rows."""
        _insert_ohlcv_rows(in_memory_connection_manager, "BTCUSDT", "1h", 100)
        loader: DataLoader = DataLoader(in_memory_connection_manager)
        analyzer: CoverageAnalyzer = CoverageAnalyzer(loader)

        records: list[CoverageRecord] = analyzer.compute_coverage(["BTCUSDT"], ["1h"])

        assert len(records) == 1
        assert records[0].total_bars == 100

    def test_compute_coverage_correct_expected(
        self,
        in_memory_connection_manager: ConnectionManager,
    ) -> None:
        """Expected bars is computed correctly from date span and timeframe.

        For 100 hourly bars starting at T and ending at T+99h, the span is
        99 hours, so expected = 99/1 + 1 = 100.
        """
        _insert_ohlcv_rows(in_memory_connection_manager, "BTCUSDT", "1h", 100)
        loader: DataLoader = DataLoader(in_memory_connection_manager)
        analyzer: CoverageAnalyzer = CoverageAnalyzer(loader)

        records: list[CoverageRecord] = analyzer.compute_coverage(["BTCUSDT"], ["1h"])

        assert len(records) == 1
        assert records[0].expected_bars == 100

    def test_detect_gaps_finds_known_gap(
        self,
        in_memory_connection_manager: ConnectionManager,
    ) -> None:
        """A 49-hour gap in hourly data is detected with default threshold (3x).

        We insert bars at hours 0..4 and 53..57, skipping indices 5..52.
        The gap spans from hour 4 to hour 53 = 49 hours.
        """
        n: int = 58
        skip: set[int] = set(range(5, 53))
        _insert_ohlcv_rows(
            in_memory_connection_manager,
            "BTCUSDT",
            "1h",
            n,
            skip_indices=skip,
        )
        loader: DataLoader = DataLoader(in_memory_connection_manager)
        analyzer: CoverageAnalyzer = CoverageAnalyzer(loader)

        gaps: list[GapRecord] = analyzer.detect_gaps("BTCUSDT", "1h")

        assert len(gaps) == 1
        assert gaps[0].gap_duration_hours == pytest.approx(49.0, abs=0.01)
        # missing_bars = int(49 / 1) - 1 = 48
        assert gaps[0].missing_bars == 48

    def test_detect_gaps_no_false_positives(
        self,
        in_memory_connection_manager: ConnectionManager,
    ) -> None:
        """Continuous hourly data has no gaps above the default threshold."""
        _insert_ohlcv_rows(in_memory_connection_manager, "BTCUSDT", "1h", 100)
        loader: DataLoader = DataLoader(in_memory_connection_manager)
        analyzer: CoverageAnalyzer = CoverageAnalyzer(loader)

        gaps: list[GapRecord] = analyzer.detect_gaps("BTCUSDT", "1h")

        assert len(gaps) == 0

    def test_filter_assets_excludes_short_data(
        self,
        in_memory_connection_manager: ConnectionManager,
    ) -> None:
        """An asset with fewer than min_days of data is excluded."""
        # 48 hourly bars = 2 days of data — well below 730 days.
        _insert_ohlcv_rows(in_memory_connection_manager, "BTCUSDT", "1h", 48)
        loader: DataLoader = DataLoader(in_memory_connection_manager)
        analyzer: CoverageAnalyzer = CoverageAnalyzer(loader)

        results: list[AssetFilterResult] = analyzer.filter_assets(["BTCUSDT"], "1h", min_days=730)

        assert len(results) == 1
        assert results[0].included is False
        assert "Insufficient data" in results[0].reason

    def test_filter_assets_includes_good_data(
        self,
        in_memory_connection_manager: ConnectionManager,
    ) -> None:
        """An asset with enough continuous data passes the filter."""
        # 800 days * 24 bars/day = 19200 hourly bars.
        n_bars: int = 800 * 24
        _insert_ohlcv_rows(in_memory_connection_manager, "BTCUSDT", "1h", n_bars)
        loader: DataLoader = DataLoader(in_memory_connection_manager)
        analyzer: CoverageAnalyzer = CoverageAnalyzer(loader)

        results: list[AssetFilterResult] = analyzer.filter_assets(["BTCUSDT"], "1h", min_days=730)

        assert len(results) == 1
        assert results[0].included is True
        assert results[0].reason == "Passed all quality filters"

    def test_build_coverage_matrix_shape(
        self,
        in_memory_connection_manager: ConnectionManager,
    ) -> None:
        """Coverage matrix has assets as rows and timeframes as columns."""
        _insert_ohlcv_rows(in_memory_connection_manager, "BTCUSDT", "1h", 50)
        _insert_ohlcv_rows(in_memory_connection_manager, "BTCUSDT", "4h", 50, interval_hours=4.0)
        _insert_ohlcv_rows(in_memory_connection_manager, "ETHUSDT", "1h", 50)
        _insert_ohlcv_rows(in_memory_connection_manager, "ETHUSDT", "4h", 50, interval_hours=4.0)
        loader: DataLoader = DataLoader(in_memory_connection_manager)
        analyzer: CoverageAnalyzer = CoverageAnalyzer(loader)

        records: list[CoverageRecord] = analyzer.compute_coverage(
            ["BTCUSDT", "ETHUSDT"],
            ["1h", "4h"],
        )
        matrix: pd.DataFrame = CoverageAnalyzer.build_coverage_matrix(records)

        assert matrix.shape == (2, 2)
        assert "BTCUSDT" in matrix.index
        assert "ETHUSDT" in matrix.index
        assert "1h" in matrix.columns
        assert "4h" in matrix.columns
