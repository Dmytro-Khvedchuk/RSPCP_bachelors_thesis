"""Smoke tests for Bokeh chart functions.

Each test verifies that the chart function executes without error and
returns the correct Bokeh model type (``Figure`` or ``DataTable``).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from bokeh.models import DataTable
from bokeh.plotting._figure import figure as Figure  # noqa: N812

from src.app.research.domain.value_objects import GapRecord, ReturnStatistics
from src.app.research.infrastructure.bokeh_charts import (
    create_bar_count_histogram,
    create_coverage_heatmap,
    create_gap_timeline,
    create_statistics_table,
    create_volume_profile,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE_TS: datetime = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture
def coverage_matrix() -> pd.DataFrame:
    """Return a small 3x3 coverage pivot table.

    Returns:
        Pivot DataFrame with assets as index and timeframes as columns.
    """
    data: dict[str, list[float]] = {
        "1h": [99.5, 97.2, 100.0],
        "4h": [98.0, 95.1, 99.9],
        "1d": [100.0, 100.0, 98.5],
    }
    df: pd.DataFrame = pd.DataFrame(data, index=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    return df


@pytest.fixture
def sample_gaps() -> list[GapRecord]:
    """Return two sample GapRecord objects.

    Returns:
        List of GapRecord value objects.
    """
    gap_1: GapRecord = GapRecord(
        asset="BTCUSDT",
        timeframe="1h",
        gap_start=_BASE_TS,
        gap_end=_BASE_TS + timedelta(hours=5),
        gap_duration_hours=5.0,
        missing_bars=5,
    )
    gap_2: GapRecord = GapRecord(
        asset="ETHUSDT",
        timeframe="1h",
        gap_start=_BASE_TS + timedelta(days=10),
        gap_end=_BASE_TS + timedelta(days=10, hours=3),
        gap_duration_hours=3.0,
        missing_bars=3,
    )
    return [gap_1, gap_2]


@pytest.fixture
def volume_df() -> pd.DataFrame:
    """Return a small DataFrame with timestamp and volume columns.

    Returns:
        DataFrame with 50 rows of OHLCV-like data.
    """
    n: int = 50
    timestamps: list[datetime] = [_BASE_TS + timedelta(hours=i) for i in range(n)]
    volumes: list[float] = [100.0 + float(i) * 0.5 for i in range(n)]
    df: pd.DataFrame = pd.DataFrame({"timestamp": timestamps, "volume": volumes})
    return df


@pytest.fixture
def weekly_counts() -> dict[str, pd.Series]:  # type: ignore[type-arg]
    """Return two bar-type weekly count Series.

    Returns:
        Dict mapping bar type name to weekly count Series.
    """
    weeks: pd.DatetimeIndex = pd.date_range("2024-01-01", periods=10, freq="W")
    time_counts: pd.Series = pd.Series(range(10, 20), index=weeks, name="time")  # type: ignore[type-arg]
    dollar_counts: pd.Series = pd.Series(range(15, 25), index=weeks, name="dollar")  # type: ignore[type-arg]
    return {"time": time_counts, "dollar": dollar_counts}


@pytest.fixture
def sample_stats() -> list[ReturnStatistics]:
    """Return two ReturnStatistics objects.

    Returns:
        List of ReturnStatistics value objects.
    """
    stat_1: ReturnStatistics = ReturnStatistics(
        asset="BTCUSDT",
        bar_type="time",
        count=1000,
        mean=0.0001,
        std=0.02,
        skewness=-0.1,
        kurtosis=3.5,
        jarque_bera_stat=50.0,
        jarque_bera_pvalue=0.001,
        is_normal=False,
    )
    stat_2: ReturnStatistics = ReturnStatistics(
        asset="ETHUSDT",
        bar_type="dollar",
        count=800,
        mean=0.0002,
        std=0.03,
        skewness=0.05,
        kurtosis=0.2,
        jarque_bera_stat=1.5,
        jarque_bera_pvalue=0.47,
        is_normal=True,
    )
    return [stat_1, stat_2]


# ---------------------------------------------------------------------------
# Tests — coverage heatmap
# ---------------------------------------------------------------------------


class TestCoverageHeatmap:
    """Smoke tests for create_coverage_heatmap."""

    def test_coverage_heatmap_returns_figure(self, coverage_matrix: pd.DataFrame) -> None:
        """Heatmap must return a Bokeh Figure."""
        result: Figure = create_coverage_heatmap(coverage_matrix)
        assert isinstance(result, Figure)

    def test_coverage_heatmap_has_renderers(self, coverage_matrix: pd.DataFrame) -> None:
        """Heatmap figure must have at least one renderer."""
        result: Figure = create_coverage_heatmap(coverage_matrix)
        assert len(result.renderers) >= 1


# ---------------------------------------------------------------------------
# Tests — gap timeline
# ---------------------------------------------------------------------------


class TestGapTimeline:
    """Smoke tests for create_gap_timeline."""

    def test_gap_timeline_returns_figure(self, sample_gaps: list[GapRecord]) -> None:
        """Gap timeline must return a Bokeh Figure."""
        result: Figure = create_gap_timeline(sample_gaps)
        assert isinstance(result, Figure)


# ---------------------------------------------------------------------------
# Tests — volume profile
# ---------------------------------------------------------------------------


class TestVolumeProfile:
    """Smoke tests for create_volume_profile."""

    def test_volume_profile_returns_figure(self, volume_df: pd.DataFrame) -> None:
        """Volume profile must return a Bokeh Figure."""
        result: Figure = create_volume_profile(volume_df)
        assert isinstance(result, Figure)


# ---------------------------------------------------------------------------
# Tests — bar count histogram
# ---------------------------------------------------------------------------


class TestBarCountHistogram:
    """Smoke tests for create_bar_count_histogram."""

    def test_bar_count_histogram_returns_figure(
        self,
        weekly_counts: dict[str, pd.Series],  # type: ignore[type-arg]
    ) -> None:
        """Bar count histogram must return a Bokeh Figure."""
        result: Figure = create_bar_count_histogram(weekly_counts)
        assert isinstance(result, Figure)


# ---------------------------------------------------------------------------
# Tests — statistics table
# ---------------------------------------------------------------------------


class TestStatisticsTable:
    """Smoke tests for create_statistics_table."""

    def test_statistics_table_returns_datatable(
        self,
        sample_stats: list[ReturnStatistics],
    ) -> None:
        """Statistics table must return a Bokeh DataTable."""
        result: DataTable = create_statistics_table(sample_stats)
        assert isinstance(result, DataTable)
