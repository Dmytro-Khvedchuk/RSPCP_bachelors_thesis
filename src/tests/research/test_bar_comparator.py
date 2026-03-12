"""Unit tests for BarComparator — bar count, duration, variability, and volume analysis."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.app.research.application.bar_comparator import BarComparator
from src.app.research.domain.value_objects import BarDurationStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars_with_durations(durations_minutes: list[float]) -> pd.DataFrame:
    """Build a minimal bar DataFrame with explicit durations.

    Args:
        durations_minutes: Duration in minutes for each bar.

    Returns:
        DataFrame with ``start_ts`` and ``end_ts`` columns.
    """
    start_ts: list[datetime] = []
    end_ts: list[datetime] = []
    current: datetime = datetime(2024, 1, 1, tzinfo=UTC)
    for dur in durations_minutes:
        start_ts.append(current)
        end: datetime = current + timedelta(minutes=dur)
        end_ts.append(end)
        current = end

    n: int = len(durations_minutes)
    return pd.DataFrame(
        {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "open": [100.0] * n,
            "high": [110.0] * n,
            "low": [90.0] * n,
            "close": [105.0] * n,
            "volume": [1.0] * n,
            "tick_count": [10] * n,
            "buy_volume": [0.5] * n,
            "sell_volume": [0.5] * n,
            "vwap": [100.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBarComparator:
    """Tests for :class:`BarComparator` methods."""

    # -- bar_count_per_period ------------------------------------------------

    def test_bar_count_per_week_correct(self) -> None:
        """Weekly bar counts match a hand-crafted distribution."""
        # Week 1: 3 bars, Week 2: 2 bars
        timestamps: list[datetime] = [
            datetime(2024, 1, 1, tzinfo=UTC),  # Mon W1
            datetime(2024, 1, 2, tzinfo=UTC),  # Tue W1
            datetime(2024, 1, 3, tzinfo=UTC),  # Wed W1
            datetime(2024, 1, 8, tzinfo=UTC),  # Mon W2
            datetime(2024, 1, 9, tzinfo=UTC),  # Tue W2
        ]
        df: pd.DataFrame = pd.DataFrame(
            {
                "start_ts": timestamps,
                "end_ts": [t + timedelta(hours=1) for t in timestamps],
                "open": [1.0] * 5,
                "high": [1.0] * 5,
                "low": [1.0] * 5,
                "close": [1.0] * 5,
                "volume": [1.0] * 5,
            }
        )

        comparator: BarComparator = BarComparator()
        counts: pd.Series = comparator.bar_count_per_period(df, freq="W")  # type: ignore[type-arg]

        total: int = int(counts.sum())
        assert total == 5
        assert len(counts) == 2
        assert int(counts.iloc[0]) == 3
        assert int(counts.iloc[1]) == 2

    def test_bar_count_per_month(self) -> None:
        """Monthly bar counts with freq='ME' aggregate correctly."""
        timestamps: list[datetime] = [
            datetime(2024, 1, 10, tzinfo=UTC),
            datetime(2024, 1, 20, tzinfo=UTC),
            datetime(2024, 2, 5, tzinfo=UTC),
        ]
        df: pd.DataFrame = pd.DataFrame(
            {
                "start_ts": timestamps,
                "end_ts": [t + timedelta(hours=1) for t in timestamps],
            }
        )

        comparator: BarComparator = BarComparator()
        counts: pd.Series = comparator.bar_count_per_period(df, freq="ME")  # type: ignore[type-arg]

        assert len(counts) == 2
        assert int(counts.iloc[0]) == 2  # January
        assert int(counts.iloc[1]) == 1  # February

    # -- compute_duration_stats ----------------------------------------------

    def test_duration_stats_known_values(self) -> None:
        """Duration stats for [10, 20, 30, 40, 50] min bars match exact values."""
        durations: list[float] = [10.0, 20.0, 30.0, 40.0, 50.0]
        df: pd.DataFrame = _make_bars_with_durations(durations)

        comparator: BarComparator = BarComparator()
        stats: BarDurationStats = comparator.compute_duration_stats(df, asset="BTCUSDT", bar_type="dollar")

        assert stats.asset == "BTCUSDT"
        assert stats.bar_type == "dollar"
        assert stats.mean_duration_minutes == pytest.approx(30.0)
        assert stats.median_duration_minutes == pytest.approx(30.0)
        assert stats.min_duration_minutes == pytest.approx(10.0)
        assert stats.max_duration_minutes == pytest.approx(50.0)

        expected_std: float = float(np.std([10.0, 20.0, 30.0, 40.0, 50.0], ddof=1))
        assert stats.std_duration_minutes == pytest.approx(expected_std, rel=1e-6)

    def test_duration_stats_cv_zero_for_uniform(self) -> None:
        """All bars with identical duration must yield CV = 0."""
        durations: list[float] = [60.0, 60.0, 60.0, 60.0]
        df: pd.DataFrame = _make_bars_with_durations(durations)

        comparator: BarComparator = BarComparator()
        stats: BarDurationStats = comparator.compute_duration_stats(df, asset="ETHUSDT", bar_type="tick")

        assert stats.cv == pytest.approx(0.0, abs=1e-12)

    def test_cv_positive_for_variable_bars(self, synthetic_bars_df: pd.DataFrame) -> None:
        """Synthetic bars with variable durations must have CV > 0."""
        comparator: BarComparator = BarComparator()
        stats: BarDurationStats = comparator.compute_duration_stats(
            synthetic_bars_df,
            asset="BTCUSDT",
            bar_type="volume",
        )

        assert stats.cv > 0.0
        assert stats.mean_duration_minutes > 0.0
        assert stats.std_duration_minutes > 0.0

    # -- compare_bar_count_variability ---------------------------------------

    def test_compare_bar_count_variability_shape(
        self,
        synthetic_bars_df: pd.DataFrame,
    ) -> None:
        """Output has one row per bar type provided."""
        comparator: BarComparator = BarComparator()
        bar_data: dict[str, pd.DataFrame] = {
            "tick": synthetic_bars_df,
            "volume": synthetic_bars_df,
            "dollar": synthetic_bars_df,
        }
        result: pd.DataFrame = comparator.compare_bar_count_variability(bar_data)

        assert len(result) == 3

    def test_compare_bar_count_variability_columns(
        self,
        synthetic_bars_df: pd.DataFrame,
    ) -> None:
        """Output DataFrame must have exactly bar_type, mean_count, std_count, cv."""
        comparator: BarComparator = BarComparator()
        bar_data: dict[str, pd.DataFrame] = {
            "tick": synthetic_bars_df,
            "volume": synthetic_bars_df,
        }
        result: pd.DataFrame = comparator.compare_bar_count_variability(bar_data)

        expected_columns: set[str] = {"bar_type", "mean_count", "std_count", "cv"}
        assert set(result.columns) == expected_columns

    # -- compute_volume_profile ----------------------------------------------

    def test_volume_profile_weekly_aggregation(self) -> None:
        """Weekly volume sums match hand-computed values across 3 weeks."""
        # Week 1: vol 10 + 20 = 30
        # Week 2: vol 5
        # Week 3: vol 15 + 25 = 40
        timestamps: list[datetime] = [
            datetime(2024, 1, 1, tzinfo=UTC),  # Mon W1
            datetime(2024, 1, 3, tzinfo=UTC),  # Wed W1
            datetime(2024, 1, 8, tzinfo=UTC),  # Mon W2
            datetime(2024, 1, 15, tzinfo=UTC),  # Mon W3
            datetime(2024, 1, 17, tzinfo=UTC),  # Wed W3
        ]
        volumes: list[float] = [10.0, 20.0, 5.0, 15.0, 25.0]
        ohlcv: pd.DataFrame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "volume": volumes,
            }
        )

        comparator: BarComparator = BarComparator()
        result: pd.DataFrame = comparator.compute_volume_profile(ohlcv, freq="W")

        assert "period" in result.columns
        assert "total_volume" in result.columns
        assert len(result) == 3

        total_volumes: list[float] = result["total_volume"].tolist()
        assert total_volumes[0] == pytest.approx(30.0)
        assert total_volumes[1] == pytest.approx(5.0)
        assert total_volumes[2] == pytest.approx(40.0)
