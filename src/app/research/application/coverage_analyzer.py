"""Coverage analysis service — gap detection and asset filtering for OHLCV data."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Final

import numpy as np
import pandas as pd

from src.app.research.application.data_loader import DataLoader
from src.app.research.domain.value_objects import AssetFilterResult, CoverageRecord, GapRecord

# ---------------------------------------------------------------------------
# Timeframe → expected interval mapping
# ---------------------------------------------------------------------------

_TIMEFRAME_HOURS: Final[dict[str, float]] = {
    "1h": 1.0,
    "4h": 4.0,
    "1d": 24.0,
}

_HOURS_PER_DAY: Final[float] = 24.0
_MIN_ROWS_FOR_DIFF: Final[int] = 2


def _interval_hours(timeframe: str) -> float:
    """Return the expected interval in hours for a timeframe string.

    Args:
        timeframe: Candlestick interval (``"1h"``, ``"4h"``, or ``"1d"``).

    Returns:
        Expected bar-to-bar interval in hours.

    Raises:
        ValueError: If the timeframe is not recognised.
    """
    interval: float | None = _TIMEFRAME_HOURS.get(timeframe)
    if interval is None:
        supported: str = ", ".join(sorted(_TIMEFRAME_HOURS))
        msg: str = f"Unknown timeframe {timeframe!r}. Supported: {supported}"
        raise ValueError(msg)
    return interval


class CoverageAnalyzer:
    """Analyses OHLCV data coverage, detects gaps, and filters assets by quality.

    All computations rely on :class:`DataLoader` for data access, keeping this
    service free of direct database dependencies.
    """

    def __init__(self, data_loader: DataLoader) -> None:
        """Initialise the analyser with a data loader.

        Args:
            data_loader: A configured :class:`DataLoader` instance.
        """
        self._data_loader: DataLoader = data_loader

    # -- coverage computation ------------------------------------------------

    def compute_coverage(
        self,
        assets: list[str],
        timeframes: list[str],
    ) -> list[CoverageRecord]:
        """Compute coverage statistics for every (asset, timeframe) combination.

        For each pair the method counts actual bars, computes the expected
        number of bars given the date span and timeframe interval, and derives
        a coverage percentage and gap count.

        Args:
            assets: List of trading pair symbols.
            timeframes: List of candlestick intervals.

        Returns:
            One :class:`CoverageRecord` per (asset, timeframe) pair that
            contains at least one bar.
        """
        records: list[CoverageRecord] = []

        for asset in assets:
            for timeframe in timeframes:
                df: pd.DataFrame = self._data_loader.load_ohlcv(asset, timeframe)
                if df.empty:
                    continue

                total_bars: int = len(df)
                start_date: datetime = df["timestamp"].iloc[0]
                end_date: datetime = df["timestamp"].iloc[-1]

                interval_h: float = _interval_hours(timeframe)
                span_hours: float = (end_date - start_date).total_seconds() / 3600.0
                expected_bars: int = int(span_hours / interval_h) + 1

                coverage_pct: float = (total_bars / expected_bars * 100.0) if expected_bars > 0 else 0.0
                # Clamp to 100% — slight over-count is possible due to rounding.
                coverage_pct = min(coverage_pct, 100.0)

                gaps: list[GapRecord] = self.detect_gaps(asset, timeframe)
                gap_count: int = len(gaps)

                record: CoverageRecord = CoverageRecord(
                    asset=asset,
                    timeframe=timeframe,
                    total_bars=total_bars,
                    start_date=start_date,
                    end_date=end_date,
                    expected_bars=expected_bars,
                    coverage_pct=coverage_pct,
                    gap_count=gap_count,
                )
                records.append(record)

        return records

    # -- gap detection -------------------------------------------------------

    def detect_gaps(
        self,
        asset: str,
        timeframe: str,
        max_gap_multiple: float = 3.0,
    ) -> list[GapRecord]:
        """Detect gaps in an OHLCV time series that exceed a threshold.

        A gap is flagged when the interval between consecutive timestamps
        exceeds ``max_gap_multiple * expected_interval``.

        Args:
            asset: Trading pair symbol.
            timeframe: Candlestick interval.
            max_gap_multiple: Multiplier of the expected interval above which
                a gap is flagged.  Defaults to ``3.0``.

        Returns:
            List of :class:`GapRecord` instances, one per detected gap.
        """
        df: pd.DataFrame = self._data_loader.load_ohlcv(asset, timeframe)
        if len(df) < _MIN_ROWS_FOR_DIFF:
            return []

        interval_h: float = _interval_hours(timeframe)
        threshold: timedelta = timedelta(hours=interval_h * max_gap_multiple)

        timestamps: pd.Series = df["timestamp"]  # type: ignore[type-arg]
        diffs: pd.Series = timestamps.diff()  # type: ignore[type-arg]

        # Vectorised mask — iterate only over actual gaps (not all rows).
        mask: pd.Series = diffs > threshold  # type: ignore[type-arg]
        gap_positions: np.ndarray = np.flatnonzero(mask.to_numpy())

        gaps: list[GapRecord] = []
        for pos in gap_positions:
            idx: int = int(pos)
            delta: timedelta = diffs.iloc[idx]
            gap_start: datetime = timestamps.iloc[idx - 1]
            gap_end: datetime = timestamps.iloc[idx]
            gap_hours: float = delta.total_seconds() / 3600.0
            missing: int = int(gap_hours / interval_h) - 1

            gap: GapRecord = GapRecord(
                asset=asset,
                timeframe=timeframe,
                gap_start=gap_start,
                gap_end=gap_end,
                gap_duration_hours=gap_hours,
                missing_bars=max(missing, 0),
            )
            gaps.append(gap)

        return gaps

    # -- asset filtering -----------------------------------------------------

    def filter_assets(
        self,
        assets: list[str],
        timeframe: str,
        min_days: int = 730,
        max_gap_ratio: float = 0.05,
    ) -> list[AssetFilterResult]:
        """Filter assets based on data quality criteria.

        An asset is *included* when it has at least ``min_days`` calendar days
        of data **and** the ratio of gap-bars to expected bars does not exceed
        ``max_gap_ratio``.

        Args:
            assets: List of trading pair symbols to evaluate.
            timeframe: Candlestick interval to check.
            min_days: Minimum calendar days of data required.  Defaults to
                ``730`` (2 years).
            max_gap_ratio: Maximum fraction of missing bars tolerated
                (0–1).  Defaults to ``0.05`` (5 %).

        Returns:
            One :class:`AssetFilterResult` per asset with inclusion decision
            and reason.
        """
        results: list[AssetFilterResult] = []

        for asset in assets:
            df: pd.DataFrame = self._data_loader.load_ohlcv(asset, timeframe)

            if df.empty:
                result: AssetFilterResult = AssetFilterResult(
                    asset=asset,
                    included=False,
                    reason="No data available",
                    total_days=0.0,
                    coverage_pct=0.0,
                )
                results.append(result)
                continue

            start_date: datetime = df["timestamp"].iloc[0]
            end_date: datetime = df["timestamp"].iloc[-1]
            total_days: float = (end_date - start_date).total_seconds() / (3600.0 * _HOURS_PER_DAY)

            interval_h: float = _interval_hours(timeframe)
            span_hours: float = (end_date - start_date).total_seconds() / 3600.0
            expected_bars: int = int(span_hours / interval_h) + 1
            total_bars: int = len(df)
            coverage_pct: float = (total_bars / expected_bars * 100.0) if expected_bars > 0 else 0.0
            coverage_pct = min(coverage_pct, 100.0)

            gap_ratio: float = 1.0 - (total_bars / expected_bars) if expected_bars > 0 else 1.0

            if total_days < min_days:
                reason: str = f"Insufficient data: {total_days:.1f} days < {min_days} required"
                included: bool = False
            elif gap_ratio > max_gap_ratio:
                reason = f"Too many gaps: {gap_ratio:.2%} missing > {max_gap_ratio:.2%} threshold"
                included = False
            else:
                reason = "Passed all quality filters"
                included = True

            result = AssetFilterResult(
                asset=asset,
                included=included,
                reason=reason,
                total_days=total_days,
                coverage_pct=coverage_pct,
            )
            results.append(result)

        return results

    # -- coverage matrix -----------------------------------------------------

    @staticmethod
    def build_coverage_matrix(records: list[CoverageRecord]) -> pd.DataFrame:
        """Build a pivot table of coverage percentages.

        Rows are assets, columns are timeframes, values are ``coverage_pct``.

        Args:
            records: Coverage records produced by :meth:`compute_coverage`.

        Returns:
            Pivot DataFrame with assets as index, timeframes as columns, and
            coverage percentages as values.
        """
        if not records:
            return pd.DataFrame()

        rows: list[dict[str, object]] = [
            {
                "asset": r.asset,
                "timeframe": r.timeframe,
                "coverage_pct": r.coverage_pct,
            }
            for r in records
        ]
        df: pd.DataFrame = pd.DataFrame(rows)
        matrix: pd.DataFrame = df.pivot(index="asset", columns="timeframe", values="coverage_pct")
        return matrix
