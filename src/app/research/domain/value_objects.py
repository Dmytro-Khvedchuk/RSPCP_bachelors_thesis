"""Research domain value objects — Pydantic result models for analysis services."""

from __future__ import annotations

from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Coverage & Gap Analysis
# ---------------------------------------------------------------------------


class CoverageRecord(BaseModel, frozen=True):
    """Coverage statistics for a single (asset, timeframe) pair.

    Attributes:
        asset: Trading pair symbol (e.g. ``"BTCUSDT"``).
        timeframe: Candlestick interval (e.g. ``"1h"``).
        total_bars: Number of bars actually present.
        start_date: Earliest timestamp in the data.
        end_date: Latest timestamp in the data.
        expected_bars: Number of bars expected given the date range and timeframe.
        coverage_pct: Percentage of expected bars that are present (0–100).
        gap_count: Number of detected gaps.
    """

    asset: str
    timeframe: str
    total_bars: int = Field(ge=0)
    start_date: datetime
    end_date: datetime
    expected_bars: int = Field(ge=0)
    coverage_pct: float = Field(ge=0.0, le=100.0)
    gap_count: int = Field(ge=0)


class GapRecord(BaseModel, frozen=True):
    """A single detected gap in an OHLCV time series.

    Attributes:
        asset: Trading pair symbol.
        timeframe: Candlestick interval.
        gap_start: Timestamp of the last bar before the gap.
        gap_end: Timestamp of the first bar after the gap.
        gap_duration_hours: Duration of the gap in hours.
        missing_bars: Estimated number of missing bars.
    """

    asset: str
    timeframe: str
    gap_start: datetime
    gap_end: datetime
    gap_duration_hours: float = Field(ge=0.0)
    missing_bars: int = Field(ge=0)


class AssetFilterResult(BaseModel, frozen=True):
    """Result of the asset filtering decision.

    Attributes:
        asset: Trading pair symbol.
        included: Whether this asset passed all quality filters.
        reason: Human-readable reason for inclusion or exclusion.
        total_days: Total calendar days spanned by available data.
        coverage_pct: Coverage percentage (0–100).
    """

    asset: str
    included: bool
    reason: str
    total_days: float = Field(ge=0.0)
    coverage_pct: float = Field(ge=0.0, le=100.0)


# ---------------------------------------------------------------------------
# Return Distribution Analysis
# ---------------------------------------------------------------------------


class ReturnStatistics(BaseModel, frozen=True):
    """Descriptive statistics for a return series.

    Attributes:
        asset: Trading pair symbol.
        bar_type: Bar aggregation type (e.g. ``"time"``, ``"dollar"``).
        count: Number of return observations.
        mean: Mean return.
        std: Standard deviation of returns.
        skewness: Fisher skewness.
        kurtosis: Fisher (excess) kurtosis.
        jarque_bera_stat: Jarque-Bera test statistic.
        jarque_bera_pvalue: Jarque-Bera p-value.
        is_normal: Whether normality was NOT rejected at alpha=0.05.
    """

    asset: str
    bar_type: str
    count: int = Field(ge=0)
    mean: float
    std: float = Field(ge=0.0)
    skewness: float
    kurtosis: float
    jarque_bera_stat: float = Field(ge=0.0)
    jarque_bera_pvalue: float = Field(ge=0.0, le=1.0)
    is_normal: bool


# ---------------------------------------------------------------------------
# Autocorrelation & Serial Dependence
# ---------------------------------------------------------------------------


class ACFResult(BaseModel, frozen=True):
    """Autocorrelation function analysis result.

    Attributes:
        asset: Trading pair symbol.
        bar_type: Bar aggregation type.
        acf_values: ACF coefficients from lag 0 to ``max_lag``.
        pacf_values: PACF coefficients from lag 0 to ``max_lag``.
        ljung_box_stat: Ljung-Box Q-statistic at the final lag.
        ljung_box_pvalue: Ljung-Box p-value at the final lag.
        has_serial_correlation: Whether serial correlation was detected (p < 0.05).
    """

    model_config = {"arbitrary_types_allowed": True}

    asset: str
    bar_type: str
    acf_values: np.ndarray  # type: ignore[type-arg]
    pacf_values: np.ndarray  # type: ignore[type-arg]
    ljung_box_stat: float = Field(ge=0.0)
    ljung_box_pvalue: float = Field(ge=0.0, le=1.0)
    has_serial_correlation: bool


# ---------------------------------------------------------------------------
# Bar Comparison
# ---------------------------------------------------------------------------


class BarDurationStats(BaseModel, frozen=True):
    """Duration statistics for bars of a given type.

    Attributes:
        asset: Trading pair symbol.
        bar_type: Bar aggregation type.
        mean_duration_minutes: Mean bar duration in minutes.
        median_duration_minutes: Median bar duration in minutes.
        std_duration_minutes: Standard deviation of bar duration in minutes.
        min_duration_minutes: Minimum bar duration in minutes.
        max_duration_minutes: Maximum bar duration in minutes.
        cv: Coefficient of variation (std / mean).
    """

    asset: str
    bar_type: str
    mean_duration_minutes: float = Field(ge=0.0)
    median_duration_minutes: float = Field(ge=0.0)
    std_duration_minutes: float = Field(ge=0.0)
    min_duration_minutes: float = Field(ge=0.0)
    max_duration_minutes: float = Field(ge=0.0)
    cv: float = Field(ge=0.0)
