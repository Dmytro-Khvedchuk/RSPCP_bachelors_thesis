"""Unit tests for ReturnAnalyzer — return distribution statistics service.

Tests cover log-return computation, descriptive statistics with Jarque-Bera
normality testing, Q-Q data extraction, and cross-bar-type comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.app.research.application.return_analyzer import ReturnAnalyzer
from src.app.research.domain.value_objects import ReturnStatistics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NORMALITY_ALPHA: float = 0.05


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer() -> ReturnAnalyzer:
    """Return a fresh ReturnAnalyzer instance.

    Returns:
        Stateless ReturnAnalyzer.
    """
    return ReturnAnalyzer()


# ---------------------------------------------------------------------------
# TestReturnAnalyzer
# ---------------------------------------------------------------------------


class TestReturnAnalyzer:
    """Tests for ReturnAnalyzer covering returns, statistics, Q-Q, and comparison."""

    # -- log returns --------------------------------------------------------

    def test_log_returns_on_known_prices(self, analyzer: ReturnAnalyzer) -> None:
        """Log returns of [100, 110, 121] must equal [ln(1.1), ln(1.1)]."""
        df: pd.DataFrame = pd.DataFrame({"close": [100.0, 110.0, 121.0]})
        result: pd.Series = analyzer.compute_log_returns(df)  # type: ignore[type-arg]
        expected: np.ndarray = np.array([np.log(1.1), np.log(1.1)])  # type: ignore[type-arg]
        np.testing.assert_allclose(result.values, expected, rtol=1e-10)

    def test_log_returns_drops_first_nan(self, analyzer: ReturnAnalyzer) -> None:
        """Output length must be n-1 (the leading NaN is dropped)."""
        n: int = 50
        df: pd.DataFrame = pd.DataFrame({"close": np.arange(1.0, n + 1.0)})
        result: pd.Series = analyzer.compute_log_returns(df)  # type: ignore[type-arg]
        assert len(result) == n - 1

    def test_log_returns_custom_price_col(self, analyzer: ReturnAnalyzer) -> None:
        """compute_log_returns must respect a custom price_col argument."""
        df: pd.DataFrame = pd.DataFrame({"adjusted_close": [100.0, 200.0, 400.0]})
        result: pd.Series = analyzer.compute_log_returns(df, price_col="adjusted_close")  # type: ignore[type-arg]
        expected: np.ndarray = np.array([np.log(2.0), np.log(2.0)])  # type: ignore[type-arg]
        np.testing.assert_allclose(result.values, expected, rtol=1e-10)

    # -- statistics ---------------------------------------------------------

    def test_statistics_on_normal_data(
        self,
        analyzer: ReturnAnalyzer,
        normal_returns: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        """N(0, 0.01) returns: skew ~ 0, excess kurtosis ~ 0, JB p > 0.05."""
        result: ReturnStatistics = analyzer.compute_statistics(normal_returns, asset="BTCUSDT", bar_type="time")
        assert abs(result.skewness) < 0.5
        assert abs(result.kurtosis) < 1.0
        assert result.jarque_bera_pvalue > _NORMALITY_ALPHA
        assert result.is_normal is True

    def test_statistics_on_fat_tails(
        self,
        analyzer: ReturnAnalyzer,
        fat_tail_returns: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        """Student-t(3) returns: excess kurtosis > 0, JB p < 0.05, is_normal=False."""
        result: ReturnStatistics = analyzer.compute_statistics(fat_tail_returns, asset="BTCUSDT", bar_type="dollar")
        assert result.kurtosis > 0.0
        assert result.jarque_bera_pvalue < _NORMALITY_ALPHA
        assert result.is_normal is False

    def test_statistics_count_matches_input(
        self,
        analyzer: ReturnAnalyzer,
        normal_returns: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        """The count field must match the length of the input series."""
        result: ReturnStatistics = analyzer.compute_statistics(normal_returns, asset="BTCUSDT", bar_type="time")
        assert result.count == len(normal_returns)

    # -- Q-Q data -----------------------------------------------------------

    def test_qq_data_shape_matches_input(
        self,
        analyzer: ReturnAnalyzer,
        normal_returns: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        """Both Q-Q arrays must have the same length as the input series."""
        theoretical: np.ndarray  # type: ignore[type-arg]
        observed: np.ndarray  # type: ignore[type-arg]
        theoretical, observed = analyzer.compute_qq_data(normal_returns)
        assert len(theoretical) == len(normal_returns)
        assert len(observed) == len(normal_returns)

    def test_qq_data_sorted(
        self,
        analyzer: ReturnAnalyzer,
        normal_returns: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        """Observed quantiles (ordered values) must be sorted ascending."""
        _theoretical: np.ndarray  # type: ignore[type-arg]
        observed: np.ndarray  # type: ignore[type-arg]
        _theoretical, observed = analyzer.compute_qq_data(normal_returns)
        assert np.all(observed[:-1] <= observed[1:])

    # -- compare_bar_types --------------------------------------------------

    def test_compare_bar_types_returns_one_per_type(
        self,
        analyzer: ReturnAnalyzer,
        synthetic_ohlcv_df: pd.DataFrame,
    ) -> None:
        """Providing 3 bar types must yield exactly 3 ReturnStatistics."""
        bar_data: dict[str, pd.DataFrame] = {
            "time": synthetic_ohlcv_df,
            "dollar": synthetic_ohlcv_df,
            "volume": synthetic_ohlcv_df,
        }
        results: list[ReturnStatistics] = analyzer.compare_bar_types(bar_data, asset="BTCUSDT")
        assert len(results) == 3
        bar_types_returned: set[str] = {r.bar_type for r in results}
        assert bar_types_returned == {"time", "dollar", "volume"}
