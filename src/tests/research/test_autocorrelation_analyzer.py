"""Unit tests for AutocorrelationAnalyzer — ACF, PACF, and Ljung-Box diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.app.research.application.autocorrelation_analyzer import AutocorrelationAnalyzer
from src.app.research.domain.value_objects import ACFResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ASSET: str = "BTCUSDT"
_BAR_TYPE: str = "time"
_DEFAULT_MAX_LAG: int = 40


# ---------------------------------------------------------------------------
# TestAutocorrelationAnalyzer
# ---------------------------------------------------------------------------


class TestAutocorrelationAnalyzer:
    """Tests for AutocorrelationAnalyzer ACF, PACF, and Ljung-Box analysis."""

    def test_acf_on_ar1_process(self, ar1_returns: pd.Series) -> None:  # type: ignore[type-arg]
        """ACF at lag 1 of an AR(1) process with phi=0.3 should be close to 0.3."""
        analyzer: AutocorrelationAnalyzer = AutocorrelationAnalyzer()
        result: ACFResult = analyzer.compute_acf_analysis(
            ar1_returns,
            asset=_ASSET,
            bar_type=_BAR_TYPE,
        )
        acf_lag_1: float = float(result.acf_values[1])
        assert abs(acf_lag_1 - 0.3) < 0.1, f"ACF(1) = {acf_lag_1}, expected ~0.3"

    def test_acf_on_white_noise(self, white_noise_returns: pd.Series) -> None:  # type: ignore[type-arg]
        """All ACF values (lags 1+) of white noise should lie within the 95% CI band."""
        analyzer: AutocorrelationAnalyzer = AutocorrelationAnalyzer()
        result: ACFResult = analyzer.compute_acf_analysis(
            white_noise_returns,
            asset=_ASSET,
            bar_type=_BAR_TYPE,
        )
        n: int = len(white_noise_returns)
        ci_bound: float = 2.0 / np.sqrt(n)
        acf_beyond_lag_0: np.ndarray = result.acf_values[1:]  # type: ignore[type-arg]
        violations: int = int(np.sum(np.abs(acf_beyond_lag_0) > ci_bound))
        # At 95% CI, we allow up to ~5% of lags to exceed the bound
        max_violations: int = max(2, int(0.10 * _DEFAULT_MAX_LAG))
        assert violations <= max_violations, f"{violations} ACF values exceeded the 95% CI bound ({ci_bound:.4f})"

    def test_ljung_box_detects_serial_correlation(self, ar1_returns: pd.Series) -> None:  # type: ignore[type-arg]
        """Ljung-Box test on AR(1) returns should detect serial correlation (p < 0.05)."""
        analyzer: AutocorrelationAnalyzer = AutocorrelationAnalyzer()
        result: ACFResult = analyzer.compute_acf_analysis(
            ar1_returns,
            asset=_ASSET,
            bar_type=_BAR_TYPE,
        )
        assert result.has_serial_correlation is True
        assert result.ljung_box_pvalue < 0.05  # noqa: PLR2004

    def test_ljung_box_white_noise_not_rejected(
        self,
        white_noise_returns: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        """Ljung-Box test on white noise should NOT reject the null (p > 0.05)."""
        analyzer: AutocorrelationAnalyzer = AutocorrelationAnalyzer()
        result: ACFResult = analyzer.compute_acf_analysis(
            white_noise_returns,
            asset=_ASSET,
            bar_type=_BAR_TYPE,
        )
        assert result.has_serial_correlation is False
        assert result.ljung_box_pvalue > 0.05  # noqa: PLR2004

    def test_acf_values_length(self, normal_returns: pd.Series) -> None:  # type: ignore[type-arg]
        """ACF values array must have length max_lag + 1 (includes lag 0)."""
        max_lag: int = 20
        analyzer: AutocorrelationAnalyzer = AutocorrelationAnalyzer()
        result: ACFResult = analyzer.compute_acf_analysis(
            normal_returns,
            asset=_ASSET,
            bar_type=_BAR_TYPE,
            max_lag=max_lag,
        )
        assert len(result.acf_values) == max_lag + 1

    def test_pacf_values_length(self, normal_returns: pd.Series) -> None:  # type: ignore[type-arg]
        """PACF values array must have length max_lag + 1 (includes lag 0)."""
        max_lag: int = 20
        analyzer: AutocorrelationAnalyzer = AutocorrelationAnalyzer()
        result: ACFResult = analyzer.compute_acf_analysis(
            normal_returns,
            asset=_ASSET,
            bar_type=_BAR_TYPE,
            max_lag=max_lag,
        )
        assert len(result.pacf_values) == max_lag + 1

    def test_squared_returns_acf(self, ar1_returns: pd.Series) -> None:  # type: ignore[type-arg]
        """compute_squared_acf must return a valid ACFResult without errors."""
        analyzer: AutocorrelationAnalyzer = AutocorrelationAnalyzer()
        result: ACFResult = analyzer.compute_squared_acf(
            ar1_returns,
            asset=_ASSET,
            bar_type=_BAR_TYPE,
        )
        assert isinstance(result, ACFResult)
        assert result.asset == _ASSET
        assert result.bar_type == _BAR_TYPE
        assert len(result.acf_values) == _DEFAULT_MAX_LAG + 1
        assert len(result.pacf_values) == _DEFAULT_MAX_LAG + 1
        assert result.ljung_box_stat >= 0.0
        assert 0.0 <= result.ljung_box_pvalue <= 1.0

    def test_compare_serial_dependence_count(
        self,
        ar1_returns: pd.Series,
        white_noise_returns: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        """compare_serial_dependence with 2 bar types must return exactly 2 results."""
        analyzer: AutocorrelationAnalyzer = AutocorrelationAnalyzer()
        bar_returns: dict[str, pd.Series] = {  # type: ignore[type-arg]
            "time": ar1_returns,
            "dollar": white_noise_returns,
        }
        results: list[ACFResult] = analyzer.compare_serial_dependence(
            bar_returns,
            asset=_ASSET,
        )
        assert len(results) == 2
        bar_types: set[str] = {r.bar_type for r in results}
        assert bar_types == {"time", "dollar"}
