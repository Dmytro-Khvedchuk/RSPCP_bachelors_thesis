"""Autocorrelation and serial dependence analysis for return series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf

from src.app.research.domain.value_objects import ACFResult


class AutocorrelationAnalyzer:
    """Stateless analyzer for autocorrelation structure in financial return series.

    Computes ACF, PACF, and Ljung-Box tests to detect serial dependence
    in raw returns (momentum/mean-reversion signals) and squared returns
    (volatility clustering).
    """

    def compute_acf_analysis(  # noqa: PLR6301
        self,
        returns: pd.Series,  # type: ignore[type-arg]
        asset: str,
        bar_type: str,
        *,
        max_lag: int = 40,
    ) -> ACFResult:
        """Compute ACF, PACF, and Ljung-Box test for a return series.

        Uses ``statsmodels.tsa.stattools.acf`` with 95 % confidence bands
        and ``statsmodels.stats.diagnostic.acorr_ljungbox`` at the final lag.

        Args:
            returns: Return series (e.g. log returns or simple returns).
            asset: Trading pair symbol (e.g. ``"BTCUSDT"``).
            bar_type: Bar aggregation type (e.g. ``"time"``, ``"dollar"``).
            max_lag: Maximum lag order for ACF/PACF. Defaults to 40.

        Returns:
            ACFResult containing ACF/PACF arrays and Ljung-Box diagnostics.
        """
        # PACF requires nlags < n // 2; cap max_lag accordingly
        _n_obs: int = len(returns)
        _pacf_limit: int = _n_obs // 2 - 1
        _effective_lag: int = min(max_lag, _pacf_limit)

        _min_acf_samples: int = 4  # need at least a handful of observations
        if _n_obs < _min_acf_samples or _effective_lag < 1:
            acf_values: np.ndarray = np.array([], dtype=np.float64)  # type: ignore[type-arg]
            pacf_values: np.ndarray = np.array([], dtype=np.float64)  # type: ignore[type-arg]
            ljung_box_stat: float = 0.0
            ljung_box_pvalue: float = 1.0
            has_serial_correlation: bool = False
            return ACFResult(
                asset=asset,
                bar_type=bar_type,
                acf_values=acf_values,
                pacf_values=pacf_values,
                ljung_box_stat=ljung_box_stat,
                ljung_box_pvalue=ljung_box_pvalue,
                has_serial_correlation=has_serial_correlation,
            )

        acf_result: tuple[np.ndarray, np.ndarray] = acf(returns.to_numpy(), nlags=_effective_lag, alpha=0.05)  # type: ignore[type-arg, assignment]
        acf_values = acf_result[0]
        pacf_values = pacf(returns.to_numpy(), nlags=_effective_lag, method="ywm")  # type: ignore[type-arg]

        ljung_box_df: pd.DataFrame = acorr_ljungbox(returns, lags=[_effective_lag])  # type: ignore[type-arg]
        ljung_box_stat = float(ljung_box_df["lb_stat"].iloc[0])
        ljung_box_pvalue = float(ljung_box_df["lb_pvalue"].iloc[0])
        if np.isnan(ljung_box_stat):
            ljung_box_stat = 0.0
        if np.isnan(ljung_box_pvalue):
            ljung_box_pvalue = 1.0

        has_serial_correlation: bool = ljung_box_pvalue < 0.05  # noqa: PLR2004

        return ACFResult(
            asset=asset,
            bar_type=bar_type,
            acf_values=acf_values,
            pacf_values=pacf_values,
            ljung_box_stat=ljung_box_stat,
            ljung_box_pvalue=ljung_box_pvalue,
            has_serial_correlation=has_serial_correlation,
        )

    def compute_squared_acf(
        self,
        returns: pd.Series,  # type: ignore[type-arg]
        asset: str,
        bar_type: str,
        *,
        max_lag: int = 40,
    ) -> ACFResult:
        """Compute ACF analysis on squared returns for volatility clustering detection.

        Significant autocorrelation in squared returns indicates ARCH/GARCH
        effects — time-varying volatility that many financial time series exhibit.

        Args:
            returns: Return series (e.g. log returns or simple returns).
            asset: Trading pair symbol (e.g. ``"BTCUSDT"``).
            bar_type: Bar aggregation type (e.g. ``"time"``, ``"dollar"``).
            max_lag: Maximum lag order for ACF/PACF. Defaults to 40.

        Returns:
            ACFResult for the squared return series.
        """
        squared_returns: pd.Series = returns**2  # type: ignore[type-arg]
        return self.compute_acf_analysis(
            squared_returns,
            asset=asset,
            bar_type=bar_type,
            max_lag=max_lag,
        )

    def compare_serial_dependence(
        self,
        bar_returns: dict[str, pd.Series],  # type: ignore[type-arg]
        asset: str,
    ) -> list[ACFResult]:
        """Compare serial dependence across multiple bar types for a single asset.

        Runs ACF analysis on each bar type's return series and returns
        the collected results. Useful for evaluating which bar sampling
        method reduces serial correlation (a desirable property per
        Lopez de Prado, *Advances in Financial Machine Learning*).

        Args:
            bar_returns: Mapping from bar type name to its return series.
            asset: Trading pair symbol (e.g. ``"BTCUSDT"``).

        Returns:
            List of ACFResult, one per bar type.
        """
        results: list[ACFResult] = []
        for bar_type, returns_series in bar_returns.items():
            result: ACFResult = self.compute_acf_analysis(
                returns_series,
                asset=asset,
                bar_type=bar_type,
            )
            results.append(result)
        return results
