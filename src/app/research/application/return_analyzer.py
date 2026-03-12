"""Return distribution analysis — descriptive statistics and normality tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.app.research.domain.value_objects import ReturnStatistics

# ---------------------------------------------------------------------------
# Significance threshold
# ---------------------------------------------------------------------------

_NORMALITY_ALPHA: float = 0.05


class ReturnAnalyzer:
    """Stateless service for computing return distribution statistics.

    Provides log-return computation, descriptive statistics with
    Jarque-Bera normality testing, Q-Q plot data extraction, and
    cross-bar-type comparison utilities.
    """

    def compute_log_returns(  # noqa: PLR6301
        self,
        df: pd.DataFrame,
        price_col: str = "close",
    ) -> pd.Series:  # type: ignore[type-arg]
        """Compute log returns from a price column.

        Calculates ``ln(P_t / P_{t-1})`` and drops the leading NaN.

        Args:
            df: DataFrame containing at least the ``price_col`` column.
            price_col: Name of the column holding close prices.

        Returns:
            Series of log returns with length ``len(df) - 1``.

        """
        prices: pd.Series = df[price_col]  # type: ignore[type-arg]
        raw_returns: pd.Series = np.log(prices / prices.shift(1))  # type: ignore[type-arg]
        log_returns: pd.Series = raw_returns.dropna()  # type: ignore[type-arg]
        return log_returns

    def compute_statistics(  # noqa: PLR6301
        self,
        returns: pd.Series,  # type: ignore[type-arg]
        asset: str,
        bar_type: str,
    ) -> ReturnStatistics:
        """Compute descriptive statistics and Jarque-Bera normality test.

        Args:
            returns: Series of log returns (or any return series).
            asset: Trading pair symbol (e.g. ``"BTCUSDT"``).
            bar_type: Bar aggregation type (e.g. ``"time"``, ``"dollar"``).

        Returns:
            Frozen ``ReturnStatistics`` value object with all fields populated.
        """
        count: int = len(returns)

        def _safe(val: float, default: float = 0.0) -> float:
            """Replace NaN/Inf with *default*.

            Returns:
                Sanitised float value.
            """
            return default if (np.isnan(val) or np.isinf(val)) else val

        mean: float = _safe(float(returns.mean())) if count > 0 else 0.0
        std: float = _safe(float(returns.std())) if count > 0 else 0.0
        skewness: float = _safe(float(stats.skew(returns))) if count > 0 else 0.0
        kurtosis: float = _safe(float(stats.kurtosis(returns, fisher=True))) if count > 0 else 0.0
        jb_stat: float
        jb_pvalue: float
        _min_jb_samples: int = 3
        if count >= _min_jb_samples and std > 0:
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
            jb_stat = _safe(float(jb_stat))
            jb_pvalue = _safe(float(jb_pvalue), default=1.0)
        else:
            jb_stat = 0.0
            jb_pvalue = 1.0
        is_normal: bool = jb_pvalue >= _NORMALITY_ALPHA

        return ReturnStatistics(
            asset=asset,
            bar_type=bar_type,
            count=count,
            mean=mean,
            std=std,
            skewness=skewness,
            kurtosis=kurtosis,
            jarque_bera_stat=float(jb_stat),
            jarque_bera_pvalue=float(jb_pvalue),
            is_normal=is_normal,
        )

    def compute_qq_data(  # noqa: PLR6301
        self,
        returns: pd.Series,  # type: ignore[type-arg]
    ) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
        """Extract Q-Q plot data against a normal distribution.

        Uses ``scipy.stats.probplot`` to compute theoretical quantiles
        and ordered sample values for a Q-Q (quantile-quantile) plot.

        Args:
            returns: Series of return observations.

        Returns:
            Tuple of ``(theoretical_quantiles, ordered_values)`` where both
            arrays have the same length as ``returns``.
        """
        _min_qq_samples: int = 3
        if len(returns.dropna()) < _min_qq_samples:
            _empty: np.ndarray = np.array([], dtype=np.float64)  # type: ignore[type-arg]
            return _empty, _empty

        probplot_result: tuple[  # type: ignore[type-arg]
            tuple[np.ndarray, np.ndarray], tuple[float, float, float]
        ] = stats.probplot(returns, dist="norm")
        theoretical_quantiles: np.ndarray = probplot_result[0][0]  # type: ignore[type-arg]
        ordered_values: np.ndarray = probplot_result[0][1]  # type: ignore[type-arg]
        return theoretical_quantiles, ordered_values

    def compare_bar_types(
        self,
        bar_data: dict[str, pd.DataFrame],
        asset: str,
    ) -> list[ReturnStatistics]:
        """Compare return distributions across different bar types.

        For each bar type, computes log returns from the ``"close"`` column
        and then derives descriptive statistics with normality testing.

        Args:
            bar_data: Mapping of bar-type name to its OHLCV DataFrame.
                Each DataFrame must contain a ``"close"`` column.
            asset: Trading pair symbol (e.g. ``"BTCUSDT"``).

        Returns:
            List of ``ReturnStatistics``, one per bar type in ``bar_data``.
        """
        results: list[ReturnStatistics] = []
        for bar_type, df in bar_data.items():
            log_returns: pd.Series = self.compute_log_returns(df)  # type: ignore[type-arg]
            stat_result: ReturnStatistics = self.compute_statistics(log_returns, asset=asset, bar_type=bar_type)
            results.append(stat_result)
        return results
