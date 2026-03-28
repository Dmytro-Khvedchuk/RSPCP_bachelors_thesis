"""Autocorrelation, variance ratio, and Granger causality analysis.

Implements tier-gated serial dependence profiling using the ML-research
path (Pandas / NumPy / statsmodels / arch) per CLAUDE.md.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from arch.unitroot import VarianceRatio  # type: ignore[import-untyped]
from loguru import logger
from scipy.stats import norm  # type: ignore[import-untyped]
from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore[import-untyped]
from statsmodels.tsa.stattools import acf, grangercausalitytests, pacf  # type: ignore[import-untyped]

from src.app.profiling.domain.value_objects import (
    AutocorrelationConfig,
    AutocorrelationProfile,
    GrangerResult,
    LjungBoxResult,
    SampleTier,
    VarianceRatioResult,
)

# ---------------------------------------------------------------------------
# Minimum observation thresholds
# ---------------------------------------------------------------------------

_MIN_ACF_SAMPLES: int = 4
"""Minimum samples required for meaningful ACF/PACF computation."""

_TIER_B_MAX_VR_HORIZON_DAYS: float = 7.0
"""Maximum variance ratio calendar horizon (days) allowed for Tier B."""


class SerialDependenceAnalyzer:
    """Stateless service for autocorrelation, variance ratio, and Granger causality analysis.

    Tier-gated:
        - **All tiers:** ACF/PACF, multi-lag Ljung-Box.
        - **Tier A/B:** Lo-MacKinlay variance ratio (robust Z2), Chow-Denning.
        - **Tier A only:** Granger causality.
        - **Tier B:** VR capped at 1-week horizon.
        - **Tier C:** ACF/PACF + Ljung-Box only.
    """

    def analyze(  # noqa: PLR6301, PLR0913, PLR0917
        self,
        returns: pd.Series,  # type: ignore[type-arg]
        asset: str,
        bar_type: str,
        tier: SampleTier,
        bars_per_day: float,
        config: AutocorrelationConfig | None = None,
        squared_returns: pd.Series | None = None,  # type: ignore[type-arg]
    ) -> AutocorrelationProfile:
        """Compute a full autocorrelation and serial dependence profile.

        Args:
            returns: Pandas Series of log returns (NaN-free).
            asset: Trading pair symbol (e.g. ``"BTCUSDT"``).
            bar_type: Bar aggregation type (e.g. ``"dollar"``).
            tier: Sample-size tier controlling analysis depth.
            bars_per_day: Average number of bars per calendar day, used
                to convert variance ratio calendar horizons to bar counts.
            config: Autocorrelation analysis configuration.  Uses defaults
                when ``None``.
            squared_returns: Pre-computed squared returns.  If ``None``,
                computed as ``returns ** 2``.

        Returns:
            Frozen ``AutocorrelationProfile`` value object.
        """
        if config is None:
            config = AutocorrelationConfig()

        n_obs: int = len(returns)
        logger.debug(
            "Analysing serial dependence: asset={}, bar_type={}, tier={}, n={}",
            asset,
            bar_type,
            tier.value,
            n_obs,
        )

        # Compute squared returns if not provided
        sq_returns: pd.Series = squared_returns if squared_returns is not None else returns**2  # type: ignore[type-arg]

        # ACF / PACF for raw and squared returns (all tiers)
        acf_vals: np.ndarray  # type: ignore[type-arg]
        pacf_vals: np.ndarray  # type: ignore[type-arg]
        acf_vals, pacf_vals = _compute_acf_pacf(returns, config.max_lag)

        acf_sq_vals: np.ndarray  # type: ignore[type-arg]
        pacf_sq_vals: np.ndarray  # type: ignore[type-arg]
        acf_sq_vals, pacf_sq_vals = _compute_acf_pacf(sq_returns, config.max_lag)

        # Ljung-Box tests (all tiers)
        lb_returns: tuple[LjungBoxResult, ...] = _compute_ljung_box(returns, config.ljung_box_lags, config.alpha)
        lb_squared: tuple[LjungBoxResult, ...] = _compute_ljung_box(sq_returns, config.ljung_box_lags, config.alpha)

        has_serial: bool = any(r.significant for r in lb_returns)
        has_vol_clustering: bool = any(r.significant for r in lb_squared)

        # Variance ratio (Tier A / B)
        vr_results: tuple[VarianceRatioResult, ...] | None = None
        cd_stat: float | None = None
        cd_pvalue: float | None = None

        if tier in {SampleTier.A, SampleTier.B}:
            vr_results, cd_stat, cd_pvalue = _compute_variance_ratios(
                returns=returns,
                bars_per_day=bars_per_day,
                calendar_horizons=config.vr_calendar_horizons_days,
                tier=tier,
                robust=config.vr_robust,
                alpha=config.alpha,
            )

        # Granger causality (Tier A only) -- handled externally via test_granger_pairs
        granger_results: tuple[GrangerResult, ...] | None = None

        return AutocorrelationProfile(
            asset=asset,
            bar_type=bar_type,
            tier=tier,
            n_observations=n_obs,
            acf_values=acf_vals,
            pacf_values=pacf_vals,
            acf_squared_values=acf_sq_vals,
            pacf_squared_values=pacf_sq_vals,
            ljung_box_returns=lb_returns,
            ljung_box_squared=lb_squared,
            has_serial_correlation=has_serial,
            has_volatility_clustering=has_vol_clustering,
            vr_results=vr_results,
            chow_denning_stat=cd_stat,
            chow_denning_pvalue=cd_pvalue,
            granger_results=granger_results,
        )

    def test_granger_pairs(  # noqa: PLR6301
        self,
        returns_dict: dict[str, pd.Series],  # type: ignore[type-arg]
        lags: tuple[int, ...],
        alpha: float,
    ) -> tuple[GrangerResult, ...]:
        """Test Granger causality across all ordered pairs from a returns dictionary.

        For a dictionary with keys {A, B, C}, tests all ordered pairs:
        A->B, A->C, B->A, B->C, C->A, C->B.

        Args:
            returns_dict: Mapping from series label to return series.
            lags: Lag values to test for each pair.
            alpha: Significance level for the F-test.

        Returns:
            Tuple of all ``GrangerResult`` objects across all pairs and lags.
        """
        all_results: list[GrangerResult] = []
        names: list[str] = list(returns_dict.keys())

        for i, source_name in enumerate(names):
            for j, target_name in enumerate(names):
                if i == j:
                    continue
                source: pd.Series = returns_dict[source_name]  # type: ignore[type-arg]
                target: pd.Series = returns_dict[target_name]  # type: ignore[type-arg]
                pair_results: tuple[GrangerResult, ...] = _compute_granger(
                    source=source,
                    target=target,
                    source_name=source_name,
                    target_name=target_name,
                    lags=lags,
                    alpha=alpha,
                )
                all_results.extend(pair_results)

        return tuple(all_results)


# ---------------------------------------------------------------------------
# Internal computation functions
# ---------------------------------------------------------------------------


def _compute_acf_pacf(
    series: pd.Series,  # type: ignore[type-arg]
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Compute ACF and PACF arrays for a series.

    Caps ``max_lag`` at ``n_obs // 2 - 1`` to satisfy PACF requirements.
    Returns empty arrays when the series is too short.

    Args:
        series: Pandas Series of observations.
        max_lag: Requested maximum lag order.

    Returns:
        Tuple of ``(acf_array, pacf_array)``.
    """
    n_obs: int = len(series)
    pacf_limit: int = n_obs // 2 - 1
    effective_lag: int = min(max_lag, pacf_limit)

    empty: np.ndarray = np.array([], dtype=np.float64)  # type: ignore[type-arg]
    if n_obs < _MIN_ACF_SAMPLES or effective_lag < 1:
        return empty, empty

    data: np.ndarray = series.to_numpy(dtype=np.float64)  # type: ignore[type-arg]

    # acf returns (values, confint) when alpha is specified; we only need values
    acf_result: tuple[np.ndarray, np.ndarray] = acf(data, nlags=effective_lag, alpha=0.05)  # type: ignore[type-arg, assignment]
    acf_values: np.ndarray = acf_result[0]  # type: ignore[type-arg]

    pacf_values: np.ndarray = pacf(data, nlags=effective_lag, method="ywm")  # type: ignore[type-arg]  # ty: ignore[invalid-assignment]

    return acf_values, pacf_values


def _compute_ljung_box(
    series: pd.Series,  # type: ignore[type-arg]
    lags: tuple[int, ...],
    alpha: float,
) -> tuple[LjungBoxResult, ...]:
    """Run Ljung-Box test at multiple lag values.

    Filters out lags that are >= ``n_obs // 2`` to avoid degenerate results.
    Replaces NaN statistics with 0.0 and NaN p-values with 1.0.

    Args:
        series: Pandas Series of observations.
        lags: Requested lag values.
        alpha: Significance level.

    Returns:
        Tuple of ``LjungBoxResult``, one per valid lag.
    """
    n_obs: int = len(series)
    max_valid_lag: int = n_obs // 2 - 1
    valid_lags: list[int] = [lag for lag in lags if lag <= max_valid_lag and lag >= 1]

    if not valid_lags:
        return ()

    lb_df: pd.DataFrame = acorr_ljungbox(series, lags=valid_lags)  # type: ignore[arg-type]

    results: list[LjungBoxResult] = []
    for lag_val in valid_lags:
        q_stat: float = float(lb_df.loc[lag_val, "lb_stat"])
        p_val: float = float(lb_df.loc[lag_val, "lb_pvalue"])

        # Sanitise NaN values
        if np.isnan(q_stat):
            q_stat = 0.0
        if np.isnan(p_val):
            p_val = 1.0

        significant: bool = p_val < alpha
        results.append(
            LjungBoxResult(
                lag=lag_val,
                q_statistic=q_stat,
                p_value=p_val,
                significant=significant,
            )
        )

    return tuple(results)


def _compute_variance_ratios(  # noqa: PLR0913, PLR0914, PLR0917
    returns: pd.Series,  # type: ignore[type-arg]
    bars_per_day: float,
    calendar_horizons: tuple[float, ...],
    tier: SampleTier,
    robust: bool,
    alpha: float,
) -> tuple[tuple[VarianceRatioResult, ...], float | None, float | None]:
    """Compute Lo-MacKinlay variance ratios and Chow-Denning joint test.

    Converts calendar-day horizons to bar counts using ``bars_per_day``.
    For Tier B, horizons exceeding 7 days are excluded.  Deduplicates
    ``q`` values that collapse when ``bars_per_day`` is low.

    The Chow-Denning statistic is ``max(|z_i|)`` across all tested horizons,
    with p-value computed via Sidak correction.

    Args:
        returns: Pandas Series of log returns.
        bars_per_day: Average bars per calendar day.
        calendar_horizons: Calendar-day horizons to test.
        tier: Sample-size tier (Tier B caps at 7-day horizon).
        robust: Whether to use heteroscedasticity-robust Z2 statistic.
        alpha: Significance level.

    Returns:
        Tuple of ``(vr_results, chow_denning_stat, chow_denning_pvalue)``.
    """
    returns_arr: np.ndarray = returns.to_numpy(dtype=np.float64)  # type: ignore[type-arg]
    # arch.unitroot.VarianceRatio expects *price levels* (cumulative sum),
    # not returns.  It internally differences to compute the variance ratio.
    prices: np.ndarray = np.cumsum(returns_arr)  # type: ignore[type-arg]

    # Filter horizons for Tier B
    effective_horizons: list[float]
    if tier == SampleTier.B:
        effective_horizons = [h for h in calendar_horizons if h <= _TIER_B_MAX_VR_HORIZON_DAYS]
    else:
        effective_horizons = list(calendar_horizons)

    # Convert calendar horizons to bar counts, deduplicate
    seen_q: set[int] = set()
    horizon_q_pairs: list[tuple[float, int]] = []
    for horizon in effective_horizons:
        q: int = max(2, round(horizon * bars_per_day))
        if q not in seen_q:
            seen_q.add(q)
            horizon_q_pairs.append((horizon, q))

    if not horizon_q_pairs:
        return (), None, None

    vr_results: list[VarianceRatioResult] = []
    z_abs_values: list[float] = []

    for cal_horizon, q_val in horizon_q_pairs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vr_obj: VarianceRatio = VarianceRatio(prices, lags=q_val, robust=robust)  # type: ignore[no-untyped-call]

            vr_value: float = float(vr_obj.vr)
            z_stat: float = float(vr_obj.stat)
            p_val: float = float(vr_obj.pvalue)

            # Clamp p-value to [0, 1]
            p_val = max(0.0, min(1.0, p_val))

            significant: bool = p_val < alpha
            z_abs_values.append(abs(z_stat))

            vr_results.append(
                VarianceRatioResult(
                    calendar_horizon_days=cal_horizon,
                    bar_count_q=q_val,
                    variance_ratio=vr_value,
                    z_statistic=z_stat,
                    p_value=p_val,
                    significant=significant,
                )
            )
        except Exception:
            logger.warning(
                "Variance ratio failed for q={} (horizon={:.1f}d), skipping",
                q_val,
                cal_horizon,
            )

    if not vr_results:
        return (), None, None

    # Chow-Denning joint test: max(|z_i|) with Sidak correction
    cd_stat: float = max(z_abs_values)
    n_tests: int = len(z_abs_values)
    # Sidak: P(reject at least one) = 1 - (1 - p_individual)^n
    # where p_individual = 2 * (1 - Phi(|z_max|)) for two-sided test
    p_individual: float = 2.0 * (1.0 - float(norm.cdf(cd_stat)))
    cd_pvalue: float = 1.0 - (1.0 - p_individual) ** n_tests
    cd_pvalue = max(0.0, min(1.0, cd_pvalue))

    return tuple(vr_results), cd_stat, cd_pvalue


def _compute_granger(  # noqa: PLR0913, PLR0917
    source: pd.Series,  # type: ignore[type-arg]
    target: pd.Series,  # type: ignore[type-arg]
    source_name: str,
    target_name: str,
    lags: tuple[int, ...],
    alpha: float,
) -> tuple[GrangerResult, ...]:
    """Run Granger causality tests for a single (source, target) pair at multiple lags.

    Uses the SSR F-test from ``statsmodels.tsa.stattools.grangercausalitytests``.

    Args:
        source: Source (potential cause) return series.
        target: Target (potential effect) return series.
        source_name: Label for the source series.
        target_name: Label for the target series.
        lags: Lag values to test.
        alpha: Significance level.

    Returns:
        Tuple of ``GrangerResult``, one per requested lag.
    """
    if len(source) != len(target):
        logger.warning(
            "Granger test: source ({}) and target ({}) have different lengths, skipping",
            len(source),
            len(target),
        )
        return ()

    max_lag: int = max(lags)
    # grangercausalitytests needs columns [target, source]
    df: pd.DataFrame = pd.DataFrame(
        {target_name: target.values, source_name: source.values},
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc_result: dict = grangercausalitytests(df, maxlag=max_lag, verbose=False)  # type: ignore[type-arg, no-untyped-call]
    except Exception:
        logger.warning(
            "Granger causality test failed for {} -> {}, skipping",
            source_name,
            target_name,
        )
        return ()

    results: list[GrangerResult] = []
    for lag_val in lags:
        if lag_val not in gc_result:
            continue
        ssr_ftest: tuple[float, float, float, int] = gc_result[lag_val][0]["ssr_ftest"]  # type: ignore[index]
        f_stat: float = float(ssr_ftest[0])
        p_val: float = float(ssr_ftest[1])
        p_val = max(0.0, min(1.0, p_val))
        significant: bool = p_val < alpha

        results.append(
            GrangerResult(
                source_name=source_name,
                target_name=target_name,
                lag=lag_val,
                f_statistic=f_stat,
                p_value=p_val,
                significant=significant,
            )
        )

    return tuple(results)
