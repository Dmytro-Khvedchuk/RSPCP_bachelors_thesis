"""Return distribution profiling via Jarque-Bera, Student-t MLE fitting, and AIC/BIC comparison.

Uses the ML-research path (Pandas / NumPy / SciPy) per CLAUDE.md.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from loguru import logger
from scipy import stats  # type: ignore[import-untyped]

from src.app.profiling.domain.value_objects import (
    DistributionConfig,
    DistributionProfile,
    SampleTier,
)


# ---------------------------------------------------------------------------
# Number of free parameters for AIC / BIC computation
# ---------------------------------------------------------------------------

_N_PARAMS_NORMAL: int = 2  # mu, sigma
_N_PARAMS_STUDENT_T: int = 3  # nu, loc, scale


class _DescriptiveStats(NamedTuple):
    """Container for descriptive statistics computed from a return series."""

    mean_return: float
    std_return: float
    skewness: float
    excess_kurtosis: float


class _JBResult(NamedTuple):
    """Container for Jarque-Bera test results."""

    jb_stat: float
    jb_pvalue: float
    is_normal: bool


class _FitResult(NamedTuple):
    """Container for Student-t MLE fit and model comparison results."""

    student_t_nu: float
    student_t_loc: float
    student_t_scale: float
    aic_normal: float
    aic_student_t: float
    bic_normal: float
    bic_student_t: float
    best_fit: str
    ks_statistic: float


def _safe_float(val: float, default: float = 0.0) -> float:
    """Replace NaN / Inf with *default*.

    Args:
        val: Value to sanitise.
        default: Fallback when ``val`` is NaN or Inf.

    Returns:
        Sanitised float.
    """
    if math.isnan(val) or math.isinf(val):
        return default
    return val


def _compute_aic(k: int, log_likelihood: float) -> float:
    """Compute Akaike Information Criterion.

    ``AIC = 2k - 2 * log_likelihood``

    Args:
        k: Number of free parameters.
        log_likelihood: Maximised log-likelihood.

    Returns:
        AIC value (lower is better).
    """
    return 2.0 * k - 2.0 * log_likelihood


def _compute_bic(k: int, n: int, log_likelihood: float) -> float:
    """Compute Bayesian Information Criterion.

    ``BIC = k * ln(n) - 2 * log_likelihood``

    Args:
        k: Number of free parameters.
        n: Number of observations.
        log_likelihood: Maximised log-likelihood.

    Returns:
        BIC value (lower is better).
    """
    return k * math.log(n) - 2.0 * log_likelihood


def _compute_descriptive_stats(returns: pd.Series, n: int) -> _DescriptiveStats:  # type: ignore[type-arg]
    """Compute mean, std, skewness, and excess kurtosis from a return series.

    Args:
        returns: Pandas Series of log returns.
        n: Number of observations (avoids recomputation).

    Returns:
        Container with descriptive statistics.
    """
    mean_return: float = _safe_float(float(returns.mean())) if n > 0 else 0.0
    std_return: float = _safe_float(float(returns.std())) if n > 0 else 0.0
    skewness: float = _safe_float(float(stats.skew(returns))) if n > 0 else 0.0
    excess_kurtosis: float = _safe_float(float(stats.kurtosis(returns, fisher=True))) if n > 0 else 0.0
    return _DescriptiveStats(mean_return, std_return, skewness, excess_kurtosis)


def _compute_jb_test(
    returns: pd.Series,  # type: ignore[type-arg]
    n: int,
    std_return: float,
    config: DistributionConfig,
) -> _JBResult:
    """Run the Jarque-Bera normality test.

    Args:
        returns: Pandas Series of log returns.
        n: Number of observations.
        std_return: Standard deviation (skip test if zero).
        config: Configuration with alpha and min-sample thresholds.

    Returns:
        Container with JB stat, p-value, and normality decision.
    """
    if n >= config.min_samples_jb and std_return > 0:
        jb_raw: object = stats.jarque_bera(returns)
        jb_stat_raw: float = float(jb_raw[0])  # type: ignore[index]
        jb_pvalue_raw: float = float(jb_raw[1])  # type: ignore[index]
        jb_stat: float = _safe_float(jb_stat_raw)
        jb_pvalue: float = _safe_float(jb_pvalue_raw, default=1.0)
    else:
        jb_stat = 0.0
        jb_pvalue = 1.0

    is_normal: bool = jb_pvalue >= config.jb_alpha
    return _JBResult(jb_stat, jb_pvalue, is_normal)


def _fit_distributions(
    data: np.ndarray[tuple[int], np.dtype[np.float64]],
    n: int,
) -> _FitResult:
    """Fit Normal and Student-t distributions, compute AIC/BIC and KS statistic.

    Args:
        data: 1-D NumPy array of return observations.
        n: Number of observations.

    Returns:
        Container with all fitting and model-comparison results.
    """
    # Student-t MLE fit
    nu_fit: float
    loc_fit: float
    scale_fit: float
    nu_fit, loc_fit, scale_fit = stats.t.fit(data)  # type: ignore[no-untyped-call]

    # Normal MLE fit (analytical)
    mu_fit: float = float(np.mean(data))
    sigma_fit: float = float(np.std(data, ddof=0))

    # Log-likelihoods
    ll_normal: float = float(np.sum(stats.norm.logpdf(data, loc=mu_fit, scale=sigma_fit)))
    ll_student_t: float = float(np.sum(stats.t.logpdf(data, df=nu_fit, loc=loc_fit, scale=scale_fit)))

    # AIC / BIC
    aic_normal: float = _compute_aic(_N_PARAMS_NORMAL, ll_normal)
    aic_student_t: float = _compute_aic(_N_PARAMS_STUDENT_T, ll_student_t)
    bic_normal: float = _compute_bic(_N_PARAMS_NORMAL, n, ll_normal)
    bic_student_t: float = _compute_bic(_N_PARAMS_STUDENT_T, n, ll_student_t)
    best_fit: str = "student_t" if aic_student_t < aic_normal else "normal"

    # KS statistic against fitted Normal
    ks_result: tuple[float, float] = stats.kstest(data, "norm", args=(mu_fit, sigma_fit))  # type: ignore[no-untyped-call]
    ks_statistic: float = float(ks_result[0])

    return _FitResult(
        student_t_nu=float(nu_fit),
        student_t_loc=float(loc_fit),
        student_t_scale=float(scale_fit),
        aic_normal=aic_normal,
        aic_student_t=aic_student_t,
        bic_normal=bic_normal,
        bic_student_t=bic_student_t,
        best_fit=best_fit,
        ks_statistic=ks_statistic,
    )


class DistributionAnalyzer:
    """Stateless service for return distribution profiling.

    Performs tier-gated analysis:

    - **All tiers:** log return computation, descriptive stats, Jarque-Bera test.
    - **Tier A / B:** Student-t MLE fitting, AIC/BIC comparison, KS distance.
    - **Tier C:** descriptive stats only (Student-t fields set to ``None``).
    """

    def analyze(  # noqa: PLR6301
        self,
        returns: pd.Series,  # type: ignore[type-arg]
        asset: str,
        bar_type: str,
        tier: SampleTier,
        config: DistributionConfig | None = None,
    ) -> DistributionProfile:
        """Compute a full distribution profile for a return series.

        Args:
            returns: Pandas Series of log returns (NaN-free).
            asset: Trading pair symbol (e.g. ``"BTCUSDT"``).
            bar_type: Bar aggregation type (e.g. ``"dollar"``).
            tier: Sample-size tier controlling analysis depth.
            config: Distribution analysis configuration. Uses defaults
                when ``None``.

        Returns:
            Frozen ``DistributionProfile`` value object.
        """
        if config is None:
            config = DistributionConfig()

        n: int = len(returns)
        logger.debug(
            "Analysing return distribution: asset={}, bar_type={}, tier={}, n={}",
            asset,
            bar_type,
            tier.value,
            n,
        )

        desc: _DescriptiveStats = _compute_descriptive_stats(returns, n)
        jb: _JBResult = _compute_jb_test(returns, n, desc.std_return, config)

        # Tier A / B: Student-t MLE, AIC/BIC, KS
        fit: _FitResult | None = None
        can_fit: bool = tier in {SampleTier.A, SampleTier.B} and n >= config.min_samples_fit and desc.std_return > 0
        if can_fit:
            data: np.ndarray[tuple[int], np.dtype[np.float64]] = returns.to_numpy(dtype=np.float64)
            fit = _fit_distributions(data, n)
            logger.debug(
                "Fit results: nu={:.2f}, AIC(N)={:.1f}, AIC(t)={:.1f}, best={}, KS_Dn={:.4f}",
                fit.student_t_nu,
                fit.aic_normal,
                fit.aic_student_t,
                fit.best_fit,
                fit.ks_statistic,
            )

        return DistributionProfile(
            asset=asset,
            bar_type=bar_type,
            tier=tier,
            n_observations=n,
            mean_return=desc.mean_return,
            std_return=desc.std_return,
            skewness=desc.skewness,
            excess_kurtosis=desc.excess_kurtosis,
            jb_stat=jb.jb_stat,
            jb_pvalue=jb.jb_pvalue,
            is_normal=jb.is_normal,
            student_t_nu=fit.student_t_nu if fit else None,
            student_t_loc=fit.student_t_loc if fit else None,
            student_t_scale=fit.student_t_scale if fit else None,
            aic_normal=fit.aic_normal if fit else None,
            aic_student_t=fit.aic_student_t if fit else None,
            bic_normal=fit.bic_normal if fit else None,
            bic_student_t=fit.bic_student_t if fit else None,
            best_fit=fit.best_fit if fit else None,
            ks_statistic=fit.ks_statistic if fit else None,
        )

    def compute_qq_data_student_t(  # noqa: PLR6301
        self,
        returns: pd.Series,  # type: ignore[type-arg]
        nu: float,
        loc: float,
        scale: float,
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]]:
        """Extract Q-Q plot data against a fitted Student-t distribution.

        Computes theoretical quantiles from the fitted Student-t
        distribution and pairs them with the ordered sample values.

        Args:
            returns: Series of return observations.
            nu: Fitted degrees of freedom.
            loc: Fitted location parameter.
            scale: Fitted scale parameter.

        Returns:
            Tuple of ``(theoretical_quantiles, ordered_values)`` where both
            arrays have the same length as ``returns``.
        """
        _min_qq_samples: int = 3
        clean: pd.Series[float] = returns.dropna()  # type: ignore[type-arg]

        if len(clean) < _min_qq_samples:
            empty: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([], dtype=np.float64)
            return empty, empty

        n: int = len(clean)
        ordered: np.ndarray[tuple[int], np.dtype[np.float64]] = np.sort(clean.to_numpy(dtype=np.float64))

        # Compute theoretical quantiles using plotting positions
        # Blom's approximation: (i - 3/8) / (n + 1/4)
        positions: np.ndarray[tuple[int], np.dtype[np.float64]] = (np.arange(1, n + 1, dtype=np.float64) - 0.375) / (
            n + 0.25
        )
        theoretical: np.ndarray[tuple[int], np.dtype[np.float64]] = np.asarray(
            stats.t.ppf(positions, df=nu, loc=loc, scale=scale),  # type: ignore[no-untyped-call]
            dtype=np.float64,
        )

        return theoretical, ordered
