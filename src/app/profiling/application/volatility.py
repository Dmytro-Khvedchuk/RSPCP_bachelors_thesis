"""Volatility modeling and regime classification via GARCH, sign bias, ARCH-LM, and BDS tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from arch import arch_model  # type: ignore[import-untyped]
from loguru import logger
from scipy import stats as sp_stats  # type: ignore[import-untyped]
from statsmodels.stats.diagnostic import het_arch  # type: ignore[import-untyped]
from statsmodels.tsa.stattools import bds  # type: ignore[import-untyped]

from src.app.profiling.domain.value_objects import (
    BDSResult,
    GARCHFitResult,
    SampleTier,
    SignBiasResult,
    VolatilityConfig,
    VolatilityProfile,
    VolatilityRegime,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fit_single_garch(  # noqa: PLR0913, PLR0917
    returns: pd.Series,  # type: ignore[type-arg]
    p: int,
    q: int,
    dist: str,
) -> GARCHFitResult | None:
    """Fit a GARCH(p,q) model with the given innovation distribution.

    Returns ``None`` when the optimiser fails to converge or any exception
    occurs during fitting.

    Args:
        returns: Pandas Series of log returns.
        p: GARCH lag order for the conditional variance.
        q: GARCH lag order for the squared innovations.
        dist: Innovation distribution (``"normal"``, ``"t"``, ``"skewt"``).

    Returns:
        Frozen ``GARCHFitResult`` or ``None`` on failure.
    """
    try:
        returns_pct: pd.Series = returns * 100  # type: ignore[type-arg]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(returns_pct, vol="Garch", p=p, q=q, dist=dist, mean="Zero")  # type: ignore[no-untyped-call]  # ty: ignore[invalid-argument-type]
            res = am.fit(disp="off", show_warning=False)  # type: ignore[no-untyped-call]

        omega_raw: float = float(res.params["omega"])
        alpha_val: float = float(res.params[f"alpha[{1}]"])
        beta_val: float = float(res.params[f"beta[{1}]"])
        persistence: float = alpha_val + beta_val

        # Unscale omega: arch fits on pct returns, so variance is in pct^2.
        # Divide by 10000 to get back to decimal return scale.
        omega_unscaled: float = omega_raw / 10000.0

        aic_val: float = float(res.aic)
        bic_val: float = float(res.bic)
        ll_val: float = float(res.loglikelihood)
        converged: bool = res.convergence_flag == 0

        # Extract distribution-specific parameters
        nu: float | None = None
        skew_lambda: float | None = None
        if dist in {"t", "skewt"}:
            nu = float(res.params["nu"])
        if dist == "skewt":
            skew_lambda = float(res.params["lambda"])

        return GARCHFitResult(
            distribution=dist,
            omega=omega_unscaled,
            alpha=alpha_val,
            beta=beta_val,
            persistence=persistence,
            aic=aic_val,
            bic=bic_val,
            log_likelihood=ll_val,
            nu=nu,
            skew_lambda=skew_lambda,
            converged=converged,
        )
    except Exception:
        logger.warning("GARCH({},{}) with dist={} failed to fit", p, q, dist)
        return None


def _compute_sign_bias(  # noqa: PLR0914
    std_resid: np.ndarray,  # type: ignore[type-arg]
    alpha: float,
) -> SignBiasResult:
    """Compute the Engle-Ng sign bias test for asymmetric volatility effects.

    Manual OLS implementation: regresses squared standardised residuals
    on a constant, a sign indicator, and positive/negative size bias
    terms.

    Args:
        std_resid: Standardised residuals from a fitted GARCH model.
        alpha: Significance level for individual bias tests.

    Returns:
        Frozen ``SignBiasResult`` value object.
    """
    n: int = len(std_resid)

    # Dependent variable: squared standardised residuals at time t
    z_sq: np.ndarray = std_resid[1:] ** 2  # type: ignore[type-arg]
    z_lag: np.ndarray = std_resid[:-1]  # type: ignore[type-arg]

    # Indicator: S^- = 1 if z_{t-1} < 0
    s_neg: np.ndarray = (z_lag < 0).astype(np.float64)  # type: ignore[type-arg]

    # Regressors: [const, S^-_{t-1}, S^-_{t-1} * z_{t-1}, (1 - S^-_{t-1}) * z_{t-1}]
    ones: np.ndarray = np.ones(n - 1, dtype=np.float64)  # type: ignore[type-arg]
    x_mat: np.ndarray = np.column_stack([ones, s_neg, s_neg * z_lag, (1 - s_neg) * z_lag])  # type: ignore[type-arg]

    # OLS via least squares
    coeffs: np.ndarray  # type: ignore[type-arg]
    residuals_ols: np.ndarray  # type: ignore[type-arg]
    coeffs, residuals_ols, _, _ = np.linalg.lstsq(x_mat, z_sq, rcond=None)

    # Residuals and standard errors
    fitted: np.ndarray = x_mat @ coeffs  # type: ignore[type-arg]
    resid_vec: np.ndarray = z_sq - fitted  # type: ignore[type-arg]
    n_params: int = 4
    dof: int = n - 1 - n_params
    rss_u: float = float(np.sum(resid_vec**2))
    sigma2: float = rss_u / max(dof, 1)

    # Covariance matrix of coefficients
    xtx_inv: np.ndarray = np.linalg.inv(x_mat.T @ x_mat)  # type: ignore[type-arg]
    se: np.ndarray = np.sqrt(np.diag(sigma2 * xtx_inv))  # type: ignore[type-arg]

    # t-statistics (indices: 0=const, 1=sign_bias, 2=neg_size_bias, 3=pos_size_bias)
    t_stats: np.ndarray = coeffs / np.maximum(se, 1e-15)  # type: ignore[type-arg]

    sign_bias_tstat: float = float(t_stats[1])
    neg_size_bias_tstat: float = float(t_stats[2])
    pos_size_bias_tstat: float = float(t_stats[3])

    # Two-sided p-values from Student-t distribution
    t_dist = sp_stats.t(df=max(dof, 1))
    sign_bias_pvalue: float = float(2.0 * (1.0 - t_dist.cdf(abs(sign_bias_tstat))))
    neg_size_bias_pvalue: float = float(2.0 * (1.0 - t_dist.cdf(abs(neg_size_bias_tstat))))
    pos_size_bias_pvalue: float = float(2.0 * (1.0 - t_dist.cdf(abs(pos_size_bias_tstat))))

    # Joint F-test: restricted model (constant only) vs unrestricted
    y_mean: float = float(np.mean(z_sq))
    rss_r: float = float(np.sum((z_sq - y_mean) ** 2))
    n_restrictions: int = 3
    joint_f_stat: float = ((rss_r - rss_u) / n_restrictions) / (rss_u / max(dof, 1))
    joint_f_pvalue: float = float(1.0 - sp_stats.f.cdf(joint_f_stat, n_restrictions, max(dof, 1)))

    has_leverage: bool = sign_bias_pvalue < alpha or neg_size_bias_pvalue < alpha or pos_size_bias_pvalue < alpha

    return SignBiasResult(
        sign_bias_tstat=sign_bias_tstat,
        sign_bias_pvalue=sign_bias_pvalue,
        neg_size_bias_tstat=neg_size_bias_tstat,
        neg_size_bias_pvalue=neg_size_bias_pvalue,
        pos_size_bias_tstat=pos_size_bias_tstat,
        pos_size_bias_pvalue=pos_size_bias_pvalue,
        joint_f_stat=joint_f_stat,
        joint_f_pvalue=joint_f_pvalue,
        has_leverage_effect=has_leverage,
    )


def _compute_arch_lm(
    residuals: np.ndarray,  # type: ignore[type-arg]
    nlags: int,
) -> tuple[float, float]:
    """Run the ARCH-LM heteroscedasticity test on residuals.

    Args:
        residuals: Residuals (or standardised residuals) from a fitted model.
        nlags: Number of lags for the LM test.

    Returns:
        Tuple of ``(lm_stat, lm_pvalue)``.
    """
    result: tuple[float, ...] = het_arch(residuals, nlags=nlags)  # type: ignore[no-untyped-call]
    lm_stat: float = float(result[0])
    lm_pvalue: float = float(result[1])
    return lm_stat, lm_pvalue


def _compute_bds(
    std_resid: np.ndarray,  # type: ignore[type-arg]
    max_dim: int,
    alpha: float,
) -> tuple[BDSResult, ...]:
    """Run the BDS independence test on standardised residuals.

    Args:
        std_resid: Standardised residuals from a fitted GARCH model.
        max_dim: Maximum embedding dimension (tests dimensions 2..max_dim).
        alpha: Significance level for individual dimension tests.

    Returns:
        Tuple of ``BDSResult``, one per dimension from 2 to max_dim.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bds_stats: np.ndarray  # type: ignore[type-arg]
        bds_pvalues: np.ndarray  # type: ignore[type-arg]
        bds_stats, bds_pvalues = bds(std_resid, max_dim=max_dim)  # type: ignore[no-untyped-call]

    results: list[BDSResult] = []
    for i in range(len(bds_stats)):
        dim: int = i + 2
        stat_val: float = float(bds_stats[i])
        p_val: float = float(bds_pvalues[i])
        # Clamp p-value
        p_val = max(0.0, min(1.0, p_val))
        significant: bool = p_val < alpha
        results.append(
            BDSResult(
                dimension=dim,
                bds_statistic=stat_val,
                p_value=p_val,
                significant=significant,
            )
        )

    return tuple(results)


def _compute_regime_labels(
    realized_vol: np.ndarray,  # type: ignore[type-arg]
    low_q: float,
    high_q: float,
) -> tuple[np.ndarray, tuple[float, float]]:  # type: ignore[type-arg]
    """Classify realized volatility into LOW / NORMAL / HIGH regimes.

    Args:
        realized_vol: Array of realized volatility estimates (may contain NaN).
        low_q: Lower quantile threshold (e.g. 0.25).
        high_q: Upper quantile threshold (e.g. 0.75).

    Returns:
        Tuple of ``(labels_array, (low_threshold, high_threshold))`` where
        labels_array is an object-dtype array of ``VolatilityRegime`` enum values.
    """
    low_thresh: float = float(np.nanquantile(realized_vol, low_q))
    high_thresh: float = float(np.nanquantile(realized_vol, high_q))

    n: int = len(realized_vol)
    labels: np.ndarray = np.empty(n, dtype=object)  # type: ignore[type-arg]

    for i in range(n):
        val: float = realized_vol[i]
        if np.isnan(val):
            labels[i] = VolatilityRegime.NORMAL
        elif val < low_thresh:
            labels[i] = VolatilityRegime.LOW
        elif val > high_thresh:
            labels[i] = VolatilityRegime.HIGH
        else:
            labels[i] = VolatilityRegime.NORMAL

    return labels, (low_thresh, high_thresh)


# ---------------------------------------------------------------------------
# Public analyzer
# ---------------------------------------------------------------------------


class VolatilityAnalyzer:
    """Stateless service for GARCH volatility modeling and regime classification.

    Tier-gated:
        - **Tier A/B + time bars:** GARCH(1,1) multi-distribution, sign bias, ARCH-LM.
        - **Tier A + time bars + leverage detected:** GJR-GARCH gamma.
        - **Tier A + time bars:** BDS nonlinearity test.
        - **All tiers and bar types:** Quantile-based regime labeling.
    """

    def analyze(  # noqa: PLR6301, PLR0913, PLR0917, PLR0914
        self,
        returns: pd.Series,  # type: ignore[type-arg]
        asset: str,
        bar_type: str,
        tier: SampleTier,
        config: VolatilityConfig | None = None,
        realized_vol: np.ndarray | None = None,  # type: ignore[type-arg]
    ) -> VolatilityProfile:
        """Compute a full volatility modeling profile for a return series.

        Args:
            returns: Pandas Series of log returns (NaN-free).
            asset: Trading pair symbol (e.g. ``"BTCUSDT"``).
            bar_type: Bar aggregation type (e.g. ``"time_1h"``).
            tier: Sample-size tier controlling analysis depth.
            config: Volatility analysis configuration.  Uses defaults
                when ``None``.
            realized_vol: Pre-computed realized volatility array.  If ``None``,
                computed as rolling 20-period standard deviation.

        Returns:
            Frozen ``VolatilityProfile`` value object.
        """
        if config is None:
            config = VolatilityConfig()

        n_obs: int = len(returns)
        is_time_bar: bool = bar_type.startswith("time_")

        logger.debug(
            "Analysing volatility: asset={}, bar_type={}, tier={}, n={}, is_time_bar={}",
            asset,
            bar_type,
            tier.value,
            n_obs,
            is_time_bar,
        )

        # Compute realized volatility if not provided
        rvol: np.ndarray = realized_vol if realized_vol is not None else returns.rolling(20).std().to_numpy()  # type: ignore[type-arg]

        # Regime labeling (all tiers and bar types)
        regime_labels: np.ndarray  # type: ignore[type-arg]
        thresholds: tuple[float, float]
        regime_labels, thresholds = _compute_regime_labels(
            rvol, config.regime_low_quantile, config.regime_high_quantile
        )

        # Initialize optional fields
        garch_fits: tuple[GARCHFitResult, ...] | None = None
        best_distribution: str | None = None
        persistence: float | None = None
        is_igarch: bool | None = None
        sign_bias: SignBiasResult | None = None
        gjr_gamma: float | None = None
        arch_lm_stat: float | None = None
        arch_lm_pvalue: float | None = None
        bds_results: tuple[BDSResult, ...] | None = None
        nonlinear_structure_detected: bool | None = None

        # GARCH modeling: time bars + Tier A/B + sufficient samples
        can_garch: bool = is_time_bar and tier in {SampleTier.A, SampleTier.B} and n_obs >= config.min_samples_garch

        if can_garch:
            # Fit GARCH(1,1) with each configured distribution
            all_fits: list[GARCHFitResult] = []
            for dist in config.innovation_distributions:
                fit: GARCHFitResult | None = _fit_single_garch(returns, config.garch_p, config.garch_q, dist)
                if fit is not None:
                    all_fits.append(fit)

            # Filter to converged fits
            converged_fits: list[GARCHFitResult] = [f for f in all_fits if f.converged]

            if converged_fits:
                garch_fits = tuple(converged_fits)

                # Select best by AIC
                best_fit: GARCHFitResult = min(converged_fits, key=lambda f: f.aic)
                best_distribution = best_fit.distribution
                persistence = best_fit.persistence
                is_igarch = persistence >= config.persistence_threshold

                logger.debug(
                    "Best GARCH dist={}, persistence={:.4f}, IGARCH={}",
                    best_distribution,
                    persistence,
                    is_igarch,
                )

                # Get standardized residuals from the best model (refit)
                std_resid: np.ndarray | None = _get_std_resid(  # type: ignore[type-arg]
                    returns, config.garch_p, config.garch_q, best_distribution
                )

                if std_resid is not None:
                    # Sign bias test (Tier A/B)
                    sign_bias = _compute_sign_bias(std_resid, config.sign_bias_alpha)

                    # GJR-GARCH: Tier A only + leverage detected
                    if tier == SampleTier.A and sign_bias.has_leverage_effect:
                        gjr_gamma = _fit_gjr_gamma(returns, best_distribution)

                    # ARCH-LM on standardized residuals
                    arch_lm_stat, arch_lm_pvalue = _compute_arch_lm(std_resid, config.arch_lm_nlags)

                    # BDS test: Tier A only
                    if tier == SampleTier.A:
                        bds_results = _compute_bds(std_resid, config.bds_max_dim, config.sign_bias_alpha)
                        n_significant_dims: int = sum(1 for r in bds_results if r.significant)
                        _min_nonlinear_dims: int = 2
                        nonlinear_structure_detected = n_significant_dims >= _min_nonlinear_dims

        return VolatilityProfile(
            asset=asset,
            bar_type=bar_type,
            tier=tier,
            n_observations=n_obs,
            is_time_bar=is_time_bar,
            garch_fits=garch_fits,
            best_distribution=best_distribution,
            persistence=persistence,
            is_igarch=is_igarch,
            sign_bias=sign_bias,
            gjr_gamma=gjr_gamma,
            arch_lm_stat=arch_lm_stat,
            arch_lm_pvalue=arch_lm_pvalue,
            bds_results=bds_results,
            nonlinear_structure_detected=nonlinear_structure_detected,
            regime_labels=regime_labels,
            regime_low_threshold=thresholds[0],
            regime_high_threshold=thresholds[1],
        )


# ---------------------------------------------------------------------------
# Additional internal helpers
# ---------------------------------------------------------------------------


def _get_std_resid(
    returns: pd.Series,  # type: ignore[type-arg]
    p: int,
    q: int,
    dist: str,
) -> np.ndarray | None:  # type: ignore[type-arg]
    """Refit the best GARCH model and extract standardised residuals.

    Args:
        returns: Pandas Series of log returns.
        p: GARCH lag order for the conditional variance.
        q: GARCH lag order for the squared innovations.
        dist: Best innovation distribution.

    Returns:
        1-D NumPy array of standardised residuals, or ``None`` on failure.
    """
    try:
        returns_pct: pd.Series = returns * 100  # type: ignore[type-arg]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(returns_pct, vol="Garch", p=p, q=q, dist=dist, mean="Zero")  # type: ignore[no-untyped-call]  # ty: ignore[invalid-argument-type]
            res = am.fit(disp="off", show_warning=False)  # type: ignore[no-untyped-call]
        return np.asarray(res.std_resid, dtype=np.float64)
    except Exception:
        logger.warning("Failed to extract standardised residuals for GARCH({},{}) dist={}", p, q, dist)
        return None


def _fit_gjr_gamma(
    returns: pd.Series,  # type: ignore[type-arg]
    dist: str,
) -> float | None:
    """Fit a GJR-GARCH(1,1,1) model and return the asymmetric leverage coefficient.

    Args:
        returns: Pandas Series of log returns.
        dist: Innovation distribution for the GJR-GARCH model.

    Returns:
        Gamma coefficient (asymmetric term), or ``None`` on failure.
    """
    try:
        returns_pct: pd.Series = returns * 100  # type: ignore[type-arg]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(returns_pct, vol="GARCH", p=1, o=1, q=1, dist=dist, mean="Zero")  # type: ignore[no-untyped-call]  # ty: ignore[invalid-argument-type]
            res = am.fit(disp="off", show_warning=False)  # type: ignore[no-untyped-call]
    except Exception:
        logger.warning("GJR-GARCH fitting failed with dist={}", dist)
        return None
    else:
        gamma: float = float(res.params["gamma[1]"])
        return gamma
