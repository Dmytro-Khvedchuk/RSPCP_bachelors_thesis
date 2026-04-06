"""Calibration and conformal prediction for regression forecasters.

Implements Adaptive Conformal Inference (Gibbs & Candes 2021), reliability
diagrams, residual diagnostics, and per-regime coverage analysis.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from scipy import stats as sp_stats
from statsmodels.stats.diagnostic import het_breuschpagan  # type: ignore[import-untyped]

from src.app.forecasting.domain.value_objects import (
    ACIConfig,
    ConformalInterval,
    QuantilePrediction,
    RegimeCoverage,
    ReliabilityDiagramResult,
    ResidualDiagnostics,
)


# ---------------------------------------------------------------------------
# Adaptive Conformal Inference (ACI) — Gibbs & Candes 2021
# ---------------------------------------------------------------------------


class AdaptiveConformalPredictor:
    """Online-adaptive conformal prediction intervals for regression.

    Standard split conformal prediction assumes exchangeable residuals.
    Crypto residuals violate this due to regime-dependent volatility and
    autocorrelated squared returns.  ACI adapts the miscoverage rate
    ``alpha_t`` online so that intervals track the target coverage even
    under distribution shift.

    **Update rule** (Gibbs & Candes 2021)::

        alpha_{t+1} = clip(alpha_t + gamma * (err_t - alpha_target),
                           min_alpha, max_alpha)

    where ``err_t = 1`` if the actual falls outside the interval at time *t*.

    Attributes:
        config: ACI hyperparameters.
    """

    def __init__(self, config: ACIConfig) -> None:
        """Initialise the adaptive conformal predictor.

        Args:
            config: ACI configuration (target coverage, gamma, alpha bounds).
        """
        self.config: ACIConfig = config
        self._scores: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None
        effective_alpha: float = (
            config.initial_alpha if config.initial_alpha is not None else (1.0 - config.target_coverage)
        )
        self._alpha_t: float = effective_alpha
        self._alpha_history: list[float] = [effective_alpha]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha_t(self) -> float:
        """Current adaptive miscoverage rate.

        Returns:
            The current alpha_t value.
        """
        return self._alpha_t

    @property
    def alpha_history(self) -> list[float]:
        """Full history of alpha_t values (including the initial value).

        Returns:
            List of alpha_t values over time.
        """
        return list(self._alpha_history)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        residuals: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Compute nonconformity scores from a held-out calibration set.

        Nonconformity scores are defined as ``|residual|`` (absolute residuals),
        which yields symmetric prediction intervals centred on the point
        prediction.

        Args:
            residuals: Calibration residuals of shape ``(n_cal,)``,
                computed as ``actuals - predictions`` on held-out data.

        Raises:
            ValueError: If residuals array is empty.
        """
        n_cal: int = residuals.shape[0]
        if n_cal == 0:
            msg: str = "residuals must contain at least one sample"
            raise ValueError(msg)

        self._scores = np.sort(np.abs(residuals)).astype(np.float64)
        logger.info(
            "ACI calibrated on {} residuals | score range [{:.6f}, {:.6f}]", n_cal, self._scores[0], self._scores[-1]
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_interval(
        self,
        predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
        actuals: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
    ) -> ConformalInterval:
        """Produce conformal prediction intervals, optionally adapting online.

        When ``actuals`` are provided the predictor updates ``alpha_t``
        sequentially for each sample (online mode).  Without actuals the
        current ``alpha_t`` is held fixed (batch mode).

        Args:
            predictions: Point predictions of shape ``(n_samples,)``.
            actuals: Observed values of shape ``(n_samples,)``.
                If provided, ACI will adapt ``alpha_t`` per time step.

        Returns:
            Conformal prediction interval with lower/upper bounds and
            optional empirical coverage.

        Raises:
            RuntimeError: If ``calibrate()`` has not been called.
            ValueError: If predictions is empty or actuals shape mismatches.
        """
        if self._scores is None:
            msg: str = "calibrate() must be called before predict_interval()"
            raise RuntimeError(msg)

        n_pred: int = predictions.shape[0]
        if n_pred == 0:
            msg = "predictions must contain at least one sample"
            raise ValueError(msg)

        if actuals is not None and actuals.shape[0] != n_pred:
            msg = f"actuals length {actuals.shape[0]} != predictions length {n_pred}"
            raise ValueError(msg)

        lower: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(n_pred, dtype=np.float64)
        upper: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(n_pred, dtype=np.float64)

        alpha_target: float = 1.0 - self.config.target_coverage

        for t in range(n_pred):
            # Compute quantile of nonconformity scores at current alpha_t
            q_level: float = 1.0 - self._alpha_t
            q_value: float = float(self._compute_quantile(q_level))

            lower[t] = predictions[t] - q_value
            upper[t] = predictions[t] + q_value

            # Adapt alpha_t if actuals are available
            if actuals is not None:
                err_t: float = 1.0 if (actuals[t] < lower[t] or actuals[t] > upper[t]) else 0.0
                new_alpha: float = self._alpha_t + self.config.gamma * (err_t - alpha_target)
                self._alpha_t = float(np.clip(new_alpha, self.config.min_alpha, self.config.max_alpha))
                self._alpha_history.append(self._alpha_t)

        # Compute empirical coverage if actuals were provided
        coverage: float | None = None
        if actuals is not None:
            inside: np.ndarray[tuple[int], np.dtype[np.bool_]] = (actuals >= lower) & (actuals <= upper)
            coverage = float(np.mean(inside))

        return ConformalInterval(lower=lower, upper=upper, coverage=coverage)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_quantile(self, level: float) -> float:
        """Compute the quantile of stored nonconformity scores.

        Uses linear interpolation between scores consistent with the
        standard conformal prediction recipe (ceil-based finite-sample
        correction is implicitly handled by numpy's interpolation).

        Args:
            level: Quantile level in ``[0, 1]``.

        Returns:
            Quantile value of the nonconformity score distribution.
        """
        # _scores is guaranteed non-None when this is called
        scores: np.ndarray[tuple[int], np.dtype[np.float64]] = self._scores  # type: ignore[assignment] # ty: ignore[invalid-assignment]
        clamped_level: float = float(np.clip(level, 0.0, 1.0))
        return float(np.quantile(scores, clamped_level))


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------


def compute_reliability_diagram(
    quantile_predictions: QuantilePrediction,
    actuals: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> ReliabilityDiagramResult:
    """Compute a reliability (calibration) diagram for quantile predictions.

    For each nominal quantile level *q*, computes the fraction of actual
    observations that fall below the predicted *q*-quantile.  Perfect
    calibration gives observed_coverage[i] == expected_coverage[i] for all *i*.

    Args:
        quantile_predictions: Quantile predictions with shape
            ``(n_samples, n_quantiles)`` and associated quantile levels.
        actuals: Observed values of shape ``(n_samples,)``.

    Returns:
        Reliability diagram result with expected and observed coverage arrays.

    Raises:
        ValueError: If actuals length does not match prediction rows.
    """
    n_samples: int = quantile_predictions.values.shape[0]
    n_quantiles: int = len(quantile_predictions.quantiles)

    if actuals.shape[0] != n_samples:
        msg: str = f"actuals length {actuals.shape[0]} != predictions rows {n_samples}"
        raise ValueError(msg)

    expected: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(quantile_predictions.quantiles, dtype=np.float64)
    observed: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(n_quantiles, dtype=np.float64)

    for i in range(n_quantiles):
        q_values: np.ndarray[tuple[int], np.dtype[np.float64]] = quantile_predictions.values[:, i]
        below: np.ndarray[tuple[int], np.dtype[np.bool_]] = actuals <= q_values
        observed[i] = float(np.mean(below))

    logger.debug(
        "Reliability diagram: {} quantiles, {} samples | max deviation {:.4f}",
        n_quantiles,
        n_samples,
        float(np.max(np.abs(expected - observed))),
    )

    return ReliabilityDiagramResult(
        expected_coverage=expected,
        observed_coverage=observed,
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# Residual diagnostics — private helpers
# ---------------------------------------------------------------------------

_SIGNIFICANCE_LEVEL: float = 0.05
"""Significance level for normality and homoscedasticity tests."""

_MAX_SHAPIRO_N: int = 5000
"""Maximum sample size for Shapiro-Wilk (scipy hard limit)."""

_MIN_SAMPLES_STAT_TEST: int = 3
"""Minimum samples required for Shapiro-Wilk and Breusch-Pagan tests."""


def _shapiro_wilk_test(
    residuals: np.ndarray[tuple[int], np.dtype[np.float64]],
    n: int,
) -> tuple[float, float, bool]:
    """Run Shapiro-Wilk normality test on residuals.

    Args:
        residuals: Residual array of shape ``(n,)``.
        n: Length of the residual array.

    Returns:
        Tuple of ``(statistic, p_value, is_normal)``.
    """
    if n < _MIN_SAMPLES_STAT_TEST:
        return float("nan"), float("nan"), False

    if n > _MAX_SHAPIRO_N:
        rng: np.random.Generator = np.random.default_rng(42)
        idx: np.ndarray[tuple[int], np.dtype[np.intp]] = rng.choice(n, size=_MAX_SHAPIRO_N, replace=False)
        shapiro_input: np.ndarray[tuple[int], np.dtype[np.float64]] = residuals[idx]
    else:
        shapiro_input = residuals

    shapiro_out = sp_stats.shapiro(shapiro_input)
    stat: float = float(shapiro_out.statistic)
    pvalue: float = float(shapiro_out.pvalue)
    return stat, pvalue, pvalue >= _SIGNIFICANCE_LEVEL


def _breusch_pagan_test(
    residuals: np.ndarray[tuple[int], np.dtype[np.float64]],
    predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
    n: int,
) -> tuple[float, float, bool]:
    """Run Breusch-Pagan homoscedasticity test.

    Args:
        residuals: Residual array of shape ``(n,)``.
        predictions: Predictions array of shape ``(n,)``.
        n: Length of the arrays.

    Returns:
        Tuple of ``(statistic, p_value, is_homoscedastic)``.
    """
    if n < _MIN_SAMPLES_STAT_TEST:
        return float("nan"), float("nan"), False

    exog: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.column_stack(
        [np.ones(n, dtype=np.float64), predictions]
    )
    bp_result: tuple[float, float, float, float] = het_breuschpagan(residuals, exog)
    stat: float = float(bp_result[0])
    pvalue: float = float(bp_result[1])
    return stat, pvalue, pvalue >= _SIGNIFICANCE_LEVEL


# ---------------------------------------------------------------------------
# Residual diagnostics — public API
# ---------------------------------------------------------------------------


def compute_residual_diagnostics(
    residuals: np.ndarray[tuple[int], np.dtype[np.float64]],
    predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> ResidualDiagnostics:
    """Diagnose residual properties relevant to conformal prediction validity.

    Tests:
    - **Normality** via Shapiro-Wilk (``scipy.stats.shapiro``).
      Shapiro-Wilk is limited to 5000 samples; if ``n > 5000`` we
      subsample to avoid the SciPy hard limit.
    - **Homoscedasticity** via Breusch-Pagan (``statsmodels``), regressing
      squared residuals on the predictions.

    Args:
        residuals: Model residuals ``actuals - predictions`` of shape ``(n,)``.
        predictions: Model point predictions of shape ``(n,)``.

    Returns:
        Residual diagnostics with normality and homoscedasticity test results.

    Raises:
        ValueError: If residuals or predictions are empty or length-mismatched.
    """
    n: int = residuals.shape[0]
    if n == 0:
        msg: str = "residuals must contain at least one sample"
        raise ValueError(msg)
    if predictions.shape[0] != n:
        msg = f"predictions length {predictions.shape[0]} != residuals length {n}"
        raise ValueError(msg)

    mean_resid: float = float(np.mean(residuals))
    std_resid: float = float(np.std(residuals, ddof=1)) if n > 1 else 0.0

    shapiro_w, shapiro_p, is_normal = _shapiro_wilk_test(residuals, n)
    bp_stat, bp_p, is_homoscedastic = _breusch_pagan_test(residuals, predictions, n)

    logger.debug(
        "Residual diagnostics: n={} | Shapiro W={:.4f} p={:.4f} | BP stat={:.4f} p={:.4f}",
        n,
        shapiro_w,
        shapiro_p,
        bp_stat,
        bp_p,
    )

    return ResidualDiagnostics(
        shapiro_stat=shapiro_w,
        shapiro_pvalue=shapiro_p,
        breusch_pagan_stat=bp_stat,
        breusch_pagan_pvalue=bp_p,
        mean_residual=mean_resid,
        std_residual=std_resid,
        is_normal=is_normal,
        is_homoscedastic=is_homoscedastic,
    )


# ---------------------------------------------------------------------------
# Per-regime coverage
# ---------------------------------------------------------------------------


def compute_regime_coverage(
    lower: np.ndarray[tuple[int], np.dtype[np.float64]],
    upper: np.ndarray[tuple[int], np.dtype[np.float64]],
    actuals: np.ndarray[tuple[int], np.dtype[np.float64]],
    volatility: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> RegimeCoverage:
    """Compute conformal interval coverage split by volatility regime.

    Splits samples at the **median** volatility into high-vol and low-vol
    groups and reports coverage for each.  This exposes the coverage
    degradation that standard (exchangeability-assuming) conformal methods
    suffer during volatile regimes.

    Args:
        lower: Lower interval bounds of shape ``(n,)``.
        upper: Upper interval bounds of shape ``(n,)``.
        actuals: Observed values of shape ``(n,)``.
        volatility: Volatility proxy of shape ``(n,)`` (e.g. realised vol,
            rolling std, or GARCH conditional vol).

    Returns:
        Per-regime coverage result.

    Raises:
        ValueError: If array lengths do not match or are empty.
    """
    n: int = lower.shape[0]
    if n == 0:
        msg: str = "arrays must contain at least one sample"
        raise ValueError(msg)
    if not (upper.shape[0] == n and actuals.shape[0] == n and volatility.shape[0] == n):
        msg = (
            f"All arrays must have the same length, got "
            f"lower={lower.shape[0]}, upper={upper.shape[0]}, "
            f"actuals={actuals.shape[0]}, volatility={volatility.shape[0]}"
        )
        raise ValueError(msg)

    inside: np.ndarray[tuple[int], np.dtype[np.bool_]] = (actuals >= lower) & (actuals <= upper)
    overall: float = float(np.mean(inside))

    vol_threshold: float = float(np.median(volatility))
    high_mask: np.ndarray[tuple[int], np.dtype[np.bool_]] = volatility > vol_threshold
    low_mask: np.ndarray[tuple[int], np.dtype[np.bool_]] = ~high_mask

    high_count: int = int(np.sum(high_mask))
    low_count: int = int(np.sum(low_mask))

    high_cov: float = float(np.mean(inside[high_mask])) if high_count > 0 else 0.0
    low_cov: float = float(np.mean(inside[low_mask])) if low_count > 0 else 0.0

    logger.info(
        "Regime coverage: overall={:.4f} | high_vol={:.4f} (n={}) | low_vol={:.4f} (n={}) | threshold={:.6f}",
        overall,
        high_cov,
        high_count,
        low_cov,
        low_count,
        vol_threshold,
    )

    return RegimeCoverage(
        overall_coverage=overall,
        high_vol_coverage=high_cov,
        low_vol_coverage=low_cov,
        high_vol_count=high_count,
        low_vol_count=low_count,
        vol_threshold=vol_threshold,
    )
