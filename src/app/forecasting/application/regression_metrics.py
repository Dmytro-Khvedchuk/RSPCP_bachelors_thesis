"""Regression metrics for return and volatility forecasters.

Provides standalone metrics (MAE, RMSE, R-squared, implicit DA, CRPS) and
volatility-specific metrics (QLIKE, Mincer-Zarnowitz, log-vol MAE).  Pipeline
metrics that require the Phase 11 classifier are stubbed with
``NotImplementedError``.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class WinsorizeConfig(BaseModel, frozen=True):
    """Configuration for prediction winsorization (clipping at percentile bounds).

    Attributes:
        lower_percentile: Lower percentile bound for clipping (e.g. 1.0 = 1st percentile).
        upper_percentile: Upper percentile bound for clipping (e.g. 99.0 = 99th percentile).
    """

    lower_percentile: Annotated[
        float,
        PydanticField(default=1.0, ge=0.0, le=100.0, description="Lower percentile bound for clipping"),
    ]

    upper_percentile: Annotated[
        float,
        PydanticField(default=99.0, ge=0.0, le=100.0, description="Upper percentile bound for clipping"),
    ]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


class RegressionMetrics(BaseModel, frozen=True):
    """Core regression metrics on all samples (not direction-conditional).

    Attributes:
        mae: Mean Absolute Error.
        rmse: Root Mean Squared Error.
        r_squared: Coefficient of determination (R-squared).
        implicit_da: Directional accuracy — fraction of samples where
            ``sign(predicted) == sign(actual)``.
        n_samples: Number of samples used for computation.
    """

    mae: float
    """Mean Absolute Error."""

    rmse: float
    """Root Mean Squared Error."""

    r_squared: float
    """Coefficient of determination (R-squared)."""

    implicit_da: float
    """Directional accuracy of sign(predicted) vs sign(actual)."""

    n_samples: int
    """Number of samples used."""


class CRPSResult(BaseModel, frozen=True):
    """Continuous Ranked Probability Score for Gaussian predictions.

    The CRPS measures the quality of probabilistic forecasts.  For Gaussian
    predictions, the closed-form is used (Gneiting & Raftery 2007).

    Attributes:
        mean_crps: Average CRPS across all samples (lower is better).
        per_sample_crps: Per-sample CRPS values of shape ``(n_samples,)``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mean_crps: float
    """Average CRPS across all samples."""

    per_sample_crps: np.ndarray[tuple[int], np.dtype[np.float64]]
    """Shape ``(n_samples,)`` — per-sample CRPS values."""


class VolatilityMetrics(BaseModel, frozen=True):
    """Volatility forecasting evaluation metrics.

    Attributes:
        qlike: Quasi-Likelihood loss (lower is better, 0 = perfect).
        mincer_zarnowitz_r2: R-squared from OLS ``actual_vol ~ alpha + beta * predicted_vol``.
        mincer_zarnowitz_slope: OLS slope (beta); perfect forecast has beta=1.
        mincer_zarnowitz_intercept: OLS intercept (alpha); perfect forecast has alpha=0.
        log_vol_mae: MAE on log-volatility space.
    """

    qlike: float
    """Quasi-Likelihood loss."""

    mincer_zarnowitz_r2: float
    """R-squared from the Mincer-Zarnowitz regression."""

    mincer_zarnowitz_slope: float
    """OLS slope (beta ≈ 1 for unbiased forecasts)."""

    mincer_zarnowitz_intercept: float
    """OLS intercept (alpha ≈ 0 for unbiased forecasts)."""

    log_vol_mae: float
    """MAE on log(vol) space."""


# ---------------------------------------------------------------------------
# Winsorization
# ---------------------------------------------------------------------------

_EPSILON: float = 1e-12
"""Small constant to avoid division by zero and log(0)."""


def winsorize_predictions(
    predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
    config: WinsorizeConfig,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Clip predictions at configurable percentile bounds.

    Winsorization limits the influence of extreme outlier predictions
    on downstream metric computation without discarding any samples.

    Args:
        predictions: Raw predictions of shape ``(n_samples,)``.
        config: Winsorization percentile bounds.

    Returns:
        Clipped predictions of the same shape.

    Raises:
        ValueError: If predictions array is empty.
    """
    n: int = predictions.shape[0]
    if n == 0:
        msg: str = "predictions must contain at least one sample"
        raise ValueError(msg)

    lower_bound: float = float(np.percentile(predictions, config.lower_percentile))
    upper_bound: float = float(np.percentile(predictions, config.upper_percentile))

    clipped: np.ndarray[tuple[int], np.dtype[np.float64]] = np.clip(predictions, lower_bound, upper_bound).astype(
        np.float64
    )

    logger.debug(
        "Winsorized {} predictions: bounds [{:.6f}, {:.6f}] from percentiles [{}, {}]",
        n,
        lower_bound,
        upper_bound,
        config.lower_percentile,
        config.upper_percentile,
    )

    return clipped


# ---------------------------------------------------------------------------
# Standalone regression metrics
# ---------------------------------------------------------------------------


def compute_regression_metrics(
    actuals: np.ndarray[tuple[int], np.dtype[np.float64]],
    predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> RegressionMetrics:
    """Compute core regression metrics on all samples.

    Metrics computed:
    - **MAE**: ``mean(|actual - predicted|)``
    - **RMSE**: ``sqrt(mean((actual - predicted)^2))``
    - **R-squared**: ``1 - SS_res / SS_tot``
    - **Implicit DA**: fraction where ``sign(predicted) == sign(actual)``

    The implicit DA is valuable for regression models: it measures whether
    the model captures directional signal even though it was not trained
    for classification.

    Args:
        actuals: Observed values of shape ``(n_samples,)``.
        predictions: Predicted values of shape ``(n_samples,)``.

    Returns:
        RegressionMetrics with MAE, RMSE, R-squared, implicit DA, and n_samples.

    Raises:
        ValueError: If arrays are empty or lengths differ.
    """
    n: int = actuals.shape[0]
    if n == 0:
        msg: str = "actuals must contain at least one sample"
        raise ValueError(msg)
    if predictions.shape[0] != n:
        msg = f"predictions length {predictions.shape[0]} != actuals length {n}"
        raise ValueError(msg)

    residuals: np.ndarray[tuple[int], np.dtype[np.float64]] = actuals - predictions

    # MAE
    mae: float = float(np.mean(np.abs(residuals)))

    # RMSE
    rmse: float = float(np.sqrt(np.mean(residuals**2)))

    # R-squared: 1 - SS_res / SS_tot
    ss_res: float = float(np.sum(residuals**2))
    ss_tot: float = float(np.sum((actuals - np.mean(actuals)) ** 2))
    r_squared: float = 1.0 - ss_res / ss_tot if ss_tot > _EPSILON else 0.0

    # Implicit DA: sign agreement (zeros treated as positive via np.sign convention)
    sign_actual: np.ndarray[tuple[int], np.dtype[np.float64]] = np.sign(actuals)
    sign_pred: np.ndarray[tuple[int], np.dtype[np.float64]] = np.sign(predictions)
    implicit_da: float = float(np.mean(sign_actual == sign_pred))

    logger.debug(
        "Regression metrics (n={}): MAE={:.6f}, RMSE={:.6f}, R²={:.4f}, DA={:.4f}",
        n,
        mae,
        rmse,
        r_squared,
        implicit_da,
    )

    return RegressionMetrics(
        mae=mae,
        rmse=rmse,
        r_squared=r_squared,
        implicit_da=implicit_da,
        n_samples=n,
    )


# ---------------------------------------------------------------------------
# CRPS — Continuous Ranked Probability Score (Gaussian closed-form)
# ---------------------------------------------------------------------------


def compute_crps_gaussian(
    actuals: np.ndarray[tuple[int], np.dtype[np.float64]],
    mean: np.ndarray[tuple[int], np.dtype[np.float64]],
    std: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> CRPSResult:
    """Compute the CRPS for Gaussian predictive distributions (closed-form).

    Uses the analytical formula (Gneiting & Raftery 2007)::

        CRPS(F, y) = sigma * [z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]

    where ``z = (y - mu) / sigma``, ``Phi`` is the standard normal CDF,
    and ``phi`` is the standard normal PDF.

    For degenerate predictions (``std ≈ 0``), CRPS reduces to ``|y - mean|``.

    Args:
        actuals: Observed values of shape ``(n_samples,)``.
        mean: Predicted means of shape ``(n_samples,)``.
        std: Predicted standard deviations of shape ``(n_samples,)``.
            Must be non-negative.

    Returns:
        CRPSResult with mean CRPS and per-sample values.

    Raises:
        ValueError: If arrays are empty, lengths differ, or std contains negatives.
    """
    n: int = actuals.shape[0]
    if n == 0:
        msg: str = "actuals must contain at least one sample"
        raise ValueError(msg)
    if mean.shape[0] != n or std.shape[0] != n:
        msg = f"array lengths must match: actuals={n}, mean={mean.shape[0]}, std={std.shape[0]}"
        raise ValueError(msg)
    if np.any(std < 0.0):
        msg = "std must be non-negative"
        raise ValueError(msg)

    # Handle degenerate case (std ≈ 0) separately
    is_degenerate: np.ndarray[tuple[int], np.dtype[np.bool_]] = std < _EPSILON

    # Safe division: replace near-zero std with 1.0 to avoid division by zero,
    # then overwrite degenerate entries afterwards.
    safe_std: np.ndarray[tuple[int], np.dtype[np.float64]] = np.where(is_degenerate, 1.0, std).astype(np.float64)
    z: np.ndarray[tuple[int], np.dtype[np.float64]] = ((actuals - mean) / safe_std).astype(np.float64)

    # Phi(z) = standard normal CDF, phi(z) = standard normal PDF
    phi_z: np.ndarray[tuple[int], np.dtype[np.float64]] = sp_stats.norm.pdf(z).astype(np.float64)
    big_phi_z: np.ndarray[tuple[int], np.dtype[np.float64]] = sp_stats.norm.cdf(z).astype(np.float64)

    inv_sqrt_pi: float = 1.0 / np.sqrt(np.pi)

    per_sample: np.ndarray[tuple[int], np.dtype[np.float64]] = (
        safe_std * (z * (2.0 * big_phi_z - 1.0) + 2.0 * phi_z - inv_sqrt_pi)
    ).astype(np.float64)

    # Degenerate case: CRPS = |y - mean|
    degenerate_crps: np.ndarray[tuple[int], np.dtype[np.float64]] = np.abs(actuals - mean).astype(np.float64)
    per_sample = np.where(is_degenerate, degenerate_crps, per_sample).astype(np.float64)

    mean_crps: float = float(np.mean(per_sample))

    logger.debug("CRPS Gaussian (n={}): mean_crps={:.6f}", n, mean_crps)

    return CRPSResult(mean_crps=mean_crps, per_sample_crps=per_sample)


# ---------------------------------------------------------------------------
# Volatility metrics
# ---------------------------------------------------------------------------


def compute_volatility_metrics(
    actual_vol: np.ndarray[tuple[int], np.dtype[np.float64]],
    predicted_vol: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> VolatilityMetrics:
    """Compute volatility forecast evaluation metrics.

    Metrics computed:

    - **QLIKE** (Quasi-Likelihood)::

        QLIKE = mean(h_actual / h_predicted - log(h_actual / h_predicted) - 1)

      where ``h = sigma^2`` (variance).  QLIKE is robust and consistent
      for comparing variance forecasts (Patton 2011).

    - **Mincer-Zarnowitz R-squared** — OLS regression of
      ``actual_vol ~ alpha + beta * predicted_vol``.  R-squared measures
      explanatory power; unbiased forecasts have ``beta ≈ 1``, ``alpha ≈ 0``.

    - **Log-vol MAE** — ``mean(|log(actual_vol) - log(predicted_vol)|)``.
      Operating in log space prevents large-variance regimes from
      dominating the error metric.

    Args:
        actual_vol: Realised volatility (sigma, not variance) of shape ``(n_samples,)``.
            Must be strictly positive.
        predicted_vol: Forecast volatility (sigma) of shape ``(n_samples,)``.
            Must be strictly positive.

    Returns:
        VolatilityMetrics with QLIKE, Mincer-Zarnowitz results, and log-vol MAE.

    Raises:
        ValueError: If arrays are empty, lengths differ, or values are non-positive.
    """
    n: int = actual_vol.shape[0]
    if n == 0:
        msg: str = "actual_vol must contain at least one sample"
        raise ValueError(msg)
    if predicted_vol.shape[0] != n:
        msg = f"predicted_vol length {predicted_vol.shape[0]} != actual_vol length {n}"
        raise ValueError(msg)
    if np.any(actual_vol <= 0.0):
        msg = "actual_vol must be strictly positive"
        raise ValueError(msg)
    if np.any(predicted_vol <= 0.0):
        msg = "predicted_vol must be strictly positive"
        raise ValueError(msg)

    # --- QLIKE (operates on variances: h = sigma^2) ---
    ratio: np.ndarray[tuple[int], np.dtype[np.float64]] = ((actual_vol**2) / (predicted_vol**2)).astype(np.float64)
    qlike: float = float(np.mean(ratio - np.log(ratio) - 1.0))

    # --- Mincer-Zarnowitz regression: actual_vol = alpha + beta * predicted_vol ---
    # np.polyfit returns [highest-degree coef, constant] = [beta, alpha]
    coeffs: np.ndarray[tuple[int], np.dtype[np.float64]] = np.polyfit(predicted_vol, actual_vol, deg=1).astype(
        np.float64
    )
    beta: float = float(coeffs[0])
    alpha: float = float(coeffs[1])

    # R-squared from the MZ regression
    fitted: np.ndarray[tuple[int], np.dtype[np.float64]] = (alpha + beta * predicted_vol).astype(np.float64)
    ss_res_mz: float = float(np.sum((actual_vol - fitted) ** 2))
    ss_tot_mz: float = float(np.sum((actual_vol - np.mean(actual_vol)) ** 2))
    mz_r2: float = 1.0 - ss_res_mz / ss_tot_mz if ss_tot_mz > _EPSILON else 0.0

    # --- Log-vol MAE ---
    log_vol_mae: float = float(np.mean(np.abs(np.log(actual_vol) - np.log(predicted_vol))))

    logger.debug(
        "Volatility metrics (n={}): QLIKE={:.6f}, MZ_R²={:.4f}, MZ_beta={:.4f}, MZ_alpha={:.6f}, log_MAE={:.6f}",
        n,
        qlike,
        mz_r2,
        beta,
        alpha,
        log_vol_mae,
    )

    return VolatilityMetrics(
        qlike=qlike,
        mincer_zarnowitz_r2=mz_r2,
        mincer_zarnowitz_slope=beta,
        mincer_zarnowitz_intercept=alpha,
        log_vol_mae=log_vol_mae,
    )


# ---------------------------------------------------------------------------
# Pipeline metrics — stubs (requires Phase 11 classifier)
# ---------------------------------------------------------------------------


def compute_dc_mae(
    actuals: np.ndarray[tuple[int], np.dtype[np.float64]],
    predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
    direction_correct: np.ndarray[tuple[int], np.dtype[np.bool_]],
) -> float:
    """Compute Direction-Conditional MAE (DC-MAE).

    DC-MAE restricts MAE to samples where the classifier's direction
    prediction was correct.  This separates magnitude accuracy from
    directional accuracy and is the primary regression metric for this
    thesis.

    Args:
        actuals: Observed values of shape ``(n_samples,)``.
        predictions: Predicted magnitudes of shape ``(n_samples,)``.
        direction_correct: Boolean mask of shape ``(n_samples,)`` indicating
            which samples had correct directional predictions.

    Returns:
        Mean absolute error restricted to direction-correct samples.

    Raises:
        ValueError: If arrays are empty, lengths differ, or no direction-correct
            samples exist.
    """
    n: int = actuals.shape[0]
    if n == 0:
        msg: str = "actuals must contain at least one sample"
        raise ValueError(msg)
    if predictions.shape[0] != n:
        msg = f"predictions length {predictions.shape[0]} != actuals length {n}"
        raise ValueError(msg)
    if direction_correct.shape[0] != n:
        msg = f"direction_correct length {direction_correct.shape[0]} != actuals length {n}"
        raise ValueError(msg)

    n_correct: int = int(np.sum(direction_correct))
    if n_correct == 0:
        msg = "no direction-correct samples — DC-MAE is undefined"
        raise ValueError(msg)

    residuals: np.ndarray[tuple[int], np.dtype[np.float64]] = np.abs(
        actuals[direction_correct] - predictions[direction_correct]
    )
    dc_mae: float = float(np.mean(residuals))

    logger.debug(
        "DC-MAE: {:.6f} (n_correct={}/{}, fraction={:.4f})",
        dc_mae,
        n_correct,
        n,
        n_correct / n,
    )

    return dc_mae


def compute_dc_rmse(
    actuals: np.ndarray[tuple[int], np.dtype[np.float64]],
    predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
    direction_correct: np.ndarray[tuple[int], np.dtype[np.bool_]],
) -> float:
    """Compute Direction-Conditional RMSE (DC-RMSE).

    DC-RMSE restricts RMSE to samples where the classifier's direction
    prediction was correct.  Complements DC-MAE by penalising large
    magnitude errors more heavily.

    Args:
        actuals: Observed values of shape ``(n_samples,)``.
        predictions: Predicted magnitudes of shape ``(n_samples,)``.
        direction_correct: Boolean mask of shape ``(n_samples,)`` indicating
            which samples had correct directional predictions.

    Returns:
        Root mean squared error restricted to direction-correct samples.

    Raises:
        ValueError: If arrays are empty, lengths differ, or no direction-correct
            samples exist.
    """
    n: int = actuals.shape[0]
    if n == 0:
        msg: str = "actuals must contain at least one sample"
        raise ValueError(msg)
    if predictions.shape[0] != n:
        msg = f"predictions length {predictions.shape[0]} != actuals length {n}"
        raise ValueError(msg)
    if direction_correct.shape[0] != n:
        msg = f"direction_correct length {direction_correct.shape[0]} != actuals length {n}"
        raise ValueError(msg)

    n_correct: int = int(np.sum(direction_correct))
    if n_correct == 0:
        msg = "no direction-correct samples — DC-RMSE is undefined"
        raise ValueError(msg)

    squared_errors: np.ndarray[tuple[int], np.dtype[np.float64]] = (
        actuals[direction_correct] - predictions[direction_correct]
    ) ** 2
    dc_rmse: float = float(np.sqrt(np.mean(squared_errors)))

    logger.debug(
        "DC-RMSE: {:.6f} (n_correct={}/{}, fraction={:.4f})",
        dc_rmse,
        n_correct,
        n,
        n_correct / n,
    )

    return dc_rmse


def compute_wdl(
    actuals: np.ndarray[tuple[int], np.dtype[np.float64]],
    predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
    direction_predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> dict[str, float]:
    """Compute Win/Draw/Loss ratios from combined direction + magnitude.

    A trade is a *win* when both direction and magnitude exceed the
    transaction cost threshold; a *draw* when the net P&L is within
    the threshold; and a *loss* otherwise.

    Note:
        Requires Phase 11 classifier for directional predictions and
        Phase 8 backtest engine for transaction cost modelling.
        This stub will be replaced in Phase 13 integration.

    Args:
        actuals: Observed values of shape ``(n_samples,)``.
        predictions: Predicted magnitudes of shape ``(n_samples,)``.
        direction_predictions: Directional predictions of shape ``(n_samples,)``.

    Raises:
        NotImplementedError: Always — requires Phase 11 classifier.
    """
    raise NotImplementedError("WDL requires Phase 11 classifier — available in Phase 13 integration")


def compute_pdr(
    actuals: np.ndarray[tuple[int], np.dtype[np.float64]],
    predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
    direction_correct: np.ndarray[tuple[int], np.dtype[np.bool_]],
    magnitude_threshold: float = 0.01,
    realized_threshold: float = 0.005,
) -> float:
    """Compute Prediction Directional Reliability (PDR).

    PDR measures: when the classifier says "up" AND the regressor predicts
    magnitude above ``magnitude_threshold``, how often is the realized
    return positive and above ``realized_threshold``?

    This captures the *reliability of confident joint predictions* — exactly
    the operating regime of the recommendation system.

    Args:
        actuals: Observed returns of shape ``(n_samples,)``.
        predictions: Predicted return magnitudes of shape ``(n_samples,)``.
        direction_correct: Boolean mask of shape ``(n_samples,)`` indicating
            which samples had correct directional predictions from the
            classifier.
        magnitude_threshold: Minimum absolute predicted magnitude to consider
            the regressor "confident" (default 1%).
        realized_threshold: Minimum realized absolute return to count as a
            "successful" prediction (default 0.5%).

    Returns:
        Fraction of confident joint predictions that were successful.
        Returns 0.0 if no predictions exceed the magnitude threshold.

    Raises:
        ValueError: If arrays are empty or lengths differ.
    """
    n: int = actuals.shape[0]
    if n == 0:
        msg: str = "actuals must contain at least one sample"
        raise ValueError(msg)
    if predictions.shape[0] != n or direction_correct.shape[0] != n:
        msg = "all arrays must have the same length"
        raise ValueError(msg)

    # Confident predictions: classifier correct AND regressor magnitude > threshold
    confident_mask: np.ndarray[tuple[int], np.dtype[np.bool_]] = direction_correct & (
        np.abs(predictions) > magnitude_threshold
    )

    n_confident: int = int(np.sum(confident_mask))
    if n_confident == 0:
        logger.debug("PDR: no confident predictions above threshold {:.4f}", magnitude_threshold)
        return 0.0

    # Success: realized return has same sign AND exceeds realized threshold
    confident_actuals: np.ndarray[tuple[int], np.dtype[np.float64]] = actuals[confident_mask]
    confident_preds: np.ndarray[tuple[int], np.dtype[np.float64]] = predictions[confident_mask]

    same_sign: np.ndarray[tuple[int], np.dtype[np.bool_]] = np.sign(confident_actuals) == np.sign(confident_preds)
    exceeds_threshold: np.ndarray[tuple[int], np.dtype[np.bool_]] = np.abs(confident_actuals) > realized_threshold

    successful: np.ndarray[tuple[int], np.dtype[np.bool_]] = same_sign & exceeds_threshold
    pdr: float = float(np.mean(successful))

    logger.debug(
        "PDR: {:.4f} (n_confident={}/{}, mag_thresh={:.4f}, real_thresh={:.4f})",
        pdr,
        n_confident,
        n,
        magnitude_threshold,
        realized_threshold,
    )

    return pdr


def _lo_2002_correction(
    pnl_series: np.ndarray[tuple[int], np.dtype[np.float64]],
    max_lags: int,
) -> float:
    """Compute the Lo (2002) autocorrelation correction factor.

    The correction adjusts the Sharpe ratio for serial correlation in
    P&L returns: ``SR_corrected = SR_raw / sqrt(1 + 2 * sum(rho_k))``.

    Args:
        pnl_series: Net P&L series of shape ``(n,)``.
        max_lags: Maximum number of autocorrelation lags to include.

    Returns:
        Correction factor (>= 1.0 for positively autocorrelated series).
    """
    if max_lags <= 0:
        return 1.0

    mean_pnl: float = float(np.mean(pnl_series))
    centered: np.ndarray[tuple[int], np.dtype[np.float64]] = (pnl_series - mean_pnl).astype(np.float64)
    var_pnl: float = float(np.var(centered, ddof=0))

    if var_pnl < _EPSILON:
        return 1.0

    rho_sum: float = 0.0
    for lag in range(1, max_lags + 1):
        autocov: float = float(np.mean(centered[lag:] * centered[:-lag]))
        rho_sum += autocov / var_pnl

    denominator: float = 1.0 + 2.0 * rho_sum
    return float(np.sqrt(denominator)) if denominator > _EPSILON else 1.0


def compute_economic_sharpe(
    actuals: np.ndarray[tuple[int], np.dtype[np.float64]],
    predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
    direction_predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
    transaction_cost: float = 0.001,
) -> float:
    """Compute the Economic Sharpe Ratio from combined predictions.

    The Economic Sharpe treats each prediction as a trading signal,
    computes net P&L after transaction costs, and reports the Sharpe
    ratio with Lo (2002) autocorrelation correction for IID violation.

    Strategy logic:
        - Position = ``sign(direction_prediction)`` (flat when direction = 0)
        - Per-bar P&L = ``position * actual_return``
        - Round-trip cost incurred on every position change:
          ``cost = transaction_cost * |position_change|``
        - Sharpe = ``mean(net_pnl) / std(net_pnl)``
        - Lo (2002) correction: ``SR_corrected = SR / sqrt(1 + 2 * sum(rho_k))``

    Args:
        actuals: Observed returns of shape ``(n_samples,)``.
        predictions: Predicted magnitudes of shape ``(n_samples,)``
            (used for position sizing in future extensions; currently unused
            beyond direction).
        direction_predictions: Directional predictions of shape ``(n_samples,)``
            with values +1, -1, or 0.
        transaction_cost: One-way transaction cost (default 10 bps).

    Returns:
        Economic Sharpe ratio (not annualised — per-bar basis).
        Returns 0.0 if net P&L has zero variance.

    Raises:
        ValueError: If arrays are empty or lengths differ.
    """
    n: int = actuals.shape[0]
    if n == 0:
        msg: str = "actuals must contain at least one sample"
        raise ValueError(msg)
    if predictions.shape[0] != n or direction_predictions.shape[0] != n:
        msg = "all arrays must have the same length"
        raise ValueError(msg)

    # Positions: sign of direction prediction
    positions: np.ndarray[tuple[int], np.dtype[np.float64]] = np.sign(direction_predictions)

    # Net P&L = gross P&L - transaction costs on position changes
    gross_pnl: np.ndarray[tuple[int], np.dtype[np.float64]] = (positions * actuals).astype(np.float64)
    costs: np.ndarray[tuple[int], np.dtype[np.float64]] = (
        np.abs(np.diff(positions, prepend=0.0)) * transaction_cost
    ).astype(np.float64)
    net_pnl: np.ndarray[tuple[int], np.dtype[np.float64]] = (gross_pnl - costs).astype(np.float64)

    mean_pnl: float = float(np.mean(net_pnl))
    std_pnl: float = float(np.std(net_pnl, ddof=1)) if n > 1 else 0.0

    if std_pnl < _EPSILON:
        logger.debug("Economic Sharpe: zero variance in net P&L")
        return 0.0

    raw_sharpe: float = mean_pnl / std_pnl
    correction: float = _lo_2002_correction(net_pnl, max_lags=min(5, n // 4))
    corrected_sharpe: float = raw_sharpe / correction if correction > _EPSILON else raw_sharpe

    logger.debug(
        "Economic Sharpe: raw={:.4f}, corrected={:.4f}, lo_correction={:.4f}",
        raw_sharpe,
        corrected_sharpe,
        correction,
    )

    return corrected_sharpe
