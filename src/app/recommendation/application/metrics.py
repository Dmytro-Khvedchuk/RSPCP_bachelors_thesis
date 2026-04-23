"""Recommendation metrics — decision quality, economic value, and conformal deployment."""

from __future__ import annotations

from typing import Annotated

import numpy as np
from loguru import logger
from pydantic import BaseModel
from pydantic import Field as PydanticField

from src.app.recommendation.domain.value_objects import Recommendation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SECONDS_PER_YEAR: float = 365.25 * 24.0 * 3_600.0
_MIN_PERIODS_FOR_STATS: int = 2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ConformalDeployConfig(BaseModel, frozen=True):
    """Configuration for split conformal prediction-based deployment.

    Controls the significance level (alpha) for the conformal interval.
    Deployment is recommended only when the lower bound of the prediction
    interval exceeds zero, i.e. ``y_pred - q > 0`` where ``q`` is the
    ``(1 - alpha)`` quantile of the calibration nonconformity scores.

    Attributes:
        alpha: Significance level for the conformal interval.
            Lower alpha produces wider intervals (more conservative).
    """

    alpha: Annotated[
        float,
        PydanticField(
            default=0.1,
            gt=0.0,
            lt=1.0,
            description="Significance level for conformal interval",
        ),
    ]


# ---------------------------------------------------------------------------
# RecommendationMetrics
# ---------------------------------------------------------------------------


class RecommendationMetrics(BaseModel, frozen=True):
    """Immutable container for recommendation system performance metrics.

    All ``float`` fields default to *None* when insufficient data is
    available for computation.

    Attributes:
        n_decisions: Total number of decision points evaluated.
        n_deployed: Number of decisions where ``deploy=True``.
        deploy_rate: Fraction of decisions that deployed.
        deploy_precision: Precision of deploy=True: fraction of deployed
            decisions with positive realised strategy return.
        sharpe_with_sizing: Annualised Sharpe of ``position_size x strategy_return``
            portfolio (Lo 2002 corrected).
        sharpe_without_sizing: Annualised Sharpe of binary deploy/skip
            portfolio (1.0 x strategy_return if deploy else 0).
        sizing_value: Sharpe difference (with - without) quantifying the
            value of continuous position sizing over binary bet/no-bet.
        mean_portfolio_return: Mean per-period return of the sized portfolio.
        cumulative_return: Cumulative return of the sized portfolio.
        lo_correction_factor: Lo (2002) eta(q) autocorrelation correction
            factor applied to the raw Sharpe.
    """

    n_decisions: Annotated[int, PydanticField(ge=0)]
    n_deployed: Annotated[int, PydanticField(ge=0)]
    deploy_rate: Annotated[float | None, PydanticField(default=None)]
    deploy_precision: Annotated[float | None, PydanticField(default=None)]
    sharpe_with_sizing: Annotated[float | None, PydanticField(default=None)]
    sharpe_without_sizing: Annotated[float | None, PydanticField(default=None)]
    sizing_value: Annotated[float | None, PydanticField(default=None)]
    mean_portfolio_return: Annotated[float | None, PydanticField(default=None)]
    cumulative_return: Annotated[float | None, PydanticField(default=None)]
    lo_correction_factor: Annotated[float | None, PydanticField(default=None)]


# ---------------------------------------------------------------------------
# Public API — compute_recommendation_metrics
# ---------------------------------------------------------------------------


def compute_recommendation_metrics(  # noqa: PLR0914
    predictions: list[Recommendation],
    actual_returns: list[float],
    *,
    lo_correction_lags: int = 6,
    periods_per_year: float = 365.25 * 24.0,
) -> RecommendationMetrics:
    """Compute comprehensive recommendation system performance metrics.

    Builds two portfolio return series:

    1. **Sized**: ``position_size x actual_return`` when deployed, else 0.
    2. **Binary**: ``1.0 x actual_return`` when deployed, else 0.

    Then computes annualised Sharpe ratios with Lo (2002) autocorrelation
    correction for both.  The difference quantifies the value of continuous
    position sizing (the "generalised meta-labeling" claim).

    Args:
        predictions: OOS recommendations from the recommender.
        actual_returns: Realised strategy returns aligned 1:1 with predictions.
        lo_correction_lags: Number of autocorrelation lags for Lo correction.
        periods_per_year: Number of decision periods per year.  Defaults to
            hourly bars (365.25 * 24).

    Returns:
        Frozen :class:`RecommendationMetrics` instance.

    Raises:
        ValueError: If ``predictions`` and ``actual_returns`` have different
            lengths.
    """
    n: int = len(predictions)
    if n != len(actual_returns):
        msg: str = f"predictions and actual_returns must have the same length, got {n} and {len(actual_returns)}"
        raise ValueError(msg)

    if n == 0:
        return RecommendationMetrics(n_decisions=0, n_deployed=0)  # ty: ignore[missing-argument]

    # -- Build portfolio return arrays ---
    actual_arr: np.ndarray[tuple[int], np.dtype[np.float64]] = np.asarray(
        actual_returns,
        dtype=np.float64,
    )
    deploy_mask: np.ndarray[tuple[int], np.dtype[np.bool_]] = np.array(
        [r.deploy for r in predictions],
        dtype=np.bool_,
    )
    sizes: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(
        [r.position_size for r in predictions],
        dtype=np.float64,
    )

    # Sized portfolio: position_size * actual_return when deployed, else 0
    sized_returns: np.ndarray[tuple[int], np.dtype[np.float64]] = np.where(
        deploy_mask,
        sizes * actual_arr,
        0.0,
    )
    # Binary portfolio: 1.0 * actual_return when deployed, else 0
    binary_returns: np.ndarray[tuple[int], np.dtype[np.float64]] = np.where(
        deploy_mask,
        actual_arr,
        0.0,
    )

    n_deployed: int = int(deploy_mask.sum())
    deploy_rate: float = n_deployed / n

    # -- Decision quality ---
    deploy_precision: float | None = _compute_deploy_precision(
        deploy_mask,
        actual_arr,
    )

    # -- Economic value: Sharpe with Lo correction ---
    sharpe_sized: float | None = _annualised_sharpe(sized_returns, periods_per_year)
    sharpe_binary: float | None = _annualised_sharpe(binary_returns, periods_per_year)
    lo_factor: float | None = _lo_correction_factor(sized_returns, lo_correction_lags)

    sharpe_lo_sized: float | None = None
    if sharpe_sized is not None and lo_factor is not None:
        sharpe_lo_sized = sharpe_sized * lo_factor

    sharpe_lo_binary: float | None = None
    if sharpe_binary is not None and lo_factor is not None:
        sharpe_lo_binary = sharpe_binary * lo_factor

    sizing_value: float | None = None
    if sharpe_lo_sized is not None and sharpe_lo_binary is not None:
        sizing_value = sharpe_lo_sized - sharpe_lo_binary

    mean_ret: float = float(np.mean(sized_returns))
    cum_ret: float = float(np.prod(1.0 + sized_returns) - 1.0)

    logger.info(
        "Recommendation metrics: n={} deployed={} precision={} sharpe_sized={} sharpe_binary={} sizing_value={}",
        n,
        n_deployed,
        f"{deploy_precision:.4f}" if deploy_precision is not None else "N/A",
        f"{sharpe_lo_sized:.4f}" if sharpe_lo_sized is not None else "N/A",
        f"{sharpe_lo_binary:.4f}" if sharpe_lo_binary is not None else "N/A",
        f"{sizing_value:.4f}" if sizing_value is not None else "N/A",
    )

    return RecommendationMetrics(
        n_decisions=n,
        n_deployed=n_deployed,
        deploy_rate=deploy_rate,
        deploy_precision=deploy_precision,
        sharpe_with_sizing=sharpe_lo_sized,
        sharpe_without_sizing=sharpe_lo_binary,
        sizing_value=sizing_value,
        mean_portfolio_return=mean_ret,
        cumulative_return=cum_ret,
        lo_correction_factor=lo_factor,
    )


# ---------------------------------------------------------------------------
# Public API — build_conformal_intervals
# ---------------------------------------------------------------------------


def build_conformal_intervals(
    calibration_predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
    calibration_actuals: np.ndarray[tuple[int], np.dtype[np.float64]],
    new_predictions: np.ndarray[tuple[int], np.dtype[np.float64]],
    config: ConformalDeployConfig | None = None,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.bool_]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
    float,
]:
    """Build split conformal prediction intervals and deploy decisions.

    Implements split conformal prediction (Vovk et al., 2005):

    1. Compute nonconformity scores on calibration data:
       ``|y_actual - y_predicted|``
    2. Compute quantile ``q = quantile(scores, 1 - alpha)``
    3. For each new prediction, interval is ``[y_pred - q, y_pred + q]``
    4. Deploy when ``lower_bound = y_pred - q > 0``

    This provides finite-sample coverage guarantees without distributional
    assumptions.

    Args:
        calibration_predictions: Predicted returns on calibration set.
        calibration_actuals: Realised returns on calibration set.
        new_predictions: Predicted returns for new deployment decisions.
        config: Conformal deployment configuration. Uses defaults if ``None``.

    Returns:
        Tuple of:
        - ``deploy_decisions``: Boolean array, ``True`` where lower bound > 0.
        - ``lower_bounds``: Lower bounds of the conformal intervals.
        - ``upper_bounds``: Upper bounds of the conformal intervals.
        - ``quantile_q``: The computed nonconformity quantile.

    Raises:
        ValueError: If calibration arrays have different lengths or are empty.
    """
    cfg: ConformalDeployConfig = config or ConformalDeployConfig()  # ty: ignore[missing-argument]
    n_cal: int = len(calibration_predictions)

    if n_cal != len(calibration_actuals):
        msg: str = f"calibration arrays must have the same length, got {n_cal} and {len(calibration_actuals)}"
        raise ValueError(msg)

    if n_cal == 0:
        msg = "calibration arrays must not be empty"
        raise ValueError(msg)

    # Step 1: Nonconformity scores = |actual - predicted|
    nonconformity_scores: np.ndarray[tuple[int], np.dtype[np.float64]] = np.abs(
        calibration_actuals - calibration_predictions,
    )

    # Step 2: Quantile at (1 - alpha) level
    # Use ceil((n+1)(1-alpha))/n index for finite-sample validity
    quantile_level: float = min(1.0, (1.0 - cfg.alpha) * (1.0 + 1.0 / n_cal))
    q: float = float(np.quantile(nonconformity_scores, quantile_level))

    # Step 3: Prediction intervals
    lower_bounds: np.ndarray[tuple[int], np.dtype[np.float64]] = new_predictions - q
    upper_bounds: np.ndarray[tuple[int], np.dtype[np.float64]] = new_predictions + q

    # Step 4: Deploy when lower bound > 0
    deploy_decisions: np.ndarray[tuple[int], np.dtype[np.bool_]] = lower_bounds > 0.0

    n_deploy: int = int(deploy_decisions.sum())
    logger.info(
        "Conformal deployment: alpha={:.2f} q={:.6f} | {}/{} predictions pass lower-bound > 0 threshold",
        cfg.alpha,
        q,
        n_deploy,
        len(new_predictions),
    )

    return deploy_decisions, lower_bounds, upper_bounds, q


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _compute_deploy_precision(
    deploy_mask: np.ndarray[tuple[int], np.dtype[np.bool_]],
    actual_returns: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> float | None:
    """Compute precision of deploy=True decisions.

    Precision = fraction of deployed decisions with positive realised return.

    Args:
        deploy_mask: Boolean array of deploy decisions.
        actual_returns: Realised strategy returns.

    Returns:
        Precision in ``[0, 1]``, or ``None`` if no decisions were deployed.
    """
    n_deployed: int = int(deploy_mask.sum())
    if n_deployed == 0:
        return None
    deployed_returns: np.ndarray[tuple[int], np.dtype[np.float64]] = actual_returns[deploy_mask]
    n_positive: int = int((deployed_returns > 0.0).sum())
    precision: float = n_positive / n_deployed
    return precision


def _annualised_sharpe(
    returns: np.ndarray[tuple[int], np.dtype[np.float64]],
    periods_per_year: float,
) -> float | None:
    """Compute annualised Sharpe ratio (risk-free rate = 0).

    Args:
        returns: Array of period returns.
        periods_per_year: Number of periods per year.

    Returns:
        Annualised Sharpe ratio, or ``None`` if insufficient data or
        zero volatility.
    """
    if len(returns) < _MIN_PERIODS_FOR_STATS:
        return None
    std: float = float(np.std(returns, ddof=1))
    if std == 0.0:
        return None
    sharpe: float = float(np.mean(returns)) / std * np.sqrt(periods_per_year)
    return sharpe


def _lo_correction_factor(
    returns: np.ndarray[tuple[int], np.dtype[np.float64]],
    q: int,
) -> float | None:
    """Compute Lo (2002) autocorrelation correction factor eta(q).

    The formula is::

        eta(q) = sqrt(q / (q + 2 * sum_{k=1}^{q} (q - k) * rho_k))

    Where ``rho_k`` is the k-th autocorrelation of returns.  Positive
    autocorrelation yields ``eta < 1`` (more conservative Sharpe).

    Args:
        returns: Array of period returns.
        q: Number of autocorrelation lags.

    Returns:
        Correction factor, or ``None`` if insufficient data or degenerate
        denominator.
    """
    n_ret: int = len(returns)
    if n_ret < q + 1:
        return None

    mean_r: float = float(np.mean(returns))
    demeaned: np.ndarray[tuple[int], np.dtype[np.float64]] = returns - mean_r
    var_r: float = float(np.dot(demeaned, demeaned)) / n_ret
    if var_r == 0.0:
        return None

    weighted_sum: float = 0.0
    for k in range(1, q + 1):
        auto_cov: float = float(np.dot(demeaned[k:], demeaned[:-k])) / n_ret
        rho_k: float = auto_cov / var_r
        weight: int = q - k
        weighted_sum += weight * rho_k

    denominator: float = float(q) + 2.0 * weighted_sum
    if denominator <= 0.0:
        return None

    eta: float = float(np.sqrt(float(q) / denominator))
    return eta
