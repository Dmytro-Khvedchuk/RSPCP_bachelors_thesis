"""Backtest metrics layer — performance, risk, and trade statistics."""

from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import BaseModel
from pydantic import Field as PydanticField

from src.app.backtest.domain.entities import EquityCurve, Trade


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SECONDS_PER_YEAR: float = 365.25 * 24.0 * 3600.0
_MIN_PERIODS_FOR_STATS: int = 2


# ---------------------------------------------------------------------------
# BacktestMetrics
# ---------------------------------------------------------------------------


class BacktestMetrics(BaseModel, frozen=True):
    """Immutable container for backtest performance statistics.

    All ``float`` fields default to *None* when insufficient data is
    available.  Trade-level metrics (``win_rate``, ``profit_factor``,
    etc.) are populated only when ``n_trades >= min_trade_count``.

    Attributes:
        total_return: Cumulative return over the evaluation period.
        annualized_return: Annualised compound return.
        cagr: Compound annual growth rate (alias for annualized_return).
        max_drawdown: Largest peak-to-trough decline as a negative fraction.
        max_drawdown_duration_days: Longest drawdown duration in calendar days.
        annualized_volatility: Annualised standard deviation of returns.
        downside_volatility: Annualised standard deviation of negative returns.
        sharpe_ratio: Annualised Sharpe ratio (risk-free rate default 0).
        sharpe_ratio_lo_corrected: Lo (2002) autocorrelation-corrected Sharpe.
        lo_correction_factor: The eta(q) factor from Lo (2002).
        sortino_ratio: Annualised Sortino ratio.
        calmar_ratio: Annualised return divided by absolute max drawdown.
        n_trades: Total number of completed trades.
        win_rate: Fraction of trades with positive net P&L.
        profit_factor: Sum of winning P&L over absolute sum of losing P&L.
        avg_win_loss_ratio: Mean winning P&L over absolute mean losing P&L.
        max_consecutive_losses: Longest streak of losing trades.
        sufficient_sample: Whether n_trades meets the minimum threshold.
        min_trade_count: Threshold used for sample sufficiency.
    """

    # -- Return metrics ------------------------------------------------------
    total_return: Annotated[float | None, PydanticField(default=None)]
    annualized_return: Annotated[float | None, PydanticField(default=None)]
    cagr: Annotated[float | None, PydanticField(default=None)]

    # -- Risk metrics --------------------------------------------------------
    max_drawdown: Annotated[float | None, PydanticField(default=None)]
    max_drawdown_duration_days: Annotated[float | None, PydanticField(default=None)]
    annualized_volatility: Annotated[float | None, PydanticField(default=None)]
    downside_volatility: Annotated[float | None, PydanticField(default=None)]

    # -- Risk-adjusted metrics -----------------------------------------------
    sharpe_ratio: Annotated[float | None, PydanticField(default=None)]
    sharpe_ratio_lo_corrected: Annotated[float | None, PydanticField(default=None)]
    lo_correction_factor: Annotated[float | None, PydanticField(default=None)]
    sortino_ratio: Annotated[float | None, PydanticField(default=None)]
    calmar_ratio: Annotated[float | None, PydanticField(default=None)]

    # -- Trade metrics -------------------------------------------------------
    n_trades: Annotated[int | None, PydanticField(default=None)]
    win_rate: Annotated[float | None, PydanticField(default=None)]
    profit_factor: Annotated[float | None, PydanticField(default=None)]
    avg_win_loss_ratio: Annotated[float | None, PydanticField(default=None)]
    max_consecutive_losses: Annotated[int | None, PydanticField(default=None)]

    # -- Metadata ------------------------------------------------------------
    sufficient_sample: Annotated[bool, PydanticField(default=False)]
    min_trade_count: Annotated[int, PydanticField(default=30, gt=0)]


# ---------------------------------------------------------------------------
# Public API — compute_metrics
# ---------------------------------------------------------------------------


def compute_metrics(  # noqa: PLR0913, PLR0914
    equity_curve: EquityCurve,
    trades: list[Trade],
    *,
    min_trade_count: int = 30,
    risk_free_rate: float = 0.0,
    lo_correction_lags: int = 6,
) -> BacktestMetrics:
    """Compute comprehensive backtest performance metrics.

    Derives return, risk, and risk-adjusted statistics from an equity
    curve.  Trade-level statistics are computed only when the number
    of completed trades meets the ``min_trade_count`` threshold.

    Args:
        equity_curve: Time-indexed equity series from a backtest run.
        trades: List of completed trades.
        min_trade_count: Minimum trades required for trade-level metrics.
        risk_free_rate: Annualised risk-free rate (default 0 for crypto).
        lo_correction_lags: Number of autocorrelation lags for the Lo
            (2002) Sharpe correction.

    Returns:
        Frozen :class:`BacktestMetrics` instance.
    """
    values: np.ndarray = np.asarray(equity_curve.values, dtype=np.float64)
    n_points: int = len(values)

    if n_points < _MIN_PERIODS_FOR_STATS:
        return _empty_metrics(
            n_trades=len(trades),
            min_trade_count=min_trade_count,
        )

    # -- Period returns ------------------------------------------------------
    returns: np.ndarray = values[1:] / values[:-1] - 1.0

    # -- Periods per year ----------------------------------------------------
    periods_per_year: float = _estimate_periods_per_year(equity_curve)

    # -- Return metrics ------------------------------------------------------
    total_ret: float = float(values[-1] / values[0] - 1.0)
    duration_days: float = _duration_days(equity_curve)
    ann_ret: float | None = _annualized_return(total_ret, duration_days)

    # -- Risk metrics --------------------------------------------------------
    max_dd: float = _max_drawdown(values)
    max_dd_dur: float = _max_drawdown_duration_days(equity_curve, values)
    ann_vol: float | None = _annualized_volatility(returns, periods_per_year)
    ds_vol: float | None = _downside_volatility(returns, periods_per_year)

    # -- Risk-adjusted metrics -----------------------------------------------
    sharpe: float | None = _sharpe_ratio(returns, risk_free_rate, periods_per_year)
    lo_factor: float | None = _lo_correction_factor(returns, lo_correction_lags)
    sharpe_lo: float | None = None
    if sharpe is not None and lo_factor is not None:
        sharpe_lo = sharpe * lo_factor
    sortino: float | None = _sortino_ratio(returns, risk_free_rate, periods_per_year)
    calmar: float | None = None
    if ann_ret is not None and max_dd < 0.0:
        calmar = ann_ret / abs(max_dd)

    # -- Trade metrics -------------------------------------------------------
    n_trades: int = len(trades)
    sufficient: bool = n_trades >= min_trade_count

    win_rate_val: float | None = None
    pf_val: float | None = None
    awl_val: float | None = None
    mcl_val: int | None = None

    if sufficient:
        trade_stats: _TradeStats = _compute_trade_metrics(trades)
        win_rate_val = trade_stats.win_rate
        pf_val = trade_stats.profit_factor
        awl_val = trade_stats.avg_win_loss_ratio
        mcl_val = trade_stats.max_consecutive_losses

    return BacktestMetrics(
        total_return=total_ret,
        annualized_return=ann_ret,
        cagr=ann_ret,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_dur,
        annualized_volatility=ann_vol,
        downside_volatility=ds_vol,
        sharpe_ratio=sharpe,
        sharpe_ratio_lo_corrected=sharpe_lo,
        lo_correction_factor=lo_factor,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        n_trades=n_trades,
        win_rate=win_rate_val,
        profit_factor=pf_val,
        avg_win_loss_ratio=awl_val,
        max_consecutive_losses=mcl_val,
        sufficient_sample=sufficient,
        min_trade_count=min_trade_count,
    )


# ---------------------------------------------------------------------------
# Public API — compute_buy_and_hold_metrics
# ---------------------------------------------------------------------------


def compute_buy_and_hold_metrics(  # noqa: PLR0914
    equity_curve: EquityCurve,
    *,
    risk_free_rate: float = 0.0,
    lo_correction_lags: int = 6,
) -> BacktestMetrics:
    """Compute return and risk metrics for a buy-and-hold benchmark.

    Trade-level metrics are set to *None* because buy-and-hold has no
    discrete trades.  ``sufficient_sample`` is always ``True`` and
    ``n_trades`` is ``0``.

    Args:
        equity_curve: Time-indexed equity series (price-level or
            portfolio equity).
        risk_free_rate: Annualised risk-free rate.
        lo_correction_lags: Autocorrelation lags for Lo (2002) correction.

    Returns:
        Frozen :class:`BacktestMetrics` instance with trade metrics as
        *None*.
    """
    values: np.ndarray = np.asarray(equity_curve.values, dtype=np.float64)
    n_points: int = len(values)

    if n_points < _MIN_PERIODS_FOR_STATS:
        return BacktestMetrics(  # ty: ignore[missing-argument]
            n_trades=0,
            sufficient_sample=True,
            min_trade_count=1,
        )

    # -- Period returns ------------------------------------------------------
    returns: np.ndarray = values[1:] / values[:-1] - 1.0

    # -- Periods per year ----------------------------------------------------
    periods_per_year: float = _estimate_periods_per_year(equity_curve)

    # -- Return metrics ------------------------------------------------------
    total_ret: float = float(values[-1] / values[0] - 1.0)
    duration_days: float = _duration_days(equity_curve)
    ann_ret: float | None = _annualized_return(total_ret, duration_days)

    # -- Risk metrics --------------------------------------------------------
    max_dd: float = _max_drawdown(values)
    max_dd_dur: float = _max_drawdown_duration_days(equity_curve, values)
    ann_vol: float | None = _annualized_volatility(returns, periods_per_year)
    ds_vol: float | None = _downside_volatility(returns, periods_per_year)

    # -- Risk-adjusted metrics -----------------------------------------------
    sharpe: float | None = _sharpe_ratio(returns, risk_free_rate, periods_per_year)
    lo_factor: float | None = _lo_correction_factor(returns, lo_correction_lags)
    sharpe_lo: float | None = None
    if sharpe is not None and lo_factor is not None:
        sharpe_lo = sharpe * lo_factor
    sortino: float | None = _sortino_ratio(returns, risk_free_rate, periods_per_year)
    calmar: float | None = None
    if ann_ret is not None and max_dd < 0.0:
        calmar = ann_ret / abs(max_dd)

    return BacktestMetrics(
        total_return=total_ret,
        annualized_return=ann_ret,
        cagr=ann_ret,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_dur,
        annualized_volatility=ann_vol,
        downside_volatility=ds_vol,
        sharpe_ratio=sharpe,
        sharpe_ratio_lo_corrected=sharpe_lo,
        lo_correction_factor=lo_factor,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        n_trades=0,
        win_rate=None,
        profit_factor=None,
        avg_win_loss_ratio=None,
        max_consecutive_losses=None,
        sufficient_sample=True,
        min_trade_count=1,
    )


# ---------------------------------------------------------------------------
# Module-private helpers — return metrics
# ---------------------------------------------------------------------------


def _duration_days(equity_curve: EquityCurve) -> float:
    """Compute total duration of the equity curve in calendar days.

    Args:
        equity_curve: Equity curve with at least two timestamps.

    Returns:
        Duration in fractional days.
    """
    delta_seconds: float = (equity_curve.timestamps[-1] - equity_curve.timestamps[0]).total_seconds()
    return delta_seconds / 86_400.0


def _annualized_return(total_return: float, duration_days: float) -> float | None:
    """Compute annualised compound return.

    Returns *None* when the duration is zero (single-point curve).

    Args:
        total_return: Cumulative return (e.g. 0.10 for 10 %).
        duration_days: Duration in calendar days.

    Returns:
        Annualised return or *None*.
    """
    if duration_days <= 0.0:
        return None
    exponent: float = 365.25 / duration_days
    if total_return < -1.0:
        return -1.0
    ann: float = (1.0 + total_return) ** exponent - 1.0
    return ann


# ---------------------------------------------------------------------------
# Module-private helpers — risk metrics
# ---------------------------------------------------------------------------


def _max_drawdown(values: np.ndarray) -> float:
    """Compute maximum drawdown as a negative fraction.

    Args:
        values: Equity curve values array.

    Returns:
        Maximum drawdown (non-positive number).
    """
    running_max: np.ndarray = np.maximum.accumulate(values)
    drawdowns: np.ndarray = (values - running_max) / np.where(running_max > 0.0, running_max, 1.0)
    dd: float = float(np.min(drawdowns))
    return min(dd, 0.0)


def _max_drawdown_duration_days(
    equity_curve: EquityCurve,
    values: np.ndarray,
) -> float:
    """Compute the longest drawdown duration in calendar days.

    A drawdown period starts when equity drops below the running peak
    and ends when equity equals or exceeds the previous peak.

    Args:
        equity_curve: Equity curve (for timestamps).
        values: Equity curve values array.

    Returns:
        Longest drawdown duration in fractional days.
    """
    timestamps: list[float] = [ts.timestamp() for ts in equity_curve.timestamps]
    running_max: np.ndarray = np.maximum.accumulate(values)
    n: int = len(values)

    max_duration_seconds: float = 0.0
    dd_start_seconds: float | None = None

    for i in range(n):
        in_drawdown: bool = values[i] < running_max[i]
        if in_drawdown:
            if dd_start_seconds is None:
                dd_start_seconds = timestamps[i]
        elif dd_start_seconds is not None:
            duration: float = timestamps[i] - dd_start_seconds
            max_duration_seconds = max(max_duration_seconds, duration)
            dd_start_seconds = None

    # Handle ongoing drawdown at end of series
    if dd_start_seconds is not None:
        duration = timestamps[-1] - dd_start_seconds
        max_duration_seconds = max(max_duration_seconds, duration)

    return max_duration_seconds / 86_400.0


def _annualized_volatility(
    returns: np.ndarray,
    periods_per_year: float,
) -> float | None:
    """Compute annualised volatility of returns.

    Args:
        returns: Array of period returns.
        periods_per_year: Number of return periods per year.

    Returns:
        Annualised volatility or *None* if fewer than 2 returns.
    """
    if len(returns) < _MIN_PERIODS_FOR_STATS:
        return None
    vol: float = float(np.std(returns, ddof=1)) * np.sqrt(periods_per_year)
    return vol


def _downside_volatility(
    returns: np.ndarray,
    periods_per_year: float,
) -> float | None:
    """Compute annualised downside volatility (semi-deviation).

    Only returns below zero contribute.

    Args:
        returns: Array of period returns.
        periods_per_year: Number of return periods per year.

    Returns:
        Annualised downside volatility or *None* if no negative returns.
    """
    negative_returns: np.ndarray = returns[returns < 0.0]
    if len(negative_returns) < _MIN_PERIODS_FOR_STATS:
        return None
    ds_vol: float = float(np.std(negative_returns, ddof=1)) * np.sqrt(periods_per_year)
    return ds_vol


# ---------------------------------------------------------------------------
# Module-private helpers — risk-adjusted metrics
# ---------------------------------------------------------------------------


def _sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float,
    periods_per_year: float,
) -> float | None:
    """Compute annualised Sharpe ratio.

    Args:
        returns: Array of period returns.
        risk_free_rate: Annualised risk-free rate.
        periods_per_year: Number of return periods per year.

    Returns:
        Annualised Sharpe ratio or *None* when volatility is zero.
    """
    if len(returns) < _MIN_PERIODS_FOR_STATS:
        return None
    rf_per_period: float = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess: np.ndarray = returns - rf_per_period
    std: float = float(np.std(returns, ddof=1))
    if std == 0.0:
        return None
    sharpe: float = float(np.mean(excess)) / std * np.sqrt(periods_per_year)
    return sharpe


def _lo_correction_factor(
    returns: np.ndarray,
    q: int,
) -> float | None:
    """Compute Lo (2002) autocorrelation correction factor eta(q).

    The formula is::

        eta(q) = sqrt(q / (q + 2 * sum_{k=1}^{q} (q - k) * rho_k))

    Where ``rho_k`` is the k-th autocorrelation of returns.  The
    corrected Sharpe ratio is ``sharpe * eta(q)``.  When positive
    autocorrelation exists, ``eta(q) < 1`` (more conservative).

    Args:
        returns: Array of period returns.
        q: Number of autocorrelation lags.

    Returns:
        Correction factor or *None* when insufficient data or the
        denominator is non-positive.
    """
    n_ret: int = len(returns)
    if n_ret < q + 1:
        return None

    mean_r: float = float(np.mean(returns))
    demeaned: np.ndarray = returns - mean_r
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

    eta: float = np.sqrt(float(q) / denominator)
    return eta


def _sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float,
    periods_per_year: float,
) -> float | None:
    """Compute annualised Sortino ratio.

    Args:
        returns: Array of period returns.
        risk_free_rate: Annualised risk-free rate.
        periods_per_year: Number of return periods per year.

    Returns:
        Annualised Sortino ratio or *None* when downside vol is zero.
    """
    if len(returns) < _MIN_PERIODS_FOR_STATS:
        return None
    negative_returns: np.ndarray = returns[returns < 0.0]
    if len(negative_returns) < _MIN_PERIODS_FOR_STATS:
        return None
    ds_std: float = float(np.std(negative_returns, ddof=1))
    if ds_std == 0.0:
        return None
    rf_per_period: float = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess: np.ndarray = returns - rf_per_period
    sortino: float = float(np.mean(excess)) / ds_std * np.sqrt(periods_per_year)
    return sortino


# ---------------------------------------------------------------------------
# Module-private helpers — trade metrics
# ---------------------------------------------------------------------------


class _TradeStats(BaseModel, frozen=True):
    """Internal container for trade-level statistics."""

    win_rate: float
    profit_factor: float | None
    avg_win_loss_ratio: float | None
    max_consecutive_losses: int


def _compute_trade_metrics(trades: list[Trade]) -> _TradeStats:
    """Compute trade-level performance statistics.

    Called only when ``len(trades) >= min_trade_count``.

    Args:
        trades: List of completed trades.

    Returns:
        Typed trade statistics container.
    """
    n: int = len(trades)
    pnls: list[float] = [t.net_pnl for t in trades]

    winners: list[float] = [p for p in pnls if p > 0.0]
    losers: list[float] = [p for p in pnls if p <= 0.0]

    win_rate: float = len(winners) / n if n > 0 else 0.0

    gross_win: float = sum(winners)
    gross_loss: float = sum(losers)
    profit_factor: float | None = None
    if gross_loss < 0.0:
        profit_factor = gross_win / abs(gross_loss)

    avg_win_loss: float | None = None
    if len(winners) > 0 and len(losers) > 0:
        avg_win: float = gross_win / len(winners)
        avg_loss: float = abs(gross_loss / len(losers))
        if avg_loss > 0.0:
            avg_win_loss = avg_win / avg_loss

    max_consec: int = _max_consecutive_losses(pnls)

    return _TradeStats(
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win_loss_ratio=avg_win_loss,
        max_consecutive_losses=max_consec,
    )


def _max_consecutive_losses(pnls: list[float]) -> int:
    """Count the longest streak of consecutive losing trades.

    A losing trade has ``net_pnl <= 0``.

    Args:
        pnls: List of net P&L values for each trade.

    Returns:
        Length of the longest losing streak.
    """
    max_streak: int = 0
    current_streak: int = 0
    for pnl in pnls:
        if pnl <= 0.0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak


# ---------------------------------------------------------------------------
# Module-private helpers — time estimation
# ---------------------------------------------------------------------------


def _estimate_periods_per_year(equity_curve: EquityCurve) -> float:
    """Estimate the number of return periods per year from timestamp gaps.

    Uses the median gap between consecutive timestamps to avoid
    sensitivity to outlier gaps (e.g. exchange downtime, missing bars).

    Args:
        equity_curve: Equity curve with at least two timestamps.

    Returns:
        Estimated number of periods per year.
    """
    timestamps: list[float] = [ts.timestamp() for ts in equity_curve.timestamps]
    gaps: np.ndarray = np.diff(np.asarray(timestamps, dtype=np.float64))
    median_gap: float = float(np.median(gaps))
    if median_gap <= 0.0:
        median_gap = 1.0
    ppy: float = _SECONDS_PER_YEAR / median_gap
    return ppy


# ---------------------------------------------------------------------------
# Module-private helpers — empty metrics
# ---------------------------------------------------------------------------


def _empty_metrics(
    *,
    n_trades: int,
    min_trade_count: int,
) -> BacktestMetrics:
    """Build a metrics instance with all computed fields set to *None*.

    Used when the equity curve has fewer than two data points.

    Args:
        n_trades: Number of trades observed.
        min_trade_count: Minimum trade threshold.

    Returns:
        :class:`BacktestMetrics` with only metadata populated.
    """
    return BacktestMetrics(  # ty: ignore[missing-argument]
        n_trades=n_trades,
        sufficient_sample=n_trades >= min_trade_count,
        min_trade_count=min_trade_count,
    )
