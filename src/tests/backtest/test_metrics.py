"""Unit tests for backtest metrics: Sharpe, Lo correction, drawdown, trade stats."""

from __future__ import annotations

from datetime import datetime, timedelta, UTC

import numpy as np
import pytest

from src.app.backtest.application.metrics import (
    BacktestMetrics,
    compute_buy_and_hold_metrics,
    compute_metrics,
)
from src.app.backtest.domain.entities import EquityCurve, Trade
from src.app.backtest.domain.value_objects import Side
from src.app.ohlcv.domain.value_objects import Asset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_T0: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_ONE_HOUR: timedelta = timedelta(hours=1)
_ONE_DAY: timedelta = timedelta(days=1)
_BTC: Asset = Asset(symbol="BTCUSDT")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_equity_curve(values: list[float], start: datetime = _T0) -> EquityCurve:
    """Build an EquityCurve from a list of equity values at daily intervals.

    Uses daily spacing to avoid OverflowError in _annualized_return when the
    duration is sub-hour (exponent 365.25/duration_days would be enormous).
    """
    timestamps: list[datetime] = [start + i * _ONE_DAY for i in range(len(values))]
    return EquityCurve(timestamps=timestamps, values=values)


def _make_hourly_equity_curve(values: list[float], start: datetime = _T0) -> EquityCurve:
    """Build an EquityCurve at hourly intervals (for tests that need sub-day spacing)."""
    timestamps: list[datetime] = [start + i * _ONE_HOUR for i in range(len(values))]
    return EquityCurve(timestamps=timestamps, values=values)


def _make_trade(
    *,
    net_pnl: float,
    gross_pnl: float | None = None,
    entry_price: float = 40_000.0,
    exit_price: float = 41_000.0,
    side: Side = Side.LONG,
    size: float = 0.25,
    offset_hours: int = 0,
) -> Trade:
    """Build a minimal Trade for metrics tests."""
    gp: float = gross_pnl if gross_pnl is not None else net_pnl
    comm: float = max(0.0, gp - net_pnl)
    entry_time: datetime = _T0 + timedelta(hours=offset_hours)
    exit_time: datetime = entry_time + _ONE_HOUR
    return Trade(
        asset=_BTC,
        side=side,
        size=size,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_time=entry_time,
        exit_time=exit_time,
        gross_pnl=gp,
        net_pnl=net_pnl,
        commission_paid=comm,
    )


# ---------------------------------------------------------------------------
# TestComputeMetrics — return metrics
# ---------------------------------------------------------------------------


class TestComputeMetricsReturn:
    """Tests for return metric calculations."""

    def test_total_return_flat_equity(self) -> None:
        """Flat equity curve produces zero total return."""
        ec: EquityCurve = _make_equity_curve([100_000.0, 100_000.0, 100_000.0])
        metrics: BacktestMetrics = compute_metrics(ec, [])
        assert metrics.total_return == pytest.approx(0.0)

    def test_total_return_doubling(self) -> None:
        """Equity doubling produces total_return == 1.0 (100%)."""
        ec: EquityCurve = _make_equity_curve([100_000.0, 200_000.0])
        metrics: BacktestMetrics = compute_metrics(ec, [])
        assert metrics.total_return == pytest.approx(1.0)

    def test_total_return_halving(self) -> None:
        """Equity halving produces total_return == -0.5."""
        ec: EquityCurve = _make_equity_curve([100_000.0, 50_000.0])
        metrics: BacktestMetrics = compute_metrics(ec, [])
        assert metrics.total_return == pytest.approx(-0.5)

    def test_single_point_returns_none_for_computed_metrics(self) -> None:
        """Single-point equity curve returns None for all computed metrics."""
        ec: EquityCurve = _make_equity_curve([100_000.0])
        metrics: BacktestMetrics = compute_metrics(ec, [])
        assert metrics.total_return is None
        assert metrics.sharpe_ratio is None
        assert metrics.max_drawdown is None

    def test_cagr_equals_annualized_return(self) -> None:
        """cagr field is always equal to annualized_return."""
        ec: EquityCurve = _make_equity_curve([100_000.0, 110_000.0, 105_000.0])
        metrics: BacktestMetrics = compute_metrics(ec, [])
        assert metrics.cagr == metrics.annualized_return


# ---------------------------------------------------------------------------
# TestComputeMetrics — max drawdown
# ---------------------------------------------------------------------------


class TestComputeMetricsDrawdown:
    """Tests for drawdown metric calculations."""

    def test_no_drawdown_on_monotone_increasing_equity(self) -> None:
        """Monotone increasing equity has max drawdown == 0."""
        ec: EquityCurve = _make_equity_curve([100_000.0, 101_000.0, 102_000.0, 103_000.0])
        metrics: BacktestMetrics = compute_metrics(ec, [])
        assert metrics.max_drawdown == pytest.approx(0.0)

    def test_drawdown_known_value(self) -> None:
        """Max drawdown computed correctly for a known equity sequence."""
        # peak=120, trough=90 → drawdown = (90-120)/120 = -0.25
        ec: EquityCurve = _make_equity_curve([100_000.0, 120_000.0, 90_000.0, 100_000.0])
        metrics: BacktestMetrics = compute_metrics(ec, [])
        assert metrics.max_drawdown is not None
        assert metrics.max_drawdown == pytest.approx(-0.25, rel=1e-6)

    def test_max_drawdown_is_non_positive(self) -> None:
        """max_drawdown is always <= 0."""
        ec: EquityCurve = _make_equity_curve([100_000.0, 90_000.0, 80_000.0, 95_000.0])
        metrics: BacktestMetrics = compute_metrics(ec, [])
        assert metrics.max_drawdown is not None
        assert metrics.max_drawdown <= 0.0

    def test_max_drawdown_duration_days_positive(self) -> None:
        """max_drawdown_duration_days is positive when there is a drawdown."""
        ec: EquityCurve = _make_equity_curve([100_000.0, 120_000.0, 90_000.0, 90_000.0, 115_000.0])
        metrics: BacktestMetrics = compute_metrics(ec, [])
        assert metrics.max_drawdown_duration_days is not None
        assert metrics.max_drawdown_duration_days > 0.0


# ---------------------------------------------------------------------------
# TestComputeMetrics — Sharpe and Lo correction
# ---------------------------------------------------------------------------


class TestSharpeAndLoCorrection:
    """Tests for Sharpe ratio and Lo (2002) autocorrelation correction."""

    def test_sharpe_none_for_constant_returns(self) -> None:
        """Constant equity (zero variance) returns None Sharpe."""
        ec: EquityCurve = _make_equity_curve([100_000.0] * 20)
        metrics: BacktestMetrics = compute_metrics(ec, [])
        assert metrics.sharpe_ratio is None

    def test_positive_sharpe_for_trending_up(self) -> None:
        """Steadily rising equity yields a positive Sharpe ratio."""
        ec: EquityCurve = _make_equity_curve([100_000.0 + i * 100.0 for i in range(50)])
        metrics: BacktestMetrics = compute_metrics(ec, [])
        assert metrics.sharpe_ratio is not None
        assert metrics.sharpe_ratio > 0.0

    def test_negative_sharpe_for_trending_down(self) -> None:
        """Steadily falling equity yields a negative Sharpe ratio."""
        ec: EquityCurve = _make_equity_curve([100_000.0 - i * 100.0 for i in range(50) if 100_000.0 - i * 100.0 > 0])
        metrics: BacktestMetrics = compute_metrics(ec, [])
        assert metrics.sharpe_ratio is not None
        assert metrics.sharpe_ratio < 0.0

    def test_lo_correction_factor_less_than_one_for_positive_autocorr(self) -> None:
        """Positive autocorrelation produces lo_correction_factor < 1.0."""
        # Generate AR(1) returns with phi=0.8 (strong positive autocorrelation)
        rng: np.random.Generator = np.random.default_rng(42)
        n: int = 300
        phi: float = 0.8
        noise: np.ndarray = rng.normal(0.0, 0.001, n)
        returns: np.ndarray = np.zeros(n)
        for i in range(1, n):
            returns[i] = phi * returns[i - 1] + noise[i]
        # Convert to equity curve
        equity: list[float] = [100_000.0]
        for r in returns[1:]:
            equity.append(equity[-1] * (1.0 + r))
        ec: EquityCurve = _make_equity_curve(equity)
        metrics: BacktestMetrics = compute_metrics(ec, [], lo_correction_lags=6)

        assert metrics.lo_correction_factor is not None
        assert metrics.lo_correction_factor < 1.0

    def test_lo_corrected_sharpe_less_than_naive_for_positive_autocorr(self) -> None:
        """With positive autocorrelation, Lo-corrected Sharpe < naive Sharpe."""
        rng: np.random.Generator = np.random.default_rng(99)
        n: int = 300
        phi: float = 0.7
        noise: np.ndarray = rng.normal(0.0005, 0.001, n)
        returns: np.ndarray = np.zeros(n)
        returns[0] = noise[0]
        for i in range(1, n):
            returns[i] = phi * returns[i - 1] + noise[i]
        equity: list[float] = [100_000.0]
        for r in returns[1:]:
            equity.append(equity[-1] * (1.0 + r))
        ec: EquityCurve = _make_equity_curve(equity)
        metrics: BacktestMetrics = compute_metrics(ec, [], lo_correction_lags=6)

        assert metrics.sharpe_ratio is not None
        assert metrics.sharpe_ratio_lo_corrected is not None
        assert metrics.sharpe_ratio_lo_corrected < metrics.sharpe_ratio

    def test_lo_correction_factor_near_one_for_iid_returns(self) -> None:
        """IID returns have Lo correction factor close to 1.0."""
        rng: np.random.Generator = np.random.default_rng(0)
        n: int = 500
        returns: np.ndarray = rng.normal(0.0002, 0.001, n)
        equity: list[float] = [100_000.0]
        for r in returns:
            equity.append(equity[-1] * (1.0 + r))
        ec: EquityCurve = _make_equity_curve(equity)
        metrics: BacktestMetrics = compute_metrics(ec, [], lo_correction_lags=6)

        assert metrics.lo_correction_factor is not None
        # IID returns have near-zero autocorrelation → eta ≈ 1
        assert metrics.lo_correction_factor == pytest.approx(1.0, abs=0.2)


# ---------------------------------------------------------------------------
# TestComputeMetrics — trade statistics
# ---------------------------------------------------------------------------


class TestTradeStatistics:
    """Tests for trade-level metrics (win rate, profit factor, etc.)."""

    def test_trade_metrics_none_below_min_trade_count(self) -> None:
        """Trade metrics are None when trades < min_trade_count."""
        ec: EquityCurve = _make_equity_curve([100_000.0, 101_000.0])
        trades: list[Trade] = [_make_trade(net_pnl=500.0)]
        metrics: BacktestMetrics = compute_metrics(ec, trades, min_trade_count=30)
        assert metrics.win_rate is None
        assert metrics.profit_factor is None

    def test_win_rate_all_winners(self) -> None:
        """All winning trades → win_rate == 1.0."""
        ec: EquityCurve = _make_equity_curve([100_000.0 + i * 100 for i in range(35)])
        trades: list[Trade] = [_make_trade(net_pnl=100.0, offset_hours=i) for i in range(30)]
        metrics: BacktestMetrics = compute_metrics(ec, trades, min_trade_count=30)
        assert metrics.win_rate is not None
        assert metrics.win_rate == pytest.approx(1.0)

    def test_win_rate_all_losers(self) -> None:
        """All losing trades → win_rate == 0.0."""
        ec: EquityCurve = _make_equity_curve([100_000.0 - i * 100 for i in range(35) if 100_000.0 - i * 100 > 0])
        trades: list[Trade] = [_make_trade(net_pnl=-100.0, offset_hours=i) for i in range(30)]
        metrics: BacktestMetrics = compute_metrics(ec, trades, min_trade_count=30)
        assert metrics.win_rate is not None
        assert metrics.win_rate == pytest.approx(0.0)

    def test_win_rate_half_and_half(self) -> None:
        """Equal winners and losers → win_rate == 0.5."""
        ec: EquityCurve = _make_equity_curve([100_000.0] * 35)
        winners: list[Trade] = [_make_trade(net_pnl=100.0, offset_hours=i * 2) for i in range(15)]
        losers: list[Trade] = [_make_trade(net_pnl=-100.0, offset_hours=i * 2 + 1) for i in range(15)]
        trades: list[Trade] = winners + losers
        metrics: BacktestMetrics = compute_metrics(ec, trades, min_trade_count=30)
        assert metrics.win_rate is not None
        assert metrics.win_rate == pytest.approx(0.5)

    def test_profit_factor_known_value(self) -> None:
        """Profit factor computed correctly: gross_wins / abs(gross_losses)."""
        ec: EquityCurve = _make_equity_curve([100_000.0] * 35)
        # 20 winners at +200 each, 10 losers at -100 each
        win_trades: list[Trade] = [_make_trade(net_pnl=200.0, offset_hours=i) for i in range(20)]
        loss_trades: list[Trade] = [_make_trade(net_pnl=-100.0, offset_hours=20 + i) for i in range(10)]
        trades: list[Trade] = win_trades + loss_trades
        metrics: BacktestMetrics = compute_metrics(ec, trades, min_trade_count=30)
        assert metrics.profit_factor is not None
        expected_pf: float = (20 * 200.0) / (10 * 100.0)  # 4.0
        assert metrics.profit_factor == pytest.approx(expected_pf)

    def test_max_consecutive_losses_known_sequence(self) -> None:
        """max_consecutive_losses computed correctly for known PnL sequence."""
        ec: EquityCurve = _make_equity_curve([100_000.0] * 35)
        # W L L L W W L W L L  (3 consecutive losses is the max run)
        pnls: list[float] = [100, -50, -50, -50, 100, 100, -50, 100, -50, -50]
        pnls.extend([100.0] * 20)  # Pad to reach min_trade_count=30
        trades: list[Trade] = [_make_trade(net_pnl=p, offset_hours=i) for i, p in enumerate(pnls)]
        metrics: BacktestMetrics = compute_metrics(ec, trades, min_trade_count=30)
        assert metrics.max_consecutive_losses is not None
        assert metrics.max_consecutive_losses == 3

    def test_n_trades_count_matches(self) -> None:
        """n_trades matches the length of the trades list."""
        ec: EquityCurve = _make_equity_curve([100_000.0, 101_000.0])
        n: int = 7
        trades: list[Trade] = [_make_trade(net_pnl=50.0, offset_hours=i) for i in range(n)]
        metrics: BacktestMetrics = compute_metrics(ec, trades)
        assert metrics.n_trades == n

    def test_sufficient_sample_true_at_threshold(self) -> None:
        """sufficient_sample is True when n_trades == min_trade_count."""
        ec: EquityCurve = _make_equity_curve([100_000.0] * 32)
        min_count: int = 30
        trades: list[Trade] = [_make_trade(net_pnl=10.0, offset_hours=i) for i in range(min_count)]
        metrics: BacktestMetrics = compute_metrics(ec, trades, min_trade_count=min_count)
        assert metrics.sufficient_sample is True


# ---------------------------------------------------------------------------
# TestComputeBuyAndHoldMetrics
# ---------------------------------------------------------------------------


class TestComputeBuyAndHoldMetrics:
    """Tests for compute_buy_and_hold_metrics."""

    def test_buy_and_hold_n_trades_zero(self) -> None:
        """Buy-and-hold metrics always have n_trades == 0."""
        ec: EquityCurve = _make_equity_curve([100_000.0, 110_000.0, 105_000.0])
        metrics: BacktestMetrics = compute_buy_and_hold_metrics(ec)
        assert metrics.n_trades == 0

    def test_buy_and_hold_sufficient_sample_always_true(self) -> None:
        """sufficient_sample is always True for buy-and-hold."""
        ec: EquityCurve = _make_equity_curve([100_000.0, 110_000.0])
        metrics: BacktestMetrics = compute_buy_and_hold_metrics(ec)
        assert metrics.sufficient_sample is True

    def test_buy_and_hold_trade_metrics_are_none(self) -> None:
        """Trade-level metrics are always None for buy-and-hold."""
        ec: EquityCurve = _make_equity_curve([100_000.0 + i * 500 for i in range(20)])
        metrics: BacktestMetrics = compute_buy_and_hold_metrics(ec)
        assert metrics.win_rate is None
        assert metrics.profit_factor is None
        assert metrics.avg_win_loss_ratio is None
        assert metrics.max_consecutive_losses is None

    def test_buy_and_hold_total_return_consistent(self) -> None:
        """Buy-and-hold total return matches equivalent strategy run."""
        values: list[float] = [100_000.0, 105_000.0, 110_000.0]
        ec: EquityCurve = _make_equity_curve(values)
        bh_metrics: BacktestMetrics = compute_buy_and_hold_metrics(ec)
        strat_metrics: BacktestMetrics = compute_metrics(ec, [])
        assert bh_metrics.total_return == pytest.approx(strat_metrics.total_return or 0.0)

    def test_single_point_buy_and_hold_returns_empty_metrics(self) -> None:
        """Single-point equity curve returns minimal buy-and-hold metrics."""
        ec: EquityCurve = _make_equity_curve([100_000.0])
        metrics: BacktestMetrics = compute_buy_and_hold_metrics(ec)
        assert metrics.n_trades == 0
        assert metrics.sufficient_sample is True
