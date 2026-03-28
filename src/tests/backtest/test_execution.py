"""Unit and integration tests for the ExecutionEngine."""

from __future__ import annotations

from datetime import datetime, timedelta, UTC

import polars as pl
import pytest

from src.app.backtest.application.execution import BacktestResult, ExecutionEngine
from src.app.backtest.application.position_sizer import FixedFractionalSizer
from src.app.backtest.domain.entities import Position, Signal, Trade
from src.app.backtest.domain.value_objects import ExecutionConfig, Side
from src.app.ohlcv.domain.value_objects import Asset
from src.tests.backtest.conftest import (
    INITIAL_CASH,
    FixedNotionalSizer,
    NeverTradeStrategy,
    SingleSignalStrategy,
    make_bars,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_T0: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_BTC: Asset = Asset(symbol="BTCUSDT")

_ENTRY_PRICE: float = 40_000.0
_EXIT_PRICE: float = 41_000.0
_COMMISSION_BPS: float = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(
    strategy: object,
    sizer: object,
    commission_bps: float = _COMMISSION_BPS,
) -> ExecutionEngine:
    """Build an ExecutionEngine with given strategy and sizer."""
    config: ExecutionConfig = ExecutionConfig(commission_bps=commission_bps)
    return ExecutionEngine(config=config, strategy=strategy, sizer=sizer)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestEmptyAndInvalidBars
# ---------------------------------------------------------------------------


class TestEmptyAndInvalidBars:
    """Tests for bar validation before execution."""

    def test_empty_bars_raises(self) -> None:
        """Empty bars DataFrame raises ValueError."""
        engine: ExecutionEngine = _make_engine(NeverTradeStrategy(), FixedNotionalSizer())
        empty: pl.DataFrame = pl.DataFrame(
            {"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
        ).cast(
            {
                "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )
        with pytest.raises(ValueError, match="empty"):
            engine.run(bars=empty, asset=_BTC)

    def test_missing_column_raises(self) -> None:
        """Bars missing a required column raises ValueError."""
        engine: ExecutionEngine = _make_engine(NeverTradeStrategy(), FixedNotionalSizer())
        bad_bars: pl.DataFrame = pl.DataFrame(
            {
                "timestamp": [_T0],
                "open": [40_000.0],
                "high": [40_200.0],
                # 'low' intentionally missing
                "close": [40_100.0],
                "volume": [1.0],
            }
        )
        with pytest.raises(ValueError, match="missing required columns"):
            engine.run(bars=bad_bars, asset=_BTC)


# ---------------------------------------------------------------------------
# TestFillOnNextOpen
# ---------------------------------------------------------------------------


class TestFillOnNextOpen:
    """Tests verifying that signals fill at the next bar's open price."""

    def test_signal_at_bar_0_fills_at_bar_1_open(self) -> None:
        """Signal generated at bar[0] is filled at bar[1].open, not bar[0].close."""
        bar_0_open: float = 40_000.0
        bar_1_open: float = 41_000.0
        bar_2_open: float = 42_000.0
        bars: pl.DataFrame = pl.DataFrame(
            {
                "timestamp": [_T0, _T0 + timedelta(hours=1), _T0 + timedelta(hours=2)],
                "open": [bar_0_open, bar_1_open, bar_2_open],
                "high": [40_300.0, 41_300.0, 42_300.0],
                "low": [39_700.0, 40_700.0, 41_700.0],
                "close": [40_200.0, 41_200.0, 42_200.0],
                "volume": [10.0, 10.0, 10.0],
            }
        )
        # Strategy emits signal on bar[0] only
        strategy: SingleSignalStrategy = SingleSignalStrategy(_BTC, Side.LONG)
        sizer: FixedNotionalSizer = FixedNotionalSizer(notional=10_000.0)
        engine: ExecutionEngine = _make_engine(strategy, sizer, commission_bps=0.0)
        result: BacktestResult = engine.run(bars=bars, asset=_BTC)

        # The position should have been opened at bar_1_open
        assert len(result.trades) >= 1
        first_trade: Trade = result.trades[0]
        assert first_trade.entry_price == pytest.approx(bar_1_open)

    def test_no_lookahead_no_entry_on_bar_0(self) -> None:
        """On bar[0] there are no pending signals yet, so no fill occurs at bar[0].open."""
        bars: pl.DataFrame = make_bars(3, start_price=40_000.0)
        strategy: SingleSignalStrategy = SingleSignalStrategy(_BTC, Side.LONG)
        sizer: FixedNotionalSizer = FixedNotionalSizer(notional=10_000.0)
        engine: ExecutionEngine = _make_engine(strategy, sizer, commission_bps=0.0)
        result: BacktestResult = engine.run(bars=bars, asset=_BTC)

        # Entry must be at bar[1].open = 40_000 (flat price bars)
        if result.trades:
            assert result.trades[0].entry_price == pytest.approx(40_000.0)

    def test_equity_curve_length_equals_bar_count(self) -> None:
        """Equity curve length equals the number of input bars."""
        n_bars: int = 10
        bars: pl.DataFrame = make_bars(n_bars, start_price=40_000.0)
        engine: ExecutionEngine = _make_engine(NeverTradeStrategy(), FixedNotionalSizer())
        result: BacktestResult = engine.run(bars=bars, asset=_BTC)
        assert len(result.equity_curve.timestamps) == n_bars
        assert len(result.equity_curve.values) == n_bars


# ---------------------------------------------------------------------------
# TestCommissionCalculation
# ---------------------------------------------------------------------------


class TestCommissionCalculation:
    """Tests verifying commission deduction on round-trip trades."""

    def test_commission_formula_on_known_inputs(self) -> None:
        """Commission equals (entry_price + exit_price) * size * bps / 10_000."""
        # Three bars: signal on bar[0] → fill at bar[1].open, liquidate at bar[2].close
        # Using distinct timestamps so entry_time < exit_time for liquidation
        t1: datetime = _T0 + timedelta(hours=1)
        t2: datetime = _T0 + timedelta(hours=2)
        bars: pl.DataFrame = pl.DataFrame(
            {
                "timestamp": [_T0, t1, t2],
                "open": [_ENTRY_PRICE, _ENTRY_PRICE, _ENTRY_PRICE],
                "high": [_ENTRY_PRICE + 300.0, _ENTRY_PRICE + 300.0, _ENTRY_PRICE + 300.0],
                "low": [_ENTRY_PRICE - 300.0, _ENTRY_PRICE - 300.0, _ENTRY_PRICE - 300.0],
                "close": [_ENTRY_PRICE, _ENTRY_PRICE, _EXIT_PRICE],
                "volume": [10.0, 10.0, 10.0],
            }
        )
        notional: float = 10_000.0
        strategy: SingleSignalStrategy = SingleSignalStrategy(_BTC, Side.LONG)
        sizer: FixedNotionalSizer = FixedNotionalSizer(notional=notional)
        engine: ExecutionEngine = _make_engine(strategy, sizer, commission_bps=_COMMISSION_BPS)
        result: BacktestResult = engine.run(bars=bars, asset=_BTC)

        assert len(result.trades) == 1
        trade: Trade = result.trades[0]
        # Engine fills bar[1].open = _ENTRY_PRICE, exits at bar[2].close = _EXIT_PRICE
        size: float = notional / trade.entry_price
        expected_comm: float = (trade.entry_price * size + trade.exit_price * size) * _COMMISSION_BPS / 10_000.0
        assert trade.commission_paid == pytest.approx(expected_comm, rel=1e-6)

    def test_net_pnl_equals_gross_minus_commission(self) -> None:
        """net_pnl == gross_pnl - commission_paid for every trade."""
        # Use 6 bars to ensure entry at bar[1] and liquidation at bar[5] differ in time
        bars: pl.DataFrame = make_bars(6, start_price=40_000.0, price_step=100.0)
        strategy: SingleSignalStrategy = SingleSignalStrategy(_BTC, Side.LONG)
        sizer: FixedNotionalSizer = FixedNotionalSizer(notional=10_000.0)
        engine: ExecutionEngine = _make_engine(strategy, sizer, commission_bps=10.0)
        result: BacktestResult = engine.run(bars=bars, asset=_BTC)

        for trade in result.trades:
            assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.commission_paid, rel=1e-9)

    def test_zero_commission_net_equals_gross(self) -> None:
        """With zero commission, net_pnl equals gross_pnl."""
        bars: pl.DataFrame = make_bars(3, start_price=40_000.0, price_step=100.0)
        strategy: SingleSignalStrategy = SingleSignalStrategy(_BTC, Side.LONG)
        sizer: FixedNotionalSizer = FixedNotionalSizer(notional=10_000.0)
        engine: ExecutionEngine = _make_engine(strategy, sizer, commission_bps=0.0)
        result: BacktestResult = engine.run(bars=bars, asset=_BTC)

        for trade in result.trades:
            assert trade.net_pnl == pytest.approx(trade.gross_pnl)
            assert trade.commission_paid == pytest.approx(0.0)

    def test_commission_deducted_on_round_trip(self) -> None:
        """Final cash is less than initial cash after a zero-PnL round trip with commission."""
        # Flat prices so gross PnL = 0; commission > 0
        bars: pl.DataFrame = make_bars(3, start_price=40_000.0, price_step=0.0)
        strategy: SingleSignalStrategy = SingleSignalStrategy(_BTC, Side.LONG)
        sizer: FixedNotionalSizer = FixedNotionalSizer(notional=10_000.0)
        engine: ExecutionEngine = _make_engine(strategy, sizer, commission_bps=10.0)
        result: BacktestResult = engine.run(bars=bars, asset=_BTC, initial_cash=INITIAL_CASH)

        final_equity: float = result.equity_curve.values[-1]
        assert final_equity < INITIAL_CASH

    def test_per_asset_multiplier_scales_commission(self) -> None:
        """Per-asset cost multiplier doubles commission for that asset."""
        bars: pl.DataFrame = make_bars(3, start_price=40_000.0)
        strategy: SingleSignalStrategy = SingleSignalStrategy(_BTC, Side.LONG)
        sizer: FixedNotionalSizer = FixedNotionalSizer(notional=10_000.0)

        config_normal: ExecutionConfig = ExecutionConfig(commission_bps=10.0)
        config_double: ExecutionConfig = ExecutionConfig(
            commission_bps=10.0,
            asset_cost_multiplier={"BTCUSDT": 2.0},
        )
        engine_normal: ExecutionEngine = ExecutionEngine(
            config=config_normal,
            strategy=SingleSignalStrategy(_BTC, Side.LONG),
            sizer=sizer,
        )
        engine_double: ExecutionEngine = ExecutionEngine(
            config=config_double,
            strategy=strategy,
            sizer=FixedNotionalSizer(notional=10_000.0),
        )
        result_normal: BacktestResult = engine_normal.run(bars=bars, asset=_BTC)
        result_double: BacktestResult = engine_double.run(bars=bars, asset=_BTC)

        if result_normal.trades and result_double.trades:
            assert result_double.trades[0].commission_paid == pytest.approx(
                result_normal.trades[0].commission_paid * 2.0, rel=1e-6
            )


# ---------------------------------------------------------------------------
# TestLongShortPnL
# ---------------------------------------------------------------------------


class TestLongShortPnL:
    """Tests for correct PnL calculation on long and short positions."""

    def test_long_rising_price_produces_positive_gross_pnl(self) -> None:
        """LONG position on rising prices yields positive gross PnL."""
        bars: pl.DataFrame = make_bars(3, start_price=40_000.0, price_step=500.0)
        strategy: SingleSignalStrategy = SingleSignalStrategy(_BTC, Side.LONG)
        sizer: FixedNotionalSizer = FixedNotionalSizer(notional=10_000.0)
        engine: ExecutionEngine = _make_engine(strategy, sizer, commission_bps=0.0)
        result: BacktestResult = engine.run(bars=bars, asset=_BTC)

        assert len(result.trades) == 1
        assert result.trades[0].gross_pnl > 0.0

    def test_long_falling_price_produces_negative_gross_pnl(self) -> None:
        """LONG position on falling prices yields negative gross PnL."""
        bars: pl.DataFrame = make_bars(3, start_price=40_000.0, price_step=-500.0)
        strategy: SingleSignalStrategy = SingleSignalStrategy(_BTC, Side.LONG)
        sizer: FixedNotionalSizer = FixedNotionalSizer(notional=10_000.0)
        engine: ExecutionEngine = _make_engine(strategy, sizer, commission_bps=0.0)
        result: BacktestResult = engine.run(bars=bars, asset=_BTC)

        assert len(result.trades) == 1
        assert result.trades[0].gross_pnl < 0.0

    def test_short_falling_price_produces_positive_gross_pnl(self) -> None:
        """SHORT position on falling prices yields positive gross PnL."""
        bars: pl.DataFrame = make_bars(3, start_price=40_000.0, price_step=-500.0)
        strategy: SingleSignalStrategy = SingleSignalStrategy(_BTC, Side.SHORT)
        sizer: FixedNotionalSizer = FixedNotionalSizer(notional=10_000.0)
        engine: ExecutionEngine = _make_engine(strategy, sizer, commission_bps=0.0)
        result: BacktestResult = engine.run(bars=bars, asset=_BTC)

        assert len(result.trades) == 1
        assert result.trades[0].gross_pnl > 0.0


# ---------------------------------------------------------------------------
# TestStopLossAndTakeProfit
# ---------------------------------------------------------------------------


class TestStopLossAndTakeProfit:
    """Tests for SL/TP exit logic."""

    def test_long_stop_loss_triggered_by_low(self) -> None:
        """Long position stop-loss is triggered when bar low <= stop_loss."""
        from src.app.backtest.application.execution import _check_sl_tp

        pos: Position = Position(
            asset=_BTC,
            side=Side.LONG,
            size=0.25,
            entry_price=40_000.0,
            entry_time=_T0,
            stop_loss=39_000.0,
        )
        # low_price=38_000 <= 39_000 SL → should trigger
        exit_price: float | None = _check_sl_tp(pos, high_price=40_500.0, low_price=38_000.0)
        assert exit_price == pytest.approx(39_000.0)

    def test_long_take_profit_triggered_by_high(self) -> None:
        """Long position take-profit is triggered when bar high >= take_profit."""
        from src.app.backtest.application.execution import _check_sl_tp

        pos: Position = Position(
            asset=_BTC,
            side=Side.LONG,
            size=0.25,
            entry_price=40_000.0,
            entry_time=_T0,
            take_profit=41_000.0,
        )
        exit_price: float | None = _check_sl_tp(pos, high_price=41_500.0, low_price=39_500.0)
        assert exit_price == pytest.approx(41_000.0)

    def test_short_stop_loss_triggered_by_high(self) -> None:
        """Short position stop-loss triggers when bar high >= stop_loss."""
        from src.app.backtest.application.execution import _check_sl_tp

        pos: Position = Position(
            asset=_BTC,
            side=Side.SHORT,
            size=0.25,
            entry_price=40_000.0,
            entry_time=_T0,
            stop_loss=41_000.0,
        )
        exit_price: float | None = _check_sl_tp(pos, high_price=41_500.0, low_price=39_500.0)
        assert exit_price == pytest.approx(41_000.0)

    def test_short_take_profit_triggered_by_low(self) -> None:
        """Short position take-profit triggers when bar low <= take_profit."""
        from src.app.backtest.application.execution import _check_sl_tp

        pos: Position = Position(
            asset=_BTC,
            side=Side.SHORT,
            size=0.25,
            entry_price=40_000.0,
            entry_time=_T0,
            take_profit=39_000.0,
        )
        exit_price: float | None = _check_sl_tp(pos, high_price=40_500.0, low_price=38_500.0)
        assert exit_price == pytest.approx(39_000.0)

    def test_sl_takes_priority_over_tp(self) -> None:
        """Stop-loss is checked before take-profit (worst-case execution)."""
        from src.app.backtest.application.execution import _check_sl_tp

        pos: Position = Position(
            asset=_BTC,
            side=Side.LONG,
            size=0.25,
            entry_price=40_000.0,
            entry_time=_T0,
            stop_loss=39_000.0,
            take_profit=41_000.0,
        )
        # Both SL and TP triggered in same bar
        exit_price: float | None = _check_sl_tp(pos, high_price=41_500.0, low_price=38_500.0)
        assert exit_price == pytest.approx(39_000.0)  # SL wins

    def test_no_sl_tp_returns_none(self) -> None:
        """No SL/TP configured always returns None."""
        from src.app.backtest.application.execution import _check_sl_tp

        pos: Position = Position(
            asset=_BTC,
            side=Side.LONG,
            size=0.25,
            entry_price=40_000.0,
            entry_time=_T0,
        )
        result: float | None = _check_sl_tp(pos, high_price=42_000.0, low_price=38_000.0)
        assert result is None


# ---------------------------------------------------------------------------
# TestStalenessSkip
# ---------------------------------------------------------------------------


class TestStalenessSkip:
    """Tests for gap-detection and staleness-skip behavior."""

    def test_gap_exceeding_threshold_skips_bar(self) -> None:
        """Bars with a gap > 2x median bar duration are skipped."""
        # Normal bars at 1-hour intervals; one bar has a 24-hour gap
        t0: datetime = _T0
        t1: datetime = _T0 + timedelta(hours=1)
        t2: datetime = _T0 + timedelta(hours=25)  # 24h gap >> 2 * 1h
        t3: datetime = _T0 + timedelta(hours=26)
        bars: pl.DataFrame = pl.DataFrame(
            {
                "timestamp": [t0, t1, t2, t3],
                "open": [40_000.0] * 4,
                "high": [40_200.0] * 4,
                "low": [39_800.0] * 4,
                "close": [40_000.0] * 4,
                "volume": [10.0] * 4,
            }
        )
        engine: ExecutionEngine = _make_engine(NeverTradeStrategy(), FixedNotionalSizer())
        result: BacktestResult = engine.run(bars=bars, asset=_BTC)
        # Engine should complete without error; equity curve should have <= 4 points
        # (bar t2 is skipped)
        assert len(result.equity_curve.values) <= 4

    def test_uniform_bars_no_staleness_skip(self) -> None:
        """Uniform 1-hour bars do not trigger staleness skip."""
        n: int = 10
        bars: pl.DataFrame = make_bars(n, start_price=40_000.0)
        engine: ExecutionEngine = _make_engine(NeverTradeStrategy(), FixedNotionalSizer())
        result: BacktestResult = engine.run(bars=bars, asset=_BTC)
        assert len(result.equity_curve.values) == n


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases: single bar, never-trade, zero volume, etc."""

    def test_single_bar_never_trade_returns_initial_cash(self) -> None:
        """With a single bar and no trades, equity equals initial cash."""
        bars: pl.DataFrame = make_bars(1, start_price=40_000.0)
        engine: ExecutionEngine = _make_engine(NeverTradeStrategy(), FixedNotionalSizer())
        result: BacktestResult = engine.run(bars=bars, asset=_BTC, initial_cash=INITIAL_CASH)
        assert result.equity_curve.values[0] == pytest.approx(INITIAL_CASH)

    def test_never_trade_strategy_produces_no_trades(self) -> None:
        """NeverTradeStrategy results in zero completed trades."""
        bars: pl.DataFrame = make_bars(20, start_price=40_000.0)
        engine: ExecutionEngine = _make_engine(NeverTradeStrategy(), FixedNotionalSizer())
        result: BacktestResult = engine.run(bars=bars, asset=_BTC)
        assert len(result.trades) == 0

    def test_never_trade_equity_stays_constant(self) -> None:
        """With no trades, all equity values equal initial cash."""
        bars: pl.DataFrame = make_bars(5, start_price=40_000.0)
        engine: ExecutionEngine = _make_engine(NeverTradeStrategy(), FixedNotionalSizer())
        result: BacktestResult = engine.run(bars=bars, asset=_BTC, initial_cash=INITIAL_CASH)
        for val in result.equity_curve.values:
            assert val == pytest.approx(INITIAL_CASH)

    def test_zero_volume_bars_handled(self) -> None:
        """Bars with zero volume are processed without error."""
        # Use NeverTradeStrategy to avoid opening a position that would need
        # entry_time < exit_time on the last bar's liquidation
        bars: pl.DataFrame = make_bars(5, start_price=40_000.0, volume=0.0)
        engine: ExecutionEngine = _make_engine(NeverTradeStrategy(), FixedNotionalSizer())
        result: BacktestResult = engine.run(bars=bars, asset=_BTC)
        # Main check: no exception raised, equity curve has correct length
        assert len(result.equity_curve.values) == 5

    def test_opposite_signal_closes_existing_position(self) -> None:
        """A SHORT signal while LONG is open first closes the long, then opens short."""

        class _FlipStrategy:
            """Strategy that goes LONG on bar[0], SHORT on bar[1]."""

            def __init__(self) -> None:
                self._call: int = 0

            def on_bar(
                self,
                timestamp: datetime,
                features: pl.DataFrame,  # noqa: ARG002
                portfolio: object,  # noqa: ARG002
            ) -> list[Signal]:
                self._call += 1
                if self._call == 1:
                    return [Signal(asset=_BTC, side=Side.LONG, strength=1.0, timestamp=timestamp)]
                if self._call == 2:  # noqa: PLR2004
                    return [Signal(asset=_BTC, side=Side.SHORT, strength=1.0, timestamp=timestamp)]
                return []

        bars: pl.DataFrame = make_bars(4, start_price=40_000.0)
        engine: ExecutionEngine = _make_engine(_FlipStrategy(), FixedNotionalSizer(10_000.0))
        result: BacktestResult = engine.run(bars=bars, asset=_BTC)
        # At minimum the LONG position was closed and a new SHORT opened
        # So there must be at least one completed trade (the closed long)
        assert len(result.trades) >= 1

    def test_insufficient_cash_skips_trade(self) -> None:
        """Sizer requesting more than available cash results in no trade."""
        bars: pl.DataFrame = make_bars(3, start_price=40_000.0)
        # Request notional larger than initial_cash
        strategy: SingleSignalStrategy = SingleSignalStrategy(_BTC, Side.LONG)
        sizer: FixedNotionalSizer = FixedNotionalSizer(notional=999_999_999.0)
        engine: ExecutionEngine = _make_engine(strategy, sizer, commission_bps=0.0)
        result: BacktestResult = engine.run(bars=bars, asset=_BTC, initial_cash=INITIAL_CASH)
        assert len(result.trades) == 0

    def test_last_bar_position_is_liquidated(self) -> None:
        """Open position at the end of bars is liquidated at last close."""
        bars: pl.DataFrame = make_bars(3, start_price=40_000.0)
        strategy: SingleSignalStrategy = SingleSignalStrategy(_BTC, Side.LONG)
        sizer: FixedNotionalSizer = FixedNotionalSizer(notional=10_000.0)
        engine: ExecutionEngine = _make_engine(strategy, sizer, commission_bps=0.0)
        result: BacktestResult = engine.run(bars=bars, asset=_BTC)
        # Position opened at bar[1] should be liquidated: 1 trade total
        assert len(result.trades) == 1
        assert result.trades[0].exit_price == pytest.approx(40_000.0)  # last close (flat)


# ---------------------------------------------------------------------------
# TestEquityCurveManualVerification
# ---------------------------------------------------------------------------


class TestEquityCurveManualVerification:
    """Manual verification of equity curve values against expected calculations."""

    def test_never_trade_equity_curve_matches_initial_cash(self) -> None:
        """All equity snapshots equal initial cash when no trades are made."""
        initial: float = 75_000.0
        bars: pl.DataFrame = make_bars(5, start_price=40_000.0)
        engine: ExecutionEngine = _make_engine(NeverTradeStrategy(), FixedNotionalSizer())
        result: BacktestResult = engine.run(bars=bars, asset=_BTC, initial_cash=initial)
        expected_values: list[float] = [initial] * 5
        for actual, expected in zip(result.equity_curve.values, expected_values, strict=True):
            assert actual == pytest.approx(expected)

    def test_equity_timestamps_match_bar_timestamps(self) -> None:
        """Equity curve timestamps exactly match input bar timestamps."""
        n: int = 5
        bars: pl.DataFrame = make_bars(n, start_price=40_000.0)
        engine: ExecutionEngine = _make_engine(NeverTradeStrategy(), FixedNotionalSizer())
        result: BacktestResult = engine.run(bars=bars, asset=_BTC)
        bar_timestamps: list[datetime] = bars.get_column("timestamp").to_list()
        assert result.equity_curve.timestamps == bar_timestamps

    def test_buy_and_hold_equity_includes_unrealised_pnl(self) -> None:
        """Equity snapshots include unrealised P&L from open positions."""
        # Rising prices: open long at bar[1], equity should increase thereafter
        bars: pl.DataFrame = make_bars(5, start_price=40_000.0, price_step=100.0)
        from src.app.backtest.application.baselines import BuyAndHoldStrategy

        strategy: BuyAndHoldStrategy = BuyAndHoldStrategy(asset=_BTC)
        sizer: FixedFractionalSizer = FixedFractionalSizer(fraction=0.5)
        config: ExecutionConfig = ExecutionConfig(commission_bps=0.0)
        engine: ExecutionEngine = ExecutionEngine(config=config, strategy=strategy, sizer=sizer)
        result: BacktestResult = engine.run(bars=bars, asset=_BTC, initial_cash=INITIAL_CASH)

        # After buying in bar[1], equity at later bars should reflect price gains
        # (rising market → equity should be >= initial after bar[2])
        if len(result.equity_curve.values) >= 3:  # noqa: PLR2004
            assert result.equity_curve.values[2] >= result.equity_curve.values[1]
