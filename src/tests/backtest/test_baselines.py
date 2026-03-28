"""Unit tests for BuyAndHoldStrategy and RandomStrategy baselines."""

from __future__ import annotations

from datetime import datetime, timedelta, UTC

import polars as pl
import pytest

from src.app.backtest.application.baselines import BuyAndHoldStrategy, RandomStrategy
from src.app.backtest.application.execution import BacktestResult, ExecutionEngine
from src.app.backtest.application.position_sizer import FixedFractionalSizer
from src.app.backtest.domain.entities import Signal
from src.app.backtest.domain.value_objects import ExecutionConfig, PortfolioSnapshot, Side
from src.app.ohlcv.domain.value_objects import Asset
from src.tests.backtest.conftest import INITIAL_CASH, make_bars, make_snapshot


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_T0: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_ONE_HOUR: timedelta = timedelta(hours=1)
_BTC: Asset = Asset(symbol="BTCUSDT")
_ETH: Asset = Asset(symbol="ETHUSDT")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_empty_features() -> pl.DataFrame:
    """Return an empty DataFrame as a stand-in for features."""
    return pl.DataFrame()


def _make_portfolio(*, has_position: bool = False) -> PortfolioSnapshot:
    """Build a PortfolioSnapshot with or without an open position."""
    positions: dict[str, float] = {"BTCUSDT": 0.1} if has_position else {}
    return make_snapshot(positions=positions)


# ---------------------------------------------------------------------------
# TestBuyAndHoldStrategy
# ---------------------------------------------------------------------------


class TestBuyAndHoldStrategy:
    """Tests for BuyAndHoldStrategy."""

    def test_first_bar_with_no_position_emits_signal(self) -> None:
        """Strategy emits a LONG signal when no position is held."""
        strategy: BuyAndHoldStrategy = BuyAndHoldStrategy(asset=_BTC)
        portfolio: PortfolioSnapshot = _make_portfolio(has_position=False)
        signals: list[Signal] = strategy.on_bar(_T0, _make_empty_features(), portfolio)
        assert len(signals) == 1
        assert signals[0].side == Side.LONG

    def test_signal_strength_is_full_conviction(self) -> None:
        """The LONG signal has strength == 1.0 (full conviction)."""
        strategy: BuyAndHoldStrategy = BuyAndHoldStrategy(asset=_BTC)
        portfolio: PortfolioSnapshot = _make_portfolio(has_position=False)
        signals: list[Signal] = strategy.on_bar(_T0, _make_empty_features(), portfolio)
        assert signals[0].strength == pytest.approx(1.0)

    def test_signal_asset_matches_strategy_asset(self) -> None:
        """Emitted signal's asset matches the strategy's configured asset."""
        strategy: BuyAndHoldStrategy = BuyAndHoldStrategy(asset=_ETH)
        portfolio: PortfolioSnapshot = _make_portfolio(has_position=False)
        signals: list[Signal] = strategy.on_bar(_T0, _make_empty_features(), portfolio)
        assert signals[0].asset == _ETH

    def test_subsequent_bars_emit_no_signal_when_position_held(self) -> None:
        """Strategy emits no signal when a position is already open."""
        strategy: BuyAndHoldStrategy = BuyAndHoldStrategy(asset=_BTC)
        portfolio_with_pos: PortfolioSnapshot = _make_portfolio(has_position=True)
        signals: list[Signal] = strategy.on_bar(_T0, _make_empty_features(), portfolio_with_pos)
        assert len(signals) == 0

    def test_only_emits_on_first_bar_in_engine_run(self) -> None:
        """In a full engine run, BuyAndHold emits exactly one trade signal."""
        n_bars: int = 10
        bars: pl.DataFrame = make_bars(n_bars, start_price=40_000.0)
        strategy: BuyAndHoldStrategy = BuyAndHoldStrategy(asset=_BTC)
        sizer: FixedFractionalSizer = FixedFractionalSizer(fraction=0.1)
        config: ExecutionConfig = ExecutionConfig(commission_bps=0.0)
        engine: ExecutionEngine = ExecutionEngine(config=config, strategy=strategy, sizer=sizer)
        result: BacktestResult = engine.run(bars=bars, asset=_BTC, initial_cash=INITIAL_CASH)
        # Buy-and-hold opens position on bar[1] (fill) and holds; liquidates at last bar
        # → 1 completed trade
        assert len(result.trades) == 1

    def test_buy_and_hold_equity_rises_on_uptrend(self) -> None:
        """Buy-and-hold equity increases when prices trend upward (no commission)."""
        n_bars: int = 10
        bars: pl.DataFrame = make_bars(n_bars, start_price=40_000.0, price_step=200.0)
        strategy: BuyAndHoldStrategy = BuyAndHoldStrategy(asset=_BTC)
        sizer: FixedFractionalSizer = FixedFractionalSizer(fraction=0.5)
        config: ExecutionConfig = ExecutionConfig(commission_bps=0.0)
        engine: ExecutionEngine = ExecutionEngine(config=config, strategy=strategy, sizer=sizer)
        result: BacktestResult = engine.run(bars=bars, asset=_BTC, initial_cash=INITIAL_CASH)
        final_equity: float = result.equity_curve.values[-1]
        assert final_equity > INITIAL_CASH

    def test_buy_and_hold_is_frozen_model(self) -> None:
        """BuyAndHoldStrategy is a frozen Pydantic model."""
        from pydantic import ValidationError

        strategy: BuyAndHoldStrategy = BuyAndHoldStrategy(asset=_BTC)
        with pytest.raises(ValidationError):
            strategy.asset = _ETH  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestRandomStrategy
# ---------------------------------------------------------------------------


class TestRandomStrategy:
    """Tests for RandomStrategy."""

    def test_seed_reproducibility(self) -> None:
        """Same seed produces the same signal sequence."""
        n_bars: int = 50
        bars: pl.DataFrame = make_bars(n_bars)

        strategy_a: RandomStrategy = RandomStrategy(asset=_BTC, signal_frequency=0.3, seed=42)
        strategy_b: RandomStrategy = RandomStrategy(asset=_BTC, signal_frequency=0.3, seed=42)
        portfolio: PortfolioSnapshot = make_snapshot()
        signals_a: list[list[Signal]] = [
            strategy_a.on_bar(bars["timestamp"][i], bars, portfolio) for i in range(n_bars)
        ]
        signals_b: list[list[Signal]] = [
            strategy_b.on_bar(bars["timestamp"][i], bars, portfolio) for i in range(n_bars)
        ]
        # Signal counts must be identical
        counts_a: list[int] = [len(s) for s in signals_a]
        counts_b: list[int] = [len(s) for s in signals_b]
        assert counts_a == counts_b

    def test_different_seeds_produce_different_sequences(self) -> None:
        """Different seeds produce different signal sequences (with high probability)."""
        n_bars: int = 100
        bars: pl.DataFrame = make_bars(n_bars)
        portfolio: PortfolioSnapshot = make_snapshot()

        strategy_a: RandomStrategy = RandomStrategy(asset=_BTC, signal_frequency=0.3, seed=1)
        strategy_b: RandomStrategy = RandomStrategy(asset=_BTC, signal_frequency=0.3, seed=999)
        sides_a: list[Side | None] = []
        sides_b: list[Side | None] = []
        for i in range(n_bars):
            ts: datetime = bars["timestamp"][i]
            sigs_a: list[Signal] = strategy_a.on_bar(ts, bars, portfolio)
            sigs_b: list[Signal] = strategy_b.on_bar(ts, bars, portfolio)
            sides_a.append(sigs_a[0].side if sigs_a else None)
            sides_b.append(sigs_b[0].side if sigs_b else None)
        # At least some difference expected with overwhelming probability
        assert sides_a != sides_b

    def test_frequency_zero_never_emits(self) -> None:
        """signal_frequency == 0.0 never emits a signal."""
        n_bars: int = 50
        bars: pl.DataFrame = make_bars(n_bars)
        strategy: RandomStrategy = RandomStrategy(asset=_BTC, signal_frequency=0.0, seed=0)
        portfolio: PortfolioSnapshot = make_snapshot()
        for i in range(n_bars):
            signals: list[Signal] = strategy.on_bar(bars["timestamp"][i], bars, portfolio)
            assert len(signals) == 0

    def test_frequency_one_always_emits(self) -> None:
        """signal_frequency == 1.0 emits a signal on every bar."""
        n_bars: int = 50
        bars: pl.DataFrame = make_bars(n_bars)
        strategy: RandomStrategy = RandomStrategy(asset=_BTC, signal_frequency=1.0, seed=0)
        portfolio: PortfolioSnapshot = make_snapshot()
        for i in range(n_bars):
            signals: list[Signal] = strategy.on_bar(bars["timestamp"][i], bars, portfolio)
            assert len(signals) == 1

    def test_signal_frequency_preserved_approximately(self) -> None:
        """Observed signal rate is close to the configured signal_frequency."""
        n_bars: int = 1_000
        freq: float = 0.2
        bars: pl.DataFrame = make_bars(n_bars)
        strategy: RandomStrategy = RandomStrategy(asset=_BTC, signal_frequency=freq, seed=7)
        portfolio: PortfolioSnapshot = make_snapshot()
        total_signals: int = 0
        for i in range(n_bars):
            signals: list[Signal] = strategy.on_bar(bars["timestamp"][i], bars, portfolio)
            total_signals += len(signals)
        observed_rate: float = total_signals / n_bars
        # Allow ±5% tolerance
        assert observed_rate == pytest.approx(freq, abs=0.05)

    def test_random_strategy_emits_both_sides(self) -> None:
        """RandomStrategy emits both LONG and SHORT signals over many bars."""
        n_bars: int = 500
        bars: pl.DataFrame = make_bars(n_bars)
        strategy: RandomStrategy = RandomStrategy(asset=_BTC, signal_frequency=1.0, seed=21)
        portfolio: PortfolioSnapshot = make_snapshot()
        sides: set[Side] = set()
        for i in range(n_bars):
            signals: list[Signal] = strategy.on_bar(bars["timestamp"][i], bars, portfolio)
            for sig in signals:
                sides.add(sig.side)
        assert Side.LONG in sides
        assert Side.SHORT in sides

    def test_random_strategy_signal_has_full_strength(self) -> None:
        """Emitted signals always have strength == 1.0."""
        n_bars: int = 50
        bars: pl.DataFrame = make_bars(n_bars)
        strategy: RandomStrategy = RandomStrategy(asset=_BTC, signal_frequency=1.0, seed=5)
        portfolio: PortfolioSnapshot = make_snapshot()
        for i in range(n_bars):
            signals: list[Signal] = strategy.on_bar(bars["timestamp"][i], bars, portfolio)
            for sig in signals:
                assert sig.strength == pytest.approx(1.0)

    def test_invalid_frequency_raises(self) -> None:
        """signal_frequency > 1.0 raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RandomStrategy(asset=_BTC, signal_frequency=1.5)

    def test_negative_frequency_raises(self) -> None:
        """signal_frequency < 0.0 raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RandomStrategy(asset=_BTC, signal_frequency=-0.1)
