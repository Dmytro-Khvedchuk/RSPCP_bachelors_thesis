"""Tests for cost-sensitivity sweep across multiple commission levels."""

from __future__ import annotations

from datetime import datetime, UTC

import pytest

from src.app.backtest.application.cost_sweep import run_with_cost_sweep
from src.app.backtest.application.execution import BacktestResult, ExecutionEngine
from src.app.backtest.domain.value_objects import ExecutionConfig, Side
from src.app.ohlcv.domain.value_objects import Asset
from src.tests.backtest.conftest import (
    INITIAL_CASH,
    AlwaysLongStrategy,
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

_FEE_LEVELS: list[float] = [5.0, 10.0, 15.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# TestRunWithCostSweep — result structure
# ---------------------------------------------------------------------------


class TestCostSweepResultStructure:
    """Tests verifying the structure of cost-sweep results."""

    def test_returns_dict_keyed_by_fee_level(self) -> None:
        """run_with_cost_sweep returns a dict with fee levels as keys."""
        bars = make_bars(5, start_price=40_000.0)
        fees: list[float] = [5.0, 10.0, 20.0]
        results: dict[float, BacktestResult] = run_with_cost_sweep(
            strategy=NeverTradeStrategy(),
            sizer=FixedNotionalSizer(),
            bars=bars,
            asset=_BTC,
            fees=fees,
        )
        assert set(results.keys()) == {5.0, 10.0, 20.0}

    def test_result_count_matches_fee_levels(self) -> None:
        """Number of results equals the number of fee levels."""
        bars = make_bars(5, start_price=40_000.0)
        fees: list[float] = [5.0, 10.0, 15.0, 20.0, 30.0]
        results: dict[float, BacktestResult] = run_with_cost_sweep(
            strategy=NeverTradeStrategy(),
            sizer=FixedNotionalSizer(),
            bars=bars,
            asset=_BTC,
            fees=fees,
        )
        assert len(results) == len(fees)

    def test_each_result_is_backtest_result(self) -> None:
        """Every value in the returned dict is a BacktestResult."""
        bars = make_bars(5, start_price=40_000.0)
        results: dict[float, BacktestResult] = run_with_cost_sweep(
            strategy=NeverTradeStrategy(),
            sizer=FixedNotionalSizer(),
            bars=bars,
            asset=_BTC,
            fees=[10.0],
        )
        for result in results.values():
            assert isinstance(result, BacktestResult)

    def test_config_commission_matches_fee_level(self) -> None:
        """Each result's config.commission_bps matches the corresponding fee level."""
        bars = make_bars(5, start_price=40_000.0)
        fees: list[float] = [5.0, 15.0, 25.0]
        results: dict[float, BacktestResult] = run_with_cost_sweep(
            strategy=NeverTradeStrategy(),
            sizer=FixedNotionalSizer(),
            bars=bars,
            asset=_BTC,
            fees=fees,
        )
        for fee in fees:
            assert results[fee].config.commission_bps == pytest.approx(fee)

    def test_default_fees_when_none(self) -> None:
        """Passing fees=None uses the default 5-level sweep."""
        bars = make_bars(5, start_price=40_000.0)
        results: dict[float, BacktestResult] = run_with_cost_sweep(
            strategy=NeverTradeStrategy(),
            sizer=FixedNotionalSizer(),
            bars=bars,
            asset=_BTC,
            fees=None,
        )
        assert len(results) == 5


# ---------------------------------------------------------------------------
# TestCostSweepEconomics — higher fees reduce final equity
# ---------------------------------------------------------------------------


class TestCostSweepEconomics:
    """Tests verifying that higher fees produce lower final equity."""

    def test_higher_fee_lowers_final_equity(self) -> None:
        """With active trading, higher commission results in lower final equity.

        Each fee-level run gets its own strategy instance (AlwaysLongStrategy is
        stateless) so all runs see identical signal sequences.
        """
        # 5 bars so position opens at bar[1] and is liquidated at bar[4] (different time)
        bars = make_bars(5, start_price=40_000.0)
        fees: list[float] = [5.0, 10.0, 20.0]
        # run_with_cost_sweep shares a single strategy instance; use AlwaysLongStrategy
        # which has no per-call state so each run behaves identically
        equities: list[float] = []
        for fee in sorted(fees):
            config: ExecutionConfig = ExecutionConfig(commission_bps=fee)
            engine: ExecutionEngine = ExecutionEngine(
                config=config,
                strategy=AlwaysLongStrategy(_BTC),
                sizer=FixedNotionalSizer(notional=10_000.0),
            )
            result: BacktestResult = engine.run(bars=bars, asset=_BTC, initial_cash=INITIAL_CASH)
            equities.append(result.equity_curve.values[-1])
        # Each step should be non-increasing
        assert equities[0] >= equities[1] >= equities[2]

    def test_zero_commission_highest_equity(self) -> None:
        """Zero commission result has the highest final equity among all fee levels."""
        bars = make_bars(5, start_price=40_000.0, price_step=100.0)
        fees: list[float] = [0.0, 5.0, 10.0, 20.0]
        results: dict[float, BacktestResult] = run_with_cost_sweep(
            strategy=SingleSignalStrategy(_BTC, Side.LONG),
            sizer=FixedNotionalSizer(notional=10_000.0),
            bars=bars,
            asset=_BTC,
            fees=fees,
            initial_cash=INITIAL_CASH,
        )
        zero_equity: float = results[0.0].equity_curve.values[-1]
        for fee in [5.0, 10.0, 20.0]:
            assert zero_equity >= results[fee].equity_curve.values[-1]

    def test_no_trades_equity_identical_across_fees(self) -> None:
        """With no trades, final equity is the same regardless of fee level."""
        bars = make_bars(5, start_price=40_000.0)
        fees: list[float] = [5.0, 10.0, 30.0]
        results: dict[float, BacktestResult] = run_with_cost_sweep(
            strategy=NeverTradeStrategy(),
            sizer=FixedNotionalSizer(),
            bars=bars,
            asset=_BTC,
            fees=fees,
            initial_cash=INITIAL_CASH,
        )
        equities: list[float] = [results[fee].equity_curve.values[-1] for fee in fees]
        for eq in equities:
            assert eq == pytest.approx(equities[0])


# ---------------------------------------------------------------------------
# TestCostSweepBaseConfig — inheritance from base_config
# ---------------------------------------------------------------------------


class TestCostSweepBaseConfig:
    """Tests for base_config inheritance in cost sweeps."""

    def test_base_config_min_trade_count_inherited(self) -> None:
        """Base config min_trade_count is preserved in all sweep results."""
        bars = make_bars(5, start_price=40_000.0)
        base: ExecutionConfig = ExecutionConfig(commission_bps=10.0, min_trade_count=50)
        fees: list[float] = [5.0, 15.0]
        results: dict[float, BacktestResult] = run_with_cost_sweep(
            strategy=NeverTradeStrategy(),
            sizer=FixedNotionalSizer(),
            bars=bars,
            asset=_BTC,
            fees=fees,
            base_config=base,
        )
        for result in results.values():
            assert result.config.min_trade_count == 50

    def test_base_config_asset_multiplier_inherited(self) -> None:
        """Base config per-asset multiplier is inherited in all sweep results."""
        bars = make_bars(5, start_price=40_000.0)
        base: ExecutionConfig = ExecutionConfig(
            commission_bps=10.0,
            asset_cost_multiplier={"BTCUSDT": 1.5},
        )
        fees: list[float] = [5.0, 10.0]
        results: dict[float, BacktestResult] = run_with_cost_sweep(
            strategy=NeverTradeStrategy(),
            sizer=FixedNotionalSizer(),
            bars=bars,
            asset=_BTC,
            fees=fees,
            base_config=base,
        )
        for result in results.values():
            assert result.config.asset_cost_multiplier.get("BTCUSDT") == pytest.approx(1.5)

    def test_none_base_config_uses_defaults(self) -> None:
        """None base_config uses ExecutionConfig defaults."""
        bars = make_bars(5, start_price=40_000.0)
        fees: list[float] = [10.0]
        results: dict[float, BacktestResult] = run_with_cost_sweep(
            strategy=NeverTradeStrategy(),
            sizer=FixedNotionalSizer(),
            bars=bars,
            asset=_BTC,
            fees=fees,
            base_config=None,
        )
        assert results[10.0].config.min_trade_count == 30


# ---------------------------------------------------------------------------
# TestCostSweepEdgeCases
# ---------------------------------------------------------------------------


class TestCostSweepEdgeCases:
    """Edge case tests for cost sweep."""

    def test_single_fee_level_produces_single_result(self) -> None:
        """A single fee level returns exactly one result."""
        bars = make_bars(5, start_price=40_000.0)
        results: dict[float, BacktestResult] = run_with_cost_sweep(
            strategy=NeverTradeStrategy(),
            sizer=FixedNotionalSizer(),
            bars=bars,
            asset=_BTC,
            fees=[7.5],
        )
        assert len(results) == 1
        assert 7.5 in results

    def test_equity_curve_present_in_all_results(self) -> None:
        """All sweep results have an equity curve with correct length."""
        n_bars: int = 8
        bars = make_bars(n_bars, start_price=40_000.0)
        results: dict[float, BacktestResult] = run_with_cost_sweep(
            strategy=NeverTradeStrategy(),
            sizer=FixedNotionalSizer(),
            bars=bars,
            asset=_BTC,
            fees=[5.0, 10.0],
        )
        for result in results.values():
            assert len(result.equity_curve.values) == n_bars

    def test_initial_cash_propagated_to_all_runs(self) -> None:
        """Custom initial_cash is used in all fee-level runs."""
        custom_cash: float = 50_000.0
        bars = make_bars(5, start_price=40_000.0)
        results: dict[float, BacktestResult] = run_with_cost_sweep(
            strategy=NeverTradeStrategy(),
            sizer=FixedNotionalSizer(),
            bars=bars,
            asset=_BTC,
            fees=[5.0, 10.0],
            initial_cash=custom_cash,
        )
        for result in results.values():
            assert result.equity_curve.values[0] == pytest.approx(custom_cash)
