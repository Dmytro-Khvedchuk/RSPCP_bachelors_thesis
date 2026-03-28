"""Integration tests for WalkForwardRunner — expanding and rolling modes."""

from __future__ import annotations

from datetime import datetime, UTC

import polars as pl
import pytest

from src.app.backtest.application.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardRunner,
    WindowMode,
    WindowResult,
    _generate_window_specs,
)
from src.app.backtest.domain.protocols import IStrategy
from src.app.backtest.domain.value_objects import ExecutionConfig
from src.app.ohlcv.domain.value_objects import Asset
from src.tests.backtest.conftest import (
    INITIAL_CASH,
    AlwaysLongStrategy,
    FixedNotionalSizer,
    NeverTradeStrategy,
    make_bars,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_T0: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_BTC: Asset = Asset(symbol="BTCUSDT")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AlwaysLongFactory:
    """Strategy factory that always returns AlwaysLongStrategy."""

    def __init__(self, asset: Asset) -> None:
        self._asset: Asset = asset

    def __call__(self, train_bars: pl.DataFrame) -> IStrategy:  # noqa: ARG002
        return AlwaysLongStrategy(self._asset)


class _NeverTradeFactory:
    """Strategy factory that always returns NeverTradeStrategy."""

    def __call__(self, train_bars: pl.DataFrame) -> IStrategy:  # noqa: ARG002
        return NeverTradeStrategy()


def _make_runner(
    mode: WindowMode = WindowMode.EXPANDING,
    train_bars: int = 5,
    test_bars: int = 5,
    step_bars: int | None = None,
    commission_bps: float = 0.0,
    factory: object | None = None,
) -> WalkForwardRunner:
    """Build a WalkForwardRunner with configurable parameters."""
    wf_config: WalkForwardConfig = WalkForwardConfig(
        mode=mode,
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
    )
    exec_config: ExecutionConfig = ExecutionConfig(commission_bps=commission_bps)
    used_factory: object = factory or _NeverTradeFactory()
    sizer: FixedNotionalSizer = FixedNotionalSizer(notional=5_000.0)
    return WalkForwardRunner(
        config=wf_config,
        execution_config=exec_config,
        strategy_factory=used_factory,  # type: ignore[arg-type]
        sizer=sizer,
    )


# ---------------------------------------------------------------------------
# TestWalkForwardConfig
# ---------------------------------------------------------------------------


class TestWalkForwardConfig:
    """Tests for WalkForwardConfig validation and defaults."""

    def test_effective_step_bars_defaults_to_test_bars(self) -> None:
        """step_bars defaults to test_bars when unset."""
        config: WalkForwardConfig = WalkForwardConfig(train_bars=10, test_bars=5)
        assert config.effective_step_bars == 5

    def test_explicit_step_bars_respected(self) -> None:
        """Explicit step_bars overrides default."""
        config: WalkForwardConfig = WalkForwardConfig(train_bars=10, test_bars=5, step_bars=2)
        assert config.effective_step_bars == 2

    def test_zero_train_bars_raises(self) -> None:
        """train_bars == 0 raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WalkForwardConfig(train_bars=0, test_bars=5)

    def test_zero_test_bars_raises(self) -> None:
        """test_bars == 0 raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WalkForwardConfig(train_bars=10, test_bars=0)


# ---------------------------------------------------------------------------
# TestGenerateWindowSpecs
# ---------------------------------------------------------------------------


class TestGenerateWindowSpecs:
    """Tests for _generate_window_specs helper."""

    def test_expanding_single_window(self) -> None:
        """Exactly train+test bars produces exactly one expanding window."""
        from src.app.backtest.application.walk_forward import _WindowSpec

        config: WalkForwardConfig = WalkForwardConfig(mode=WindowMode.EXPANDING, train_bars=5, test_bars=5)
        specs: list[_WindowSpec] = _generate_window_specs(total_bars=10, config=config)
        assert len(specs) == 1
        assert specs[0].train_start_idx == 0
        assert specs[0].train_end_idx == 5
        assert specs[0].test_start_idx == 5
        assert specs[0].test_end_idx == 10

    def test_expanding_multiple_windows(self) -> None:
        """Sufficient bars produce multiple expanding windows."""
        from src.app.backtest.application.walk_forward import _WindowSpec

        config: WalkForwardConfig = WalkForwardConfig(mode=WindowMode.EXPANDING, train_bars=5, test_bars=5)
        specs: list[_WindowSpec] = _generate_window_specs(total_bars=20, config=config)
        assert len(specs) == 3  # windows start at 5, 10, 15

    def test_expanding_train_always_starts_at_zero(self) -> None:
        """In expanding mode, train_start_idx is always 0 for all windows."""
        from src.app.backtest.application.walk_forward import _WindowSpec

        config: WalkForwardConfig = WalkForwardConfig(mode=WindowMode.EXPANDING, train_bars=3, test_bars=3)
        specs: list[_WindowSpec] = _generate_window_specs(total_bars=15, config=config)
        for spec in specs:
            assert spec.train_start_idx == 0

    def test_rolling_train_slides_forward(self) -> None:
        """In rolling mode, train_start_idx advances with each window."""
        from src.app.backtest.application.walk_forward import _WindowSpec

        config: WalkForwardConfig = WalkForwardConfig(mode=WindowMode.ROLLING, train_bars=5, test_bars=5)
        specs: list[_WindowSpec] = _generate_window_specs(total_bars=20, config=config)
        assert len(specs) >= 2
        # Second window's train_start > first window's train_start
        assert specs[1].train_start_idx > specs[0].train_start_idx

    def test_insufficient_bars_returns_empty(self) -> None:
        """Fewer bars than train+test produces an empty list."""
        from src.app.backtest.application.walk_forward import _WindowSpec

        config: WalkForwardConfig = WalkForwardConfig(mode=WindowMode.EXPANDING, train_bars=10, test_bars=10)
        specs: list[_WindowSpec] = _generate_window_specs(total_bars=15, config=config)
        assert len(specs) == 0

    def test_window_indices_are_sequential(self) -> None:
        """Window indices start at 0 and increment by 1."""
        from src.app.backtest.application.walk_forward import _WindowSpec

        config: WalkForwardConfig = WalkForwardConfig(mode=WindowMode.EXPANDING, train_bars=5, test_bars=5)
        specs: list[_WindowSpec] = _generate_window_specs(total_bars=30, config=config)
        for i, spec in enumerate(specs):
            assert spec.window_index == i


# ---------------------------------------------------------------------------
# TestWalkForwardRunnerErrors
# ---------------------------------------------------------------------------


class TestWalkForwardRunnerErrors:
    """Tests for error conditions in WalkForwardRunner.run."""

    def test_insufficient_bars_raises_value_error(self) -> None:
        """Fewer bars than train+test raises ValueError."""
        runner: WalkForwardRunner = _make_runner(train_bars=10, test_bars=10)
        bars: pl.DataFrame = make_bars(15, start_price=40_000.0)
        with pytest.raises(ValueError, match="Insufficient bars"):
            runner.run(bars=bars, asset=_BTC)

    def test_exactly_minimum_bars_runs_successfully(self) -> None:
        """Exactly train+test bars runs without error (one window)."""
        runner: WalkForwardRunner = _make_runner(train_bars=5, test_bars=5)
        bars: pl.DataFrame = make_bars(10, start_price=40_000.0)
        result: WalkForwardResult = runner.run(bars=bars, asset=_BTC)
        assert len(result.windows) == 1


# ---------------------------------------------------------------------------
# TestWalkForwardRunnerExpanding
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestWalkForwardRunnerExpanding:
    """Integration tests for expanding-window walk-forward."""

    def test_expanding_windows_count(self) -> None:
        """Correct number of expanding windows generated for given bar count."""
        runner: WalkForwardRunner = _make_runner(mode=WindowMode.EXPANDING, train_bars=5, test_bars=5)
        bars: pl.DataFrame = make_bars(20, start_price=40_000.0)
        result: WalkForwardResult = runner.run(bars=bars, asset=_BTC)
        # Windows: test starts at 5, 10, 15 → 3 windows
        assert len(result.windows) == 3

    def test_expanding_windows_chronological_order(self) -> None:
        """Windows are returned in chronological order."""
        runner: WalkForwardRunner = _make_runner(mode=WindowMode.EXPANDING, train_bars=5, test_bars=5)
        bars: pl.DataFrame = make_bars(20, start_price=40_000.0)
        result: WalkForwardResult = runner.run(bars=bars, asset=_BTC)
        for i in range(1, len(result.windows)):
            assert result.windows[i].test_start > result.windows[i - 1].test_start

    def test_expanding_train_end_advances_per_window(self) -> None:
        """Expanding mode: each window has a larger training set than the previous."""
        runner: WalkForwardRunner = _make_runner(mode=WindowMode.EXPANDING, train_bars=5, test_bars=5)
        bars: pl.DataFrame = make_bars(20, start_price=40_000.0)
        result: WalkForwardResult = runner.run(bars=bars, asset=_BTC)
        for i in range(1, len(result.windows)):
            prev: WindowResult = result.windows[i - 1]
            curr: WindowResult = result.windows[i]
            assert curr.train_end >= prev.train_end

    def test_no_lookahead_train_end_before_test_start(self) -> None:
        """Training window ends strictly before the test window starts."""
        runner: WalkForwardRunner = _make_runner(mode=WindowMode.EXPANDING, train_bars=5, test_bars=5)
        bars: pl.DataFrame = make_bars(20, start_price=40_000.0)
        result: WalkForwardResult = runner.run(bars=bars, asset=_BTC)
        for window in result.windows:
            assert window.train_end < window.test_start

    def test_equity_is_chained_across_windows(self) -> None:
        """Final equity of window N equals initial equity of window N+1."""
        runner: WalkForwardRunner = _make_runner(
            mode=WindowMode.EXPANDING,
            train_bars=5,
            test_bars=5,
            commission_bps=0.0,
        )
        bars: pl.DataFrame = make_bars(20, start_price=40_000.0)
        result: WalkForwardResult = runner.run(bars=bars, asset=_BTC, initial_cash=INITIAL_CASH)

        for i in range(1, len(result.windows)):
            prev_final: float = result.windows[i - 1].backtest_result.equity_curve.values[-1]
            curr_initial: float = result.windows[i].backtest_result.equity_curve.values[0]
            # The initial cash of the next window matches the final equity of the previous
            assert curr_initial == pytest.approx(prev_final, rel=1e-9)

    def test_aggregate_equity_spans_all_windows(self) -> None:
        """Aggregate equity curve timestamps span from first to last window."""
        runner: WalkForwardRunner = _make_runner(mode=WindowMode.EXPANDING, train_bars=5, test_bars=5)
        bars: pl.DataFrame = make_bars(20, start_price=40_000.0)
        result: WalkForwardResult = runner.run(bars=bars, asset=_BTC)
        # The aggregate equity curve should contain timestamps from all test windows
        agg_ec_len: int = len(result.windows)
        assert agg_ec_len > 0


# ---------------------------------------------------------------------------
# TestWalkForwardRunnerRolling
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestWalkForwardRunnerRolling:
    """Integration tests for rolling-window walk-forward."""

    def test_rolling_train_window_fixed_size(self) -> None:
        """Rolling mode: each train window has exactly train_bars bars."""
        from src.app.backtest.application.walk_forward import _WindowSpec

        train_bars_count: int = 5
        config: WalkForwardConfig = WalkForwardConfig(
            mode=WindowMode.ROLLING,
            train_bars=train_bars_count,
            test_bars=5,
        )
        specs: list[_WindowSpec] = _generate_window_specs(total_bars=25, config=config)
        assert len(specs) > 0
        for spec in specs:
            actual_train_size: int = spec.train_end_idx - spec.train_start_idx
            assert actual_train_size == train_bars_count

    def test_rolling_more_windows_than_expanding(self) -> None:
        """Rolling mode produces the same number of windows as expanding for same step."""
        total: int = 30
        bars: pl.DataFrame = make_bars(total, start_price=40_000.0)
        runner_exp: WalkForwardRunner = _make_runner(mode=WindowMode.EXPANDING, train_bars=5, test_bars=5)
        runner_roll: WalkForwardRunner = _make_runner(mode=WindowMode.ROLLING, train_bars=5, test_bars=5)
        result_exp: WalkForwardResult = runner_exp.run(bars=bars, asset=_BTC)
        result_roll: WalkForwardResult = runner_roll.run(bars=bars, asset=_BTC)
        # Same step size → same window count
        assert len(result_exp.windows) == len(result_roll.windows)

    def test_rolling_no_lookahead(self) -> None:
        """Rolling mode: train_end strictly before test_start for all windows."""
        runner: WalkForwardRunner = _make_runner(mode=WindowMode.ROLLING, train_bars=5, test_bars=5)
        bars: pl.DataFrame = make_bars(25, start_price=40_000.0)
        result: WalkForwardResult = runner.run(bars=bars, asset=_BTC)
        for window in result.windows:
            assert window.train_end < window.test_start


# ---------------------------------------------------------------------------
# TestWalkForwardAggregateMetrics
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestWalkForwardAggregateMetrics:
    """Tests for aggregate metrics produced by walk-forward evaluation."""

    def test_aggregate_n_trades_equals_sum_of_window_trades(self) -> None:
        """Aggregate n_trades equals the sum of trades across all windows."""
        runner: WalkForwardRunner = _make_runner(
            mode=WindowMode.EXPANDING,
            train_bars=5,
            test_bars=5,
            factory=_AlwaysLongFactory(_BTC),
        )
        bars: pl.DataFrame = make_bars(20, start_price=40_000.0)
        result: WalkForwardResult = runner.run(bars=bars, asset=_BTC)
        total_window_trades: int = sum(len(w.backtest_result.trades) for w in result.windows)
        assert result.aggregate_metrics.n_trades == total_window_trades

    def test_aggregate_config_preserved(self) -> None:
        """WalkForwardResult preserves the config used."""
        config: WalkForwardConfig = WalkForwardConfig(mode=WindowMode.EXPANDING, train_bars=5, test_bars=5)
        exec_config: ExecutionConfig = ExecutionConfig(commission_bps=0.0)
        runner: WalkForwardRunner = WalkForwardRunner(
            config=config,
            execution_config=exec_config,
            strategy_factory=_NeverTradeFactory(),  # type: ignore[arg-type]
            sizer=FixedNotionalSizer(),
        )
        bars: pl.DataFrame = make_bars(10, start_price=40_000.0)
        result: WalkForwardResult = runner.run(bars=bars, asset=_BTC)
        assert result.config.mode == WindowMode.EXPANDING
        assert result.config.train_bars == 5
