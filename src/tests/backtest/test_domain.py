"""Unit tests for backtest domain value objects and entities."""

from __future__ import annotations

from datetime import datetime, timedelta, UTC

import pytest
from pydantic import ValidationError

from src.app.backtest.domain.entities import EquityCurve, Position, Signal, Trade
from src.app.backtest.domain.value_objects import (
    ExecutionConfig,
    PortfolioSnapshot,
    Side,
    TradeResult,
)
from src.app.ohlcv.domain.value_objects import Asset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_T0: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_T1: datetime = datetime(2024, 1, 1, 1, 0, 0, tzinfo=UTC)
_T2: datetime = datetime(2024, 1, 1, 2, 0, 0, tzinfo=UTC)
_T3: datetime = datetime(2024, 1, 1, 3, 0, 0, tzinfo=UTC)

_BTC: Asset = Asset(symbol="BTCUSDT")


# ---------------------------------------------------------------------------
# Side
# ---------------------------------------------------------------------------


class TestSide:
    """Tests for the Side enum."""

    def test_long_value(self) -> None:
        """Side.LONG has value 'long'."""
        assert Side.LONG == "long"

    def test_short_value(self) -> None:
        """Side.SHORT has value 'short'."""
        assert Side.SHORT == "short"

    def test_side_is_str_enum(self) -> None:
        """Side inherits from str so it can be compared directly to strings."""
        assert isinstance(Side.LONG, str)


# ---------------------------------------------------------------------------
# ExecutionConfig
# ---------------------------------------------------------------------------


class TestExecutionConfig:
    """Tests for ExecutionConfig validation and defaults."""

    def test_default_commission_bps(self) -> None:
        """Default commission is 10 bps."""
        config: ExecutionConfig = ExecutionConfig()
        assert config.commission_bps == pytest.approx(10.0)

    def test_custom_commission_bps(self) -> None:
        """Custom commission bps is stored correctly."""
        config: ExecutionConfig = ExecutionConfig(commission_bps=5.0)
        assert config.commission_bps == pytest.approx(5.0)

    def test_zero_commission_bps_allowed(self) -> None:
        """Commission of 0 bps is valid."""
        config: ExecutionConfig = ExecutionConfig(commission_bps=0.0)
        assert config.commission_bps == pytest.approx(0.0)

    def test_negative_commission_raises(self) -> None:
        """Negative commission bps raises ValidationError."""
        with pytest.raises(ValidationError):
            ExecutionConfig(commission_bps=-1.0)

    def test_default_cost_sweep_bps(self) -> None:
        """Default cost sweep has 5 levels."""
        config: ExecutionConfig = ExecutionConfig()
        assert len(config.cost_sweep_bps) == 5

    def test_asset_cost_multiplier_defaults_empty(self) -> None:
        """asset_cost_multiplier defaults to empty dict."""
        config: ExecutionConfig = ExecutionConfig()
        assert config.asset_cost_multiplier == {}

    def test_per_asset_multiplier_stored(self) -> None:
        """Per-asset cost multiplier is stored and retrievable."""
        config: ExecutionConfig = ExecutionConfig(asset_cost_multiplier={"BTCUSDT": 1.5})
        assert config.asset_cost_multiplier["BTCUSDT"] == pytest.approx(1.5)

    def test_min_trade_count_must_be_positive(self) -> None:
        """min_trade_count <= 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            ExecutionConfig(min_trade_count=0)

    def test_frozen_immutability(self) -> None:
        """ExecutionConfig is frozen — assignment raises."""
        config: ExecutionConfig = ExecutionConfig()
        with pytest.raises(ValidationError):
            config.commission_bps = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TradeResult
# ---------------------------------------------------------------------------


class TestTradeResult:
    """Tests for TradeResult value object."""

    def test_valid_trade_result(self) -> None:
        """TradeResult with valid fields is created correctly."""
        tr: TradeResult = TradeResult(
            entry_price=40_000.0,
            exit_price=41_000.0,
            side=Side.LONG,
            size=0.25,
            entry_time=_T0,
            exit_time=_T1,
            gross_pnl=250.0,
            net_pnl=240.0,
            commission_paid=10.0,
        )
        assert tr.gross_pnl == pytest.approx(250.0)
        assert tr.net_pnl == pytest.approx(240.0)

    def test_exit_before_entry_raises(self) -> None:
        """exit_time <= entry_time raises ValueError via model_validator."""
        with pytest.raises(ValidationError):
            TradeResult(
                entry_price=40_000.0,
                exit_price=41_000.0,
                side=Side.LONG,
                size=0.25,
                entry_time=_T1,
                exit_time=_T0,
                gross_pnl=250.0,
                net_pnl=240.0,
                commission_paid=10.0,
            )

    def test_equal_entry_exit_time_raises(self) -> None:
        """exit_time == entry_time (not strictly after) raises ValueError."""
        with pytest.raises(ValidationError):
            TradeResult(
                entry_price=40_000.0,
                exit_price=41_000.0,
                side=Side.LONG,
                size=0.25,
                entry_time=_T0,
                exit_time=_T0,
                gross_pnl=0.0,
                net_pnl=0.0,
                commission_paid=0.0,
            )

    def test_negative_entry_price_raises(self) -> None:
        """Negative entry_price raises ValidationError."""
        with pytest.raises(ValidationError):
            TradeResult(
                entry_price=-1.0,
                exit_price=41_000.0,
                side=Side.LONG,
                size=0.25,
                entry_time=_T0,
                exit_time=_T1,
                gross_pnl=0.0,
                net_pnl=0.0,
                commission_paid=0.0,
            )

    def test_zero_size_raises(self) -> None:
        """size == 0 raises ValidationError (must be > 0)."""
        with pytest.raises(ValidationError):
            TradeResult(
                entry_price=40_000.0,
                exit_price=41_000.0,
                side=Side.LONG,
                size=0.0,
                entry_time=_T0,
                exit_time=_T1,
                gross_pnl=0.0,
                net_pnl=0.0,
                commission_paid=0.0,
            )

    def test_negative_commission_raises(self) -> None:
        """Negative commission_paid raises ValidationError."""
        with pytest.raises(ValidationError):
            TradeResult(
                entry_price=40_000.0,
                exit_price=41_000.0,
                side=Side.LONG,
                size=0.25,
                entry_time=_T0,
                exit_time=_T1,
                gross_pnl=0.0,
                net_pnl=0.0,
                commission_paid=-1.0,
            )


# ---------------------------------------------------------------------------
# PortfolioSnapshot
# ---------------------------------------------------------------------------


class TestPortfolioSnapshot:
    """Tests for PortfolioSnapshot value object."""

    def test_valid_snapshot(self) -> None:
        """Valid snapshot is created without error."""
        snap: PortfolioSnapshot = PortfolioSnapshot(
            timestamp=_T0,
            equity=100_000.0,
            cash=100_000.0,
        )
        assert snap.equity == pytest.approx(100_000.0)

    def test_negative_equity_raises(self) -> None:
        """Negative equity raises ValidationError."""
        with pytest.raises(ValidationError):
            PortfolioSnapshot(
                timestamp=_T0,
                equity=-1.0,
                cash=100_000.0,
            )

    def test_positive_drawdown_raises(self) -> None:
        """Positive drawdown raises ValidationError (must be <= 0)."""
        with pytest.raises(ValidationError):
            PortfolioSnapshot(
                timestamp=_T0,
                equity=100_000.0,
                cash=100_000.0,
                drawdown=0.05,
            )

    def test_drawdown_zero_is_valid(self) -> None:
        """drawdown == 0.0 is valid."""
        snap: PortfolioSnapshot = PortfolioSnapshot(
            timestamp=_T0,
            equity=100_000.0,
            cash=100_000.0,
            drawdown=0.0,
        )
        assert snap.drawdown == pytest.approx(0.0)

    def test_positions_default_empty(self) -> None:
        """positions dict defaults to empty when not supplied."""
        snap: PortfolioSnapshot = PortfolioSnapshot(
            timestamp=_T0,
            equity=100_000.0,
            cash=100_000.0,
        )
        assert snap.positions == {}


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


class TestSignal:
    """Tests for Signal entity."""

    def test_valid_long_signal(self) -> None:
        """Valid LONG signal is created correctly."""
        sig: Signal = Signal(
            asset=_BTC,
            side=Side.LONG,
            strength=0.8,
            timestamp=_T0,
        )
        assert sig.side == Side.LONG
        assert sig.strength == pytest.approx(0.8)

    def test_strength_above_one_raises(self) -> None:
        """strength > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            Signal(
                asset=_BTC,
                side=Side.LONG,
                strength=1.1,
                timestamp=_T0,
            )

    def test_strength_below_zero_raises(self) -> None:
        """strength < 0.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            Signal(
                asset=_BTC,
                side=Side.LONG,
                strength=-0.1,
                timestamp=_T0,
            )

    def test_strength_exactly_zero_is_valid(self) -> None:
        """strength == 0.0 is valid (closed interval)."""
        sig: Signal = Signal(
            asset=_BTC,
            side=Side.LONG,
            strength=0.0,
            timestamp=_T0,
        )
        assert sig.strength == pytest.approx(0.0)

    def test_strength_exactly_one_is_valid(self) -> None:
        """strength == 1.0 is valid (closed interval)."""
        sig: Signal = Signal(
            asset=_BTC,
            side=Side.SHORT,
            strength=1.0,
            timestamp=_T0,
        )
        assert sig.strength == pytest.approx(1.0)

    def test_signal_is_frozen(self) -> None:
        """Signal is frozen — field assignment raises."""
        sig: Signal = Signal(
            asset=_BTC,
            side=Side.LONG,
            strength=1.0,
            timestamp=_T0,
        )
        with pytest.raises(ValidationError):
            sig.strength = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


class TestPosition:
    """Tests for Position entity."""

    def test_valid_position_creation(self) -> None:
        """Valid long position is created correctly."""
        pos: Position = Position(
            asset=_BTC,
            side=Side.LONG,
            size=0.5,
            entry_price=40_000.0,
            entry_time=_T0,
        )
        assert pos.size == pytest.approx(0.5)
        assert pos.unrealized_pnl == pytest.approx(0.0)

    def test_negative_size_raises(self) -> None:
        """Negative position size raises ValidationError."""
        with pytest.raises(ValidationError):
            Position(
                asset=_BTC,
                side=Side.LONG,
                size=-0.5,
                entry_price=40_000.0,
                entry_time=_T0,
            )

    def test_zero_size_raises(self) -> None:
        """Zero position size raises ValidationError."""
        with pytest.raises(ValidationError):
            Position(
                asset=_BTC,
                side=Side.LONG,
                size=0.0,
                entry_price=40_000.0,
                entry_time=_T0,
            )

    def test_negative_entry_price_raises(self) -> None:
        """Negative entry_price raises ValidationError."""
        with pytest.raises(ValidationError):
            Position(
                asset=_BTC,
                side=Side.LONG,
                size=0.5,
                entry_price=-1.0,
                entry_time=_T0,
            )

    def test_position_is_mutable(self) -> None:
        """Position is mutable — unrealized_pnl can be updated."""
        pos: Position = Position(
            asset=_BTC,
            side=Side.LONG,
            size=0.5,
            entry_price=40_000.0,
            entry_time=_T0,
        )
        pos.unrealized_pnl = 500.0
        assert pos.unrealized_pnl == pytest.approx(500.0)

    def test_optional_stop_loss_default_none(self) -> None:
        """stop_loss defaults to None."""
        pos: Position = Position(
            asset=_BTC,
            side=Side.LONG,
            size=0.5,
            entry_price=40_000.0,
            entry_time=_T0,
        )
        assert pos.stop_loss is None

    def test_optional_take_profit_default_none(self) -> None:
        """take_profit defaults to None."""
        pos: Position = Position(
            asset=_BTC,
            side=Side.LONG,
            size=0.5,
            entry_price=40_000.0,
            entry_time=_T0,
        )
        assert pos.take_profit is None


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------


class TestTrade:
    """Tests for Trade entity."""

    def test_valid_trade_creation(self) -> None:
        """Valid trade is created with correct fields."""
        trade: Trade = Trade(
            asset=_BTC,
            side=Side.LONG,
            size=0.25,
            entry_price=40_000.0,
            exit_price=41_000.0,
            entry_time=_T0,
            exit_time=_T1,
            gross_pnl=250.0,
            net_pnl=240.0,
            commission_paid=10.0,
        )
        assert trade.gross_pnl == pytest.approx(250.0)
        assert trade.net_pnl == pytest.approx(240.0)

    def test_exit_before_entry_raises(self) -> None:
        """exit_time before entry_time raises ValidationError."""
        with pytest.raises(ValidationError):
            Trade(
                asset=_BTC,
                side=Side.LONG,
                size=0.25,
                entry_price=40_000.0,
                exit_price=41_000.0,
                entry_time=_T1,
                exit_time=_T0,
                gross_pnl=0.0,
                net_pnl=0.0,
                commission_paid=0.0,
            )

    def test_to_result_strips_asset(self) -> None:
        """to_result() returns a TradeResult without Asset reference."""
        trade: Trade = Trade(
            asset=_BTC,
            side=Side.LONG,
            size=0.25,
            entry_price=40_000.0,
            exit_price=41_000.0,
            entry_time=_T0,
            exit_time=_T1,
            gross_pnl=250.0,
            net_pnl=240.0,
            commission_paid=10.0,
        )
        result: TradeResult = trade.to_result()
        assert result.entry_price == pytest.approx(40_000.0)
        assert result.exit_price == pytest.approx(41_000.0)
        assert result.gross_pnl == pytest.approx(250.0)
        assert not hasattr(result, "asset")

    def test_trade_is_frozen(self) -> None:
        """Trade is frozen — field assignment raises."""
        trade: Trade = Trade(
            asset=_BTC,
            side=Side.LONG,
            size=0.25,
            entry_price=40_000.0,
            exit_price=41_000.0,
            entry_time=_T0,
            exit_time=_T1,
            gross_pnl=250.0,
            net_pnl=240.0,
            commission_paid=10.0,
        )
        with pytest.raises(ValidationError):
            trade.gross_pnl = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EquityCurve
# ---------------------------------------------------------------------------


class TestEquityCurve:
    """Tests for EquityCurve entity validation."""

    def test_valid_equity_curve(self) -> None:
        """Valid equity curve with aligned, monotone timestamps is created."""
        ec: EquityCurve = EquityCurve(
            timestamps=[_T0, _T1, _T2],
            values=[100_000.0, 101_000.0, 102_000.0],
        )
        assert len(ec.timestamps) == 3
        assert len(ec.values) == 3

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched timestamps/values length raises ValidationError."""
        with pytest.raises(ValidationError):
            EquityCurve(
                timestamps=[_T0, _T1],
                values=[100_000.0],
            )

    def test_non_monotone_timestamps_raises(self) -> None:
        """Non-monotone timestamps raise ValidationError."""
        with pytest.raises(ValidationError):
            EquityCurve(
                timestamps=[_T0, _T2, _T1],
                values=[100_000.0, 102_000.0, 101_000.0],
            )

    def test_duplicate_timestamps_raises(self) -> None:
        """Duplicate timestamps (not strictly increasing) raise ValidationError."""
        with pytest.raises(ValidationError):
            EquityCurve(
                timestamps=[_T0, _T0, _T2],
                values=[100_000.0, 100_000.0, 102_000.0],
            )

    def test_single_element_curve_is_valid(self) -> None:
        """Single-element equity curve (no ordering check) is valid."""
        ec: EquityCurve = EquityCurve(
            timestamps=[_T0],
            values=[100_000.0],
        )
        assert ec.values[0] == pytest.approx(100_000.0)

    def test_empty_curve_is_valid(self) -> None:
        """Empty equity curve (zero elements) is valid."""
        ec: EquityCurve = EquityCurve(timestamps=[], values=[])
        assert len(ec.timestamps) == 0

    def test_equity_curve_is_frozen(self) -> None:
        """EquityCurve is frozen — field assignment raises."""
        ec: EquityCurve = EquityCurve(
            timestamps=[_T0, _T1],
            values=[100_000.0, 101_000.0],
        )
        with pytest.raises(ValidationError):
            ec.values = [999.0, 999.0]  # type: ignore[misc]

    def test_values_can_decrease(self) -> None:
        """Equity values may decrease (drawdowns are valid)."""
        ec: EquityCurve = EquityCurve(
            timestamps=[_T0, _T1, _T2],
            values=[100_000.0, 95_000.0, 98_000.0],
        )
        assert ec.values[1] == pytest.approx(95_000.0)

    def test_large_curve_monotone_check(self) -> None:
        """Large equity curve with all increasing timestamps is accepted."""
        n: int = 1000
        base: datetime = datetime(2024, 1, 1, tzinfo=UTC)
        timestamps: list[datetime] = [base + timedelta(hours=i) for i in range(n)]
        values: list[float] = [100_000.0 + i for i in range(n)]
        ec: EquityCurve = EquityCurve(timestamps=timestamps, values=values)
        assert len(ec.timestamps) == n
