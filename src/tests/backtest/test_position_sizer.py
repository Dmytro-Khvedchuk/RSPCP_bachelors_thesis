"""Unit tests for FixedFractionalSizer and RegimeConditionalSizer."""

from __future__ import annotations

from datetime import datetime, UTC

import pytest
from pydantic import ValidationError

from src.app.backtest.application.position_sizer import (
    FixedFractionalSizer,
    RegimeConditionalSizer,
)
from src.app.backtest.domain.entities import Signal
from src.app.backtest.domain.value_objects import PortfolioSnapshot, Side
from src.app.ohlcv.domain.value_objects import Asset
from src.tests.backtest.conftest import make_snapshot


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_T0: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_BTC: Asset = Asset(symbol="BTCUSDT")
_DEFAULT_EQUITY: float = 100_000.0
_DEFAULT_FRACTION: float = 0.02


def _make_signal(strength: float = 1.0, side: Side = Side.LONG) -> Signal:
    """Build a test signal with given strength."""
    return Signal(asset=_BTC, side=side, strength=strength, timestamp=_T0)


# ---------------------------------------------------------------------------
# FixedFractionalSizer
# ---------------------------------------------------------------------------


class TestFixedFractionalSizer:
    """Tests for FixedFractionalSizer."""

    def test_default_fraction_is_two_percent(self) -> None:
        """Default fraction is 0.02."""
        sizer: FixedFractionalSizer = FixedFractionalSizer()
        assert sizer.fraction == pytest.approx(_DEFAULT_FRACTION)

    def test_size_full_strength_returns_fraction_of_equity(self) -> None:
        """At strength=1.0, notional equals equity * fraction."""
        sizer: FixedFractionalSizer = FixedFractionalSizer(fraction=0.02)
        snapshot: PortfolioSnapshot = make_snapshot(equity=_DEFAULT_EQUITY)
        signal: Signal = _make_signal(strength=1.0)
        notional: float = sizer.size(signal, snapshot, 0.0)
        expected: float = _DEFAULT_EQUITY * 0.02 * 1.0
        assert notional == pytest.approx(expected)

    def test_size_half_strength_returns_half_notional(self) -> None:
        """At strength=0.5, notional is half of full-strength notional."""
        sizer: FixedFractionalSizer = FixedFractionalSizer(fraction=0.02)
        snapshot: PortfolioSnapshot = make_snapshot(equity=_DEFAULT_EQUITY)
        full_signal: Signal = _make_signal(strength=1.0)
        half_signal: Signal = _make_signal(strength=0.5)
        full_notional: float = sizer.size(full_signal, snapshot, 0.0)
        half_notional: float = sizer.size(half_signal, snapshot, 0.0)
        assert half_notional == pytest.approx(full_notional * 0.5)

    def test_size_zero_equity_returns_zero(self) -> None:
        """Zero portfolio equity yields zero notional."""
        sizer: FixedFractionalSizer = FixedFractionalSizer(fraction=0.1)
        snapshot: PortfolioSnapshot = make_snapshot(equity=0.0, cash=0.0)
        signal: Signal = _make_signal(strength=1.0)
        notional: float = sizer.size(signal, snapshot, 0.0)
        assert notional == pytest.approx(0.0)

    def test_size_large_equity(self) -> None:
        """Sizer scales correctly with large equity values."""
        equity: float = 10_000_000.0
        fraction: float = 0.05
        sizer: FixedFractionalSizer = FixedFractionalSizer(fraction=fraction)
        snapshot: PortfolioSnapshot = make_snapshot(equity=equity)
        signal: Signal = _make_signal(strength=1.0)
        notional: float = sizer.size(signal, snapshot, 0.0)
        assert notional == pytest.approx(equity * fraction)

    def test_volatility_parameter_is_ignored(self) -> None:
        """FixedFractionalSizer ignores volatility — result is identical for any vol."""
        sizer: FixedFractionalSizer = FixedFractionalSizer(fraction=0.02)
        snapshot: PortfolioSnapshot = make_snapshot(equity=_DEFAULT_EQUITY)
        signal: Signal = _make_signal(strength=1.0)
        notional_low_vol: float = sizer.size(signal, snapshot, 0.001)
        notional_high_vol: float = sizer.size(signal, snapshot, 1.0)
        assert notional_low_vol == pytest.approx(notional_high_vol)

    def test_fraction_above_one_raises(self) -> None:
        """fraction > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            FixedFractionalSizer(fraction=1.1)

    def test_fraction_zero_raises(self) -> None:
        """fraction == 0.0 raises ValidationError (must be > 0)."""
        with pytest.raises(ValidationError):
            FixedFractionalSizer(fraction=0.0)

    def test_fraction_one_is_valid(self) -> None:
        """fraction == 1.0 is a valid edge value."""
        sizer: FixedFractionalSizer = FixedFractionalSizer(fraction=1.0)
        assert sizer.fraction == pytest.approx(1.0)

    def test_different_assets_same_notional(self) -> None:
        """Sizer does not differentiate by asset — notional depends only on equity."""
        sizer: FixedFractionalSizer = FixedFractionalSizer(fraction=0.02)
        snapshot: PortfolioSnapshot = make_snapshot(equity=_DEFAULT_EQUITY)
        btc_signal: Signal = Signal(
            asset=Asset(symbol="BTCUSDT"),
            side=Side.LONG,
            strength=1.0,
            timestamp=_T0,
        )
        eth_signal: Signal = Signal(
            asset=Asset(symbol="ETHUSDT"),
            side=Side.LONG,
            strength=1.0,
            timestamp=_T0,
        )
        assert sizer.size(btc_signal, snapshot, 0.0) == pytest.approx(sizer.size(eth_signal, snapshot, 0.0))

    @pytest.mark.parametrize(
        ("fraction", "equity", "strength", "expected"),
        [
            (0.01, 100_000.0, 1.0, 1_000.0),
            (0.05, 50_000.0, 0.5, 1_250.0),
            (0.10, 200_000.0, 0.25, 5_000.0),
            (1.00, 100_000.0, 1.0, 100_000.0),
        ],
    )
    def test_size_parametrized_known_values(
        self,
        fraction: float,
        equity: float,
        strength: float,
        expected: float,
    ) -> None:
        """Parametrized check: notional matches equity * fraction * strength."""
        sizer: FixedFractionalSizer = FixedFractionalSizer(fraction=fraction)
        snapshot: PortfolioSnapshot = make_snapshot(equity=equity)
        signal: Signal = _make_signal(strength=strength)
        notional: float = sizer.size(signal, snapshot, 0.0)
        assert notional == pytest.approx(expected)


# ---------------------------------------------------------------------------
# RegimeConditionalSizer
# ---------------------------------------------------------------------------


class TestRegimeConditionalSizer:
    """Tests for RegimeConditionalSizer."""

    def _make_sizer(
        self,
        *,
        da_estimate: float = 0.60,
        pe_estimate: float = 0.90,
        break_even_da: float = 0.55,
        high_entropy_threshold: float = 0.98,
        high_entropy_reduction: float = 0.5,
        base_fraction: float = 0.02,
        is_tier_b: bool = False,
        tier_b_kelly_multiplier: float = 0.5,
    ) -> RegimeConditionalSizer:
        """Build a RegimeConditionalSizer with given parameters."""
        return RegimeConditionalSizer(
            base_fraction=base_fraction,
            break_even_da=break_even_da,
            high_entropy_threshold=high_entropy_threshold,
            high_entropy_reduction=high_entropy_reduction,
            da_estimate=da_estimate,
            pe_estimate=pe_estimate,
            tier_b_kelly_multiplier=tier_b_kelly_multiplier,
            is_tier_b=is_tier_b,
        )

    def test_above_breakeven_da_returns_positive(self) -> None:
        """DA above break-even yields positive notional."""
        sizer: RegimeConditionalSizer = self._make_sizer(da_estimate=0.60, break_even_da=0.55)
        snapshot: PortfolioSnapshot = make_snapshot(equity=_DEFAULT_EQUITY)
        signal: Signal = _make_signal(strength=1.0)
        notional: float = sizer.size(signal, snapshot, 0.0)
        assert notional > 0.0

    def test_at_breakeven_da_returns_zero(self) -> None:
        """DA exactly at break-even returns zero (no edge)."""
        sizer: RegimeConditionalSizer = self._make_sizer(da_estimate=0.55, break_even_da=0.55)
        snapshot: PortfolioSnapshot = make_snapshot(equity=_DEFAULT_EQUITY)
        signal: Signal = _make_signal(strength=1.0)
        notional: float = sizer.size(signal, snapshot, 0.0)
        assert notional == pytest.approx(0.0)

    def test_below_breakeven_da_returns_zero(self) -> None:
        """DA below break-even returns zero (no-trade signal)."""
        sizer: RegimeConditionalSizer = self._make_sizer(da_estimate=0.50, break_even_da=0.55)
        snapshot: PortfolioSnapshot = make_snapshot(equity=_DEFAULT_EQUITY)
        signal: Signal = _make_signal(strength=1.0)
        notional: float = sizer.size(signal, snapshot, 0.0)
        assert notional == pytest.approx(0.0)

    def test_zero_equity_returns_zero(self) -> None:
        """Zero equity yields zero notional even with good DA."""
        sizer: RegimeConditionalSizer = self._make_sizer(da_estimate=0.70, break_even_da=0.55)
        snapshot: PortfolioSnapshot = make_snapshot(equity=0.0, cash=0.0)
        signal: Signal = _make_signal(strength=1.0)
        notional: float = sizer.size(signal, snapshot, 0.0)
        assert notional == pytest.approx(0.0)

    def test_high_entropy_applies_reduction(self) -> None:
        """PE above threshold halves notional (reduction=0.5)."""
        sizer_normal: RegimeConditionalSizer = self._make_sizer(
            da_estimate=0.60,
            pe_estimate=0.95,
            high_entropy_threshold=0.98,
            high_entropy_reduction=0.5,
        )
        sizer_high_pe: RegimeConditionalSizer = self._make_sizer(
            da_estimate=0.60,
            pe_estimate=0.99,
            high_entropy_threshold=0.98,
            high_entropy_reduction=0.5,
        )
        snapshot: PortfolioSnapshot = make_snapshot(equity=_DEFAULT_EQUITY)
        signal: Signal = _make_signal(strength=1.0)
        normal_notional: float = sizer_normal.size(signal, snapshot, 0.0)
        high_pe_notional: float = sizer_high_pe.size(signal, snapshot, 0.0)
        assert high_pe_notional == pytest.approx(normal_notional * 0.5)

    def test_low_entropy_no_reduction(self) -> None:
        """PE at or below threshold does not apply entropy reduction."""
        sizer: RegimeConditionalSizer = self._make_sizer(
            da_estimate=0.60,
            pe_estimate=0.97,
            high_entropy_threshold=0.98,
            high_entropy_reduction=0.5,
        )
        snapshot: PortfolioSnapshot = make_snapshot(equity=_DEFAULT_EQUITY)
        signal: Signal = _make_signal(strength=1.0)
        notional: float = sizer.size(signal, snapshot, 0.0)
        # Should equal base * strength * da_scale without reduction
        da_scale: float = (0.60 - 0.55) / 0.55
        expected: float = _DEFAULT_EQUITY * 0.02 * 1.0 * da_scale
        assert notional == pytest.approx(expected, rel=1e-6)

    def test_tier_b_applies_kelly_multiplier(self) -> None:
        """Tier-B flag reduces notional by tier_b_kelly_multiplier."""
        sizer_a: RegimeConditionalSizer = self._make_sizer(
            da_estimate=0.60, is_tier_b=False, tier_b_kelly_multiplier=0.5
        )
        sizer_b: RegimeConditionalSizer = self._make_sizer(
            da_estimate=0.60, is_tier_b=True, tier_b_kelly_multiplier=0.5
        )
        snapshot: PortfolioSnapshot = make_snapshot(equity=_DEFAULT_EQUITY)
        signal: Signal = _make_signal(strength=1.0)
        tier_a_notional: float = sizer_a.size(signal, snapshot, 0.0)
        tier_b_notional: float = sizer_b.size(signal, snapshot, 0.0)
        assert tier_b_notional == pytest.approx(tier_a_notional * 0.5)

    def test_both_reductions_stack_multiplicatively(self) -> None:
        """High entropy AND tier-B reductions multiply together."""
        sizer: RegimeConditionalSizer = self._make_sizer(
            da_estimate=0.60,
            pe_estimate=0.99,
            high_entropy_threshold=0.98,
            high_entropy_reduction=0.5,
            is_tier_b=True,
            tier_b_kelly_multiplier=0.5,
        )
        sizer_base: RegimeConditionalSizer = self._make_sizer(
            da_estimate=0.60,
            pe_estimate=0.90,
            high_entropy_threshold=0.98,
            high_entropy_reduction=0.5,
            is_tier_b=False,
            tier_b_kelly_multiplier=0.5,
        )
        snapshot: PortfolioSnapshot = make_snapshot(equity=_DEFAULT_EQUITY)
        signal: Signal = _make_signal(strength=1.0)
        base_notional: float = sizer_base.size(signal, snapshot, 0.0)
        reduced_notional: float = sizer.size(signal, snapshot, 0.0)
        assert reduced_notional == pytest.approx(base_notional * 0.5 * 0.5)

    def test_strength_scales_notional(self) -> None:
        """Lower signal strength proportionally reduces notional."""
        sizer: RegimeConditionalSizer = self._make_sizer(da_estimate=0.60)
        snapshot: PortfolioSnapshot = make_snapshot(equity=_DEFAULT_EQUITY)
        full: Signal = _make_signal(strength=1.0)
        half: Signal = _make_signal(strength=0.5)
        full_notional: float = sizer.size(full, snapshot, 0.0)
        half_notional: float = sizer.size(half, snapshot, 0.0)
        assert half_notional == pytest.approx(full_notional * 0.5)

    def test_higher_da_increases_notional(self) -> None:
        """Higher DA estimate produces larger notional (greater edge)."""
        sizer_low_da: RegimeConditionalSizer = self._make_sizer(da_estimate=0.57)
        sizer_high_da: RegimeConditionalSizer = self._make_sizer(da_estimate=0.65)
        snapshot: PortfolioSnapshot = make_snapshot(equity=_DEFAULT_EQUITY)
        signal: Signal = _make_signal(strength=1.0)
        low_notional: float = sizer_low_da.size(signal, snapshot, 0.0)
        high_notional: float = sizer_high_da.size(signal, snapshot, 0.0)
        assert high_notional > low_notional

    def test_da_exactly_above_breakeven_produces_positive(self) -> None:
        """Infinitesimally above break-even DA yields positive notional."""
        sizer: RegimeConditionalSizer = self._make_sizer(da_estimate=0.550_001, break_even_da=0.55)
        snapshot: PortfolioSnapshot = make_snapshot(equity=_DEFAULT_EQUITY)
        signal: Signal = _make_signal(strength=1.0)
        notional: float = sizer.size(signal, snapshot, 0.0)
        assert notional > 0.0
