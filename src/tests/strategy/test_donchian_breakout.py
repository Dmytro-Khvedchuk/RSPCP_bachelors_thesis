"""Unit tests for the DonchianBreakout strategy."""

from __future__ import annotations


import pytest

from src.app.features.domain.value_objects import FeatureSet
from src.app.strategy.application.donchian_breakout import (
    DonchianBreakout,
    DonchianBreakoutConfig,
)
from src.tests.strategy.conftest import make_feature_set, _make_ohlcv_base

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N: int = 60
_CHANNEL_PERIOD: int = 5  # small period so warmup is only 5+1 bars
_TYPICAL_ATR: float = 200.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_breakout_feature_set(
    warmup: int,
    extra: int,
    n: int,
    *,
    breakout_close: float,
    breakout_high: float,
    lows: list[float],
) -> FeatureSet:
    """Build a FeatureSet with flat warmup bars followed by breakout bars."""
    closes = [40_000.0] * warmup + [breakout_close] * extra
    highs = [40_200.0] * warmup + [breakout_high] * extra
    df = _make_ohlcv_base(n, closes, highs, lows)
    return make_feature_set(df, atr_values=[_TYPICAL_ATR] * n)


def _make_strategy(
    channel_period: int = _CHANNEL_PERIOD,
    atr_column: str = "atr_14",
) -> DonchianBreakout:
    return DonchianBreakout(config=DonchianBreakoutConfig(channel_period=channel_period, atr_column=atr_column))


def _flat_fs(n: int = _N) -> FeatureSet:
    closes = [40_000.0] * n
    df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
    return make_feature_set(df, atr_values=[_TYPICAL_ATR] * n)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDonchianBreakoutName:
    def test_name_returns_expected_string(self) -> None:
        strategy = _make_strategy()
        assert strategy.name == "donchian_breakout"

    def test_name_unchanged_with_custom_config(self) -> None:
        strategy = DonchianBreakout(config=DonchianBreakoutConfig(channel_period=20))
        assert strategy.name == "donchian_breakout"


class TestDonchianBreakoutOutputSchema:
    def test_output_has_required_columns(self) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(_flat_fs())
        assert "timestamp" in result.columns
        assert "side" in result.columns
        assert "strength" in result.columns

    def test_output_row_count_matches_input(self) -> None:
        strategy = _make_strategy()
        fs = _flat_fs()
        result = strategy.generate_signals(fs)
        assert len(result) == len(fs.df)

    def test_strength_always_in_zero_one_range(self) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(_flat_fs())
        for s in result["strength"].to_list():
            if s is not None:
                assert 0.0 <= s <= 1.0

    def test_side_values_only_long_or_flat(self) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(_flat_fs())
        valid_sides = {"long", "flat"}
        for side in result["side"].to_list():
            if side is not None:
                assert side in valid_sides


class TestDonchianBreakoutLongOnlyConstraint:
    def test_never_produces_short_signal(self, trending_down_feature_set: FeatureSet) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(trending_down_feature_set)
        sides = [s for s in result["side"].to_list() if s is not None]
        assert "short" not in sides

    def test_never_produces_short_on_flat_data(self) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(_flat_fs())
        sides = [s for s in result["side"].to_list() if s is not None]
        assert "short" not in sides


class TestDonchianBreakoutChannelLogic:
    def test_close_exceeds_prior_high_produces_long(self) -> None:
        # Build a price series: first channel_period bars at 40000,
        # then a breakout bar at 42000 (above rolling max of highs=40200)
        n = _CHANNEL_PERIOD + 5
        closes = [40_000.0] * _CHANNEL_PERIOD + [42_000.0] * 5
        highs = [c + 200.0 for c in closes]
        lows = [c - 200.0 for c in closes]
        df = _make_ohlcv_base(n, closes, highs, lows)
        fs = make_feature_set(df, atr_values=[_TYPICAL_ATR] * n)
        strategy = _make_strategy(channel_period=_CHANNEL_PERIOD)
        result = strategy.generate_signals(fs)
        # After warmup (channel_period bars after shift), breakout bars should be "long"
        last_sides = [s for s in result["side"].to_list()[_CHANNEL_PERIOD:] if s is not None]
        assert any(s == "long" for s in last_sides)

    def test_close_below_channel_produces_flat(self) -> None:
        n = _N
        # All closes at 40000, highs at 40200, so channel = 40200
        # If close stays at 40000 it never exceeds the channel
        closes = [40_000.0] * n
        highs = [40_200.0] * n
        lows = [39_800.0] * n
        df = _make_ohlcv_base(n, closes, highs, lows)
        fs = make_feature_set(df, atr_values=[_TYPICAL_ATR] * n)
        strategy = _make_strategy(channel_period=_CHANNEL_PERIOD)
        result = strategy.generate_signals(fs)
        non_null_sides = [s for s in result["side"].to_list() if s is not None]
        assert all(s == "flat" for s in non_null_sides)

    def test_first_bars_before_warmup_are_flat(self) -> None:
        # The first channel_period + 1 bars have no valid prior channel (shift+rolling)
        n = _CHANNEL_PERIOD + 2
        closes = [40_000.0] * n
        highs = [c + 200.0 for c in closes]
        lows = [c - 200.0 for c in closes]
        df = _make_ohlcv_base(n, closes, highs, lows)
        fs = make_feature_set(df, atr_values=[_TYPICAL_ATR] * n)
        strategy = _make_strategy(channel_period=_CHANNEL_PERIOD)
        result = strategy.generate_signals(fs)
        # Before warmup, _dc_upper is null -> otherwise branch -> "flat"
        early_sides = [s for s in result["side"].to_list()[:_CHANNEL_PERIOD] if s is not None]
        assert all(s == "flat" for s in early_sides)


class TestDonchianBreakoutShiftPreventsLookahead:
    def test_bar_zero_is_always_flat(self) -> None:
        # Bar 0 has no prior data after shift(1), so channel is null -> flat
        n = _N
        closes = [50_000.0] * n  # close starts high
        highs = [c + 100.0 for c in closes]
        lows = [c - 100.0 for c in closes]
        df = _make_ohlcv_base(n, closes, highs, lows)
        fs = make_feature_set(df, atr_values=[_TYPICAL_ATR] * n)
        strategy = _make_strategy(channel_period=_CHANNEL_PERIOD)
        result = strategy.generate_signals(fs)
        first_side = result["side"][0]
        assert first_side == "flat"

    def test_sudden_spike_at_bar_one_does_not_trigger_breakout(self) -> None:
        # Bar 0: close 40000, Bar 1: close 100000 (huge spike)
        # Since shift(1) is applied first, bar 1's channel is based on shifted-bar-0 high
        # which is shift(1) of a 1-bar window — we don't have enough bars for rolling_max
        n = _CHANNEL_PERIOD + 2
        closes = [40_000.0] + [100_000.0] + [40_000.0] * (_CHANNEL_PERIOD)
        highs = [c + 200.0 for c in closes]
        lows = [c - 200.0 for c in closes]
        df = _make_ohlcv_base(n, closes, highs, lows)
        fs = make_feature_set(df, atr_values=[_TYPICAL_ATR] * n)
        strategy = _make_strategy(channel_period=_CHANNEL_PERIOD)
        result = strategy.generate_signals(fs)
        # Bar 1 has only 1 prior bar after shift, not enough for channel_period rolling
        bar_1_side = result["side"][1]
        assert bar_1_side == "flat"


class TestDonchianBreakoutStrength:
    def test_strength_zero_when_no_breakout(self) -> None:
        n = _N
        closes = [40_000.0] * n
        highs = [40_200.0] * n
        lows = [39_800.0] * n
        df = _make_ohlcv_base(n, closes, highs, lows)
        fs = make_feature_set(df, atr_values=[_TYPICAL_ATR] * n)
        strategy = _make_strategy()
        result = strategy.generate_signals(fs)
        non_null_strengths = [s for s in result["strength"].to_list() if s is not None]
        assert all(s == pytest.approx(0.0) for s in non_null_strengths)

    def test_larger_breakout_produces_higher_strength(self) -> None:
        # Build two scenarios:
        # Warmup: CHANNEL_PERIOD bars with flat high=40200 (channel=40200 after shift+rolling)
        # Breakout bars: close exceeds 40200 by different amounts
        warmup = _CHANNEL_PERIOD
        extra = 5
        n = warmup + extra
        lows = [39_800.0] * n

        fs_s = _build_breakout_feature_set(
            warmup, extra, n, breakout_close=40_300.0, breakout_high=40_500.0, lows=lows
        )
        fs_l = _build_breakout_feature_set(
            warmup, extra, n, breakout_close=40_800.0, breakout_high=41_000.0, lows=lows
        )

        strategy = _make_strategy(channel_period=_CHANNEL_PERIOD)
        result_s = strategy.generate_signals(fs_s)
        result_l = strategy.generate_signals(fs_l)

        strengths_s = [s for s in result_s["strength"].to_list()[-extra:] if s is not None and s > 0]
        strengths_l = [s for s in result_l["strength"].to_list()[-extra:] if s is not None and s > 0]

        assert strengths_s, "expected some breakout bars in small breakout scenario"
        assert strengths_l, "expected some breakout bars in large breakout scenario"
        assert max(strengths_l) > max(strengths_s)


class TestDonchianBreakoutOnSyntheticData:
    def test_trending_up_produces_long_signals_after_warmup(self) -> None:
        # Explicitly craft a breakout scenario:
        # - warmup bars have flat high=40100 so the channel settles at 40100
        # - breakout bars have close=40200 which exceeds the channel
        warmup = _CHANNEL_PERIOD + 1
        extra = 5
        n = warmup + extra
        closes = [40_000.0] * warmup + [40_200.0] * extra
        highs = [40_100.0] * warmup + [40_400.0] * extra
        lows = [39_900.0] * n
        df = _make_ohlcv_base(n, closes, highs, lows)
        fs = make_feature_set(df, atr_values=[_TYPICAL_ATR] * n)
        strategy = _make_strategy(channel_period=_CHANNEL_PERIOD)
        result = strategy.generate_signals(fs)
        post_warmup_sides = [s for s in result["side"].to_list()[warmup:] if s is not None]
        assert any(s == "long" for s in post_warmup_sides)

    def test_flat_prices_produce_no_breakout(self) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(_flat_fs())
        sides = [s for s in result["side"].to_list() if s is not None]
        assert "long" not in sides
