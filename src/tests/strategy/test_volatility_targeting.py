"""Unit tests for the VolatilityTargeting strategy."""

from __future__ import annotations

import pytest

from src.app.features.domain.value_objects import FeatureSet
from src.app.strategy.application.volatility_targeting import (
    VolatilityTargeting,
    VolatilityTargetingConfig,
)
from src.tests.strategy.conftest import make_feature_set, _make_ohlcv_base

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N: int = 60
_TARGET_VOL: float = 0.15
_EPSILON: float = 1e-12  # matches internal _EPSILON constant in source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_strategy(target_vol: float = _TARGET_VOL, rv_col: str = "rv_24") -> VolatilityTargeting:
    return VolatilityTargeting(config=VolatilityTargetingConfig(target_vol=target_vol, rv_column=rv_col))


def _fs_with_rv(rv: float, n: int = _N) -> FeatureSet:
    closes = [40_000.0] * n
    df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
    return make_feature_set(df, rv_values=[rv] * n)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVolatilityTargetingName:
    def test_name_returns_expected_string(self) -> None:
        strategy = _make_strategy()
        assert strategy.name == "volatility_targeting"

    def test_name_unchanged_with_custom_config(self) -> None:
        strategy = VolatilityTargeting(config=VolatilityTargetingConfig(target_vol=0.2))
        assert strategy.name == "volatility_targeting"


class TestVolatilityTargetingOutputSchema:
    def test_output_has_required_columns(self, flat_feature_set: FeatureSet) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(flat_feature_set)
        assert "timestamp" in result.columns
        assert "side" in result.columns
        assert "strength" in result.columns

    def test_output_row_count_matches_input(self, flat_feature_set: FeatureSet) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(flat_feature_set)
        assert len(result) == len(flat_feature_set.df)

    def test_strength_always_in_zero_one_range(self, flat_feature_set: FeatureSet) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(flat_feature_set)
        for s in result["strength"].to_list():
            assert 0.0 <= s <= 1.0

    def test_side_dtype_is_utf8(self, flat_feature_set: FeatureSet) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(flat_feature_set)
        assert result["side"].dtype == pl.Utf8 or result["side"].dtype == pl.String


class TestVolatilityTargetingAlwaysLong:
    def test_all_bars_are_long(self, flat_feature_set: FeatureSet) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(flat_feature_set)
        sides = result["side"].to_list()
        assert all(s == "long" for s in sides)

    def test_never_produces_short(self, high_vol_feature_set: FeatureSet) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(high_vol_feature_set)
        sides = result["side"].to_list()
        assert "short" not in sides

    def test_never_produces_flat(self, low_vol_feature_set: FeatureSet) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(low_vol_feature_set)
        sides = result["side"].to_list()
        assert "flat" not in sides

    def test_all_long_on_trending_up(self, trending_up_feature_set: FeatureSet) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(trending_up_feature_set)
        sides = result["side"].to_list()
        assert all(s == "long" for s in sides)

    def test_all_long_on_trending_down(self, trending_down_feature_set: FeatureSet) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(trending_down_feature_set)
        sides = result["side"].to_list()
        assert all(s == "long" for s in sides)


class TestVolatilityTargetingStrengthCalculation:
    def test_strength_equals_target_over_rv_when_below_one(self) -> None:
        target = 0.15
        rv = 0.30  # rv > target, so strength = target/rv = 0.5
        n = 5
        fs = _fs_with_rv(rv, n)
        strategy = _make_strategy(target_vol=target)
        result = strategy.generate_signals(fs)
        expected = target / (rv + _EPSILON)
        for s in result["strength"].to_list():
            assert s == pytest.approx(expected, rel=1e-6)

    def test_low_rv_clipped_to_one(self) -> None:
        # rv << target -> target/rv >> 1 -> clipped to 1
        target = 0.15
        rv = 0.001
        n = 5
        fs = _fs_with_rv(rv, n)
        strategy = _make_strategy(target_vol=target)
        result = strategy.generate_signals(fs)
        for s in result["strength"].to_list():
            assert s == pytest.approx(1.0)

    def test_rv_equals_target_produces_strength_near_one(self) -> None:
        target = 0.15
        rv = target
        n = 5
        fs = _fs_with_rv(rv, n)
        strategy = _make_strategy(target_vol=target)
        result = strategy.generate_signals(fs)
        for s in result["strength"].to_list():
            assert s == pytest.approx(target / (rv + _EPSILON), rel=1e-5)

    def test_high_rv_produces_low_strength(self) -> None:
        target = 0.15
        rv = 3.0  # 20x target
        n = 5
        fs = _fs_with_rv(rv, n)
        strategy = _make_strategy(target_vol=target)
        result = strategy.generate_signals(fs)
        for s in result["strength"].to_list():
            assert s < 0.1

    def test_near_zero_rv_does_not_produce_infinity(self) -> None:
        rv = 0.0  # edge case: rv == 0 -> uses epsilon
        n = 5
        fs = _fs_with_rv(rv, n)
        strategy = _make_strategy()
        result = strategy.generate_signals(fs)
        for s in result["strength"].to_list():
            assert 0.0 <= s <= 1.0

    def test_high_vol_produces_lower_strength_than_low_vol(self) -> None:
        high_rv_fs = _fs_with_rv(1.0)
        low_rv_fs = _fs_with_rv(0.05)
        strategy = _make_strategy(target_vol=0.15)
        high_result = strategy.generate_signals(high_rv_fs)
        low_result = strategy.generate_signals(low_rv_fs)
        avg_high = sum(high_result["strength"].to_list()) / _N
        avg_low = sum(low_result["strength"].to_list()) / _N
        assert avg_low > avg_high

    def test_varying_rv_produces_varying_strength(self) -> None:
        n = 10
        rv_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        fs = make_feature_set(df, rv_values=rv_values)
        strategy = _make_strategy(target_vol=0.3)
        result = strategy.generate_signals(fs)
        strengths = result["strength"].to_list()
        # Increasing rv should produce decreasing strength
        for i in range(len(strengths) - 1):
            assert strengths[i] >= strengths[i + 1]


class TestVolatilityTargetingCustomConfig:
    def test_higher_target_vol_produces_higher_strength(self) -> None:
        rv = 0.3
        n = 5
        fs = _fs_with_rv(rv, n)
        low_target_strategy = _make_strategy(target_vol=0.1)
        high_target_strategy = _make_strategy(target_vol=0.5)
        low_result = low_target_strategy.generate_signals(fs)
        high_result = high_target_strategy.generate_signals(fs)
        low_avg = sum(low_result["strength"].to_list()) / n
        high_avg = sum(high_result["strength"].to_list()) / n
        assert high_avg > low_avg


# import needed for dtype check
import polars as pl  # noqa: E402
