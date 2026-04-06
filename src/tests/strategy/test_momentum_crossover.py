"""Unit tests for the MomentumCrossover strategy."""

from __future__ import annotations

import polars as pl
import pytest

from src.app.features.domain.value_objects import FeatureSet
from src.app.strategy.application.momentum_crossover import (
    MomentumCrossover,
    MomentumCrossoverConfig,
)
from src.tests.strategy.conftest import make_feature_set, _make_ohlcv_base

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N: int = 60
_THRESHOLD: float = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_ohlcv(n: int = _N) -> FeatureSet:
    closes = [40_000.0] * n
    df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
    return make_feature_set(df)


def _make_strategy(threshold: float = 0.0) -> MomentumCrossover:
    return MomentumCrossover(config=MomentumCrossoverConfig(signal_threshold=threshold))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMomentumCrossoverName:
    def test_name_returns_expected_string(self) -> None:
        strategy = _make_strategy()
        assert strategy.name == "momentum_crossover"

    def test_name_unchanged_with_custom_config(self) -> None:
        strategy = MomentumCrossover(
            config=MomentumCrossoverConfig(signal_threshold=0.2, xover_column="ema_xover_8_21")
        )
        assert strategy.name == "momentum_crossover"


class TestMomentumCrossoverOutputSchema:
    def test_output_has_required_columns(self) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(_flat_ohlcv())
        assert "timestamp" in result.columns
        assert "side" in result.columns
        assert "strength" in result.columns

    def test_output_row_count_matches_input(self) -> None:
        strategy = _make_strategy()
        fs = _flat_ohlcv()
        result = strategy.generate_signals(fs)
        assert len(result) == len(fs.df)

    def test_strength_always_in_zero_one_range(self) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(_flat_ohlcv())
        strengths = result["strength"].to_list()
        for s in strengths:
            assert 0.0 <= s <= 1.0

    def test_side_values_are_valid_strings(self) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(_flat_ohlcv())
        valid_sides = {"long", "short", "flat"}
        for side in result["side"].to_list():
            assert side in valid_sides


class TestMomentumCrossoverLongSignals:
    def test_positive_xover_above_threshold_produces_long(self) -> None:
        n = _N
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        fs = make_feature_set(df, xover_values=[0.5] * n)
        strategy = _make_strategy(threshold=0.1)
        result = strategy.generate_signals(fs)
        sides = result["side"].to_list()
        assert all(s == "long" for s in sides)

    def test_xover_exactly_at_threshold_is_flat(self) -> None:
        n = _N
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        # xover == threshold: condition is > threshold so it should be flat
        fs = make_feature_set(df, xover_values=[_THRESHOLD] * n)
        strategy = _make_strategy(threshold=_THRESHOLD)
        result = strategy.generate_signals(fs)
        sides = result["side"].to_list()
        assert all(s == "flat" for s in sides)

    def test_strength_equals_clipped_xover_for_long(self) -> None:
        n = 5
        xover_val = 0.7
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        fs = make_feature_set(df, xover_values=[xover_val] * n)
        strategy = _make_strategy(threshold=0.0)
        result = strategy.generate_signals(fs)
        for s in result["strength"].to_list():
            assert abs(s - xover_val) < 1e-9

    def test_strength_clipped_at_one_for_large_xover(self) -> None:
        n = 5
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        fs = make_feature_set(df, xover_values=[5.0] * n)  # well above 1.0
        strategy = _make_strategy(threshold=0.0)
        result = strategy.generate_signals(fs)
        for s in result["strength"].to_list():
            assert s == pytest.approx(1.0)


class TestMomentumCrossoverShortSignals:
    def test_negative_xover_below_threshold_produces_short(self) -> None:
        n = _N
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        fs = make_feature_set(df, xover_values=[-0.5] * n)
        strategy = _make_strategy(threshold=0.1)
        result = strategy.generate_signals(fs)
        sides = result["side"].to_list()
        assert all(s == "short" for s in sides)

    def test_xover_exactly_neg_threshold_is_flat(self) -> None:
        n = _N
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        # xover == -threshold: condition is < -threshold so it should be flat
        fs = make_feature_set(df, xover_values=[-_THRESHOLD] * n)
        strategy = _make_strategy(threshold=_THRESHOLD)
        result = strategy.generate_signals(fs)
        sides = result["side"].to_list()
        assert all(s == "flat" for s in sides)

    def test_strength_clipped_at_one_for_large_negative_xover(self) -> None:
        n = 5
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        fs = make_feature_set(df, xover_values=[-5.0] * n)
        strategy = _make_strategy(threshold=0.0)
        result = strategy.generate_signals(fs)
        for s in result["strength"].to_list():
            assert s == pytest.approx(1.0)


class TestMomentumCrossoverFlatSignals:
    def test_zero_xover_produces_flat(self) -> None:
        n = _N
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        fs = make_feature_set(df, xover_values=[0.0] * n)
        strategy = _make_strategy(threshold=0.0)
        result = strategy.generate_signals(fs)
        # xover==0.0 is not > 0.0 and not < 0.0 (with threshold=0), so flat
        sides = result["side"].to_list()
        assert all(s == "flat" for s in sides)

    def test_flat_bars_produce_zero_strength(self) -> None:
        n = _N
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        fs = make_feature_set(df, xover_values=[0.0] * n)
        strategy = _make_strategy(threshold=0.1)
        result = strategy.generate_signals(fs)
        for s in result["strength"].to_list():
            assert s == pytest.approx(0.0)

    def test_small_xover_within_threshold_is_flat(self) -> None:
        n = _N
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        fs = make_feature_set(df, xover_values=[0.05] * n)
        strategy = _make_strategy(threshold=0.1)
        result = strategy.generate_signals(fs)
        sides = result["side"].to_list()
        assert all(s == "flat" for s in sides)


class TestMomentumCrossoverOnSyntheticData:
    def test_trending_up_produces_mostly_long(self, trending_up_feature_set: FeatureSet) -> None:
        strategy = _make_strategy(threshold=0.1)
        result = strategy.generate_signals(trending_up_feature_set)
        sides = result["side"].to_list()
        long_count = sides.count("long")
        assert long_count > len(sides) // 2

    def test_trending_down_produces_mostly_short(self, trending_down_feature_set: FeatureSet) -> None:
        strategy = _make_strategy(threshold=0.1)
        result = strategy.generate_signals(trending_down_feature_set)
        sides = result["side"].to_list()
        short_count = sides.count("short")
        assert short_count > len(sides) // 2

    def test_flat_data_produces_all_flat(self, flat_feature_set: FeatureSet) -> None:
        strategy = _make_strategy(threshold=0.0)
        result = strategy.generate_signals(flat_feature_set)
        sides = result["side"].to_list()
        assert all(s == "flat" for s in sides)


class TestMomentumCrossoverCustomConfig:
    def test_custom_xover_column_name(self) -> None:
        n = _N
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        # Add a custom crossover column
        custom_col = "my_custom_xover"
        enriched = df.with_columns(pl.lit(0.5).alias(custom_col))
        fs = FeatureSet(
            df=enriched,
            feature_columns=(custom_col,),
            target_columns=(),
            n_rows_raw=n,
            n_rows_clean=n,
        )
        strategy = MomentumCrossover(config=MomentumCrossoverConfig(xover_column=custom_col, signal_threshold=0.1))
        result = strategy.generate_signals(fs)
        sides = result["side"].to_list()
        assert all(s == "long" for s in sides)

    def test_high_threshold_produces_fewer_signals_than_low_threshold(self) -> None:
        n = _N
        import math as _math

        # oscillating xover between -0.5 and 0.5
        xover = [0.4 * _math.sin(2 * _math.pi * i / 10) for i in range(n)]
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        fs = make_feature_set(df, xover_values=xover)

        low_thresh_result = _make_strategy(threshold=0.1).generate_signals(fs)
        high_thresh_result = _make_strategy(threshold=0.45).generate_signals(fs)

        low_directional = sum(1 for s in low_thresh_result["side"].to_list() if s != "flat")
        high_directional = sum(1 for s in high_thresh_result["side"].to_list() if s != "flat")
        assert high_directional < low_directional
