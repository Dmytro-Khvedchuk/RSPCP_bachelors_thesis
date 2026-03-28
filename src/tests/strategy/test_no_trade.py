"""Unit tests for the NoTrade strategy."""

from __future__ import annotations

import pytest

from src.app.features.domain.value_objects import FeatureSet
from src.app.strategy.application.no_trade import NoTrade, NoTradeConfig
from src.tests.strategy.conftest import make_feature_set, _make_ohlcv_base

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N: int = 60
_PE_THRESHOLD: float = 0.98
_LOW_VOL_THRESHOLD: float = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_strategy(
    pe_threshold: float = _PE_THRESHOLD,
    pe_value: float = 0.5,
    rv_column: str = "rv_24",
    low_vol_threshold: float = _LOW_VOL_THRESHOLD,
) -> NoTrade:
    return NoTrade(
        config=NoTradeConfig(
            pe_threshold=pe_threshold,
            pe_value=pe_value,
            rv_column=rv_column,
            low_vol_threshold=low_vol_threshold,
        )
    )


def _flat_fs(n: int = _N, rv: float = 0.1) -> FeatureSet:
    closes = [40_000.0] * n
    df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
    return make_feature_set(df, rv_values=[rv] * n)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoTradeName:
    def test_name_returns_expected_string(self) -> None:
        strategy = _make_strategy()
        assert strategy.name == "no_trade"

    def test_name_unchanged_with_custom_config(self) -> None:
        strategy = NoTrade(config=NoTradeConfig(pe_threshold=0.9, pe_value=0.5))
        assert strategy.name == "no_trade"


class TestNoTradeOutputSchema:
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
            assert 0.0 <= s <= 1.0


class TestNoTradeAlwaysFlat:
    def test_all_bars_are_flat_pe_gate_active(self) -> None:
        # pe_value > pe_threshold -> global gate active
        strategy = _make_strategy(pe_threshold=0.9, pe_value=0.95)
        result = strategy.generate_signals(_flat_fs())
        sides = result["side"].to_list()
        assert all(s == "flat" for s in sides)

    def test_all_bars_are_flat_pe_gate_inactive(self) -> None:
        # pe_value < pe_threshold -> per-bar filter
        strategy = _make_strategy(pe_threshold=0.98, pe_value=0.5)
        result = strategy.generate_signals(_flat_fs())
        sides = result["side"].to_list()
        assert all(s == "flat" for s in sides)

    def test_never_produces_long(self, trending_up_feature_set: FeatureSet) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(trending_up_feature_set)
        sides = result["side"].to_list()
        assert "long" not in sides

    def test_never_produces_short(self, mean_reverting_feature_set: FeatureSet) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(mean_reverting_feature_set)
        sides = result["side"].to_list()
        assert "short" not in sides


class TestNoTradeGlobalPEGate:
    def test_pe_above_threshold_all_strength_one(self) -> None:
        # pe_value (0.99) > pe_threshold (0.98) -> all bars get strength=1.0
        strategy = _make_strategy(pe_threshold=0.98, pe_value=0.99)
        result = strategy.generate_signals(_flat_fs())
        for s in result["strength"].to_list():
            assert s == pytest.approx(1.0)

    def test_pe_exactly_at_threshold_does_not_trigger_gate(self) -> None:
        # pe_value == pe_threshold: condition is > threshold so gate NOT active
        strategy = _make_strategy(pe_threshold=0.98, pe_value=0.98, low_vol_threshold=0.0)
        result = strategy.generate_signals(_flat_fs(rv=0.1))
        # Per-bar filter with low_vol_threshold=0.0 and rv=0.1 -> all strength=0.0
        for s in result["strength"].to_list():
            assert s == pytest.approx(0.0)

    def test_pe_below_threshold_does_not_activate_gate(self) -> None:
        # pe_value (0.5) < pe_threshold (0.98) -> gate not active
        strategy = _make_strategy(pe_threshold=0.98, pe_value=0.5, low_vol_threshold=0.0)
        result = strategy.generate_signals(_flat_fs(rv=0.1))
        # Per-bar with low_vol_threshold=0.0: rv >= 0 so strength=0 always
        strengths = result["strength"].to_list()
        assert all(s == pytest.approx(0.0) for s in strengths)

    def test_pe_gate_overrides_low_vol_filter(self) -> None:
        # When PE gate is active, we never evaluate the per-bar filter
        # Even with very low rv, all bars get strength=1.0 from PE gate
        strategy = _make_strategy(pe_threshold=0.9, pe_value=0.95, low_vol_threshold=10.0)
        result = strategy.generate_signals(_flat_fs(rv=0.001))
        for s in result["strength"].to_list():
            assert s == pytest.approx(1.0)


class TestNoTradePerBarLowVolFilter:
    def test_low_rv_bars_get_strength_one(self) -> None:
        # pe not triggered, low_vol_threshold > rv -> strength=1.0
        rv = 0.01
        threshold = 0.05
        strategy = _make_strategy(pe_threshold=0.98, pe_value=0.5, low_vol_threshold=threshold)
        result = strategy.generate_signals(_flat_fs(rv=rv))
        for s in result["strength"].to_list():
            assert s == pytest.approx(1.0)

    def test_high_rv_bars_get_strength_zero(self) -> None:
        # rv >> low_vol_threshold -> strength=0.0
        rv = 1.0
        threshold = 0.05
        strategy = _make_strategy(pe_threshold=0.98, pe_value=0.5, low_vol_threshold=threshold)
        result = strategy.generate_signals(_flat_fs(rv=rv))
        for s in result["strength"].to_list():
            assert s == pytest.approx(0.0)

    def test_rv_exactly_at_threshold_is_not_flagged(self) -> None:
        # rv == low_vol_threshold: condition is < threshold so not flagged
        rv = 0.05
        threshold = 0.05
        strategy = _make_strategy(pe_threshold=0.98, pe_value=0.5, low_vol_threshold=threshold)
        result = strategy.generate_signals(_flat_fs(rv=rv))
        for s in result["strength"].to_list():
            assert s == pytest.approx(0.0)

    def test_mixed_rv_values_produce_mixed_strength(self) -> None:
        n = 10
        rv_values = [0.01, 0.10, 0.01, 0.10, 0.01, 0.10, 0.01, 0.10, 0.01, 0.10]
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        fs = make_feature_set(df, rv_values=rv_values)
        threshold = 0.05
        strategy = _make_strategy(pe_threshold=0.98, pe_value=0.5, low_vol_threshold=threshold)
        result = strategy.generate_signals(fs)
        strengths = result["strength"].to_list()
        for i, rv in enumerate(rv_values):
            expected = 1.0 if rv < threshold else 0.0
            assert strengths[i] == pytest.approx(expected)

    def test_zero_low_vol_threshold_never_flags_bars(self) -> None:
        # low_vol_threshold=0.0: rv < 0 is always False -> strength=0.0 always
        strategy = _make_strategy(pe_threshold=0.98, pe_value=0.5, low_vol_threshold=0.0)
        result = strategy.generate_signals(_flat_fs(rv=0.0001))
        for s in result["strength"].to_list():
            assert s == pytest.approx(0.0)


class TestNoTradeOnSyntheticData:
    def test_pe_gate_on_random_feature_set_all_flat_strength_one(self, flat_feature_set: FeatureSet) -> None:
        strategy = _make_strategy(pe_threshold=0.5, pe_value=0.99)
        result = strategy.generate_signals(flat_feature_set)
        sides = result["side"].to_list()
        strengths = result["strength"].to_list()
        assert all(s == "flat" for s in sides)
        assert all(abs(st - 1.0) < 1e-9 for st in strengths)

    def test_high_vol_feature_set_all_strength_zero_without_pe_gate(self, high_vol_feature_set: FeatureSet) -> None:
        # high rv > low_vol_threshold -> strength=0.0
        strategy = _make_strategy(pe_threshold=0.98, pe_value=0.5, low_vol_threshold=0.05)
        result = strategy.generate_signals(high_vol_feature_set)
        for s in result["strength"].to_list():
            assert s == pytest.approx(0.0)

    def test_low_vol_feature_set_all_strength_one_without_pe_gate(self, low_vol_feature_set: FeatureSet) -> None:
        # low rv < low_vol_threshold -> strength=1.0
        strategy = _make_strategy(pe_threshold=0.98, pe_value=0.5, low_vol_threshold=0.05)
        result = strategy.generate_signals(low_vol_feature_set)
        for s in result["strength"].to_list():
            assert s == pytest.approx(1.0)
