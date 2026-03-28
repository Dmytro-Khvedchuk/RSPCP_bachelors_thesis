"""Unit tests for the MeanReversion strategy."""

from __future__ import annotations


from src.app.features.domain.value_objects import FeatureSet
from src.app.strategy.application.mean_reversion import MeanReversion, MeanReversionConfig
from src.tests.strategy.conftest import make_feature_set, _make_ohlcv_base

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N: int = 60
_BB_WINDOW: int = 10  # small window so warmup is only 10 bars
_BB_STD: float = 2.0
_HURST_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_strategy(
    bb_window: int = _BB_WINDOW,
    bb_num_std: float = _BB_STD,
    hurst_threshold: float = _HURST_THRESHOLD,
) -> MeanReversion:
    return MeanReversion(
        config=MeanReversionConfig(
            bb_window=bb_window,
            bb_num_std=bb_num_std,
            hurst_threshold=hurst_threshold,
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMeanReversionName:
    def test_name_returns_expected_string(self) -> None:
        strategy = _make_strategy()
        assert strategy.name == "mean_reversion"

    def test_name_unchanged_with_custom_config(self) -> None:
        strategy = MeanReversion(config=MeanReversionConfig(bb_window=20, hurst_threshold=0.4))
        assert strategy.name == "mean_reversion"


class TestMeanReversionOutputSchema:
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
            if s is not None:
                assert 0.0 <= s <= 1.0

    def test_side_values_are_valid_strings(self, flat_feature_set: FeatureSet) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(flat_feature_set)
        valid_sides = {"long", "short", "flat"}
        for side in result["side"].to_list():
            if side is not None:
                assert side in valid_sides


class TestMeanReversionHurstGate:
    def test_high_hurst_produces_all_flat(self) -> None:
        n = _N
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        # High hurst blocks all signals
        fs = make_feature_set(df, hurst_values=[0.8] * n)
        strategy = _make_strategy()
        result = strategy.generate_signals(fs)
        non_null_sides = [s for s in result["side"].to_list() if s is not None]
        assert all(s == "flat" for s in non_null_sides)

    def test_hurst_exactly_at_threshold_is_blocked(self) -> None:
        n = _N
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        # hurst == hurst_threshold: condition is < threshold so it should be blocked
        fs = make_feature_set(df, hurst_values=[_HURST_THRESHOLD] * n)
        strategy = _make_strategy(hurst_threshold=_HURST_THRESHOLD)
        result = strategy.generate_signals(fs)
        non_null_sides = [s for s in result["side"].to_list() if s is not None]
        assert all(s == "flat" for s in non_null_sides)

    def test_low_hurst_enables_signals(self, mean_reverting_feature_set: FeatureSet) -> None:
        strategy = _make_strategy()
        result = strategy.generate_signals(mean_reverting_feature_set)
        sides = [s for s in result["side"].to_list() if s is not None]
        directional = sum(1 for s in sides if s in {"long", "short"})
        # With low Hurst and oscillating prices, some bars should be directional
        assert directional > 0


class TestMeanReversionBandLogic:
    def test_price_below_lower_band_with_low_hurst_produces_long(self) -> None:
        # Build a price series where close is much lower than rolling mean
        n = _N
        # All prices at 40000 except the last few which drop dramatically
        closes = [40_000.0] * (n - 5) + [30_000.0] * 5
        highs = [c + 200.0 for c in closes]
        lows = [max(1.0, c - 200.0) for c in closes]
        df = _make_ohlcv_base(n, closes, highs, lows)
        fs = make_feature_set(df, hurst_values=[0.2] * n)
        strategy = _make_strategy(bb_window=_BB_WINDOW, bb_num_std=1.0)
        result = strategy.generate_signals(fs)
        # The last bars where price drops below lower band should produce "long"
        last_sides = result["side"].to_list()[-5:]
        assert any(s == "long" for s in last_sides if s is not None)

    def test_price_above_upper_band_with_low_hurst_produces_short(self) -> None:
        # Build a price series where close is much higher than rolling mean
        n = _N
        closes = [40_000.0] * (n - 5) + [55_000.0] * 5
        highs = [c + 200.0 for c in closes]
        lows = [max(1.0, c - 200.0) for c in closes]
        df = _make_ohlcv_base(n, closes, highs, lows)
        fs = make_feature_set(df, hurst_values=[0.2] * n)
        strategy = _make_strategy(bb_window=_BB_WINDOW, bb_num_std=1.0)
        result = strategy.generate_signals(fs)
        last_sides = result["side"].to_list()[-5:]
        assert any(s == "short" for s in last_sides if s is not None)

    def test_price_within_bands_produces_flat(self) -> None:
        n = _N
        closes = [40_000.0] * n
        highs = [c + 200.0 for c in closes]
        lows = [c - 200.0 for c in closes]
        df = _make_ohlcv_base(n, closes, highs, lows)
        fs = make_feature_set(df, hurst_values=[0.2] * n)
        strategy = _make_strategy(bb_window=_BB_WINDOW, bb_num_std=2.5)
        result = strategy.generate_signals(fs)
        # Constant price is at the midpoint, never outside bands
        non_null_sides = [s for s in result["side"].to_list() if s is not None]
        assert all(s == "flat" for s in non_null_sides)

    def test_no_short_produced_without_hurst_filter(self) -> None:
        n = _N
        closes = [40_000.0] * n
        df = _make_ohlcv_base(n, closes, [c + 200.0 for c in closes], [c - 200.0 for c in closes])
        fs = make_feature_set(df, hurst_values=[0.8] * n)  # Hurst blocks all
        strategy = _make_strategy()
        result = strategy.generate_signals(fs)
        sides = [s for s in result["side"].to_list() if s is not None]
        assert "short" not in sides


class TestMeanReversionOnSyntheticData:
    def test_mean_reverting_data_produces_some_directional_signals(
        self, mean_reverting_feature_set: FeatureSet
    ) -> None:
        strategy = _make_strategy(bb_window=_BB_WINDOW, bb_num_std=0.5)
        result = strategy.generate_signals(mean_reverting_feature_set)
        sides = [s for s in result["side"].to_list() if s is not None]
        directional = sum(1 for s in sides if s in {"long", "short"})
        assert directional > 0

    def test_trending_data_with_high_hurst_produces_mostly_flat(self, trending_up_feature_set: FeatureSet) -> None:
        # trending_up_feature_set has high hurst
        strategy = _make_strategy()
        result = strategy.generate_signals(trending_up_feature_set)
        sides = [s for s in result["side"].to_list() if s is not None]
        flat_count = sides.count("flat")
        assert flat_count > len(sides) // 2
