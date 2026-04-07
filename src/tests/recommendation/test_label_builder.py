"""Tests for the LabelBuilder service."""

from __future__ import annotations

from datetime import timedelta

import polars as pl
import pytest

from src.app.recommendation.application.label_builder import LabelBuilder, LabelConfig
from src.tests.recommendation.conftest import (
    ASSET_SYMBOL,
    BASE_TS,
    STRATEGY_NAME,
    make_bars,
    make_signals,
)


# ---------------------------------------------------------------------------
# LabelConfig tests
# ---------------------------------------------------------------------------


class TestLabelConfig:
    """Tests for LabelConfig value object."""

    def test_defaults(self):
        cfg = LabelConfig()
        assert cfg.label_horizon == 7
        assert cfg.commission_bps == 10.0
        assert cfg.min_bars_for_label is None
        assert cfg.effective_min_bars == 7

    def test_custom_min_bars(self):
        cfg = LabelConfig(min_bars_for_label=5, label_horizon=10)
        assert cfg.effective_min_bars == 5

    def test_frozen(self):
        cfg = LabelConfig()
        with pytest.raises(Exception, match="frozen"):  # noqa: B017
            cfg.label_horizon = 10  # type: ignore[misc]

    def test_validation_horizon_positive(self):
        with pytest.raises(Exception, match="greater than 0"):  # noqa: B017
            LabelConfig(label_horizon=0)

    def test_validation_commission_non_negative(self):
        with pytest.raises(Exception, match="greater than or equal to 0"):  # noqa: B017
            LabelConfig(commission_bps=-1.0)


# ---------------------------------------------------------------------------
# LabelBuilder — basic label computation
# ---------------------------------------------------------------------------


class TestLabelBuilderBasic:
    """Tests for correct label computation."""

    def test_long_uptrend_positive_return(self):
        """Long signal in uptrend produces positive strategy return."""
        bars = make_bars(20, price_step=100.0)
        signals = make_signals(20, sides=["long"] * 20)
        config = LabelConfig(label_horizon=5, commission_bps=0.0)
        builder = LabelBuilder(config)

        labels = builder.build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)

        # First bar: close=40000, close at t+5=40500, return = 500/40000 = 0.0125
        assert len(labels) > 0
        first_return = labels.get_column("strategy_return")[0]
        expected = 500.0 / 40_000.0
        assert abs(first_return - expected) < 1e-10

    def test_short_downtrend_positive_return(self):
        """Short signal in downtrend produces positive strategy return."""
        bars = make_bars(20, price_step=-100.0, start_price=50_000.0)
        signals = make_signals(20, sides=["short"] * 20)
        config = LabelConfig(label_horizon=5, commission_bps=0.0)
        builder = LabelBuilder(config)

        labels = builder.build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)

        assert len(labels) > 0
        first_return = labels.get_column("strategy_return")[0]
        # close=50000, close at t+5=49500, raw=-500/50000=-0.01, negated=+0.01
        expected = 500.0 / 50_000.0
        assert abs(first_return - expected) < 1e-10

    def test_long_downtrend_negative_return(self):
        """Long signal in downtrend produces negative strategy return."""
        bars = make_bars(20, price_step=-100.0, start_price=50_000.0)
        signals = make_signals(20, sides=["long"] * 20)
        config = LabelConfig(label_horizon=5, commission_bps=0.0)
        builder = LabelBuilder(config)

        labels = builder.build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)

        assert len(labels) > 0
        first_return = labels.get_column("strategy_return")[0]
        assert first_return < 0.0

    def test_transaction_costs_deducted(self):
        """Transaction costs are subtracted from strategy return."""
        bars = make_bars(20, price_step=100.0)
        signals = make_signals(20, sides=["long"] * 20)

        config_no_cost = LabelConfig(label_horizon=5, commission_bps=0.0)
        config_with_cost = LabelConfig(label_horizon=5, commission_bps=10.0)

        labels_no_cost = LabelBuilder(config_no_cost).build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)
        labels_with_cost = LabelBuilder(config_with_cost).build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)

        no_cost_ret = labels_no_cost.get_column("strategy_return")[0]
        with_cost_ret = labels_with_cost.get_column("strategy_return")[0]

        # Round-trip cost = 2 * 10 / 10000 = 0.002
        expected_diff = 2.0 * 10.0 / 10_000.0
        assert abs((no_cost_ret - with_cost_ret) - expected_diff) < 1e-10

    def test_flat_signals_excluded(self):
        """Flat signals produce no labels."""
        bars = make_bars(20, price_step=100.0)
        signals = make_signals(20, sides=["flat"] * 20)
        builder = LabelBuilder(LabelConfig(label_horizon=5))

        labels = builder.build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)
        assert len(labels) == 0

    def test_mixed_sides(self):
        """Mix of long, short, and flat signals — only directional ones produce labels."""
        bars = make_bars(20, price_step=100.0)
        sides = ["long", "short", "flat", "long", "flat"] + ["long"] * 15
        signals = make_signals(20, sides=sides)
        config = LabelConfig(label_horizon=5, commission_bps=0.0)
        builder = LabelBuilder(config)

        labels = builder.build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)

        # "flat" at indices 2 and 4 should be excluded
        # Last 5 bars can't have labels (horizon=5), so bars 0-14 can have labels
        # Among those, non-flat: 0(long), 1(short), 3(long), 5-14(long) = 13 labels
        label_sides = labels.get_column("side").to_list()
        assert "flat" not in label_sides


# ---------------------------------------------------------------------------
# LabelBuilder — horizon truncation
# ---------------------------------------------------------------------------


class TestLabelBuilderHorizon:
    """Tests for horizon truncation and forward window requirements."""

    def test_insufficient_forward_window_dropped(self):
        """Bars near the end with insufficient forward window produce no labels."""
        n_bars = 10
        horizon = 5
        bars = make_bars(n_bars, price_step=100.0)
        signals = make_signals(n_bars, sides=["long"] * n_bars)
        config = LabelConfig(label_horizon=horizon, commission_bps=0.0)
        builder = LabelBuilder(config)

        labels = builder.build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)

        # With 10 bars and horizon=5, bars at indices 0-4 can have labels (5 labels)
        assert len(labels) == 5

    def test_horizon_equals_bar_count_no_labels(self):
        """When horizon >= bar count, no labels can be computed."""
        bars = make_bars(5, price_step=100.0)
        signals = make_signals(5, sides=["long"] * 5)
        config = LabelConfig(label_horizon=5, commission_bps=0.0)
        builder = LabelBuilder(config)

        labels = builder.build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)
        assert len(labels) == 0

    def test_min_bars_for_label_stricter(self):
        """Setting min_bars_for_label > label_horizon drops more labels."""
        bars = make_bars(20, price_step=100.0)
        signals = make_signals(20, sides=["long"] * 20)

        config_normal = LabelConfig(label_horizon=5, commission_bps=0.0)
        config_strict = LabelConfig(label_horizon=5, min_bars_for_label=10, commission_bps=0.0)

        labels_normal = LabelBuilder(config_normal).build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)
        labels_strict = LabelBuilder(config_strict).build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)

        # Strict requires 10 forward bars, normal requires 5
        assert len(labels_strict) < len(labels_normal)

    def test_single_bar_no_labels(self):
        """Single bar produces no labels (no forward window)."""
        bars = make_bars(1)
        signals = make_signals(1, sides=["long"])
        builder = LabelBuilder(LabelConfig(label_horizon=1))

        labels = builder.build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)
        assert len(labels) == 0


# ---------------------------------------------------------------------------
# LabelBuilder — output schema
# ---------------------------------------------------------------------------


class TestLabelBuilderSchema:
    """Tests for output DataFrame schema and metadata columns."""

    def test_output_columns(self):
        """Output DataFrame has the correct columns."""
        bars = make_bars(20, price_step=100.0)
        signals = make_signals(20, sides=["long"] * 20)
        builder = LabelBuilder(LabelConfig(label_horizon=5))

        labels = builder.build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)

        expected_cols = {"timestamp", "strategy_return", "asset", "strategy", "side", "horizon"}
        assert set(labels.columns) == expected_cols

    def test_asset_and_strategy_metadata(self):
        """Asset and strategy columns are correctly populated."""
        bars = make_bars(20, price_step=100.0)
        signals = make_signals(20, sides=["long"] * 20)
        builder = LabelBuilder(LabelConfig(label_horizon=5))

        labels = builder.build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)

        assets = labels.get_column("asset").unique().to_list()
        strategies = labels.get_column("strategy").unique().to_list()
        horizons = labels.get_column("horizon").unique().to_list()

        assert assets == [ASSET_SYMBOL]
        assert strategies == [STRATEGY_NAME]
        assert horizons == [5]

    def test_empty_result_has_correct_schema(self):
        """Empty result from all-flat signals still has correct columns."""
        bars = make_bars(20)
        signals = make_signals(20, sides=["flat"] * 20)
        builder = LabelBuilder()

        labels = builder.build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)

        expected_cols = {"timestamp", "strategy_return", "asset", "strategy", "side", "horizon"}
        assert set(labels.columns) == expected_cols
        assert len(labels) == 0


# ---------------------------------------------------------------------------
# LabelBuilder — validation
# ---------------------------------------------------------------------------


class TestLabelBuilderValidation:
    """Tests for input validation."""

    def test_empty_bars_raises(self):
        bars = pl.DataFrame(
            {
                "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
                "close": pl.Series([], dtype=pl.Float64),
            }
        )
        signals = make_signals(5)
        builder = LabelBuilder()

        with pytest.raises(ValueError, match="empty"):
            builder.build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)

    def test_empty_signals_raises(self):
        bars = make_bars(10)
        signals = pl.DataFrame(
            {
                "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
                "side": pl.Series([], dtype=pl.String),
            }
        )
        builder = LabelBuilder()

        with pytest.raises(ValueError, match="empty"):
            builder.build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)

    def test_missing_bars_column_raises(self):
        bars = pl.DataFrame(
            {
                "timestamp": [BASE_TS],
                "price": [40_000.0],  # wrong name — should be "close"
            }
        )
        signals = make_signals(1)
        builder = LabelBuilder()

        with pytest.raises(ValueError, match="missing required columns"):
            builder.build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)

    def test_missing_signals_column_raises(self):
        bars = make_bars(10)
        signals = pl.DataFrame(
            {
                "timestamp": [BASE_TS],
                "direction": ["long"],  # wrong name — should be "side"
            }
        )
        builder = LabelBuilder()

        with pytest.raises(ValueError, match="missing required columns"):
            builder.build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)


# ---------------------------------------------------------------------------
# LabelBuilder — no timestamp overlap
# ---------------------------------------------------------------------------


class TestLabelBuilderNoOverlap:
    """Tests for when bars and signals have no timestamp overlap."""

    def test_no_overlap_returns_empty(self):
        """Non-overlapping timestamps produce empty labels."""
        bars = make_bars(10, start_time=BASE_TS)
        different_start = BASE_TS + timedelta(days=365)
        signals = make_signals(10, sides=["long"] * 10, start_time=different_start)
        builder = LabelBuilder(LabelConfig(label_horizon=3))

        labels = builder.build_labels(bars, signals, ASSET_SYMBOL, STRATEGY_NAME)
        assert len(labels) == 0


# ---------------------------------------------------------------------------
# LabelBuilder — default config
# ---------------------------------------------------------------------------


class TestLabelBuilderDefaults:
    """Tests for default configuration."""

    def test_default_config_used(self):
        builder = LabelBuilder()
        assert builder.config.label_horizon == 7
        assert builder.config.commission_bps == 10.0

    def test_custom_config_preserved(self):
        config = LabelConfig(label_horizon=14, commission_bps=5.0)
        builder = LabelBuilder(config)
        assert builder.config.label_horizon == 14
        assert builder.config.commission_bps == 5.0
