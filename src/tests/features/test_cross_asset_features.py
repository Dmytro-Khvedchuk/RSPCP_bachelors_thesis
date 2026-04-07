"""Unit tests for BTC-lagged cross-asset features.

Verifies correct computation of BTC-lagged features via asof join + shift,
absence of future leakage, and exclusion of BTC-lagged features from BTC's
own feature set.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, UTC

import polars as pl
import pytest

from src.app.features.application.cross_asset_features import (
    CrossAssetConfig,
    add_btc_lagged_features,
    get_cross_asset_column_names,
    is_btc_asset,
)


_BASE_TS: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_ONE_HOUR: timedelta = timedelta(hours=1)


def _make_ohlcv(
    n: int,
    *,
    price_start: float = 100.0,
    price_step: float = 1.0,
    base_ts: datetime = _BASE_TS,
    interval: timedelta = _ONE_HOUR,
) -> pl.DataFrame:
    """Build a minimal OHLCV DataFrame for cross-asset tests."""
    timestamps: list[datetime] = [base_ts + i * interval for i in range(n)]
    closes: list[float] = [price_start + i * price_step for i in range(n)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
        }
    )


class TestIsBtcAsset:
    """Tests for the is_btc_asset guard function."""

    def test_btcusdt_is_btc(self) -> None:
        assert is_btc_asset("BTCUSDT") is True

    def test_btcusdt_case_insensitive(self) -> None:
        assert is_btc_asset("btcusdt") is True

    def test_ethusdt_is_not_btc(self) -> None:
        assert is_btc_asset("ETHUSDT") is False

    def test_solusdt_is_not_btc(self) -> None:
        assert is_btc_asset("SOLUSDT") is False


class TestGetCrossAssetColumnNames:
    """Tests for the column name listing function."""

    def test_returns_sorted(self) -> None:
        names: list[str] = get_cross_asset_column_names()
        assert names == sorted(names)

    def test_contains_expected_columns(self) -> None:
        names: set[str] = set(get_cross_asset_column_names())
        assert "btc_logret_lag1" in names
        assert "btc_rv_lag1" in names
        assert "btc_direction_lag1" in names

    def test_exactly_three_columns(self) -> None:
        assert len(get_cross_asset_column_names()) == 3


class TestAddBtcLaggedFeatures:
    """Tests for the add_btc_lagged_features service function."""

    def test_output_has_btc_lagged_columns(self) -> None:
        """Output DataFrame should contain all BTC-lagged feature columns."""
        btc_df: pl.DataFrame = _make_ohlcv(50, price_start=40000.0, price_step=100.0)
        alt_df: pl.DataFrame = _make_ohlcv(50, price_start=2000.0, price_step=5.0)

        result: pl.DataFrame = add_btc_lagged_features(alt_df, btc_df)

        expected_cols: set[str] = set(get_cross_asset_column_names())
        assert expected_cols.issubset(set(result.columns))

    def test_preserves_original_columns(self) -> None:
        """All original altcoin columns must be preserved."""
        btc_df: pl.DataFrame = _make_ohlcv(30, price_start=40000.0)
        alt_df: pl.DataFrame = _make_ohlcv(30, price_start=2000.0)

        result: pl.DataFrame = add_btc_lagged_features(alt_df, btc_df)

        for col in alt_df.columns:
            assert col in result.columns

    def test_preserves_row_count(self) -> None:
        """Output should have the same number of rows as the altcoin input."""
        btc_df: pl.DataFrame = _make_ohlcv(40, price_start=40000.0)
        alt_df: pl.DataFrame = _make_ohlcv(40, price_start=2000.0)

        result: pl.DataFrame = add_btc_lagged_features(alt_df, btc_df)
        assert len(result) == len(alt_df)

    def test_first_row_btc_features_are_null(self) -> None:
        """First row should have null BTC-lagged features due to shift(1)."""
        btc_df: pl.DataFrame = _make_ohlcv(30, price_start=40000.0, price_step=100.0)
        alt_df: pl.DataFrame = _make_ohlcv(30, price_start=2000.0, price_step=5.0)

        result: pl.DataFrame = add_btc_lagged_features(alt_df, btc_df)

        for col in get_cross_asset_column_names():
            assert result[col][0] is None, f"{col} at row 0 should be null (shift-by-1)"

    def test_no_future_leakage_shift_by_one(self) -> None:
        """BTC-lagged feature at row t must reflect BTC data from time <= t-1.

        This is the key acceptance criterion: the shift(1) guarantees that
        even with perfectly aligned timestamps, we only use past BTC data.
        """
        n: int = 30
        btc_df: pl.DataFrame = _make_ohlcv(n, price_start=40000.0, price_step=100.0)
        alt_df: pl.DataFrame = _make_ohlcv(n, price_start=2000.0, price_step=5.0)

        result: pl.DataFrame = add_btc_lagged_features(alt_df, btc_df)

        # Manually compute BTC 1-bar log return
        btc_closes: list[float] = btc_df["close"].to_list()

        for i in range(2, n):
            # BTC logret at bar (i-1): ln(close[i-1] / close[i-2])
            expected_logret: float = math.log(btc_closes[i - 1] / btc_closes[i - 2])
            actual_logret: float | None = result["btc_logret_lag1"][i]
            assert actual_logret is not None
            assert actual_logret == pytest.approx(expected_logret, rel=1e-6), (
                f"Row {i}: expected BTC logret from bar {i - 1}, got mismatch"
            )

    def test_btc_direction_lag1_values(self) -> None:
        """btc_direction_lag1 should be +1 for rising BTC, -1 for falling."""
        # BTC: alternating up/down pattern
        n: int = 10
        timestamps: list[datetime] = [_BASE_TS + i * _ONE_HOUR for i in range(n)]
        btc_closes: list[float] = [
            40000.0,
            40100.0,
            40000.0,
            40200.0,
            40100.0,
            40300.0,
            40200.0,
            40400.0,
            40300.0,
            40500.0,
        ]
        btc_df: pl.DataFrame = pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": btc_closes,
                "high": [c + 50 for c in btc_closes],
                "low": [c - 50 for c in btc_closes],
                "close": btc_closes,
                "volume": [1000.0] * n,
            }
        )
        alt_df: pl.DataFrame = _make_ohlcv(n, price_start=2000.0, price_step=5.0)

        result: pl.DataFrame = add_btc_lagged_features(alt_df, btc_df)
        dir_vals: list[int | None] = result["btc_direction_lag1"].to_list()

        # Row 0: null (shift)
        assert dir_vals[0] is None
        # Row 1: null (BTC logret at bar 0 is null — no prior bar)
        assert dir_vals[1] is None
        # Row 2: direction of BTC logret at bar 1 = ln(40100/40000) > 0 → +1
        assert dir_vals[2] == 1
        # Row 3: direction of BTC logret at bar 2 = ln(40000/40100) < 0 → -1
        assert dir_vals[3] == -1

    def test_btc_rv_lag1_nonnegative(self) -> None:
        """btc_rv_lag1 should be non-negative where non-null."""
        btc_df: pl.DataFrame = _make_ohlcv(60, price_start=40000.0, price_step=100.0)
        alt_df: pl.DataFrame = _make_ohlcv(60, price_start=2000.0, price_step=5.0)
        config: CrossAssetConfig = CrossAssetConfig(rv_window=5)

        result: pl.DataFrame = add_btc_lagged_features(alt_df, btc_df, config=config)
        rv_vals: list[float | None] = result["btc_rv_lag1"].to_list()

        non_null: list[float] = [v for v in rv_vals if v is not None]
        assert len(non_null) > 0
        assert all(v >= 0.0 for v in non_null)

    def test_handles_misaligned_timestamps(self) -> None:
        """BTC and altcoin can have different timestamp grids (asof join handles this)."""
        btc_timestamps: list[datetime] = [_BASE_TS + i * timedelta(minutes=30) for i in range(40)]
        btc_closes: list[float] = [40000.0 + i * 10.0 for i in range(40)]
        btc_df: pl.DataFrame = pl.DataFrame(
            {
                "timestamp": btc_timestamps,
                "open": btc_closes,
                "high": [c + 5.0 for c in btc_closes],
                "low": [c - 5.0 for c in btc_closes],
                "close": btc_closes,
                "volume": [1000.0] * 40,
            }
        )

        alt_df: pl.DataFrame = _make_ohlcv(20, price_start=2000.0, price_step=5.0)

        result: pl.DataFrame = add_btc_lagged_features(alt_df, btc_df)

        assert len(result) == 20
        assert set(get_cross_asset_column_names()).issubset(set(result.columns))

    def test_handles_missing_btc_timestamps(self) -> None:
        """If BTC has gaps, asof join should propagate the latest available value."""
        # BTC has bars at hours 0, 1, 2, 5, 6 (gap at 3, 4)
        btc_timestamps: list[datetime] = [_BASE_TS + timedelta(hours=h) for h in [0, 1, 2, 5, 6]]
        btc_closes: list[float] = [40000.0, 40100.0, 40200.0, 40500.0, 40600.0]
        btc_df: pl.DataFrame = pl.DataFrame(
            {
                "timestamp": btc_timestamps,
                "open": btc_closes,
                "high": [c + 50 for c in btc_closes],
                "low": [c - 50 for c in btc_closes],
                "close": btc_closes,
                "volume": [1000.0] * 5,
            }
        )

        # Altcoin has bars at hours 0..6
        alt_df: pl.DataFrame = _make_ohlcv(7, price_start=2000.0, price_step=5.0)

        result: pl.DataFrame = add_btc_lagged_features(alt_df, btc_df)
        assert len(result) == 7
        # Row 0 and 1 should be null (shift + warmup)
        assert result["btc_logret_lag1"][0] is None

    def test_no_intermediate_btc_columns_leak(self) -> None:
        """Internal temporary columns (_btc_logret, etc.) must not remain in output."""
        btc_df: pl.DataFrame = _make_ohlcv(30, price_start=40000.0)
        alt_df: pl.DataFrame = _make_ohlcv(30, price_start=2000.0)

        result: pl.DataFrame = add_btc_lagged_features(alt_df, btc_df)

        internal_cols: list[str] = [c for c in result.columns if c.startswith("_btc_")]
        assert len(internal_cols) == 0, f"Internal columns leaked: {internal_cols}"

    def test_custom_config_rv_window(self) -> None:
        """Custom rv_window should be respected in the computation."""
        btc_df: pl.DataFrame = _make_ohlcv(60, price_start=40000.0, price_step=100.0)
        alt_df: pl.DataFrame = _make_ohlcv(60, price_start=2000.0, price_step=5.0)

        config_5: CrossAssetConfig = CrossAssetConfig(rv_window=5)
        config_10: CrossAssetConfig = CrossAssetConfig(rv_window=10)

        result_5: pl.DataFrame = add_btc_lagged_features(alt_df, btc_df, config=config_5)
        result_10: pl.DataFrame = add_btc_lagged_features(alt_df, btc_df, config=config_10)

        # Different rv_windows should produce different null patterns (10 has more warmup)
        null_count_5: int = result_5["btc_rv_lag1"].null_count()
        null_count_10: int = result_10["btc_rv_lag1"].null_count()
        assert null_count_10 > null_count_5

    def test_default_config_used_when_none(self) -> None:
        """When config is None, default CrossAssetConfig should be used."""
        btc_df: pl.DataFrame = _make_ohlcv(50, price_start=40000.0, price_step=100.0)
        alt_df: pl.DataFrame = _make_ohlcv(50, price_start=2000.0, price_step=5.0)

        # Should not raise
        result: pl.DataFrame = add_btc_lagged_features(alt_df, btc_df, config=None)
        assert set(get_cross_asset_column_names()).issubset(set(result.columns))


class TestCrossAssetConfig:
    """Tests for CrossAssetConfig validation."""

    def test_default_config(self) -> None:
        config: CrossAssetConfig = CrossAssetConfig()
        assert config.rv_window == 24
        assert config.timestamp_col == "timestamp"
        assert config.close_col == "close"

    def test_rv_window_min(self) -> None:
        """rv_window must be >= 2."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="rv_window"):
            CrossAssetConfig(rv_window=1)

    def test_frozen(self) -> None:
        """Config should be immutable."""
        from pydantic import ValidationError

        config: CrossAssetConfig = CrossAssetConfig()
        with pytest.raises(ValidationError, match="frozen"):
            config.rv_window = 10  # type: ignore[misc]
