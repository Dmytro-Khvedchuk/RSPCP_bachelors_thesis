"""Unit tests for forward-looking regression target functions in targets.py.

Verifies correct computation of forward log returns and forward volatility on
known price series, tail-null behaviour, and the orchestrator functions.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from src.app.features.application.targets import (
    compute_all_targets,
    forward_log_return,
    forward_volatility,
    get_target_column_names,
)
from src.app.features.domain.value_objects import TargetConfig

from src.tests.features.conftest import (
    make_ohlcv_df,
    make_small_target_config,
    make_trending_df,
)


_RTOL: float = 1e-6


class TestForwardLogReturn:
    """Tests for the forward_log_return function."""

    def test_forward_log_return_known_prices_h1(self) -> None:
        """h=1 forward return on [100, 110, 121, 133.1] equals ln(1.1) per bar."""
        prices: list[float] = [100.0, 110.0, 121.0, 133.1]
        df: pl.DataFrame = pl.DataFrame({"close": prices})
        result: pl.DataFrame = df.with_columns(forward_log_return(pl.col("close"), horizon=1).alias("fwd_lr"))
        vals: list[float | None] = result["fwd_lr"].to_list()

        # Last row is null (no future data)
        assert vals[3] is None

        # First 3 rows: ln(price_{t+1} / price_t)
        expected_0: float = math.log(110.0 / 100.0)
        expected_1: float = math.log(121.0 / 110.0)
        expected_2: float = math.log(133.1 / 121.0)

        assert vals[0] == pytest.approx(expected_0, rel=_RTOL)
        assert vals[1] == pytest.approx(expected_1, rel=_RTOL)
        assert vals[2] == pytest.approx(expected_2, rel=_RTOL)

    def test_forward_log_return_h2(self) -> None:
        """h=2 forward return on [100, 110, 121, 133.1, 146.41].

        fwd_logret_2[0] = ln(121/100) = 2*ln(1.1)
        fwd_logret_2[1] = ln(133.1/110) = 2*ln(1.1)
        fwd_logret_2[2] = ln(146.41/121) = 2*ln(1.1)
        Last 2 rows are null.
        """
        prices: list[float] = [100.0, 110.0, 121.0, 133.1, 146.41]
        df: pl.DataFrame = pl.DataFrame({"close": prices})
        result: pl.DataFrame = df.with_columns(forward_log_return(pl.col("close"), horizon=2).alias("fwd_lr2"))
        vals: list[float | None] = result["fwd_lr2"].to_list()

        assert vals[3] is None
        assert vals[4] is None

        two_ln_1_1: float = 2.0 * math.log(1.1)
        assert vals[0] == pytest.approx(two_ln_1_1, rel=_RTOL)
        assert vals[1] == pytest.approx(two_ln_1_1, rel=_RTOL)
        assert vals[2] == pytest.approx(two_ln_1_1, rel=_RTOL)

    def test_forward_log_return_tail_nulls_count(self) -> None:
        """Exactly h rows at the tail must be null."""
        n: int = 20
        horizons_to_test: list[int] = [1, 3, 5]
        df: pl.DataFrame = make_ohlcv_df(n, price_step=1.0)

        for h in horizons_to_test:
            result: pl.DataFrame = df.with_columns(forward_log_return(pl.col("close"), horizon=h).alias("fwd_lr"))
            vals: list[float | None] = result["fwd_lr"].to_list()
            null_count: int = sum(1 for v in vals if v is None)
            assert null_count == h, f"horizon={h}: expected {h} nulls, got {null_count}"

    def test_forward_log_return_positive_for_rising_prices(self) -> None:
        """Forward log returns should be positive for monotonically rising prices."""
        df: pl.DataFrame = make_trending_df(20, direction=1.0, step=5.0)
        result: pl.DataFrame = df.with_columns(forward_log_return(pl.col("close"), horizon=1).alias("fwd_lr"))
        vals: list[float | None] = result["fwd_lr"].to_list()
        non_null: list[float] = [v for v in vals if v is not None]
        assert all(v > 0.0 for v in non_null)

    def test_forward_log_return_negative_for_falling_prices(self) -> None:
        """Forward log returns should be negative for monotonically falling prices."""
        df: pl.DataFrame = make_trending_df(20, direction=-1.0, step=5.0, price_start=500.0)
        result: pl.DataFrame = df.with_columns(forward_log_return(pl.col("close"), horizon=1).alias("fwd_lr"))
        vals: list[float | None] = result["fwd_lr"].to_list()
        non_null: list[float] = [v for v in vals if v is not None]
        assert all(v < 0.0 for v in non_null)


class TestForwardVolatility:
    """Tests for the forward_volatility function."""

    def test_forward_volatility_constant_price_is_zero(self) -> None:
        """Constant prices have zero 1-bar returns → forward vol == 0."""
        n: int = 30
        df: pl.DataFrame = make_ohlcv_df(n, price_step=0.0)
        result: pl.DataFrame = df.with_columns(forward_volatility(pl.col("close"), horizon=5).alias("fwd_vol"))
        vals: list[float | None] = result["fwd_vol"].to_list()
        non_null: list[float] = [v for v in vals if v is not None]
        for v in non_null:
            assert v == pytest.approx(0.0, abs=1e-8)

    def test_forward_volatility_tail_nulls_count(self) -> None:
        """Exactly h rows at the tail must be null for forward_volatility."""
        n: int = 30
        horizons: list[int] = [2, 4, 6]
        df: pl.DataFrame = make_ohlcv_df(n, price_step=1.0)

        for h in horizons:
            result: pl.DataFrame = df.with_columns(forward_volatility(pl.col("close"), horizon=h).alias("fwd_vol"))
            vals: list[float | None] = result["fwd_vol"].to_list()
            null_count: int = sum(1 for v in vals if v is None)
            assert null_count == h, f"horizon={h}: expected {h} nulls, got {null_count}"

    def test_forward_volatility_nonnegative(self) -> None:
        """Forward volatility must always be non-negative."""
        from src.tests.features.conftest import make_random_walk_df

        df: pl.DataFrame = make_random_walk_df(50, seed=30)
        result: pl.DataFrame = df.with_columns(forward_volatility(pl.col("close"), horizon=5).alias("fwd_vol"))
        non_null: list[float] = [v for v in result["fwd_vol"].to_list() if v is not None]
        assert all(v >= 0.0 for v in non_null)

    def test_forward_volatility_higher_for_volatile_series(self) -> None:
        """A more volatile series should produce higher forward volatility on average."""
        from src.tests.features.conftest import make_random_walk_df

        df_low: pl.DataFrame = make_random_walk_df(100, seed=31, volatility=0.1)
        df_high: pl.DataFrame = make_random_walk_df(100, seed=31, volatility=20.0)

        res_low: pl.DataFrame = df_low.with_columns(forward_volatility(pl.col("close"), horizon=5).alias("fwd_vol"))
        res_high: pl.DataFrame = df_high.with_columns(forward_volatility(pl.col("close"), horizon=5).alias("fwd_vol"))

        mean_low: float = float(pl.Series([v for v in res_low["fwd_vol"].to_list() if v is not None]).mean())
        mean_high: float = float(pl.Series([v for v in res_high["fwd_vol"].to_list() if v is not None]).mean())
        assert mean_high > mean_low


class TestComputeAllTargets:
    """Tests for the compute_all_targets orchestrator."""

    def test_compute_all_targets_columns_present(self) -> None:
        """All expected target columns should be present in the output."""
        df: pl.DataFrame = make_ohlcv_df(30, price_step=1.0)
        config: TargetConfig = make_small_target_config()  # horizons=(1,4), vol_horizons=(2,4)
        result: pl.DataFrame = compute_all_targets(df, config)

        expected_cols: set[str] = {
            "fwd_logret_1",
            "fwd_logret_4",
            "fwd_vol_2",
            "fwd_vol_4",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_compute_all_targets_preserves_original_columns(self) -> None:
        """Original OHLCV columns must be preserved."""
        df: pl.DataFrame = make_ohlcv_df(20)
        config: TargetConfig = TargetConfig()
        result: pl.DataFrame = compute_all_targets(df, config)
        for col in df.columns:
            assert col in result.columns

    def test_compute_all_targets_row_count_preserved(self) -> None:
        """compute_all_targets must not drop or add rows."""
        n: int = 25
        df: pl.DataFrame = make_ohlcv_df(n, price_step=0.5)
        config: TargetConfig = make_small_target_config()
        result: pl.DataFrame = compute_all_targets(df, config)
        assert len(result) == n


class TestGetTargetColumnNames:
    """Tests for the get_target_column_names function."""

    def test_get_target_column_names_sorted(self) -> None:
        """get_target_column_names must return a sorted list."""
        config: TargetConfig = make_small_target_config()  # returns (1,4), vol (2,4)
        names: list[str] = get_target_column_names(config)
        assert names == sorted(names)

    def test_get_target_column_names_content(self) -> None:
        """Correct column names are returned for small config."""
        config: TargetConfig = make_small_target_config()
        names: list[str] = get_target_column_names(config)
        expected: set[str] = {"fwd_logret_1", "fwd_logret_4", "fwd_vol_2", "fwd_vol_4"}
        assert set(names) == expected

    def test_get_target_column_names_default_config(self) -> None:
        """Default TargetConfig produces expected column names."""
        config: TargetConfig = TargetConfig()
        names: list[str] = get_target_column_names(config)
        # Default forward_return_horizons = (1, 4, 24), forward_vol_horizons = (4, 24)
        expected: set[str] = {
            "fwd_logret_1",
            "fwd_logret_4",
            "fwd_logret_24",
            "fwd_vol_4",
            "fwd_vol_24",
        }
        assert set(names) == expected

    def test_get_target_column_names_matches_compute_columns(self) -> None:
        """Names from get_target_column_names must match what compute_all_targets produces."""
        df: pl.DataFrame = make_ohlcv_df(30, price_step=1.0)
        config: TargetConfig = make_small_target_config()

        computed: pl.DataFrame = compute_all_targets(df, config)
        computed_target_cols: set[str] = {c for c in computed.columns if c.startswith("fwd_")}

        expected_names: set[str] = set(get_target_column_names(config))
        assert expected_names == computed_target_cols

    def test_get_target_column_names_sign_correctness(self) -> None:
        """Rising prices produce positive forward returns for all horizons."""
        df: pl.DataFrame = make_trending_df(30, direction=1.0, step=5.0)
        config: TargetConfig = TargetConfig(forward_return_horizons=(1, 2, 3), forward_vol_horizons=(2,))
        result: pl.DataFrame = compute_all_targets(df, config)

        for h in (1, 2, 3):
            col: str = f"fwd_logret_{h}"
            non_null: list[float] = [v for v in result[col].to_list() if v is not None]
            assert all(v > 0.0 for v in non_null), f"{col}: expected all positive for uptrend"
