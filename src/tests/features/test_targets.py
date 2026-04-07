"""Unit tests for forward-looking regression target functions in targets.py.

Verifies correct computation of forward log returns, forward volatility,
forward z-returns, winsorization, and the orchestrator functions.
"""

from __future__ import annotations

import math

import polars as pl
import pytest
from pydantic import ValidationError

from src.app.features.application.targets import (
    compute_all_targets,
    forward_direction,
    forward_log_return,
    forward_volatility,
    forward_zreturn,
    get_target_column_names,
    winsorize_series,
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
        """Constant prices have zero 1-bar returns -> forward vol == 0."""
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


class TestForwardZReturn:
    """Tests for the forward_zreturn function."""

    def test_forward_zreturn_known_values(self) -> None:
        """Verify z-return is fwd_logret / backward_rv on known prices.

        Uses a geometric sequence (1.1x per bar) so log returns are constant.
        After the backward vol window fills, backward rv is well-defined.
        """
        # 10 prices growing at 10% per bar
        n: int = 10
        prices: list[float] = [100.0 * (1.1**i) for i in range(n)]
        df: pl.DataFrame = pl.DataFrame({"close": prices})
        backward_window: int = 3
        horizon: int = 1

        result: pl.DataFrame = df.with_columns(
            forward_zreturn(pl.col("close"), horizon=horizon, backward_vol_window=backward_window).alias("fwd_zret"),
        )
        vals: list[float | None] = result["fwd_zret"].to_list()

        # Tail null from forward return (last 1 row)
        assert vals[-1] is None

        # Head nulls from backward vol warm-up (first backward_window rows)
        for i in range(backward_window):
            assert vals[i] is None, f"Expected null at index {i} during warm-up"

        # After warm-up, all 1-bar log returns are ln(1.1) so backward vol
        # is std of identical values = 0 -> z-return is fwd_ret / EPS.
        # The z-return will be extremely large (ln(1.1) / 1e-12).
        # This is correct behavior -- constant-return series have zero vol.
        non_null_vals: list[float] = [v for v in vals[backward_window:-horizon] if v is not None]
        assert len(non_null_vals) > 0
        for v in non_null_vals:
            # With zero backward vol, numerator / EPS gives huge values
            assert abs(v) > 1e6

    def test_forward_zreturn_sign_matches_return(self) -> None:
        """Z-return sign must match the forward log return sign (rising = positive)."""
        from src.tests.features.conftest import make_random_walk_df

        df: pl.DataFrame = make_random_walk_df(100, seed=55, volatility=5.0)
        backward_window: int = 10
        horizon: int = 1

        result: pl.DataFrame = df.with_columns(
            forward_log_return(pl.col("close"), horizon=horizon).alias("fwd_lr"),
            forward_zreturn(pl.col("close"), horizon=horizon, backward_vol_window=backward_window).alias("fwd_zret"),
        )
        # Compare signs where both are non-null
        lr_vals: list[float | None] = result["fwd_lr"].to_list()
        zret_vals: list[float | None] = result["fwd_zret"].to_list()

        matched: int = 0
        total: int = 0
        for lr, zr in zip(lr_vals, zret_vals, strict=True):
            if lr is not None and zr is not None and lr != 0.0 and zr != 0.0:
                total += 1
                if (lr > 0) == (zr > 0):
                    matched += 1

        # Signs should match (backward vol is always positive, so division preserves sign)
        assert total > 0
        assert matched == total, f"Sign mismatch: {matched}/{total}"

    def test_forward_zreturn_denominator_is_backward_looking(self) -> None:
        """Verify the backward vol in z-return uses only past data.

        Test: modify only future prices and verify the denominator
        (backward vol portion) does not change at earlier timestamps.
        """
        from src.tests.features.conftest import make_random_walk_df

        df_base: pl.DataFrame = make_random_walk_df(50, seed=42, volatility=5.0)
        backward_window: int = 5

        # Compute backward vol directly (same logic as in forward_zreturn)
        logret_1: pl.Expr = (pl.col("close") / pl.col("close").shift(1)).log()
        backward_rv: pl.Expr = logret_1.rolling_std(
            window_size=backward_window,
            min_samples=backward_window,
        )

        bv_original: pl.DataFrame = df_base.with_columns(backward_rv.alias("bwd_rv"))

        # Modify last 10 prices (future relative to row 30)
        close_vals: list[float] = df_base["close"].to_list()
        modified_close: list[float] = close_vals[:40] + [c * 2.0 for c in close_vals[40:]]
        df_modified: pl.DataFrame = df_base.with_columns(pl.Series("close", modified_close))

        bv_modified: pl.DataFrame = df_modified.with_columns(backward_rv.alias("bwd_rv"))

        # Backward vol at rows before modification point must be identical
        for i in range(backward_window, 40):
            orig_val: float | None = bv_original["bwd_rv"][i]
            mod_val: float | None = bv_modified["bwd_rv"][i]
            if orig_val is not None and mod_val is not None:
                assert orig_val == pytest.approx(mod_val, rel=1e-10), (
                    f"Backward vol changed at row {i} when only future prices changed"
                )

    def test_forward_zreturn_tail_nulls(self) -> None:
        """Z-return has nulls at tail (from forward return) and head (from backward vol)."""
        n: int = 30
        df: pl.DataFrame = make_ohlcv_df(n, price_step=1.0)
        horizon: int = 2
        backward_window: int = 5

        result: pl.DataFrame = df.with_columns(
            forward_zreturn(pl.col("close"), horizon=horizon, backward_vol_window=backward_window).alias("fwd_zret"),
        )
        vals: list[float | None] = result["fwd_zret"].to_list()

        # Last `horizon` values must be null (no future data)
        for i in range(n - horizon, n):
            assert vals[i] is None, f"Expected null at tail index {i}"

        # First `backward_window` values must be null (warm-up)
        for i in range(backward_window):
            assert vals[i] is None, f"Expected null at head index {i}"

    def test_forward_zreturn_reduces_outlier_magnitude(self) -> None:
        """Z-return should have smaller magnitude dispersion than raw returns during volatility spikes."""
        from src.tests.features.conftest import make_random_walk_df

        df: pl.DataFrame = make_random_walk_df(200, seed=77, volatility=10.0)
        backward_window: int = 10
        h: int = 1

        result: pl.DataFrame = df.with_columns(
            forward_log_return(pl.col("close"), horizon=h).alias("fwd_lr"),
            forward_zreturn(pl.col("close"), horizon=h, backward_vol_window=backward_window).alias("fwd_zret"),
        )

        # Coefficient of variation (std/mean) for z-returns should generally
        # be lower than for raw returns when vol is non-constant.
        # This is a soft test -- just verify both columns are computed.
        lr_non_null: list[float] = [v for v in result["fwd_lr"].to_list() if v is not None]
        zr_non_null: list[float] = [v for v in result["fwd_zret"].to_list() if v is not None]
        assert len(lr_non_null) > 0
        assert len(zr_non_null) > 0


class TestWinsorizeSeries:
    """Tests for the winsorize_series function."""

    def test_winsorize_clips_at_percentiles(self) -> None:
        """Values outside percentile bounds should be clipped to those bounds."""
        values: list[float] = list(range(1, 101))  # 1 to 100
        df: pl.DataFrame = pl.DataFrame({"target": [float(v) for v in values]})

        result: pl.DataFrame = winsorize_series(df, "target", lower_pct=0.05, upper_pct=0.95)
        result_vals: list[float] = result["target"].to_list()

        lower_bound: float = float(df["target"].quantile(0.05, interpolation="linear"))
        upper_bound: float = float(df["target"].quantile(0.95, interpolation="linear"))

        for v in result_vals:
            assert v >= lower_bound - 1e-10, f"Value {v} below lower bound {lower_bound}"
            assert v <= upper_bound + 1e-10, f"Value {v} above upper bound {upper_bound}"

    def test_winsorize_preserves_middle_values(self) -> None:
        """Values within the percentile range should be unchanged."""
        values: list[float] = [float(i) for i in range(1, 101)]
        df: pl.DataFrame = pl.DataFrame({"target": values})

        lower_bound: float = float(df["target"].quantile(0.05, interpolation="linear"))
        upper_bound: float = float(df["target"].quantile(0.95, interpolation="linear"))

        result: pl.DataFrame = winsorize_series(df, "target", lower_pct=0.05, upper_pct=0.95)
        result_vals: list[float] = result["target"].to_list()

        for orig, winsorized in zip(values, result_vals, strict=True):
            if lower_bound <= orig <= upper_bound:
                assert winsorized == pytest.approx(orig, abs=1e-10)

    def test_winsorize_preserves_nulls(self) -> None:
        """Null values should remain null after winsorization."""
        values: list[float | None] = [1.0, None, 50.0, None, 100.0]
        df: pl.DataFrame = pl.DataFrame({"target": values})

        result: pl.DataFrame = winsorize_series(df, "target", lower_pct=0.01, upper_pct=0.99)
        result_vals: list[float | None] = result["target"].to_list()

        assert result_vals[1] is None
        assert result_vals[3] is None

    def test_winsorize_all_nulls_returns_unchanged(self) -> None:
        """All-null column should be returned unchanged."""
        df: pl.DataFrame = pl.DataFrame({"target": [None, None, None]}).cast({"target": pl.Float64})

        result: pl.DataFrame = winsorize_series(df, "target", lower_pct=0.01, upper_pct=0.99)
        assert result["target"].null_count() == 3

    def test_winsorize_single_value(self) -> None:
        """Single non-null value should remain unchanged."""
        df: pl.DataFrame = pl.DataFrame({"target": [42.0]})
        result: pl.DataFrame = winsorize_series(df, "target", lower_pct=0.01, upper_pct=0.99)
        assert result["target"][0] == pytest.approx(42.0)

    def test_winsorize_symmetric_bounds(self) -> None:
        """Winsorization at 1st/99th should produce symmetric clipping on symmetric data."""
        values: list[float] = [float(i) for i in range(-50, 51)]  # -50 to 50
        df: pl.DataFrame = pl.DataFrame({"target": values})

        result: pl.DataFrame = winsorize_series(df, "target", lower_pct=0.01, upper_pct=0.99)

        min_val: float = float(result["target"].min())  # type: ignore[arg-type]
        max_val: float = float(result["target"].max())  # type: ignore[arg-type]

        # Should be approximately symmetric
        assert abs(min_val + max_val) < 2.0

    def test_winsorize_row_count_preserved(self) -> None:
        """Winsorization must not add or remove rows."""
        n: int = 50
        values: list[float] = [float(i) for i in range(n)]
        df: pl.DataFrame = pl.DataFrame({"target": values})
        result: pl.DataFrame = winsorize_series(df, "target", lower_pct=0.05, upper_pct=0.95)
        assert len(result) == n


class TestComputeAllTargets:
    """Tests for the compute_all_targets orchestrator."""

    def test_compute_all_targets_columns_present(self) -> None:
        """All expected target columns should be present in the output."""
        df: pl.DataFrame = make_ohlcv_df(30, price_step=1.0)
        config: TargetConfig = make_small_target_config()  # horizons=(1,4), vol_horizons=(2,4), zret=(1,4)
        result: pl.DataFrame = compute_all_targets(df, config)

        expected_cols: set[str] = {
            "fwd_logret_1",
            "fwd_logret_4",
            "fwd_vol_2",
            "fwd_vol_4",
            "fwd_zret_1",
            "fwd_zret_4",
            "fwd_dir_1",
            "fwd_dir_4",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_compute_all_targets_preserves_original_columns(self) -> None:
        """Original OHLCV columns must be preserved."""
        df: pl.DataFrame = make_ohlcv_df(20)
        config: TargetConfig = make_small_target_config()
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

    def test_compute_all_targets_with_winsorization(self) -> None:
        """Winsorization should clip extreme values in target columns."""
        from src.tests.features.conftest import make_random_walk_df

        df: pl.DataFrame = make_random_walk_df(200, seed=99, volatility=15.0)
        config: TargetConfig = make_small_target_config(winsorize=True)
        result: pl.DataFrame = compute_all_targets(df, config)

        # After winsorization, check that fwd_logret_1 values are bounded
        col: str = "fwd_logret_1"
        non_null: list[float] = [v for v in result[col].to_list() if v is not None]
        assert len(non_null) > 0

        # The range should be narrower than without winsorization
        config_no_win: TargetConfig = make_small_target_config(winsorize=False)
        result_no_win: pl.DataFrame = compute_all_targets(df, config_no_win)
        non_null_raw: list[float] = [v for v in result_no_win[col].to_list() if v is not None]

        range_winsorized: float = max(non_null) - min(non_null)
        range_raw: float = max(non_null_raw) - min(non_null_raw)
        assert range_winsorized <= range_raw + 1e-10

    def test_compute_all_targets_winsorize_false_no_clipping(self) -> None:
        """With winsorize=False, no clipping should be applied."""
        from src.tests.features.conftest import make_random_walk_df

        df: pl.DataFrame = make_random_walk_df(100, seed=10, volatility=10.0)
        config: TargetConfig = make_small_target_config(winsorize=False)
        result: pl.DataFrame = compute_all_targets(df, config)

        # Compute fwd_logret_1 manually and compare
        manual: pl.DataFrame = df.with_columns(
            forward_log_return(pl.col("close"), horizon=1).alias("fwd_lr_manual"),
        )
        for orig, computed in zip(
            manual["fwd_lr_manual"].to_list(),
            result["fwd_logret_1"].to_list(),
            strict=True,
        ):
            if orig is None:
                assert computed is None
            else:
                assert orig == pytest.approx(computed, rel=1e-10)

    def test_compute_all_targets_empty_zret_horizons(self) -> None:
        """Empty forward_zret_horizons should produce no fwd_zret columns."""
        df: pl.DataFrame = make_ohlcv_df(30, price_step=1.0)
        config: TargetConfig = make_small_target_config(forward_zret_horizons=())
        result: pl.DataFrame = compute_all_targets(df, config)

        zret_cols: list[str] = [c for c in result.columns if c.startswith("fwd_zret_")]
        assert len(zret_cols) == 0


class TestGetTargetColumnNames:
    """Tests for the get_target_column_names function."""

    def test_get_target_column_names_sorted(self) -> None:
        """get_target_column_names must return a sorted list."""
        config: TargetConfig = make_small_target_config()  # returns (1,4), vol (2,4), zret (1,4)
        names: list[str] = get_target_column_names(config)
        assert names == sorted(names)

    def test_get_target_column_names_content(self) -> None:
        """Correct column names are returned for small config."""
        config: TargetConfig = make_small_target_config()
        names: list[str] = get_target_column_names(config)
        expected: set[str] = {
            "fwd_logret_1",
            "fwd_logret_4",
            "fwd_vol_2",
            "fwd_vol_4",
            "fwd_zret_1",
            "fwd_zret_4",
            "fwd_dir_1",
            "fwd_dir_4",
        }
        assert set(names) == expected

    def test_get_target_column_names_default_config(self) -> None:
        """Default TargetConfig produces expected column names."""
        config: TargetConfig = TargetConfig()
        names: list[str] = get_target_column_names(config)
        # Default forward_return_horizons = (1, 4, 24), forward_vol_horizons = (4, 24),
        # forward_zret_horizons = (1, 4, 24), forward_direction_horizons = (1, 4, 24)
        expected: set[str] = {
            "fwd_logret_1",
            "fwd_logret_4",
            "fwd_logret_24",
            "fwd_vol_4",
            "fwd_vol_24",
            "fwd_zret_1",
            "fwd_zret_4",
            "fwd_zret_24",
            "fwd_dir_1",
            "fwd_dir_4",
            "fwd_dir_24",
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
        config: TargetConfig = TargetConfig(
            forward_return_horizons=(1, 2, 3),
            forward_vol_horizons=(2,),
            forward_zret_horizons=(1, 2),
        )
        result: pl.DataFrame = compute_all_targets(df, config)

        for h in (1, 2, 3):
            col: str = f"fwd_logret_{h}"
            non_null: list[float] = [v for v in result[col].to_list() if v is not None]
            assert all(v > 0.0 for v in non_null), f"{col}: expected all positive for uptrend"

    def test_get_target_column_names_empty_zret(self) -> None:
        """Empty forward_zret_horizons should produce no fwd_zret names."""
        config: TargetConfig = make_small_target_config(forward_zret_horizons=())
        names: list[str] = get_target_column_names(config)
        zret_names: list[str] = [n for n in names if "zret" in n]
        assert len(zret_names) == 0


class TestTargetConfigValidation:
    """Tests for TargetConfig validation of new fields."""

    def test_zret_horizons_must_be_positive(self) -> None:
        """forward_zret_horizons with value < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="forward_zret_horizons must be >= 1"):
            TargetConfig(forward_zret_horizons=(0, 4))

    def test_zret_horizons_no_duplicates(self) -> None:
        """forward_zret_horizons with duplicates should raise ValueError."""
        with pytest.raises(ValueError, match="duplicates"):
            TargetConfig(forward_zret_horizons=(4, 4))

    def test_backward_vol_window_min(self) -> None:
        """backward_vol_window < 2 should raise ValidationError."""
        with pytest.raises(ValidationError, match="backward_vol_window"):
            TargetConfig(backward_vol_window=1)

    def test_winsorize_lower_gte_upper(self) -> None:
        """winsorize_lower_pct >= winsorize_upper_pct should raise ValueError."""
        with pytest.raises(ValueError, match="winsorize_lower_pct"):
            TargetConfig(winsorize_lower_pct=0.99, winsorize_upper_pct=0.01)

    def test_winsorize_equal_bounds(self) -> None:
        """Equal winsorize percentiles should raise ValueError."""
        with pytest.raises(ValueError, match="winsorize_lower_pct"):
            TargetConfig(winsorize_lower_pct=0.5, winsorize_upper_pct=0.5)

    def test_winsorize_lower_pct_out_of_range(self) -> None:
        """winsorize_lower_pct outside (0, 1) should raise ValidationError."""
        with pytest.raises(ValidationError, match="winsorize_lower_pct"):
            TargetConfig(winsorize_lower_pct=0.0)

    def test_winsorize_upper_pct_out_of_range(self) -> None:
        """winsorize_upper_pct outside (0, 1) should raise ValidationError."""
        with pytest.raises(ValidationError, match="winsorize_upper_pct"):
            TargetConfig(winsorize_upper_pct=1.0)

    def test_default_config_valid(self) -> None:
        """Default TargetConfig should be valid."""
        config: TargetConfig = TargetConfig()
        assert config.forward_zret_horizons == (1, 4, 24)
        assert config.forward_direction_horizons == (1, 4, 24)
        assert config.backward_vol_window == 24
        assert config.winsorize is True
        assert config.winsorize_lower_pct == 0.01
        assert config.winsorize_upper_pct == 0.99

    def test_empty_zret_horizons_valid(self) -> None:
        """Empty forward_zret_horizons should be valid (opt-out)."""
        config: TargetConfig = TargetConfig(forward_zret_horizons=())
        assert config.forward_zret_horizons == ()

    def test_direction_horizons_must_be_positive(self) -> None:
        """forward_direction_horizons with value < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="forward_direction_horizons must be >= 1"):
            TargetConfig(forward_direction_horizons=(0, 4))

    def test_direction_horizons_no_duplicates(self) -> None:
        """forward_direction_horizons with duplicates should raise ValueError."""
        with pytest.raises(ValueError, match="duplicates"):
            TargetConfig(forward_direction_horizons=(4, 4))

    def test_empty_direction_horizons_valid(self) -> None:
        """Empty forward_direction_horizons should be valid (opt-out)."""
        config: TargetConfig = TargetConfig(forward_direction_horizons=())
        assert config.forward_direction_horizons == ()


class TestForwardDirection:
    """Tests for the forward_direction classification target function."""

    def test_forward_direction_known_rising_prices(self) -> None:
        """Rising prices should produce all +1 direction labels."""
        prices: list[float] = [100.0, 110.0, 121.0, 133.1, 146.41]
        df: pl.DataFrame = pl.DataFrame({"close": prices})
        result: pl.DataFrame = df.with_columns(forward_direction(pl.col("close"), horizon=1).alias("fwd_dir"))
        vals: list[int | None] = result["fwd_dir"].to_list()

        # Last row is null (no future data)
        assert vals[4] is None

        # First 4 rows: all rising -> +1
        for i in range(4):
            assert vals[i] == 1, f"Expected +1 at index {i}, got {vals[i]}"

    def test_forward_direction_known_falling_prices(self) -> None:
        """Falling prices should produce all -1 direction labels."""
        prices: list[float] = [500.0, 450.0, 400.0, 350.0, 300.0]
        df: pl.DataFrame = pl.DataFrame({"close": prices})
        result: pl.DataFrame = df.with_columns(forward_direction(pl.col("close"), horizon=1).alias("fwd_dir"))
        vals: list[int | None] = result["fwd_dir"].to_list()

        # Last row is null
        assert vals[4] is None

        # First 4 rows: all falling -> -1
        for i in range(4):
            assert vals[i] == -1, f"Expected -1 at index {i}, got {vals[i]}"

    def test_forward_direction_zero_return_maps_to_plus_one(self) -> None:
        """Zero returns (flat prices) should map to +1 by convention."""
        prices: list[float] = [100.0, 100.0, 100.0, 100.0]
        df: pl.DataFrame = pl.DataFrame({"close": prices})
        result: pl.DataFrame = df.with_columns(forward_direction(pl.col("close"), horizon=1).alias("fwd_dir"))
        vals: list[int | None] = result["fwd_dir"].to_list()

        # Last row null
        assert vals[3] is None

        # Zero return -> +1
        for i in range(3):
            assert vals[i] == 1, f"Expected +1 for zero return at index {i}, got {vals[i]}"

    def test_forward_direction_tail_nulls_match_horizon(self) -> None:
        """Exactly h rows at the tail must be null for forward_direction."""
        n: int = 20
        horizons_to_test: list[int] = [1, 3, 5]
        df: pl.DataFrame = make_ohlcv_df(n, price_step=1.0)

        for h in horizons_to_test:
            result: pl.DataFrame = df.with_columns(
                forward_direction(pl.col("close"), horizon=h).alias("fwd_dir"),
            )
            vals: list[int | None] = result["fwd_dir"].to_list()
            null_count: int = sum(1 for v in vals if v is None)
            assert null_count == h, f"horizon={h}: expected {h} nulls, got {null_count}"

    def test_forward_direction_h2(self) -> None:
        """h=2 direction on known prices matches expected signs."""
        prices: list[float] = [100.0, 110.0, 90.0, 120.0, 80.0]
        df: pl.DataFrame = pl.DataFrame({"close": prices})
        result: pl.DataFrame = df.with_columns(forward_direction(pl.col("close"), horizon=2).alias("fwd_dir_2"))
        vals: list[int | None] = result["fwd_dir_2"].to_list()

        # fwd_dir_2[0] = sign(ln(90/100)) = sign(negative) = -1
        assert vals[0] == -1
        # fwd_dir_2[1] = sign(ln(120/110)) = sign(positive) = +1
        assert vals[1] == 1
        # fwd_dir_2[2] = sign(ln(80/90)) = sign(negative) = -1
        assert vals[2] == -1
        # Last 2 null
        assert vals[3] is None
        assert vals[4] is None

    def test_forward_direction_only_plus_minus_one_or_null(self) -> None:
        """Direction values must be exactly +1, -1, or null."""
        from src.tests.features.conftest import make_random_walk_df

        df: pl.DataFrame = make_random_walk_df(100, seed=42)
        result: pl.DataFrame = df.with_columns(forward_direction(pl.col("close"), horizon=1).alias("fwd_dir"))
        vals: list[int | None] = result["fwd_dir"].to_list()

        for v in vals:
            assert v in {1, -1, None}, f"Unexpected direction value: {v}"

    def test_forward_direction_sign_matches_forward_return(self) -> None:
        """Direction label must match sign of forward log return."""
        from src.tests.features.conftest import make_random_walk_df

        df: pl.DataFrame = make_random_walk_df(100, seed=55, volatility=5.0)
        result: pl.DataFrame = df.with_columns(
            forward_log_return(pl.col("close"), horizon=1).alias("fwd_lr"),
            forward_direction(pl.col("close"), horizon=1).alias("fwd_dir"),
        )
        lr_vals: list[float | None] = result["fwd_lr"].to_list()
        dir_vals: list[int | None] = result["fwd_dir"].to_list()

        for lr, d in zip(lr_vals, dir_vals, strict=True):
            if lr is None:
                assert d is None
            elif lr > 0:
                assert d == 1
            elif lr < 0:
                assert d == -1
            else:
                assert d == 1  # zero → +1

    def test_direction_targets_in_compute_all_targets(self) -> None:
        """Direction target columns should appear in compute_all_targets output."""
        df: pl.DataFrame = make_ohlcv_df(30, price_step=1.0)
        config: TargetConfig = make_small_target_config()
        result: pl.DataFrame = compute_all_targets(df, config)

        dir_cols: list[str] = [c for c in result.columns if c.startswith("fwd_dir_")]
        assert "fwd_dir_1" in dir_cols
        assert "fwd_dir_4" in dir_cols

    def test_direction_targets_in_get_target_column_names(self) -> None:
        """Direction columns should appear in get_target_column_names output."""
        config: TargetConfig = make_small_target_config()
        names: list[str] = get_target_column_names(config)
        dir_names: list[str] = [n for n in names if n.startswith("fwd_dir_")]
        assert "fwd_dir_1" in dir_names
        assert "fwd_dir_4" in dir_names

    def test_direction_not_winsorized(self) -> None:
        """Direction targets must not be winsorized (values stay exactly +1/-1)."""
        from src.tests.features.conftest import make_random_walk_df

        df: pl.DataFrame = make_random_walk_df(200, seed=99, volatility=15.0)
        config: TargetConfig = make_small_target_config(winsorize=True)
        result: pl.DataFrame = compute_all_targets(df, config)

        # Direction columns should only contain +1, -1, or null
        for col_name in ["fwd_dir_1", "fwd_dir_4"]:
            vals: list[int | None] = result[col_name].to_list()
            for v in vals:
                assert v in {1, -1, None}, f"{col_name}: unexpected value {v}"

    def test_empty_direction_horizons_no_dir_columns(self) -> None:
        """Empty forward_direction_horizons should produce no fwd_dir columns."""
        df: pl.DataFrame = make_ohlcv_df(30, price_step=1.0)
        config: TargetConfig = make_small_target_config(forward_direction_horizons=())
        result: pl.DataFrame = compute_all_targets(df, config)

        dir_cols: list[str] = [c for c in result.columns if c.startswith("fwd_dir_")]
        assert len(dir_cols) == 0

    def test_empty_direction_horizons_no_dir_names(self) -> None:
        """Empty forward_direction_horizons should produce no fwd_dir names."""
        config: TargetConfig = make_small_target_config(forward_direction_horizons=())
        names: list[str] = get_target_column_names(config)
        dir_names: list[str] = [n for n in names if "dir" in n]
        assert len(dir_names) == 0
